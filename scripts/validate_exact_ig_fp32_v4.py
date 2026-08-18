from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from agentic_rl.exact_ig.precision_policy import (
    ExactIGPrecisionPolicy,
    exact_ig_precision_context,
    production_precision_policy,
)
from agentic_rl.exact_ig.sequential_oracle import (
    sequential_teacher_forced_oracle,
)
from agentic_rl.exact_ig.task_builder import ExactIGTaskBuilder
from agentic_rl.exact_ig.target_schema import (
    EXACT_IG_VERSION,
    OFFICIAL_IGPO_COMMIT_SHA,
    PRODUCTION_PRECISION_MODE,
)
from agentic_rl.exact_ig.vectorized_scorer import (
    BOOLEAN_4D_MASK,
    OFFICIAL_ADDITIVE_MASK,
    OFFICIAL_FULL_LOGITS,
    SELECTED_POSITIONS,
    VectorizedExactIGScorer,
    selected_positions_capability,
)
from agentic_rl.outcome.workers import score_trajectory_outcome
from agentic_rl.selection.prompt_variance import (
    ig_prompt_variance,
    outcome_prompt_variance,
)
from agentic_rl.selection.top_p import stable_mass_top_p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "exact_ig_fp32_v4_20260730"
)
MODEL_PATH = Path(
    "/root/autodl-tmp/search-r1-workspace/models/dpo_v2_final_model"
)
RTOL = 1.0e-5
ATOL = 2.0e-5
MAX_TOKEN_ABS_DIFF = 2.0e-5
MAX_PHI_ABS_DIFF = 2.0e-5
MAX_IG_ABS_DIFF = 2.0e-5
MAX_TELESCOPING_ERROR = 1.0e-10


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sampled_model_checksum(model: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            tensor = parameter.detach().reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            if tensor.numel():
                indices = torch.linspace(
                    0,
                    tensor.numel() - 1,
                    min(8, tensor.numel()),
                    dtype=torch.float64,
                    device=tensor.device,
                ).to(torch.long)
                digest.update(
                    tensor.index_select(0, indices)
                    .float()
                    .cpu()
                    .numpy()
                    .tobytes()
                )
    return digest.hexdigest()


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer.encode(text, add_special_tokens=False)
    ]


def _build_original(
    tokenizer: Any,
    *,
    question: str,
    search_turns: int,
    variant: int,
    long_prompt_tokens: int = 0,
) -> tuple[list[int], list[int]]:
    padding_text = ""
    if long_prompt_tokens:
        base = " evidence"
        padding_text = base * int(long_prompt_tokens)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": question + padding_text,
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    original = [int(value) for value in prompt]
    endpoints = [len(original)]
    for turn in range(1, search_turns + 1):
        search = (
            f"<think>Search turn {turn}.</think>"
            f"<search>{question} evidence variant {variant} turn {turn}</search>"
        )
        information = (
            f"<information>Retrieved evidence variant {variant}, turn {turn}, "
            f"for {question}.</information>"
        )
        original.extend(_encode(tokenizer, search))
        original.extend(_encode(tokenizer, information))
        endpoints.append(len(original))
    original.extend(
        _encode(
            tokenizer,
            (
                "<think>Future terminal reasoning is outside every saved "
                f"prefix.</think><answer>future-{variant}</answer>"
            ),
        )
    )
    return original, endpoints


def _test_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "prompt_id": "p0",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "turns": 0,
        },
        {
            "prompt_id": "p1",
            "question": "Who discovered penicillin?",
            "answer": "Alexander Fleming",
            "turns": 1,
        },
        {
            "prompt_id": "p2",
            "question": "中国的首都是哪里？",
            "answer": "北京",
            "turns": 2,
        },
        {
            "prompt_id": "p3",
            "question": "Give the punctuated abbreviation for United States.",
            "answer": "U.S.A.",
            "turns": 3,
        },
        {
            "prompt_id": "p4",
            "question": "Which city is also called New York City?",
            "answer": "New York",
            "turns": 4,
        },
        {
            "prompt_id": "p5",
            "question": "Who wrote Hamlet?",
            "answer": "William Shakespeare",
            "turns": 2,
        },
    )


def _allclose(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(
        torch.allclose(
            torch.tensor(tuple(left), dtype=torch.float32),
            torch.tensor(tuple(right), dtype=torch.float32),
            rtol=RTOL,
            atol=ATOL,
        )
    )


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _ranking(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(values)),
            key=lambda index: (-float(values[index]), index),
        )
    )


def _result_pair_metrics(left: Any, right: Any) -> dict[str, Any]:
    token_differences = [
        abs(float(a) - float(b))
        for left_row, right_row in zip(
            left.answer_token_log_probs_by_prefix,
            right.answer_token_log_probs_by_prefix,
            strict=True,
        )
        for a, b in zip(left_row, right_row, strict=True)
    ]
    phi_differences = [
        abs(float(a) - float(b))
        for a, b in zip(left.score_by_prefix, right.score_by_prefix, strict=True)
    ]
    ig_differences = [
        abs(float(a) - float(b))
        for a, b in zip(left.immediate_ig, right.immediate_ig, strict=True)
    ]
    token_maximum = max(token_differences, default=0.0)
    phi_maximum = max(phi_differences, default=0.0)
    ig_maximum = max(ig_differences, default=0.0)
    sign_agreement = all(
        _sign(a) == _sign(b)
        for a, b in zip(left.immediate_ig, right.immediate_ig, strict=True)
    )
    ranking_agreement = (
        _ranking(left.immediate_ig) == _ranking(right.immediate_ig)
    )
    token_allclose = all(
        _allclose(left_row, right_row)
        for left_row, right_row in zip(
            left.answer_token_log_probs_by_prefix,
            right.answer_token_log_probs_by_prefix,
            strict=True,
        )
    )
    return {
        "token_log_probs_allclose": token_allclose,
        "maximum_token_log_prob_abs_diff": token_maximum,
        "phi_allclose": _allclose(left.score_by_prefix, right.score_by_prefix),
        "maximum_phi_abs_diff": phi_maximum,
        "ig_allclose": _allclose(left.immediate_ig, right.immediate_ig),
        "maximum_ig_abs_diff": ig_maximum,
        "ig_sign_agreement": sign_agreement,
        "turn_ranking_agreement": ranking_agreement,
        "gate_pass": (
            token_allclose
            and token_maximum <= MAX_TOKEN_ABS_DIFF
            and _allclose(left.score_by_prefix, right.score_by_prefix)
            and phi_maximum <= MAX_PHI_ABS_DIFF
            and _allclose(left.immediate_ig, right.immediate_ig)
            and ig_maximum <= MAX_IG_ABS_DIFF
            and sign_agreement
            and ranking_agreement
        ),
    }


def _direct_packed_prefix_phi(
    *,
    model: Any,
    task: Any,
    prefix_index: int,
    device: torch.device,
    policy: ExactIGPrecisionPolicy,
    mutate_hidden_regions: bool,
) -> float:
    """Score one packed prefix while optionally mutating every invisible region."""

    input_ids = torch.as_tensor(
        task.input_ids.copy(),
        dtype=torch.long,
        device=device,
    )
    span = task.score_spans[prefix_index]
    if mutate_hidden_regions:
        vocabulary_size = int(model.config.vocab_size)
        prefix_end = int(span.prefix_end_position)
        input_ids[prefix_end : task.original_token_count] = (
            input_ids[prefix_end : task.original_token_count] + 17
        ) % vocabulary_size
        for other_index, (start, length) in enumerate(
            zip(task.segment_starts, task.segment_lengths, strict=True)
        ):
            if other_index == prefix_index:
                continue
            input_ids[start : start + length] = (
                input_ids[start : start + length] + 29
            ) % vocabulary_size
    additive_mask = torch.where(
        torch.as_tensor(task.attention_mask, dtype=torch.bool, device=device),
        torch.tensor(0.0, dtype=torch.float32, device=device),
        torch.tensor(-10000.0, dtype=torch.float32, device=device),
    ).unsqueeze(0).unsqueeze(0)
    positions = torch.as_tensor(
        task.position_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad(), exact_ig_precision_context(model, policy):
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=additive_mask,
            position_ids=positions,
            use_cache=False,
        )
        logits = outputs.logits
        rows = torch.as_tensor(
            span.logit_positions,
            dtype=torch.long,
            device=device,
        )
        targets = torch.as_tensor(
            span.answer_token_ids,
            dtype=torch.long,
            device=device,
        )
        values = torch.nn.functional.log_softmax(
            logits[0].index_select(0, rows),
            dim=-1,
        ).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(values.mean().cpu().item())


def _result_comparison(
    task: Any,
    fast: Any,
    oracle: Any,
) -> dict[str, Any]:
    token_rows: list[dict[str, Any]] = []
    for prefix_index, (left, right) in enumerate(
        zip(
            fast.answer_token_log_probs_by_prefix,
            oracle.answer_token_log_probs_by_prefix,
            strict=True,
        )
    ):
        differences = [
            abs(float(a) - float(b))
            for a, b in zip(left, right, strict=True)
        ]
        token_rows.append(
            {
                "prefix_index": prefix_index,
                "fast": list(left),
                "sequential": list(right),
                "max_abs_diff": max(differences, default=0.0),
                "mean_abs_diff": (
                    float(np.mean(differences)) if differences else 0.0
                ),
                "allclose": _allclose(left, right),
            }
        )
    phi_differences = [
        abs(float(left) - float(right))
        for left, right in zip(
            fast.score_by_prefix,
            oracle.score_by_prefix,
            strict=True,
        )
    ]
    ig_differences = [
        abs(float(left) - float(right))
        for left, right in zip(
            fast.immediate_ig,
            oracle.immediate_ig,
            strict=True,
        )
    ]
    finite = all(
        math.isfinite(float(value))
        for value in (
            *fast.score_by_prefix,
            *fast.immediate_ig,
            *oracle.score_by_prefix,
            *oracle.immediate_ig,
        )
    )
    maximum_token_difference = max(
        (
            row["max_abs_diff"]
            for row in token_rows
        ),
        default=0.0,
    )
    maximum_phi_difference = max(phi_differences, default=0.0)
    maximum_ig_difference = max(ig_differences, default=0.0)
    maximum_telescoping = max(
        abs(float(fast.telescoping_error)),
        abs(float(oracle.telescoping_error)),
    )
    token_parity = (
        all(row["allclose"] for row in token_rows)
        and maximum_token_difference <= MAX_TOKEN_ABS_DIFF
    )
    phi_parity = (
        _allclose(fast.score_by_prefix, oracle.score_by_prefix)
        and maximum_phi_difference <= MAX_PHI_ABS_DIFF
    )
    ig_parity = (
        _allclose(fast.immediate_ig, oracle.immediate_ig)
        and maximum_ig_difference <= MAX_IG_ABS_DIFF
    )
    fast_metadata = dict(fast.runtime_metadata)
    oracle_metadata = dict(oracle.runtime_metadata)
    dtype_gate = all(
        metadata.get("actual_model_parameter_dtype") == "float32"
        and metadata.get("actual_logits_dtype") == "float32"
        and metadata.get("actual_log_probs_dtype") == "float32"
        and metadata.get("autocast_enabled") is False
        and float(metadata.get("temperature", math.nan)) == 1.0
        for metadata in (fast_metadata, oracle_metadata)
    )
    return {
        "prompt_global_id": task.prompt_global_id,
        "trajectory_id": task.trajectory_id,
        "prefix_count": task.prefix_count,
        "search_turn_count": task.prefix_count - 1,
        "canonical_answer": task.canonical_answer,
        "canonical_answer_equal": (
            fast.canonical_answer == oracle.canonical_answer
            == task.canonical_answer
        ),
        "target_token_ids_equal": (
            fast.target_token_ids_hash == task.target_token_ids_hash
            and oracle.target_token_ids == task.canonical_target.token_ids
        ),
        "answer_range": [
            task.canonical_target.answer_token_start,
            task.canonical_target.answer_token_end,
        ],
        "answer_range_equal": (
            oracle.answer_token_range
            == (
                task.canonical_target.answer_token_start,
                task.canonical_target.answer_token_end,
            )
            and fast.score_token_ids_by_prefix
            == oracle.score_token_ids_by_prefix
        ),
        "boundary_crossing_any": (
            task.canonical_target.boundary_crossing_any
        ),
        "token_rows": token_rows,
        "token_parity_pass": token_parity,
        "phi_parity_pass": phi_parity,
        "ig_parity_pass": ig_parity,
        "all_turns_allclose": token_parity and phi_parity and ig_parity,
        "fast_phi": list(fast.score_by_prefix),
        "sequential_phi": list(oracle.score_by_prefix),
        "phi_max_abs_diff": maximum_phi_difference,
        "fast_ig": list(fast.immediate_ig),
        "sequential_ig": list(oracle.immediate_ig),
        "ig_max_abs_diff": maximum_ig_difference,
        "ig_sign_agreement": all(
            _sign(left) == _sign(right)
            for left, right in zip(
                fast.immediate_ig,
                oracle.immediate_ig,
                strict=True,
            )
        ),
        "turn_ranking_agreement": (
            _ranking(fast.immediate_ig)
            == _ranking(oracle.immediate_ig)
        ),
        "fast_telescoping_error": float(fast.telescoping_error),
        "sequential_telescoping_error": float(oracle.telescoping_error),
        "telescoping_max_abs_error": maximum_telescoping,
        "telescoping_pass": maximum_telescoping <= MAX_TELESCOPING_ERROR,
        "finite": finite,
        "dtype_gate_pass": dtype_gate,
        "fast_runtime_metadata": fast_metadata,
        "sequential_runtime_metadata": oracle_metadata,
    }


def _ragen_projection(
    result_by_trajectory: Mapping[str, Any],
    tasks: Sequence[Any],
    outcome_by_trajectory: Mapping[str, float],
) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        grouped[str(task.prompt_global_id)].append(task)
    ig_variances: dict[str, float] = {}
    outcome_variances: dict[str, float] = {}
    for prompt_id, prompt_tasks in grouped.items():
        ig_inputs = [
            {
                index + 1: float(value)
                for index, value in enumerate(
                    result_by_trajectory[task.trajectory_id].immediate_ig
                )
            }
            for task in prompt_tasks
        ]
        ig_variances[prompt_id] = ig_prompt_variance(ig_inputs).aggregate
        outcome_variances[prompt_id] = outcome_prompt_variance(
            [
                outcome_by_trajectory[task.trajectory_id]
                for task in prompt_tasks
            ]
        )
    noise = 1.0e-12
    ig_excess = {
        key: max(value - noise, 0.0)
        for key, value in ig_variances.items()
    }
    outcome_excess = {
        key: max(value - noise, 0.0)
        for key, value in outcome_variances.items()
    }
    ig_positive = [value for value in ig_excess.values() if value > 0]
    outcome_positive = [
        value for value in outcome_excess.values() if value > 0
    ]
    ig_scale = float(np.median(ig_positive)) if ig_positive else None
    outcome_scale = (
        float(np.median(outcome_positive)) if outcome_positive else None
    )
    ig_active = len(ig_positive) >= 4 and ig_scale is not None
    outcome_active = (
        len(outcome_positive) >= 4 and outcome_scale is not None
    )
    denominator = 0.5 * int(ig_active) + 0.5 * int(outcome_active) + 1.0e-12
    scores = {
        prompt_id: (
            (
                0.5 * ig_excess[prompt_id] / (ig_scale + 1.0e-12)
                if ig_active and ig_scale is not None
                else 0.0
            )
            + (
                0.5
                * outcome_excess[prompt_id]
                / (outcome_scale + 1.0e-12)
                if outcome_active and outcome_scale is not None
                else 0.0
            )
        )
        / denominator
        for prompt_id in grouped
    }
    selection = stable_mass_top_p(scores, rho=0.9)
    prompt_ranking = sorted(
        scores,
        key=lambda prompt_id: (-float(scores[prompt_id]), prompt_id),
    )
    return {
        "ig_variance": ig_variances,
        "outcome_variance": outcome_variances,
        "ig_scale": ig_scale,
        "outcome_scale": outcome_scale,
        "ig_active": ig_active,
        "outcome_active": outcome_active,
        "scores": scores,
        "prompt_ranking": prompt_ranking,
        "selected_ids": list(selection.selected_ids),
        "selected_count": len(selection.selected_ids),
        "selected_mass_ratio": selection.selected_mass_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa_math", "eager"),
        default="sdpa_math",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA is required")
    if os.environ.get("EXACT_IG_DETERMINISTIC_DIAGNOSTIC") == "1":
        torch.use_deterministic_algorithms(True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    checksum_before = _sampled_model_checksum(model)
    dtypes_before = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    policy = production_precision_policy("fp32_exact_ig")
    if args.attention_backend == "eager":
        policy = replace(
            policy,
            attention_implementation="eager",
            sdpa_backend=None,
        )
    builder = ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=4096,
        maximum_position_id_exclusive=32768,
    )

    tasks = []
    outcome_by_trajectory: dict[str, float] = {}
    for spec in _test_specs():
        for variant in (0, 1):
            original, endpoints = _build_original(
                tokenizer,
                question=spec["question"],
                search_turns=int(spec["turns"]),
                variant=variant,
            )
            trajectory_id = f"{spec['prompt_id']}-v{variant}"
            task = builder.build(
                prompt_global_id=str(spec["prompt_id"]),
                trajectory_id=trajectory_id,
                full_trajectory_input_ids=original,
                original_attention_mask=[1] * len(original),
                prefix_end_positions=endpoints,
                canonical_answer=str(spec["answer"]),
            )
            tasks.append(task)
            predicted = (
                str(spec["answer"]) if variant == 0 else "definitely wrong"
            )
            outcome_by_trajectory[trajectory_id] = score_trajectory_outcome(
                [
                    (
                        "<think>Final.</think><answer>"
                        + predicted
                        + "</answer>"
                    )
                ],
                [str(spec["answer"])],
            ).task_outcome

    long_original, long_endpoints = _build_original(
        tokenizer,
        question="Long-context canary: who wrote Hamlet?",
        search_turns=1,
        variant=0,
        long_prompt_tokens=900,
    )
    long_task = builder.build(
        prompt_global_id="long",
        trajectory_id="long-v0",
        full_trajectory_input_ids=long_original,
        original_attention_mask=[1] * len(long_original),
        prefix_end_positions=long_endpoints,
        canonical_answer="William Shakespeare",
    )

    full_scorer = VectorizedExactIGScorer(
        precision_policy=policy,
        padding_token_id=int(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        scoring_logits_mode=OFFICIAL_FULL_LOGITS,
        attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
    )
    full_results: dict[str, Any] = {}
    oracle_results: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    full_seconds = 0.0
    oracle_seconds = 0.0
    full_peak = 0
    full_peak_reserved = 0
    all_tasks = (*tasks, long_task)
    for task in all_tasks:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        fast = full_scorer.score(model, task, device)
        torch.cuda.synchronize(device)
        full_seconds += time.perf_counter() - started
        full_peak = max(full_peak, torch.cuda.max_memory_allocated(device))
        full_peak_reserved = max(
            full_peak_reserved,
            torch.cuda.max_memory_reserved(device),
        )
        started = time.perf_counter()
        oracle = sequential_teacher_forced_oracle(
            model=model,
            tokenizer=tokenizer,
            full_trajectory_input_ids=task.input_ids[: task.original_token_count],
            original_attention_mask=task.original_attention_mask,
            original_position_ids=task.original_position_ids,
            prefix_end_positions=task.prefix_end_positions,
            canonical_answer=task.canonical_answer,
            encoded_target=task.canonical_target,
            device=device,
            precision_policy=policy,
        )
        torch.cuda.synchronize(device)
        oracle_seconds += time.perf_counter() - started
        full_results[task.trajectory_id] = fast
        oracle_results[task.trajectory_id] = oracle
        comparisons.append(_result_comparison(task, fast, oracle))

    single_results = dict(full_results)
    batched_scorer = VectorizedExactIGScorer(
        precision_policy=policy,
        padding_token_id=int(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        scoring_logits_mode=OFFICIAL_FULL_LOGITS,
        attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
    )
    batched_results = batched_scorer.score_many(
        model,
        all_tasks,
        device,
        max_records_per_forward=2,
        max_attention_cost_per_batch=100_000_000,
        max_extended_tokens_per_batch=100_000,
        max_full_logits_bytes=16 * 1024**3,
        max_selected_logits_bytes=16 * 1024**3,
    )
    batch_rows = []
    for task in all_tasks:
        single = single_results[task.trajectory_id]
        batched = batched_results[task.trajectory_id]
        row = _result_pair_metrics(single, batched)
        row["trajectory_id"] = task.trajectory_id
        batch_rows.append(row)

    future_original_a, future_endpoints = _build_original(
        tokenizer,
        question="Future leakage canary",
        search_turns=3,
        variant=0,
    )
    future_original_b, _ = _build_original(
        tokenizer,
        question="Future leakage canary",
        search_turns=3,
        variant=1,
    )
    # Keep the entire visible prefix byte-for-byte identical. Both variant
    # trajectories differ only after the no-search prompt boundary.
    fixed_endpoint = [future_endpoints[0]]
    future_tasks = [
        builder.build(
            prompt_global_id="future",
            trajectory_id=f"future-{index}",
            full_trajectory_input_ids=original,
            original_attention_mask=[1] * len(original),
            prefix_end_positions=fixed_endpoint,
            canonical_answer="fixed answer",
        )
        for index, original in enumerate(
            (future_original_a, future_original_b)
        )
    ]
    future_results = [
        full_scorer.score(model, task, device) for task in future_tasks
    ]
    future_oracles = [
        sequential_teacher_forced_oracle(
            model=model,
            tokenizer=tokenizer,
            full_trajectory_input_ids=task.input_ids[
                : task.original_token_count
            ],
            original_attention_mask=task.original_attention_mask,
            original_position_ids=task.original_position_ids,
            prefix_end_positions=task.prefix_end_positions,
            canonical_answer=task.canonical_answer,
            encoded_target=task.canonical_target,
            device=device,
            precision_policy=policy,
        )
        for task in future_tasks
    ]
    future_fast_difference = max(
        abs(float(left) - float(right))
        for left, right in zip(
            future_results[0].score_by_prefix,
            future_results[1].score_by_prefix,
            strict=True,
        )
    )
    future_oracle_difference = max(
        abs(float(left) - float(right))
        for left, right in zip(
            future_oracles[0].score_by_prefix,
            future_oracles[1].score_by_prefix,
            strict=True,
        )
    )
    other_copy_task = tasks[6]
    direct_original = _direct_packed_prefix_phi(
        model=model,
        task=other_copy_task,
        prefix_index=0,
        device=device,
        policy=policy,
        mutate_hidden_regions=False,
    )
    direct_mutated = _direct_packed_prefix_phi(
        model=model,
        task=other_copy_task,
        prefix_index=0,
        device=device,
        policy=policy,
        mutate_hidden_regions=True,
    )
    other_copy_difference = abs(direct_original - direct_mutated)
    future_leakage_pass = bool(
        future_fast_difference <= MAX_TOKEN_ABS_DIFF
        and future_oracle_difference <= MAX_TOKEN_ABS_DIFF
        and other_copy_difference <= MAX_TOKEN_ABS_DIFF
        and all(
            _result_pair_metrics(fast, oracle)["gate_pass"]
            for fast, oracle in zip(
                future_results,
                future_oracles,
                strict=True,
            )
        )
    )

    capability = selected_positions_capability(model)
    selected_payload: dict[str, Any] = {
        "capability": capability,
        "enabled": False,
        "gate_pass": False,
        "reason": "unsupported_signature",
    }
    selected_results: dict[str, Any] = {}
    if capability["supported_by_signature"]:
        selected_scorer = VectorizedExactIGScorer(
            precision_policy=policy,
            padding_token_id=int(tokenizer.pad_token_id),
            tokenizer=tokenizer,
            scoring_logits_mode=SELECTED_POSITIONS,
            attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
        )
        selected_rows = []
        selected_peak = 0
        selected_peak_reserved = 0
        selected_seconds = 0.0
        for task in all_tasks:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            selected = selected_scorer.score(model, task, device)
            torch.cuda.synchronize(device)
            selected_seconds += time.perf_counter() - started
            selected_peak = max(
                selected_peak,
                torch.cuda.max_memory_allocated(device),
            )
            selected_peak_reserved = max(
                selected_peak_reserved,
                torch.cuda.max_memory_reserved(device),
            )
            selected_results[task.trajectory_id] = selected
            full = full_results[task.trajectory_id]
            row = _result_pair_metrics(selected, full)
            row["trajectory_id"] = task.trajectory_id
            selected_rows.append(row)
        selected_payload.update(
            {
                "rows": selected_rows,
                "peak_allocated_bytes": selected_peak,
                "peak_reserved_bytes": selected_peak_reserved,
                "official_full_peak_allocated_bytes": full_peak,
                "official_full_peak_reserved_bytes": full_peak_reserved,
                "memory_decreased": selected_peak < full_peak,
                "seconds": selected_seconds,
                "gate_pass": (
                    all(
                        row["token_log_probs_allclose"]
                        and row["phi_allclose"]
                        and row["ig_allclose"]
                        and row["ig_sign_agreement"]
                        and row["turn_ranking_agreement"]
                        and row["gate_pass"]
                        for row in selected_rows
                    )
                    and selected_peak < full_peak
                ),
            }
        )
        selected_payload["enabled"] = bool(selected_payload["gate_pass"])
        selected_payload["reason"] = (
            "parity_and_memory_gate_pass"
            if selected_payload["gate_pass"]
            else "parity_or_memory_gate_failed"
        )

    mask_payload: dict[str, Any] = {
        "official_additive_peak_allocated_bytes": full_peak,
        "boolean_4d_supported": False,
        "boolean_4d_gate_pass": False,
    }
    boolean_results: dict[str, Any] = {}
    try:
        boolean_scorer = VectorizedExactIGScorer(
            precision_policy=policy,
            padding_token_id=int(tokenizer.pad_token_id),
            tokenizer=tokenizer,
            scoring_logits_mode=OFFICIAL_FULL_LOGITS,
            attention_mask_mode=BOOLEAN_4D_MASK,
        )
        boolean_rows = []
        boolean_peak = 0
        boolean_peak_reserved = 0
        for task in all_tasks:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            boolean_result = boolean_scorer.score(model, task, device)
            boolean_peak = max(
                boolean_peak,
                torch.cuda.max_memory_allocated(device),
            )
            boolean_peak_reserved = max(
                boolean_peak_reserved,
                torch.cuda.max_memory_reserved(device),
            )
            boolean_results[task.trajectory_id] = boolean_result
            additive_result = full_results[task.trajectory_id]
            row = _result_pair_metrics(boolean_result, additive_result)
            row["trajectory_id"] = task.trajectory_id
            boolean_rows.append(row)
        boolean_future = [
            boolean_scorer.score(model, task, device) for task in future_tasks
        ]
        boolean_future_difference = max(
            abs(float(left) - float(right))
            for left, right in zip(
                boolean_future[0].score_by_prefix,
                boolean_future[1].score_by_prefix,
                strict=True,
            )
        )
        mask_payload.update(
            {
                "boolean_4d_supported": True,
                "boolean_4d_peak_allocated_bytes": boolean_peak,
                "boolean_4d_peak_reserved_bytes": boolean_peak_reserved,
                "memory_not_increased": boolean_peak <= full_peak,
                "rows": boolean_rows,
                "future_leakage_max_abs_diff": boolean_future_difference,
                "future_leakage_pass": (
                    boolean_future_difference <= MAX_TOKEN_ABS_DIFF
                ),
            }
        )
        mask_payload["boolean_4d_gate_pass"] = bool(
            all(row["gate_pass"] for row in boolean_rows)
            and mask_payload["future_leakage_pass"]
            and mask_payload["memory_not_increased"]
        )
    except BaseException as error:
        mask_payload["boolean_4d_error"] = repr(error)

    combined_payload: dict[str, Any] = {
        "tested": False,
        "gate_pass": False,
    }
    combined_results: dict[str, Any] = {}
    if selected_payload.get("gate_pass") and mask_payload.get(
        "boolean_4d_gate_pass"
    ):
        combined_payload["tested"] = True
        combined_scorer = VectorizedExactIGScorer(
            precision_policy=policy,
            padding_token_id=int(tokenizer.pad_token_id),
            tokenizer=tokenizer,
            scoring_logits_mode=SELECTED_POSITIONS,
            attention_mask_mode=BOOLEAN_4D_MASK,
        )
        combined_rows = []
        combined_peak = 0
        combined_peak_reserved = 0
        for task in all_tasks:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            combined_result = combined_scorer.score(model, task, device)
            combined_peak = max(
                combined_peak,
                torch.cuda.max_memory_allocated(device),
            )
            combined_peak_reserved = max(
                combined_peak_reserved,
                torch.cuda.max_memory_reserved(device),
            )
            row = _result_pair_metrics(
                combined_result,
                full_results[task.trajectory_id],
            )
            row["trajectory_id"] = task.trajectory_id
            combined_rows.append(row)
            combined_results[task.trajectory_id] = combined_result
        combined_payload.update(
            {
                "rows": combined_rows,
                "peak_allocated_bytes": combined_peak,
                "peak_reserved_bytes": combined_peak_reserved,
                "gate_pass": all(row["gate_pass"] for row in combined_rows),
            }
        )

    oracle_ragen = _ragen_projection(
        oracle_results,
        tasks,
        outcome_by_trajectory,
    )
    full_ragen = _ragen_projection(
        full_results,
        tasks,
        outcome_by_trajectory,
    )
    oracle_ids = set(oracle_ragen["selected_ids"])
    full_ids = set(full_ragen["selected_ids"])
    union = oracle_ids | full_ids
    full_ragen_comparison = {
        "oracle": oracle_ragen,
        "official_fast_full_logits": full_ragen,
        "selected_ids_equal": oracle_ids == full_ids,
        "selected_count_equal": (
            oracle_ragen["selected_count"] == full_ragen["selected_count"]
        ),
        "prompt_ranking_equal": (
            oracle_ragen["prompt_ranking"] == full_ragen["prompt_ranking"]
        ),
        "selected_set_jaccard": (
            len(oracle_ids & full_ids) / len(union) if union else 1.0
        ),
    }
    if selected_results:
        selected_ragen = _ragen_projection(
            selected_results,
            tasks,
            outcome_by_trajectory,
        )
        selected_ids = set(selected_ragen["selected_ids"])
        selected_payload["ragen"] = {
            "projection": selected_ragen,
            "selected_ids_equal_to_full": selected_ids == full_ids,
            "selected_set_jaccard_to_full": (
                len(selected_ids & full_ids) / len(selected_ids | full_ids)
                if selected_ids | full_ids
                else 1.0
            ),
        }
        selected_payload["gate_pass"] = bool(
            selected_payload["gate_pass"]
            and selected_ids == full_ids
        )
        selected_payload["enabled"] = bool(selected_payload["gate_pass"])
    if boolean_results:
        boolean_ragen = _ragen_projection(
            boolean_results,
            tasks,
            outcome_by_trajectory,
        )
        boolean_ids = set(boolean_ragen["selected_ids"])
        mask_payload["ragen"] = {
            "projection": boolean_ragen,
            "selected_ids_equal_to_full": boolean_ids == full_ids,
            "selected_set_jaccard_to_full": (
                len(boolean_ids & full_ids) / len(boolean_ids | full_ids)
                if boolean_ids | full_ids
                else 1.0
            ),
            "prompt_ranking_equal_to_full": (
                boolean_ragen["prompt_ranking"]
                == full_ragen["prompt_ranking"]
            ),
        }
        mask_payload["boolean_4d_gate_pass"] = bool(
            mask_payload["boolean_4d_gate_pass"]
            and boolean_ids == full_ids
            and mask_payload["ragen"]["prompt_ranking_equal_to_full"]
        )
    if combined_results:
        combined_ragen = _ragen_projection(
            combined_results,
            tasks,
            outcome_by_trajectory,
        )
        combined_ids = set(combined_ragen["selected_ids"])
        combined_payload["ragen"] = {
            "projection": combined_ragen,
            "selected_ids_equal_to_full": combined_ids == full_ids,
            "selected_set_jaccard_to_full": (
                len(combined_ids & full_ids) / len(combined_ids | full_ids)
                if combined_ids | full_ids
                else 1.0
            ),
            "prompt_ranking_equal_to_full": (
                combined_ragen["prompt_ranking"]
                == full_ragen["prompt_ranking"]
            ),
        }
        combined_payload["gate_pass"] = bool(
            combined_payload["gate_pass"]
            and combined_ids == full_ids
            and combined_payload["ragen"]["prompt_ranking_equal_to_full"]
        )

    checksum_after = _sampled_model_checksum(model)
    dtypes_after = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    token_parity = all(row["token_parity_pass"] for row in comparisons)
    phi_parity = all(row["phi_parity_pass"] for row in comparisons)
    ig_parity = all(row["ig_parity_pass"] for row in comparisons)
    maximum_token_difference = max(
        (
            token_row["max_abs_diff"]
            for row in comparisons
            for token_row in row["token_rows"]
        ),
        default=0.0,
    )
    maximum_phi_difference = max(
        (row["phi_max_abs_diff"] for row in comparisons),
        default=0.0,
    )
    maximum_ig_difference = max(
        (row["ig_max_abs_diff"] for row in comparisons),
        default=0.0,
    )
    maximum_telescoping = max(
        (row["telescoping_max_abs_error"] for row in comparisons),
        default=0.0,
    )
    sign_agreement = all(row["ig_sign_agreement"] for row in comparisons)
    ranking_agreement = all(
        row["turn_ranking_agreement"] for row in comparisons
    )
    canonical_agreement = all(
        row["canonical_answer_equal"] for row in comparisons
    )
    target_ids_agreement = all(
        row["target_token_ids_equal"] for row in comparisons
    )
    answer_range_agreement = all(
        row["answer_range_equal"] for row in comparisons
    )
    finite = all(row["finite"] for row in comparisons)
    dtype_gate = all(row["dtype_gate_pass"] for row in comparisons)
    telescoping_pass = all(row["telescoping_pass"] for row in comparisons)
    batch_gate = all(row["gate_pass"] for row in batch_rows)
    ragen_gate = bool(
        full_ragen_comparison["selected_ids_equal"]
        and full_ragen_comparison["selected_count_equal"]
        and full_ragen_comparison["prompt_ranking_equal"]
        and full_ragen_comparison["selected_set_jaccard"] == 1.0
    )
    baseline_gate = bool(
        token_parity
        and phi_parity
        and ig_parity
        and maximum_token_difference <= MAX_TOKEN_ABS_DIFF
        and maximum_phi_difference <= MAX_PHI_ABS_DIFF
        and maximum_ig_difference <= MAX_IG_ABS_DIFF
        and sign_agreement
        and ranking_agreement
        and canonical_agreement
        and target_ids_agreement
        and answer_range_agreement
        and finite
        and dtype_gate
        and telescoping_pass
        and maximum_telescoping <= MAX_TELESCOPING_ERROR
        and batch_gate
        and future_leakage_pass
        and ragen_gate
        and checksum_before == checksum_after
        and dtypes_before == dtypes_after == ["float32"]
    )

    selected_gate = bool(selected_payload.get("gate_pass")) and baseline_gate
    boolean_gate = bool(mask_payload.get("boolean_4d_gate_pass")) and baseline_gate
    combined_gate = bool(combined_payload.get("gate_pass")) and baseline_gate
    if selected_gate and boolean_gate and combined_gate:
        production_combination = {
            "precision_mode": PRODUCTION_PRECISION_MODE,
            "scoring_logits_mode": SELECTED_POSITIONS,
            "attention_mask_mode": BOOLEAN_4D_MASK,
            "selection_reason": "priority_1_both_optimizations_passed",
            "peak_allocated_bytes": combined_payload["peak_allocated_bytes"],
            "peak_reserved_bytes": combined_payload["peak_reserved_bytes"],
        }
    elif selected_gate:
        production_combination = {
            "precision_mode": PRODUCTION_PRECISION_MODE,
            "scoring_logits_mode": SELECTED_POSITIONS,
            "attention_mask_mode": OFFICIAL_ADDITIVE_MASK,
            "selection_reason": "priority_2_selected_positions_passed",
            "peak_allocated_bytes": selected_payload["peak_allocated_bytes"],
            "peak_reserved_bytes": selected_payload["peak_reserved_bytes"],
        }
    elif boolean_gate:
        production_combination = {
            "precision_mode": PRODUCTION_PRECISION_MODE,
            "scoring_logits_mode": OFFICIAL_FULL_LOGITS,
            "attention_mask_mode": BOOLEAN_4D_MASK,
            "selection_reason": "priority_3_boolean_mask_passed",
            "peak_allocated_bytes": mask_payload[
                "boolean_4d_peak_allocated_bytes"
            ],
            "peak_reserved_bytes": mask_payload[
                "boolean_4d_peak_reserved_bytes"
            ],
        }
    else:
        production_combination = {
            "precision_mode": PRODUCTION_PRECISION_MODE,
            "scoring_logits_mode": OFFICIAL_FULL_LOGITS,
            "attention_mask_mode": OFFICIAL_ADDITIVE_MASK,
            "selection_reason": "validated_fp32_baseline",
            "peak_allocated_bytes": full_peak,
            "peak_reserved_bytes": full_peak_reserved,
        }

    selected_payload["optimization_parity_gate_pass"] = bool(
        selected_payload.get("gate_pass")
    )
    selected_payload["enabled"] = bool(
        production_combination["scoring_logits_mode"] == SELECTED_POSITIONS
    )
    mask_payload["boolean_4d_enabled"] = bool(
        production_combination["attention_mask_mode"] == BOOLEAN_4D_MASK
    )
    full_logits_payload = {
        "token_parity_pass": token_parity,
        "phi_parity_pass": phi_parity,
        "ig_parity_pass": ig_parity,
        "maximum_token_log_prob_abs_diff": maximum_token_difference,
        "maximum_phi_abs_diff": maximum_phi_difference,
        "maximum_ig_abs_diff": maximum_ig_difference,
        "maximum_telescoping_error": maximum_telescoping,
        "ig_sign_agreement": sign_agreement,
        "turn_ranking_agreement": ranking_agreement,
        "canonical_answer_agreement": canonical_agreement,
        "target_token_ids_agreement": target_ids_agreement,
        "answer_range_agreement": answer_range_agreement,
        "finite": finite,
        "dtype_gate_pass": dtype_gate,
        "autocast_disabled": True,
        "telescoping_pass": telescoping_pass,
        "seconds": full_seconds,
        "sequential_seconds": oracle_seconds,
        "peak_allocated_bytes": full_peak,
        "peak_reserved_bytes": full_peak_reserved,
    }
    batch_payload = {
        "schema": "exact_ig_fp32_batch_parity_v4",
        "rows": batch_rows,
        "profiles": [
            profile.as_dict()
            for profile in batched_scorer.last_microbatch_profiles
        ],
        "gate_pass": batch_gate,
    }
    future_payload = {
        "schema": "exact_ig_fp32_future_leakage_v4",
        "fast_future_region_max_abs_diff": future_fast_difference,
        "sequential_future_region_max_abs_diff": future_oracle_difference,
        "other_gt_copy_and_original_future_max_abs_diff": other_copy_difference,
        "fast_phi_a": list(future_results[0].score_by_prefix),
        "fast_phi_b": list(future_results[1].score_by_prefix),
        "gate_pass": future_leakage_pass,
    }
    full_ragen_comparison["gate_pass"] = ragen_gate
    runtime_payload = {
        "schema": "exact_ig_fp32_runtime_metadata_v4",
        "exact_ig_version": EXACT_IG_VERSION,
        "precision_mode": PRODUCTION_PRECISION_MODE,
        "actual_model_parameter_dtype": dtypes_before[0],
        "actual_logits_dtypes": sorted(
            {
                str(row["fast_runtime_metadata"]["actual_logits_dtype"])
                for row in comparisons
            }
            | {
                str(row["sequential_runtime_metadata"]["actual_logits_dtype"])
                for row in comparisons
            }
        ),
        "actual_log_probs_dtypes": sorted(
            {
                str(row["fast_runtime_metadata"]["actual_log_probs_dtype"])
                for row in comparisons
            }
            | {
                str(row["sequential_runtime_metadata"]["actual_log_probs_dtype"])
                for row in comparisons
            }
        ),
        "autocast_enabled": False,
        "autocast_dtype": None,
        "attention_backend": (
            f"{policy.attention_implementation}:"
            f"{policy.sdpa_backend or 'native'}"
        ),
        "temperature": 1.0,
        "allow_tf32": False,
        "allow_bf16_reduced_precision_reduction": False,
        "allow_fp16_reduced_precision_reduction": False,
        "float32_matmul_precision": "highest",
        "model_checksum_before": checksum_before,
        "model_checksum_after": checksum_after,
        "model_checksum_unchanged": checksum_before == checksum_after,
        "model_dtypes_before": dtypes_before,
        "model_dtypes_after": dtypes_after,
        "model_dtype_unchanged": dtypes_before == dtypes_after,
    }
    summary = {
        "schema": "exact_ig_fp32_production_gate_v4",
        "exact_ig_version": EXACT_IG_VERSION,
        "official_igpo_commit_sha": OFFICIAL_IGPO_COMMIT_SHA,
        "model_path": str(args.model_path.resolve()),
        "precision_mode": PRODUCTION_PRECISION_MODE,
        "rtol": RTOL,
        "atol": ATOL,
        "maximum_token_log_prob_abs_diff_limit": MAX_TOKEN_ABS_DIFF,
        "maximum_phi_abs_diff_limit": MAX_PHI_ABS_DIFF,
        "maximum_ig_abs_diff_limit": MAX_IG_ABS_DIFF,
        "maximum_telescoping_error_limit": MAX_TELESCOPING_ERROR,
        "comparisons": comparisons,
        "official_full_logits": full_logits_payload,
        "batch_parity": batch_payload,
        "future_leakage": future_payload,
        "ragen": full_ragen_comparison,
        "selected_positions": selected_payload,
        "boolean_4d": mask_payload,
        "selected_positions_boolean_4d": combined_payload,
        "runtime_metadata": runtime_payload,
        "production_combination": production_combination,
        "single_gpu_gate_pass": baseline_gate,
        "oracle_validated": baseline_gate,
        "fsdp4": {"gate_pass": False, "status": "PENDING"},
        "budget_tests": {"gate_pass": False, "status": "PENDING"},
        "pending_gates": ["FSDP4", "BUDGET_TESTS", "FINALIZATION"],
        "gate_pass": False,
        "allow_next_stage": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
    }

    _write_json(
        output_dir / "EXACT_IG_FP32_FAST_ORACLE_PARITY_V4.json",
        summary,
    )
    _write_json(
        output_dir / "EXACT_IG_FP32_BATCH_PARITY_V4.json",
        batch_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_FP32_FUTURE_LEAKAGE_V4.json",
        future_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_FP32_SELECTED_POSITIONS_PARITY_V4.json",
        selected_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_FP32_BOOLEAN_MASK_PARITY_V4.json",
        mask_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_FP32_RAGEN_PARITY_V4.json",
        full_ragen_comparison,
    )
    _write_json(
        output_dir / "EXACT_IG_FP32_RUNTIME_METADATA_V4.json",
        runtime_payload,
    )
    print(
        json.dumps(
            {
                "single_gpu_gate_pass": baseline_gate,
                "maximum_token_log_prob_abs_diff": maximum_token_difference,
                "maximum_phi_abs_diff": maximum_phi_difference,
                "maximum_ig_abs_diff": maximum_ig_difference,
                "ig_sign_agreement": sign_agreement,
                "turn_ranking_agreement": ranking_agreement,
                "single_batch_parity": batch_gate,
                "future_leakage": future_leakage_pass,
                "ragen_parity": ragen_gate,
                "selected_positions_enabled": selected_payload["enabled"],
                "boolean_4d_enabled": mask_payload["boolean_4d_enabled"],
                "checksum_unchanged": checksum_before == checksum_after,
            },
            sort_keys=True,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    raise SystemExit(0 if baseline_gate else 2)


if __name__ == "__main__":
    main()
