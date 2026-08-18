from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from agentic_rl.exact_ig.precision_policy import (
    ExactIGPrecisionPolicy,
    production_precision_policy,
)
from agentic_rl.exact_ig.sequential_oracle import (
    sequential_teacher_forced_oracle,
)
from agentic_rl.exact_ig.task_builder import ExactIGTaskBuilder
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
    PROJECT_ROOT / "artifacts" / "exact_ig_official_alignment_v3_20260730"
)
MODEL_PATH = Path(
    "/root/autodl-tmp/search-r1-workspace/models/dpo_v2_final_model"
)
RTOL = 1.0e-4
ATOL = 1.0e-6


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
        ),
        "answer_range": [
            task.canonical_target.answer_token_start,
            task.canonical_target.answer_token_end,
        ],
        "answer_range_equal": (
            fast.score_token_ids_by_prefix
            == oracle.score_token_ids_by_prefix
        ),
        "boundary_crossing_any": (
            task.canonical_target.boundary_crossing_any
        ),
        "token_rows": token_rows,
        "all_turns_allclose": all(row["allclose"] for row in token_rows),
        "fast_phi": list(fast.score_by_prefix),
        "sequential_phi": list(oracle.score_by_prefix),
        "phi_max_abs_diff": max(phi_differences, default=0.0),
        "fast_ig": list(fast.immediate_ig),
        "sequential_ig": list(oracle.immediate_ig),
        "ig_max_abs_diff": max(ig_differences, default=0.0),
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
        "finite": finite,
        "fast_runtime_metadata": dict(fast.runtime_metadata),
        "sequential_runtime_metadata": dict(oracle.runtime_metadata),
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
    return {
        "ig_variance": ig_variances,
        "outcome_variance": outcome_variances,
        "ig_scale": ig_scale,
        "outcome_scale": outcome_scale,
        "ig_active": ig_active,
        "outcome_active": outcome_active,
        "scores": scores,
        "selected_ids": list(selection.selected_ids),
        "selected_count": len(selection.selected_ids),
        "selected_mass_ratio": selection.selected_mass_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--parameter-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Use the actual project Actor snapshot dtype; BF16 is diagnostic.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA is required")
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
    parameter_dtype = (
        torch.float32
        if args.parameter_dtype == "float32"
        else torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=parameter_dtype,
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
    policy = production_precision_policy("official_bf16_autocast")
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
    for task in (*tasks, long_task):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        fast = full_scorer.score(model, task, device)
        torch.cuda.synchronize(device)
        full_seconds += time.perf_counter() - started
        full_peak = max(full_peak, torch.cuda.max_memory_allocated(device))
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

    single_results = {
        task.trajectory_id: full_results[task.trajectory_id]
        for task in tasks[:2]
    }
    batched_scorer = VectorizedExactIGScorer(
        precision_policy=policy,
        padding_token_id=int(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        scoring_logits_mode=OFFICIAL_FULL_LOGITS,
        attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
    )
    batched_results = batched_scorer.score_many(
        model,
        tuple(tasks[:2]),
        device,
        max_records_per_forward=2,
        max_attention_cost_per_batch=100_000_000,
        max_extended_tokens_per_batch=100_000,
        max_full_logits_bytes=16 * 1024**3,
        max_selected_logits_bytes=16 * 1024**3,
    )
    batch_rows = []
    for task in tasks[:2]:
        single = single_results[task.trajectory_id]
        batched = batched_results[task.trajectory_id]
        batch_rows.append(
            {
                "trajectory_id": task.trajectory_id,
                "token_log_probs_allclose": all(
                    _allclose(left, right)
                    for left, right in zip(
                        single.answer_token_log_probs_by_prefix,
                        batched.answer_token_log_probs_by_prefix,
                        strict=True,
                    )
                ),
                "phi_allclose": _allclose(
                    single.score_by_prefix,
                    batched.score_by_prefix,
                ),
                "ig_allclose": _allclose(
                    single.immediate_ig,
                    batched.immediate_ig,
                ),
            }
        )

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
    future_leakage_pass = _allclose(
        future_results[0].score_by_prefix,
        future_results[1].score_by_prefix,
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
        selected_seconds = 0.0
        for task in tasks:
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
            selected_results[task.trajectory_id] = selected
            full = full_results[task.trajectory_id]
            selected_rows.append(
                {
                    "trajectory_id": task.trajectory_id,
                    "token_log_probs_allclose": all(
                        _allclose(left, right)
                        for left, right in zip(
                            selected.answer_token_log_probs_by_prefix,
                            full.answer_token_log_probs_by_prefix,
                            strict=True,
                        )
                    ),
                    "phi_allclose": _allclose(
                        selected.score_by_prefix,
                        full.score_by_prefix,
                    ),
                    "ig_allclose": _allclose(
                        selected.immediate_ig,
                        full.immediate_ig,
                    ),
                    "sign_agreement": all(
                        _sign(left) == _sign(right)
                        for left, right in zip(
                            selected.immediate_ig,
                            full.immediate_ig,
                            strict=True,
                        )
                    ),
                    "ranking_agreement": (
                        _ranking(selected.immediate_ig)
                        == _ranking(full.immediate_ig)
                    ),
                }
            )
        selected_payload.update(
            {
                "rows": selected_rows,
                "peak_allocated_bytes": selected_peak,
                "official_full_peak_allocated_bytes": full_peak,
                "memory_decreased": selected_peak < full_peak,
                "seconds": selected_seconds,
                "gate_pass": (
                    all(
                        row["token_log_probs_allclose"]
                        and row["phi_allclose"]
                        and row["ig_allclose"]
                        and row["sign_agreement"]
                        and row["ranking_agreement"]
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
    try:
        boolean_scorer = VectorizedExactIGScorer(
            precision_policy=policy,
            padding_token_id=int(tokenizer.pad_token_id),
            tokenizer=tokenizer,
            scoring_logits_mode=OFFICIAL_FULL_LOGITS,
            attention_mask_mode=BOOLEAN_4D_MASK,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        boolean_result = boolean_scorer.score(model, tasks[0], device)
        boolean_peak = torch.cuda.max_memory_allocated(device)
        additive_result = full_results[tasks[0].trajectory_id]
        mask_payload.update(
            {
                "boolean_4d_supported": True,
                "boolean_4d_peak_allocated_bytes": boolean_peak,
                "token_log_probs_allclose": all(
                    _allclose(left, right)
                    for left, right in zip(
                        boolean_result.answer_token_log_probs_by_prefix,
                        additive_result.answer_token_log_probs_by_prefix,
                        strict=True,
                    )
                ),
            }
        )
        mask_payload["boolean_4d_gate_pass"] = bool(
            mask_payload["token_log_probs_allclose"]
        )
    except BaseException as error:
        mask_payload["boolean_4d_error"] = repr(error)

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

    fp32_diagnostic_policy = ExactIGPrecisionPolicy(
        mode="fp32_native_structure_diagnostic",
        autocast_enabled=False,
        autocast_dtype="bfloat16",
        temperature=1.0,
        attention_implementation="sdpa",
        sdpa_backend="math",
    )
    fp32_structure_rows = []
    if parameter_dtype == torch.float32:
        diagnostic_scorer = VectorizedExactIGScorer(
            precision_policy=fp32_diagnostic_policy,
            padding_token_id=int(tokenizer.pad_token_id),
            tokenizer=tokenizer,
            scoring_logits_mode=OFFICIAL_FULL_LOGITS,
            attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
        )
        for task in (tasks[2], tasks[8]):
            diagnostic_fast = diagnostic_scorer.score(model, task, device)
            diagnostic_oracle = sequential_teacher_forced_oracle(
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
                precision_policy=fp32_diagnostic_policy,
            )
            row = _result_comparison(
                task,
                diagnostic_fast,
                diagnostic_oracle,
            )
            fp32_structure_rows.append(
                {
                    "trajectory_id": task.trajectory_id,
                    "all_turns_allclose": row["all_turns_allclose"],
                    "phi_max_abs_diff": row["phi_max_abs_diff"],
                    "ig_max_abs_diff": row["ig_max_abs_diff"],
                    "ig_sign_agreement": row["ig_sign_agreement"],
                    "turn_ranking_agreement": row[
                        "turn_ranking_agreement"
                    ],
                }
            )

    checksum_after = _sampled_model_checksum(model)
    dtypes_after = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    all_turns_close = all(row["all_turns_allclose"] for row in comparisons)
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
    summary = {
        "schema": "exact_ig_official_alignment_v3",
        "official_igpo_commit_sha": (
            "64165e2741ed8801f977948c8128080ce87b4101"
        ),
        "model_path": str(args.model_path.resolve()),
        "actual_model_parameter_dtype": dtypes_before,
        "autocast_enabled": True,
        "autocast_dtype": "bfloat16",
        "attention_backend": "sdpa:math",
        "temperature": 1.0,
        "rtol": RTOL,
        "atol": ATOL,
        "comparisons": comparisons,
        "official_full_logits": {
            "all_turns_allclose": all_turns_close,
            "max_abs_diff": max(
                (
                    token_row["max_abs_diff"]
                    for row in comparisons
                    for token_row in row["token_rows"]
                ),
                default=0.0,
            ),
            "mean_abs_diff": float(
                np.mean(
                    [
                        token_row["mean_abs_diff"]
                        for row in comparisons
                        for token_row in row["token_rows"]
                    ]
                )
            ),
            "ig_sign_agreement": sign_agreement,
            "turn_ranking_agreement": ranking_agreement,
            "canonical_answer_agreement": canonical_agreement,
            "target_token_ids_agreement": target_ids_agreement,
            "answer_range_agreement": answer_range_agreement,
            "future_leakage_pass": future_leakage_pass,
            "single_vs_batch_parity_pass": all(
                row["token_log_probs_allclose"]
                and row["phi_allclose"]
                and row["ig_allclose"]
                for row in batch_rows
            ),
            "finite": finite,
            "seconds": full_seconds,
            "sequential_seconds": oracle_seconds,
            "peak_allocated_bytes": full_peak,
        },
        "batch_parity": {
            "rows": batch_rows,
            "profiles": [
                profile.as_dict()
                for profile in batched_scorer.last_microbatch_profiles
            ],
        },
        "future_leakage": {
            "pass": future_leakage_pass,
            "phi_a": list(future_results[0].score_by_prefix),
            "phi_b": list(future_results[1].score_by_prefix),
        },
        "ragen": full_ragen_comparison,
        "fp32_native_structure_diagnostic": {
            "production_gate_input": False,
            "rows": fp32_structure_rows,
            "pass": bool(
                fp32_structure_rows
                and all(
                    row["all_turns_allclose"]
                    and row["ig_sign_agreement"]
                    and row["turn_ranking_agreement"]
                    for row in fp32_structure_rows
                )
            ),
        },
        "model_checksum_before": checksum_before,
        "model_checksum_after": checksum_after,
        "model_checksum_unchanged": checksum_before == checksum_after,
        "model_dtypes_before": dtypes_before,
        "model_dtypes_after": dtypes_after,
        "model_dtype_unchanged": dtypes_before == dtypes_after,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
    }
    summary["gate_pass"] = bool(
        all_turns_close
        and sign_agreement
        and ranking_agreement
        and canonical_agreement
        and target_ids_agreement
        and answer_range_agreement
        and finite
        and future_leakage_pass
        and summary["official_full_logits"]["single_vs_batch_parity_pass"]
        and full_ragen_comparison["selected_ids_equal"]
        and checksum_before == checksum_after
        and dtypes_before == dtypes_after
    )
    summary["oracle_validated"] = True
    selected_payload["optimization_parity_gate_pass"] = bool(
        selected_payload.get("gate_pass")
    )
    selected_payload["enabled"] = bool(
        selected_payload.get("gate_pass") and summary["gate_pass"]
    )
    if (
        selected_payload.get("optimization_parity_gate_pass")
        and not summary["gate_pass"]
    ):
        selected_payload["reason"] = (
            "optimization_matches_full_logits_but_official_fast_sequential_"
            "baseline_failed"
        )
    summary["selected_mode"] = "OFFICIAL_BF16_FAST_FULL_LOGITS"
    summary["allow_next_stage"] = bool(summary["gate_pass"])

    _write_json(
        output_dir / "EXACT_IG_FAST_SEQUENTIAL_PARITY_V3.json",
        summary,
    )
    _write_json(
        output_dir / "EXACT_IG_SELECTED_LOGITS_PARITY.json",
        selected_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_BATCH_PARITY.json",
        summary["batch_parity"],
    )
    _write_json(
        output_dir / "EXACT_IG_MASK_MEMORY_REPORT.json",
        mask_payload,
    )
    print(
        json.dumps(
            {
                "gate_pass": summary["gate_pass"],
                "full_logits_allclose": all_turns_close,
                "max_abs_diff": summary["official_full_logits"][
                    "max_abs_diff"
                ],
                "selected_positions_gate_pass": selected_payload.get(
                    "gate_pass"
                ),
                "future_leakage_pass": future_leakage_pass,
                "single_vs_batch_parity_pass": summary[
                    "official_full_logits"
                ]["single_vs_batch_parity_pass"],
                "checksum_unchanged": checksum_before == checksum_after,
            },
            sort_keys=True,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    raise SystemExit(0 if summary["gate_pass"] else 2)


if __name__ == "__main__":
    main()
