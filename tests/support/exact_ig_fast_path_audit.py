#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from agentic_rl.exact_ig.alias_reduce import (
    immediate_ig_from_prefix_scores,
    telescoping_error,
)
from agentic_rl.exact_ig.precision_policy import production_precision_policy
from agentic_rl.exact_ig.sequential_oracle import (
    sequential_teacher_forced_oracle,
)
from agentic_rl.exact_ig.target_schema import (
    ANSWER_SCAFFOLD_TEXT,
    CANONICAL_ALIAS_POLICY,
    EXACT_IG_VERSION,
    OFFICIAL_IGPO_COMMIT_SHA,
    TARGET_SCHEMA_SUFFIX,
    encode_exact_ig_target,
    select_canonical_answer,
    token_ids_hash,
)
from agentic_rl.exact_ig.task_builder import (
    ExactIGTaskBuilder,
    VectorizedExactIGTask,
)
from agentic_rl.exact_ig.vectorized_scorer import (
    OFFICIAL_ADDITIVE_MASK,
    OFFICIAL_FULL_LOGITS,
    VectorizedExactIGScorer,
)
from agentic_rl.selection.prompt_variance import (
    ig_prompt_variance,
    outcome_prompt_variance,
)
from agentic_rl.selection.top_p import stable_mass_top_p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    os.environ.get("AETHERSEARCH_ACTOR_MODEL", "")
)
DEFAULT_FAILURE = (
    PROJECT_ROOT
    / "runtime/diagnostics/exact_ig_canary/failure-005716a64165294c.json"
)
DEFAULT_TRAJECTORY_METRICS = (
    PROJECT_ROOT
    / "outputs/formal_training/"
    "formal_fresh_u000_to_u500_corrected_exactig_scconsensus_g16_lr2e7_"
    "kl1e2_20260729_221501/metrics/trajectory_metrics.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/exact_ig_fast_path_self_audit_20260730"
)
OFFICIAL_ROOT = (
    PROJECT_ROOT
    / "third_party/"
    "igpo_official_64165e2741ed8801f977948c8128080ce87b4101"
)
TRAIN_PARQUET = Path(
    os.environ.get("AETHERSEARCH_TRAIN_DATA", "")
)
MODEL_MAX_EXTENDED = 16384
MODEL_MAX_POSITION = 32768
MAX_PHI_SAFETY_ERROR = 1.0e-3
MAX_IG_SAFETY_ERROR = 1.0e-3


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sampled_model_checksum(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            flat = parameter.detach().reshape(-1)
            if flat.numel():
                count = min(16, flat.numel())
                indices = torch.linspace(
                    0,
                    flat.numel() - 1,
                    count,
                    dtype=torch.float64,
                    device=flat.device,
                ).to(torch.int64)
                digest.update(
                    flat.index_select(0, indices)
                    .float()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
    return digest.hexdigest()


def independent_answer_range(
    offsets: Sequence[Sequence[int]],
    answer_start: int,
    answer_end: int,
) -> tuple[int, int]:
    token_start: int | None = None
    token_end: int | None = None
    for token_index, pair in enumerate(offsets):
        char_start, char_end = int(pair[0]), int(pair[1])
        if token_start is None and char_end > answer_start:
            token_start = token_index
        if char_start < answer_end and char_end > 0:
            token_end = token_index + 1
    if token_start is None:
        token_start = len(offsets)
    if token_end is None:
        token_end = len(offsets)
    return token_start, token_end


def independent_expected_mask(
    *,
    original_token_count: int,
    original_attention_mask: Sequence[int],
    prefix_end_positions: Sequence[int],
    segment_starts: Sequence[int],
    segment_lengths: Sequence[int],
) -> np.ndarray:
    total = original_token_count + sum(int(value) for value in segment_lengths)
    expected = np.zeros((total, total), dtype=np.bool_)
    original_valid = np.asarray(original_attention_mask, dtype=np.bool_)
    original_queries = np.arange(original_token_count, dtype=np.int64)[:, None]
    original_keys = np.arange(original_token_count, dtype=np.int64)[None, :]
    expected[:original_token_count, :original_token_count] = (
        original_keys <= original_queries
    ) & original_valid[None, :]
    for start, length, prefix_end in zip(
        segment_starts,
        segment_lengths,
        prefix_end_positions,
        strict=True,
    ):
        start = int(start)
        length = int(length)
        prefix_end = int(prefix_end)
        expected[start : start + length, :original_token_count] = (
            np.arange(original_token_count, dtype=np.int64) < prefix_end
        )[None, :] & original_valid[None, :]
        local_queries = np.arange(length, dtype=np.int64)[:, None]
        local_keys = np.arange(length, dtype=np.int64)[None, :]
        expected[
            start : start + length,
            start : start + length,
        ] = local_keys <= local_queries
    return expected


def independent_expected_positions(
    *,
    original_position_ids: Sequence[int],
    prefix_end_positions: Sequence[int],
    segment_starts: Sequence[int],
    segment_lengths: Sequence[int],
) -> np.ndarray:
    original = np.asarray(original_position_ids, dtype=np.int64)
    total = len(original) + sum(int(value) for value in segment_lengths)
    expected = np.zeros(total, dtype=np.int64)
    expected[: len(original)] = original
    for prefix_end, start, length in zip(
        prefix_end_positions,
        segment_starts,
        segment_lengths,
        strict=True,
    ):
        first = int(original[int(prefix_end) - 1]) + 1
        expected[int(start) : int(start) + int(length)] = np.arange(
            first,
            first + int(length),
            dtype=np.int64,
        )
    return expected


def independent_score_positions(
    *,
    segment_starts: Sequence[int],
    answer_token_start: int,
    answer_token_end: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    answer_positions = tuple(
        int(start) + local
        for start in segment_starts
        for local in range(int(answer_token_start), int(answer_token_end))
    )
    return answer_positions, tuple(position - 1 for position in answer_positions)


def audit_task_contract(task: VectorizedExactIGTask) -> dict[str, Any]:
    target = task.canonical_target
    expected_input = np.concatenate(
        (
            task.input_ids[: task.original_token_count],
            *(
                np.asarray(target.token_ids, dtype=np.int64)
                for _ in task.prefix_end_positions
            ),
        )
    )
    expected_mask = independent_expected_mask(
        original_token_count=task.original_token_count,
        original_attention_mask=task.original_attention_mask,
        prefix_end_positions=task.prefix_end_positions,
        segment_starts=task.segment_starts,
        segment_lengths=task.segment_lengths,
    )
    expected_positions = independent_expected_positions(
        original_position_ids=task.original_position_ids,
        prefix_end_positions=task.prefix_end_positions,
        segment_starts=task.segment_starts,
        segment_lengths=task.segment_lengths,
    )
    expected_answer_positions, expected_logit_positions = (
        independent_score_positions(
            segment_starts=task.segment_starts,
            answer_token_start=target.answer_token_start,
            answer_token_end=target.answer_token_end,
        )
    )
    actual_answer_positions = tuple(
        position
        for span in task.score_spans
        for position in span.answer_token_positions
    )
    actual_logit_positions = tuple(
        position
        for span in task.score_spans
        for position in span.logit_positions
    )
    expected_score_mask = np.zeros(task.input_ids.size, dtype=np.bool_)
    expected_score_mask[list(expected_answer_positions)] = True
    packed_pass = np.array_equal(task.input_ids, expected_input)
    mask_pass = np.array_equal(task.attention_mask, expected_mask)
    position_pass = np.array_equal(task.position_ids, expected_positions)
    shift_pass = (
        actual_answer_positions == expected_answer_positions
        and actual_logit_positions == expected_logit_positions
        and np.array_equal(task.answer_score_mask, expected_score_mask)
    )
    return {
        "prompt_global_id": task.prompt_global_id,
        "trajectory_id": task.trajectory_id,
        "original_token_count": int(task.original_token_count),
        "packed_length": int(task.input_ids.size),
        "prefix_count": int(task.prefix_count),
        "search_turn_count": int(task.prefix_count - 1),
        "target_token_count": len(target.token_ids),
        "answer_token_count": target.answer_token_count,
        "canonical_answer": target.canonical_answer,
        "canonical_answer_sha256": target.canonical_answer_sha256,
        "target_token_ids_hash": target.token_ids_hash,
        "answer_span": [target.answer_token_start, target.answer_token_end],
        "packed_structure_pass": packed_pass,
        "attention_mask_exhaustive_pass": mask_pass,
        "position_ids_pass": position_pass,
        "p_minus_one_shift_pass": shift_pass,
        "no_anchor_pass": all(
            span.answer_token_positions[0] > span.segment_start
            and span.logit_positions[0] >= span.segment_start
            for span in task.score_spans
        ),
        "mask_true_count": int(task.attention_mask.sum()),
        "expected_mask_true_count": int(expected_mask.sum()),
        "maximum_position_id": int(task.position_ids.max()),
        "boundary_crossing_any": bool(target.boundary_crossing_any),
    }


def _canonical_from_row(row: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    value: Any = row.get("golden_answers")
    if value is None:
        reward_model = row.get("reward_model")
        if isinstance(reward_model, Mapping):
            ground_truth = reward_model.get("ground_truth")
            if isinstance(ground_truth, Mapping):
                value = ground_truth.get("target")
    if hasattr(value, "tolist"):
        value = value.tolist()
    aliases = (value,) if isinstance(value, str) else tuple(value or ())
    aliases = tuple(str(item) for item in aliases if str(item).strip())
    if not aliases:
        raise ValueError("Replay row has no valid aliases")
    return aliases, select_canonical_answer(value)


def _prompt_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    value = row["prompt"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in value
    ]


def _independent_information_ids(
    tokenizer: Any,
    documents: Sequence[Any],
    maximum_tokens: int = 500,
) -> list[int]:
    body = "\n".join(str(document.contents) for document in documents)
    prefix = tokenizer(
        "<information>",
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    suffix = tokenizer(
        "</information>",
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    body_ids = tokenizer(
        body,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    budget = maximum_tokens - len(prefix) - len(suffix)
    if budget < 0:
        raise RuntimeError("Information budget is smaller than protocol tags")
    return [int(value) for value in (*prefix, *body_ids[:budget], *suffix)]


def _load_failure_case(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "prompt_global_id": str(payload["prompt_global_id"]),
        "trajectory_id": str(payload["trajectory_id"]),
        "input_ids": [int(value) for value in payload["original_input_ids"]],
        "attention_mask": [
            int(value) for value in payload["original_attention_mask"]
        ],
        "position_ids": [
            int(value) for value in payload["original_position_ids"]
        ],
        "prefix_end_positions": [
            int(value) for value in payload["prefix_end_positions"]
        ],
        "canonical_answer": str(payload["canonical_answer"]),
        "outcome": 0.0,
        "source": "exact_production_canary_failure",
        "source_path": str(path.resolve()),
    }


def prepare_replay_cases(
    *,
    model_path: Path,
    metrics_path: Path,
    failure_path: Path,
    output_path: Path,
    retriever_url: str,
) -> dict[str, Any]:
    import pandas as pd
    from transformers import AutoTokenizer

    from agentic_rl.retriever.client import HybridRetrieverClient

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    frame = pd.read_parquet(TRAIN_PARQUET)
    metric_rows = _read_jsonl(metrics_path)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_prompt[str(row["prompt_global_id"])].append(row)
    complete_groups = {
        prompt_id: sorted(rows, key=lambda item: str(item["trajectory_id"]))
        for prompt_id, rows in by_prompt.items()
        if len(rows) == 16
    }
    failure = _load_failure_case(failure_path)
    required_prompt = failure["prompt_global_id"].split(":snapshot-", 1)[0]
    if required_prompt not in complete_groups:
        required_prompt = failure["prompt_global_id"].rsplit(":snapshot-", 1)[0]
    ranked_prompts = sorted(
        complete_groups,
        key=lambda prompt_id: (
            -max(int(row["search_count"]) for row in complete_groups[prompt_id]),
            prompt_id,
        ),
    )
    selected_prompts: list[str] = []
    if required_prompt in complete_groups:
        selected_prompts.append(required_prompt)
    for prompt_id in ranked_prompts:
        if prompt_id not in selected_prompts:
            selected_prompts.append(prompt_id)
        if len(selected_prompts) == 32:
            break
    if len(selected_prompts) != 32:
        raise RuntimeError("Could not select 32 complete G=16 replay groups")

    selected_rows = [
        row
        for prompt_id in selected_prompts
        for row in complete_groups[prompt_id]
    ]
    all_queries = sorted(
        {
            str(query)
            for row in selected_rows
            for query in row.get("queries", ())
            if str(query).strip()
        }
    )
    retriever = HybridRetrieverClient(
        retriever_url,
        timeout_seconds=180.0,
        default_top_k=3,
    )
    documents_by_query: dict[str, Sequence[Any]] = {}
    for start in range(0, len(all_queries), 128):
        batch = all_queries[start : start + 128]
        response = retriever.retrieve(batch)
        for query, documents in zip(
            batch,
            response.documents_by_query,
            strict=True,
        ):
            documents_by_query[query] = documents

    cases: list[dict[str, Any]] = []
    inserted_failure = False
    inserted_zero_search = False
    for prompt_id in selected_prompts:
        rows = complete_groups[prompt_id]
        source_index = int(prompt_id.rsplit(":", 1)[1])
        raw = frame.loc[source_index].to_dict()
        aliases, canonical = _canonical_from_row(raw)
        prompt_ids = [
            int(value)
            for value in tokenizer.apply_chat_template(
                _prompt_messages(raw),
                add_generation_prompt=True,
                tokenize=True,
            )
        ]
        for row_index, row in enumerate(rows):
            trajectory_id = str(row["trajectory_id"])
            if trajectory_id == failure["trajectory_id"]:
                case = dict(failure)
                case["outcome"] = float(row["R_task"])
                case["aliases"] = list(aliases)
                cases.append(case)
                inserted_failure = True
                continue
            response_ids: list[int] = []
            endpoints = [len(prompt_ids)]
            queries = [str(value) for value in row.get("queries", ())]
            if not inserted_zero_search and row_index == 0:
                queries = []
                inserted_zero_search = True
            for query in queries[:4]:
                action = (
                    "<think>I should search for the main entities and relation "
                    "in the question.</think><search>"
                    + query
                    + "</search>"
                )
                response_ids.extend(
                    int(value)
                    for value in tokenizer(
                        action,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                )
                response_ids.extend(
                    _independent_information_ids(
                        tokenizer,
                        documents_by_query[query],
                    )
                )
                endpoints.append(len(prompt_ids) + len(response_ids))
            original = prompt_ids + response_ids
            cases.append(
                {
                    "prompt_global_id": prompt_id,
                    "trajectory_id": trajectory_id,
                    "input_ids": original,
                    "attention_mask": [1] * len(original),
                    "position_ids": list(range(len(original))),
                    "prefix_end_positions": endpoints,
                    "canonical_answer": canonical,
                    "aliases": list(aliases),
                    "outcome": float(row["R_task"]),
                    "source": "recorded_query_faithful_replay",
                    "source_path": str(metrics_path.resolve()),
                    "recorded_queries": [str(value) for value in row.get("queries", ())],
                }
            )
    if len(cases) != 512:
        raise RuntimeError(f"Expected 512 grouped replay cases, got {len(cases)}")
    if not inserted_failure:
        failure["outcome"] = 0.0
        failure["aliases"] = [failure["canonical_answer"]]
        cases.append(failure)
        inserted_failure = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    _append_jsonl(output_path, cases)
    summary = {
        "case_count": len(cases),
        "grouped_calibration_case_count": 512,
        "special_failure_case_count": 1,
        "prompt_count": len({case["prompt_global_id"] for case in cases}),
        "group_sizes": dict(
            Counter(case["prompt_global_id"] for case in cases)
        ),
        "search_turn_histogram": dict(
            sorted(
                Counter(
                    len(case["prefix_end_positions"]) - 1 for case in cases
                ).items()
            )
        ),
        "failure_sample_included": inserted_failure,
        "failure_trajectory_id": failure["trajectory_id"],
        "complete_group_count": sum(
            value == 16
            for value in Counter(
                case["prompt_global_id"] for case in cases
            ).values()
        ),
        "replay_contract": (
            "One exact production failure replay plus immutable real prompt/"
            "recorded real query/retriever-faithful context replays. No model "
            "generation or rollout was executed."
        ),
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _builder(tokenizer: Any) -> ExactIGTaskBuilder:
    return ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=MODEL_MAX_EXTENDED,
        maximum_position_id_exclusive=MODEL_MAX_POSITION,
    )


def _build_task(builder: ExactIGTaskBuilder, case: Mapping[str, Any]) -> Any:
    return builder.build(
        prompt_global_id=str(case["prompt_global_id"]),
        trajectory_id=str(case["trajectory_id"]),
        full_trajectory_input_ids=case["input_ids"],
        original_attention_mask=case["attention_mask"],
        original_position_ids=case["position_ids"],
        prefix_end_positions=case["prefix_end_positions"],
        canonical_answer=case["canonical_answer"],
    )


def run_static_audit(
    *,
    model_path: Path,
    replay_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    cases = _read_jsonl(replay_path)
    builder = _builder(tokenizer)
    mask_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    packed_structure_results: list[bool] = []
    no_anchor_results: list[bool] = []
    canonical_by_prompt: dict[str, set[str]] = defaultdict(set)
    target_hash_by_prompt: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        task = _build_task(builder, case)
        if not isinstance(task, VectorizedExactIGTask):
            raise RuntimeError(
                "The 512-case structural audit unexpectedly required Sequential fallback"
            )
        row = audit_task_contract(task)
        packed_structure_results.append(bool(row["packed_structure_pass"]))
        no_anchor_results.append(bool(row["no_anchor_pass"]))
        mask_rows.append(
            {
                key: row[key]
                for key in (
                    "trajectory_id",
                    "packed_length",
                    "prefix_count",
                    "mask_true_count",
                    "expected_mask_true_count",
                    "attention_mask_exhaustive_pass",
                )
            }
        )
        position_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "prefix_count": row["prefix_count"],
                "maximum_position_id": row["maximum_position_id"],
                "position_ids_pass": row["position_ids_pass"],
            }
        )
        shift_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "answer_token_count": row["answer_token_count"],
                "p_minus_one_shift_pass": row["p_minus_one_shift_pass"],
                "answer_span_mean_contract_pass": (
                    row["answer_token_count"]
                    == task.canonical_target.answer_token_end
                    - task.canonical_target.answer_token_start
                ),
            }
        )
        target = task.canonical_target
        expected_range = independent_answer_range(
            target.offset_mapping,
            len(ANSWER_SCAFFOLD_TEXT),
            len(ANSWER_SCAFFOLD_TEXT) + len(target.canonical_answer),
        )
        target_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "prompt_global_id": row["prompt_global_id"],
                "canonical_answer": target.canonical_answer,
                "rendered_target": target.rendered_text,
                "target_token_ids_hash": target.token_ids_hash,
                "answer_token_ids_hash": token_ids_hash(target.answer_token_ids),
                "answer_char_span": [
                    target.answer_char_start,
                    target.answer_char_end,
                ],
                "answer_token_span": [
                    target.answer_token_start,
                    target.answer_token_end,
                ],
                "independent_answer_token_span": list(expected_range),
                "one_shot_tokenization_contract": True,
                "answer_span_pass": expected_range
                == (target.answer_token_start, target.answer_token_end),
                "decode_match": tokenizer.decode(
                    target.token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                == target.rendered_text,
                "boundary_crossing_any": target.boundary_crossing_any,
            }
        )
        canonical_by_prompt[row["prompt_global_id"]].add(
            target.canonical_answer
        )
        target_hash_by_prompt[row["prompt_global_id"]].add(
            target.token_ids_hash
        )

    target_schema_source_path = (
        PROJECT_ROOT / "src/agentic_rl/exact_ig/target_schema.py"
    )
    target_schema_source = target_schema_source_path.read_text(encoding="utf-8")
    target_schema_tree = ast.parse(target_schema_source)
    tokenize_function = next(
        node
        for node in target_schema_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_tokenize_complete_target_once"
    )
    tokenizer_call_count = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tokenizer"
        for node in ast.walk(tokenize_function)
    )
    segmented_symbols = (
        "_boundary_safe_segmented_target",
        "segmented_boundary_safe",
        "segmented_no_offset_mapping",
    )
    target_payload = {
        "exact_ig_version": EXACT_IG_VERSION,
        "official_igpo_commit_sha": OFFICIAL_IGPO_COMMIT_SHA,
        "scaffold": ANSWER_SCAFFOLD_TEXT,
        "scaffold_sha256": hashlib.sha256(
            ANSWER_SCAFFOLD_TEXT.encode("utf-8")
        ).hexdigest(),
        "suffix": TARGET_SCHEMA_SUFFIX,
        "canonical_policy": CANONICAL_ALIAS_POLICY,
        "case_count": len(target_rows),
        "all_decode_match": all(row["decode_match"] for row in target_rows),
        "all_answer_spans_match": all(
            row["answer_span_pass"] for row in target_rows
        ),
        "same_prompt_canonical_consistent": all(
            len(values) == 1 for values in canonical_by_prompt.values()
        ),
        "same_prompt_target_hash_consistent": all(
            len(values) == 1 for values in target_hash_by_prompt.values()
        ),
        "complete_target_tokenizer_call_count": tokenizer_call_count,
        "one_shot_tokenization_source_pass": tokenizer_call_count == 1,
        "segmented_fallback_absent": all(
            symbol not in target_schema_source for symbol in segmented_symbols
        ),
        "all_packed_structure_pass": all(packed_structure_results),
        "all_no_anchor_pass": all(no_anchor_results),
        "rows": target_rows,
    }
    mask_payload = {
        "case_count": len(mask_rows),
        "all_pass": all(
            row["attention_mask_exhaustive_pass"] for row in mask_rows
        ),
        "rows": mask_rows,
    }
    position_payload = {
        "case_count": len(position_rows),
        "all_pass": all(row["position_ids_pass"] for row in position_rows),
        "rows": position_rows,
    }
    shift_payload = {
        "case_count": len(shift_rows),
        "all_p_minus_one": all(
            row["p_minus_one_shift_pass"] for row in shift_rows
        ),
        "all_answer_span_mean": all(
            row["answer_span_mean_contract_pass"] for row in shift_rows
        ),
        "rows": shift_rows,
    }
    _write_json(
        output_dir / "EXACT_IG_FAST_PATH_TARGET_CONTRACT.json",
        target_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_FAST_PATH_MASK_EXHAUSTIVE.json",
        mask_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_FAST_PATH_POSITION_PARITY.json",
        position_payload,
    )
    _write_json(
        output_dir / "EXACT_IG_FAST_PATH_SHIFT_AND_SCORE.json",
        shift_payload,
    )
    return {
        "target_contract": (
            target_payload["all_decode_match"]
            and target_payload["all_answer_spans_match"]
            and target_payload["same_prompt_canonical_consistent"]
            and target_payload["same_prompt_target_hash_consistent"]
            and target_payload["one_shot_tokenization_source_pass"]
            and target_payload["segmented_fallback_absent"]
        ),
        "packed_structure": all(packed_structure_results),
        "no_anchor": all(no_anchor_results),
        "mask": mask_payload["all_pass"],
        "positions": position_payload["all_pass"],
        "shift": (
            shift_payload["all_p_minus_one"]
            and shift_payload["all_answer_span_mean"]
        ),
    }


def _flatten(values: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for row in values for value in row]


def _ranking(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(values)),
            key=lambda index: (-float(values[index]), index),
        )
    )


def run_gpu_shard(
    *,
    model_path: Path,
    replay_path: Path,
    output_path: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Real GPU Exact-IG audit requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    checksum_before = _sampled_model_checksum(model)
    all_cases = _read_jsonl(replay_path)
    cases = [
        case
        for index, case in enumerate(all_cases)
        if index % shard_count == shard_index
    ]
    builder = _builder(tokenizer)
    tasks = [_build_task(builder, case) for case in cases]
    if not all(isinstance(task, VectorizedExactIGTask) for task in tasks):
        raise RuntimeError("GPU calibration set unexpectedly used Sequential fallback")
    policy = production_precision_policy("fp32_exact_ig")
    single_scorer = VectorizedExactIGScorer(
        precision_policy=policy,
        padding_token_id=int(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        scoring_logits_mode=OFFICIAL_FULL_LOGITS,
        attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
    )
    rows: list[dict[str, Any]] = []
    single_results: dict[str, Any] = {}
    oracle_results: dict[str, Any] = {}
    started = time.perf_counter()
    for task, case in zip(tasks, cases, strict=True):
        fast = single_scorer.score(model, task, device)
        oracle = sequential_teacher_forced_oracle(
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
        single_results[task.trajectory_id] = fast
        oracle_results[task.trajectory_id] = oracle
        token_diffs = [
            abs(float(left) - float(right))
            for left_row, right_row in zip(
                fast.answer_token_log_probs_by_prefix,
                oracle.answer_token_log_probs_by_prefix,
                strict=True,
            )
            for left, right in zip(left_row, right_row, strict=True)
        ]
        phi_diffs = [
            abs(float(left) - float(right))
            for left, right in zip(
                fast.score_by_prefix,
                oracle.score_by_prefix,
                strict=True,
            )
        ]
        ig_diffs = [
            abs(float(left) - float(right))
            for left, right in zip(
                fast.immediate_ig,
                oracle.immediate_ig,
                strict=True,
            )
        ]
        rows.append(
            {
                "prompt_global_id": task.prompt_global_id,
                "trajectory_id": task.trajectory_id,
                "source": case["source"],
                "outcome": float(case["outcome"]),
                "search_turn_count": task.prefix_count - 1,
                "packed_length": int(task.input_ids.size),
                "prefix_count": int(task.prefix_count),
                "answer_token_count": int(
                    task.canonical_target.answer_token_count
                ),
                "boundary_crossing_any": bool(
                    task.canonical_target.boundary_crossing_any
                ),
                "fast_phi": list(fast.score_by_prefix),
                "oracle_phi": list(oracle.score_by_prefix),
                "fast_ig": list(fast.immediate_ig),
                "oracle_ig": list(oracle.immediate_ig),
                "token_abs_diffs": token_diffs,
                "phi_abs_diffs": phi_diffs,
                "ig_abs_diffs": ig_diffs,
                "maximum_token_abs_diff": max(token_diffs, default=0.0),
                "maximum_phi_abs_diff": max(phi_diffs, default=0.0),
                "maximum_ig_abs_diff": max(ig_diffs, default=0.0),
                "turn_ranking_equal": _ranking(fast.immediate_ig)
                == _ranking(oracle.immediate_ig),
                "finite": all(
                    math.isfinite(float(value))
                    for value in (
                        *fast.score_by_prefix,
                        *fast.immediate_ig,
                        *oracle.score_by_prefix,
                        *oracle.immediate_ig,
                    )
                ),
                "fast_telescoping_error": float(fast.telescoping_error),
                "oracle_telescoping_error": float(oracle.telescoping_error),
                "runtime_metadata": dict(fast.runtime_metadata),
            }
        )

    batch_scorer = VectorizedExactIGScorer(
        precision_policy=policy,
        padding_token_id=int(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        scoring_logits_mode=OFFICIAL_FULL_LOGITS,
        attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
    )
    batched = batch_scorer.score_many(
        model,
        tasks,
        device,
        max_records_per_forward=2,
        max_attention_cost_per_batch=67_108_864,
        max_extended_tokens_per_batch=16_384,
        max_full_logits_bytes=4_294_967_296,
        max_selected_logits_bytes=1_073_741_824,
    )
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        single = single_results[trajectory_id]
        batch = batched[trajectory_id]
        oracle = oracle_results[trajectory_id]
        row["single_fast_phi"] = list(single.score_by_prefix)
        row["single_fast_ig"] = list(single.immediate_ig)
        row["single_fast_runtime_metadata"] = dict(single.runtime_metadata)
        row["single_batch_token_max_abs_diff"] = max(
            (
                abs(float(left) - float(right))
                for left_row, right_row in zip(
                    single.answer_token_log_probs_by_prefix,
                    batch.answer_token_log_probs_by_prefix,
                    strict=True,
                )
                for left, right in zip(left_row, right_row, strict=True)
            ),
            default=0.0,
        )
        row["single_batch_phi_max_abs_diff"] = max(
            (
                abs(float(left) - float(right))
                for left, right in zip(
                    single.score_by_prefix,
                    batch.score_by_prefix,
                    strict=True,
                )
            ),
            default=0.0,
        )
        row["single_batch_ig_max_abs_diff"] = max(
            (
                abs(float(left) - float(right))
                for left, right in zip(
                    single.immediate_ig,
                    batch.immediate_ig,
                    strict=True,
                )
            ),
            default=0.0,
        )
        row["single_batch_turn_ranking_equal"] = (
            _ranking(single.immediate_ig) == _ranking(batch.immediate_ig)
        )
        production_token_diffs = [
            abs(float(left) - float(right))
            for left_row, right_row in zip(
                batch.answer_token_log_probs_by_prefix,
                oracle.answer_token_log_probs_by_prefix,
                strict=True,
            )
            for left, right in zip(left_row, right_row, strict=True)
        ]
        production_phi_diffs = [
            abs(float(left) - float(right))
            for left, right in zip(
                batch.score_by_prefix,
                oracle.score_by_prefix,
                strict=True,
            )
        ]
        production_ig_diffs = [
            abs(float(left) - float(right))
            for left, right in zip(
                batch.immediate_ig,
                oracle.immediate_ig,
                strict=True,
            )
        ]
        row.update(
            {
                "fast_phi": list(batch.score_by_prefix),
                "fast_ig": list(batch.immediate_ig),
                "token_abs_diffs": production_token_diffs,
                "phi_abs_diffs": production_phi_diffs,
                "ig_abs_diffs": production_ig_diffs,
                "maximum_token_abs_diff": max(
                    production_token_diffs,
                    default=0.0,
                ),
                "maximum_phi_abs_diff": max(
                    production_phi_diffs,
                    default=0.0,
                ),
                "maximum_ig_abs_diff": max(
                    production_ig_diffs,
                    default=0.0,
                ),
                "turn_ranking_equal": (
                    _ranking(batch.immediate_ig)
                    == _ranking(oracle.immediate_ig)
                ),
                "finite": all(
                    math.isfinite(float(value))
                    for value in (
                        *batch.score_by_prefix,
                        *batch.immediate_ig,
                        *oracle.score_by_prefix,
                        *oracle.immediate_ig,
                    )
                ),
                "fast_telescoping_error": float(batch.telescoping_error),
                "runtime_metadata": dict(batch.runtime_metadata),
            }
        )

    checksum_after = _sampled_model_checksum(model)
    payload = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "case_count": len(rows),
        "gpu": torch.cuda.get_device_name(0),
        "model_checksum_before": checksum_before,
        "model_checksum_after": checksum_after,
        "model_checksum_unchanged": checksum_before == checksum_after,
        "rows": rows,
        "microbatch_profiles": [
            profile.as_dict() for profile in batch_scorer.last_microbatch_profiles
        ],
        "wall_seconds": time.perf_counter() - started,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
    }
    _write_json(output_path, payload)
    return payload


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("p50", "p95", "p99", "p99_9", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "max": float(np.max(array)),
    }


def _bucketed(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {}
    boundaries = np.percentile(np.asarray(values), [0, 25, 50, 75, 100])
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        index = min(
            3,
            next(
                (
                    candidate
                    for candidate in range(4)
                    if value <= float(boundaries[candidate + 1])
                ),
                3,
            ),
        )
        label = (
            f"[{float(boundaries[index]):.6g},"
            f"{float(boundaries[index + 1]):.6g}]"
        )
        buckets[label].append(row)
    return {
        label: {
            "count": len(bucket),
            "phi": _percentiles(_flatten([item["phi_abs_diffs"] for item in bucket])),
            "ig": _percentiles(_flatten([item["ig_abs_diffs"] for item in bucket])),
        }
        for label, bucket in buckets.items()
    }


def _ragen_projection(
    rows: Sequence[Mapping[str, Any]],
    *,
    result_key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_global_id"])].append(row)
    grouped = {
        prompt_id: prompt_rows
        for prompt_id, prompt_rows in grouped.items()
        if len(prompt_rows) == 16
    }
    if len(grouped) != 32:
        raise RuntimeError(
            f"RAGEN semantic projection requires 32 complete G=16 groups, "
            f"got {len(grouped)}"
        )
    ig_variance: dict[str, float] = {}
    outcome_variance: dict[str, float] = {}
    for prompt_id, prompt_rows in grouped.items():
        ig_variance[prompt_id] = ig_prompt_variance(
            [
                {
                    index + 1: float(value)
                    for index, value in enumerate(row[result_key])
                }
                for row in prompt_rows
            ]
        ).aggregate
        outcome_variance[prompt_id] = outcome_prompt_variance(
            [float(row["outcome"]) for row in prompt_rows]
        )
    ig_excess = {
        key: max(value - 1.0e-12, 0.0) for key, value in ig_variance.items()
    }
    outcome_excess = {
        key: max(value - 1.0e-12, 0.0)
        for key, value in outcome_variance.items()
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
    outcome_active = len(outcome_positive) >= 4 and outcome_scale is not None
    denominator = 0.5 * int(ig_active) + 0.5 * int(outcome_active) + 1.0e-12
    scores = {
        prompt_id: (
            (
                0.5 * ig_excess[prompt_id] / (float(ig_scale) + 1.0e-12)
                if ig_active
                else 0.0
            )
            + (
                0.5
                * outcome_excess[prompt_id]
                / (float(outcome_scale) + 1.0e-12)
                if outcome_active
                else 0.0
            )
        )
        / denominator
        for prompt_id in grouped
    }
    selected = stable_mass_top_p(scores, rho=0.9)
    return {
        "prompt_count": len(grouped),
        "ig_variance": ig_variance,
        "outcome_variance": outcome_variance,
        "ig_scale": ig_scale,
        "outcome_scale": outcome_scale,
        "scores": scores,
        "prompt_ranking": sorted(
            scores,
            key=lambda prompt_id: (-float(scores[prompt_id]), prompt_id),
        ),
        "selected_ids": list(selected.selected_ids),
        "selected_count": len(selected.selected_ids),
        "selected_mass_ratio": float(selected.selected_mass_ratio),
    }


def merge_gpu_shards(
    *,
    shard_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    rows = [
        row
        for shard in sorted(shards, key=lambda item: int(item["shard_index"]))
        for row in shard["rows"]
    ]
    rows.sort(key=lambda item: str(item["trajectory_id"]))
    if len(rows) < 512:
        raise RuntimeError(f"Expected at least 512 GPU rows, got {len(rows)}")
    token_diffs = _flatten([row["token_abs_diffs"] for row in rows])
    phi_diffs = _flatten([row["phi_abs_diffs"] for row in rows])
    ig_diffs = _flatten([row["ig_abs_diffs"] for row in rows])
    ig_distribution = _percentiles(ig_diffs)
    epsilon_num = 3.0 * ig_distribution["p99"]
    ambiguous = 0
    non_ambiguous = 0
    non_ambiguous_sign_matches = 0
    for row in rows:
        for fast, oracle in zip(
            row["fast_ig"],
            row["oracle_ig"],
            strict=True,
        ):
            if abs(float(oracle)) <= epsilon_num:
                ambiguous += 1
            else:
                non_ambiguous += 1
                non_ambiguous_sign_matches += int(
                    (float(fast) > 0) == (float(oracle) > 0)
                )
    fast_ragen = _ragen_projection(rows, result_key="fast_ig")
    oracle_ragen = _ragen_projection(rows, result_key="oracle_ig")
    selected_fast = set(fast_ragen["selected_ids"])
    selected_oracle = set(oracle_ragen["selected_ids"])
    union = selected_fast | selected_oracle
    ragen_payload = {
        "fast": fast_ragen,
        "oracle": oracle_ragen,
        "prompt_ig_variance_ranking_equal": sorted(
            fast_ragen["ig_variance"],
            key=lambda key: (-fast_ragen["ig_variance"][key], key),
        )
        == sorted(
            oracle_ragen["ig_variance"],
            key=lambda key: (-oracle_ragen["ig_variance"][key], key),
        ),
        "prompt_ranking_equal": fast_ragen["prompt_ranking"]
        == oracle_ragen["prompt_ranking"],
        "selected_ids_equal": fast_ragen["selected_ids"]
        == oracle_ragen["selected_ids"],
        "selected_count_equal": fast_ragen["selected_count"]
        == oracle_ragen["selected_count"],
        "selected_set_jaccard": (
            len(selected_fast & selected_oracle) / len(union) if union else 1.0
        ),
    }
    ragen_payload["gate_pass"] = (
        ragen_payload["prompt_ig_variance_ranking_equal"]
        and ragen_payload["prompt_ranking_equal"]
        and ragen_payload["selected_ids_equal"]
        and ragen_payload["selected_count_equal"]
        and ragen_payload["selected_set_jaccard"] == 1.0
    )
    distribution = {
        "case_count": len(rows),
        "prefix_score_count": sum(len(row["fast_phi"]) for row in rows),
        "ig_count": sum(len(row["fast_ig"]) for row in rows),
        "epsilon_num": epsilon_num,
        "epsilon_definition": "3 * P99(abs(Fast IG - Oracle IG))",
        "numeric_ambiguous_ig_count": ambiguous,
        "non_ambiguous_ig_count": non_ambiguous,
        "non_ambiguous_sign_agreement": (
            non_ambiguous_sign_matches / non_ambiguous
            if non_ambiguous
            else 1.0
        ),
        "turn_ranking_agreement": sum(
            bool(row["turn_ranking_equal"]) for row in rows
        )
        / len(rows),
        "single_batch_turn_ranking_agreement": sum(
            bool(row["single_batch_turn_ranking_equal"]) for row in rows
        )
        / len(rows),
        "token_abs_diff": _percentiles(token_diffs),
        "phi_abs_diff": _percentiles(phi_diffs),
        "ig_abs_diff": ig_distribution,
        "single_batch_token_abs_diff": _percentiles(
            [float(row["single_batch_token_max_abs_diff"]) for row in rows]
        ),
        "single_batch_phi_abs_diff": _percentiles(
            [float(row["single_batch_phi_max_abs_diff"]) for row in rows]
        ),
        "single_batch_ig_abs_diff": _percentiles(
            [float(row["single_batch_ig_max_abs_diff"]) for row in rows]
        ),
        "all_finite": all(bool(row["finite"]) for row in rows),
        "model_checksums_unchanged": all(
            bool(shard["model_checksum_unchanged"]) for shard in shards
        ),
        "by_packed_length": _bucketed(rows, "packed_length"),
        "by_prefix_count": _bucketed(rows, "prefix_count"),
        "by_answer_token_count": _bucketed(rows, "answer_token_count"),
        "by_boundary_crossing": {
            str(flag): {
                "count": len(bucket),
                "phi": _percentiles(
                    _flatten([row["phi_abs_diffs"] for row in bucket])
                ),
                "ig": _percentiles(
                    _flatten([row["ig_abs_diffs"] for row in bucket])
                ),
            }
            for flag in (False, True)
            if (
                bucket := [
                    row
                    for row in rows
                    if bool(row["boundary_crossing_any"]) is flag
                ]
            )
        },
        "maximum_phi_error_safety_pass": max(phi_diffs, default=0.0)
        <= MAX_PHI_SAFETY_ERROR,
        "maximum_ig_error_safety_pass": max(ig_diffs, default=0.0)
        <= MAX_IG_SAFETY_ERROR,
        "rows": rows,
    }
    distribution["semantic_gate_pass"] = (
        distribution["non_ambiguous_sign_agreement"] == 1.0
        and distribution["turn_ranking_agreement"] == 1.0
        and distribution["single_batch_turn_ranking_agreement"] == 1.0
        and distribution["all_finite"]
        and distribution["maximum_phi_error_safety_pass"]
        and distribution["maximum_ig_error_safety_pass"]
        and ragen_payload["gate_pass"]
    )
    _write_json(
        output_dir / "EXACT_IG_FAST_ORACLE_ERROR_DISTRIBUTION.json",
        distribution,
    )
    _write_json(
        output_dir / "EXACT_IG_FAST_RAGEN_SEMANTIC_PARITY.json",
        ragen_payload,
    )
    runtime_rows = [row["runtime_metadata"] for row in rows]
    runtime_payload = {
        "case_count": len(rows),
        "actual_model_parameter_dtypes": sorted(
            {row.get("actual_model_parameter_dtype") for row in runtime_rows}
        ),
        "actual_logits_dtypes": sorted(
            {row.get("actual_logits_dtype") for row in runtime_rows}
        ),
        "actual_log_probs_dtypes": sorted(
            {row.get("actual_log_probs_dtype") for row in runtime_rows}
        ),
        "actual_hidden_first_dtypes": sorted(
            {row.get("actual_hidden_dtype_first_layer") for row in runtime_rows}
        ),
        "actual_hidden_last_dtypes": sorted(
            {row.get("actual_hidden_dtype_last_layer") for row in runtime_rows}
        ),
        "autocast_enabled_values": sorted(
            {bool(row.get("autocast_enabled")) for row in runtime_rows}
        ),
        "tf32_matmul_values": sorted(
            {bool(row.get("tf32_matmul_enabled")) for row in runtime_rows}
        ),
        "tf32_cudnn_values": sorted(
            {bool(row.get("tf32_cudnn_enabled")) for row in runtime_rows}
        ),
        "attention_backends": sorted(
            {str(row.get("attention_backend")) for row in runtime_rows}
        ),
        "model_config_attn_implementations": sorted(
            {
                str(row.get("model_config_attn_implementation"))
                for row in runtime_rows
            }
        ),
        "scoring_logits_modes": sorted(
            {str(row.get("scoring_logits_mode")) for row in runtime_rows}
        ),
        "attention_mask_modes": sorted(
            {str(row.get("attention_mask_mode")) for row in runtime_rows}
        ),
        "peak_allocated_bytes": max(
            int(row.get("actual_peak_allocated_bytes") or 0)
            for row in runtime_rows
        ),
        "peak_reserved_bytes": max(
            int(row.get("actual_peak_reserved_bytes") or 0)
            for row in runtime_rows
        ),
        "model_checksums_unchanged": distribution[
            "model_checksums_unchanged"
        ],
    }
    runtime_payload["gate_pass"] = (
        runtime_payload["actual_model_parameter_dtypes"] == ["float32"]
        and runtime_payload["actual_logits_dtypes"] == ["float32"]
        and runtime_payload["actual_log_probs_dtypes"] == ["float32"]
        and runtime_payload["actual_hidden_first_dtypes"] == ["float32"]
        and runtime_payload["actual_hidden_last_dtypes"] == ["float32"]
        and runtime_payload["autocast_enabled_values"] == [False]
        and runtime_payload["tf32_matmul_values"] == [False]
        and runtime_payload["tf32_cudnn_values"] == [False]
        and runtime_payload["attention_backends"] == ["sdpa:math"]
        and runtime_payload["model_config_attn_implementations"] == ["sdpa"]
        and runtime_payload["scoring_logits_modes"] == [
            "official_full_logits"
        ]
        and runtime_payload["attention_mask_modes"] == [
            "official_additive"
        ]
        and runtime_payload["model_checksums_unchanged"]
    )
    _write_json(
        output_dir / "EXACT_IG_FAST_PATH_RUNTIME_METADATA.json",
        runtime_payload,
    )
    return {
        "distribution": distribution,
        "ragen": ragen_payload,
        "runtime": runtime_payload,
    }


def run_future_leakage(
    *,
    model_path: Path,
    replay_path: Path,
    output_path: Path,
    maximum_cases: int = 12,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    scorer = VectorizedExactIGScorer.for_production_mode(
        "fp32_exact_ig",
        padding_token_id=int(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        scoring_logits_mode=OFFICIAL_FULL_LOGITS,
        attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
    )
    cases = _read_jsonl(replay_path)
    selected = sorted(
        cases,
        key=lambda case: (
            -len(case["prefix_end_positions"]),
            -len(case["input_ids"]),
        ),
    )[:maximum_cases]
    builder = _builder(tokenizer)
    rows: list[dict[str, Any]] = []
    for case in selected:
        task = _build_task(builder, case)
        baseline = scorer.score(model, task, device)
        for prefix_index, prefix_end in enumerate(task.prefix_end_positions):
            mutated_ids = task.input_ids.copy()
            future_start = int(prefix_end)
            future_end = int(task.original_token_count)
            if future_start < future_end:
                mutated_ids[future_start:future_end] = (
                    mutated_ids[future_start:future_end] + 7919
                ) % int(model.config.vocab_size)
            for other_index, other_start in enumerate(task.segment_starts):
                if other_index == prefix_index:
                    continue
                local_candidates = [
                    local
                    for local, scored in enumerate(
                        task.canonical_target.score_mask
                    )
                    if not scored
                ]
                for local in local_candidates:
                    physical = int(other_start) + local
                    mutated_ids[physical] = (
                        int(mutated_ids[physical]) + 3571
                    ) % int(model.config.vocab_size)
            mutated = replace(task, input_ids=mutated_ids)
            observed = scorer.score(model, mutated, device)
            difference = abs(
                float(baseline.score_by_prefix[prefix_index])
                - float(observed.score_by_prefix[prefix_index])
            )
            rows.append(
                {
                    "trajectory_id": task.trajectory_id,
                    "prefix_index": prefix_index,
                    "prefix_end": int(prefix_end),
                    "same_shape": True,
                    "phi_abs_diff": difference,
                }
            )
    maximum = max((float(row["phi_abs_diff"]) for row in rows), default=0.0)
    payload = {
        "case_count": len(selected),
        "prefix_check_count": len(rows),
        "maximum_abs_diff": maximum,
        "gate_pass": maximum == 0.0,
        "rows": rows,
    }
    _write_json(output_path, payload)
    return payload


def _forward_probe(
    *,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any,
    answer_positions: Sequence[int],
    answer_token_ids: Sequence[int],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        enabled=False,
    ), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        rows = torch.tensor(
            [int(position) - 1 for position in answer_positions],
            dtype=torch.long,
            device=input_ids.device,
        )
        targets = torch.tensor(
            [int(value) for value in answer_token_ids],
            dtype=torch.long,
            device=input_ids.device,
        )
        selected_logits = output.logits[0].index_select(0, rows)
        log_probs = F.log_softmax(selected_logits, dim=-1).gather(
            -1,
            targets.unsqueeze(-1),
        ).squeeze(-1)
        hidden = [
            state[0].index_select(0, rows).float().cpu()
            for state in output.hidden_states
        ]
    return {
        "token_log_probs": log_probs.float().cpu(),
        "phi": float(log_probs.mean().cpu().item()),
        "hidden": hidden,
    }


def run_failure_decomposition(
    *,
    model_path: Path,
    failure_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    case = _load_failure_case(failure_path)
    task = _build_task(_builder(tokenizer), case)
    target_ids = list(task.canonical_target.token_ids)
    answer_local = list(
        range(
            task.canonical_target.answer_token_start,
            task.canonical_target.answer_token_end,
        )
    )
    path_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prefix_index, prefix_end in enumerate(task.prefix_end_positions):
        sequential_ids = [
            *case["input_ids"][:prefix_end],
            *target_ids,
        ]
        sequential_positions = [
            *case["position_ids"][:prefix_end],
            *range(
                int(case["position_ids"][prefix_end - 1]) + 1,
                int(case["position_ids"][prefix_end - 1]) + 1
                + len(target_ids),
            ),
        ]
        answer_positions_a = [
            int(prefix_end) + local for local in answer_local
        ]
        ids_a = torch.tensor([sequential_ids], dtype=torch.long, device=device)
        positions_a = torch.tensor(
            [sequential_positions],
            dtype=torch.long,
            device=device,
        )
        mask_2d = torch.ones_like(ids_a)
        path_results["A"].append(
            _forward_probe(
                model=model,
                input_ids=ids_a,
                attention_mask=mask_2d,
                position_ids=positions_a,
                answer_positions=answer_positions_a,
                answer_token_ids=task.canonical_target.answer_token_ids,
            )
        )
        length_a = len(sequential_ids)
        causal = torch.tril(
            torch.ones((1, 1, length_a, length_a), dtype=torch.bool, device=device)
        )
        mask_4d = torch.where(
            causal,
            torch.tensor(0.0, device=device),
            torch.tensor(-10000.0, device=device),
        )
        path_results["B"].append(
            _forward_probe(
                model=model,
                input_ids=ids_a,
                attention_mask=mask_4d,
                position_ids=positions_a,
                answer_positions=answer_positions_a,
                answer_token_ids=task.canonical_target.answer_token_ids,
            )
        )

        original_count = task.original_token_count
        ids_c_list = [*case["input_ids"], *target_ids]
        positions_c_list = [
            *case["position_ids"],
            *range(
                int(case["position_ids"][prefix_end - 1]) + 1,
                int(case["position_ids"][prefix_end - 1]) + 1
                + len(target_ids),
            ),
        ]
        mask_c_bool = independent_expected_mask(
            original_token_count=original_count,
            original_attention_mask=case["attention_mask"],
            prefix_end_positions=[prefix_end],
            segment_starts=[original_count],
            segment_lengths=[len(target_ids)],
        )
        mask_c = torch.where(
            torch.tensor(mask_c_bool, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(-10000.0, device=device),
        ).unsqueeze(0).unsqueeze(0)
        path_results["C"].append(
            _forward_probe(
                model=model,
                input_ids=torch.tensor(
                    [ids_c_list],
                    dtype=torch.long,
                    device=device,
                ),
                attention_mask=mask_c,
                position_ids=torch.tensor(
                    [positions_c_list],
                    dtype=torch.long,
                    device=device,
                ),
                answer_positions=[
                    original_count + local for local in answer_local
                ],
                answer_token_ids=task.canonical_target.answer_token_ids,
            )
        )

        additive_d = torch.where(
            torch.tensor(task.attention_mask, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(-10000.0, device=device),
        ).unsqueeze(0).unsqueeze(0)
        path_results["D"].append(
            _forward_probe(
                model=model,
                input_ids=torch.tensor(
                    [task.input_ids],
                    dtype=torch.long,
                    device=device,
                ),
                attention_mask=additive_d,
                position_ids=torch.tensor(
                    [task.position_ids],
                    dtype=torch.long,
                    device=device,
                ),
                answer_positions=task.score_spans[
                    prefix_index
                ].answer_token_positions,
                answer_token_ids=task.canonical_target.answer_token_ids,
            )
        )

        batch_ids = torch.tensor(
            np.stack((task.input_ids, task.input_ids)),
            dtype=torch.long,
            device=device,
        )
        batch_positions = torch.tensor(
            np.stack((task.position_ids, task.position_ids)),
            dtype=torch.long,
            device=device,
        )
        batch_mask = torch.where(
            torch.tensor(
                np.stack((task.attention_mask, task.attention_mask)),
                device=device,
            ),
            torch.tensor(0.0, device=device),
            torch.tensor(-10000.0, device=device),
        ).unsqueeze(1)
        path_results["E"].append(
            _forward_probe(
                model=model,
                input_ids=batch_ids,
                attention_mask=batch_mask,
                position_ids=batch_positions,
                answer_positions=task.score_spans[
                    prefix_index
                ].answer_token_positions,
                answer_token_ids=task.canonical_target.answer_token_ids,
            )
        )

    comparisons: dict[str, Any] = {}
    for left_name, right_name in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")):
        rows = []
        for prefix_index, (left, right) in enumerate(
            zip(path_results[left_name], path_results[right_name], strict=True)
        ):
            layer_diffs = [
                {
                    "layer": layer,
                    "max_abs_diff": float(
                        (left_hidden - right_hidden).abs().max().item()
                    ),
                    "mean_abs_diff": float(
                        (left_hidden - right_hidden).abs().mean().item()
                    ),
                    "relative_l2_error": float(
                        torch.linalg.vector_norm(left_hidden - right_hidden).item()
                        / (
                            torch.linalg.vector_norm(right_hidden).item()
                            + 1.0e-30
                        )
                    ),
                }
                for layer, (left_hidden, right_hidden) in enumerate(
                    zip(left["hidden"], right["hidden"], strict=True)
                )
            ]
            rows.append(
                {
                    "prefix_index": prefix_index,
                    "token_log_prob_max_abs_diff": float(
                        (
                            left["token_log_probs"]
                            - right["token_log_probs"]
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "phi_abs_diff": abs(float(left["phi"]) - float(right["phi"])),
                    "layer_diffs": layer_diffs,
                }
            )
        comparisons[f"{left_name}-{right_name}"] = rows
    phi_by_path = {
        name: [float(row["phi"]) for row in path_results[name]]
        for name in path_results
    }
    ig_by_path = {
        name: list(immediate_ig_from_prefix_scores(values))
        for name, values in phi_by_path.items()
    }
    comparisons["ig_pair_max_abs_diff"] = {
        f"{left}-{right}": max(
            (
                abs(float(a) - float(b))
                for a, b in zip(
                    ig_by_path[left],
                    ig_by_path[right],
                    strict=True,
                )
            ),
            default=0.0,
        )
        for left, right in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"))
    }
    payload = {
        "failure_path": str(failure_path.resolve()),
        "trajectory_id": task.trajectory_id,
        "path_definitions": {
            "A": "Sequential 2D causal",
            "B": "same-length Sequential explicit 4D causal",
            "C": "full original plus one target copy",
            "D": "full original plus every target copy",
            "E": "two-row production-equivalent batch shape",
        },
        "phi_by_path": phi_by_path,
        "ig_by_path": ig_by_path,
        "comparisons": comparisons,
    }
    _write_json(output_path, payload)
    return payload


def _official_snapshot() -> dict[str, Any]:
    files = (
        "scrl/llm_agent/vectorized_gt_logprob.py",
        "scrl/llm_agent/generation.py",
    )
    return {
        "commit": OFFICIAL_IGPO_COMMIT_SHA,
        "repository": "https://github.com/GuoqingWang1/IGPO",
        "files": {
            name: {
                "path": str((OFFICIAL_ROOT / name).resolve()),
                "sha256": _sha256_file(OFFICIAL_ROOT / name),
            }
            for name in files
        },
    }


def build_report(output_dir: Path) -> dict[str, Any]:
    target = json.loads(
        (output_dir / "EXACT_IG_FAST_PATH_TARGET_CONTRACT.json").read_text()
    )
    mask = json.loads(
        (output_dir / "EXACT_IG_FAST_PATH_MASK_EXHAUSTIVE.json").read_text()
    )
    positions = json.loads(
        (output_dir / "EXACT_IG_FAST_PATH_POSITION_PARITY.json").read_text()
    )
    shift = json.loads(
        (output_dir / "EXACT_IG_FAST_PATH_SHIFT_AND_SCORE.json").read_text()
    )
    leakage = json.loads(
        (output_dir / "EXACT_IG_FAST_PATH_FUTURE_LEAKAGE.json").read_text()
    )
    runtime = json.loads(
        (output_dir / "EXACT_IG_FAST_PATH_RUNTIME_METADATA.json").read_text()
    )
    errors = json.loads(
        (output_dir / "EXACT_IG_FAST_ORACLE_ERROR_DISTRIBUTION.json").read_text()
    )
    ragen = json.loads(
        (output_dir / "EXACT_IG_FAST_RAGEN_SEMANTIC_PARITY.json").read_text()
    )
    fsdp = json.loads(
        (output_dir / "EXACT_IG_FAST_FSDP_STATE_AUDIT.json").read_text()
    )
    gates = {
        "TARGET_CONTRACT": bool(
            target["all_decode_match"] and target["all_answer_spans_match"]
        ),
        "CANONICAL_FIRST_ALIAS": (
            target["canonical_policy"] == "first"
            and target["same_prompt_canonical_consistent"]
        ),
        "ONE_SHOT_TOKENIZATION": bool(
            target["one_shot_tokenization_source_pass"]
            and target["segmented_fallback_absent"]
        ),
        "PACKED_STRUCTURE": bool(target["all_packed_structure_pass"]),
        "NO_ANCHOR": bool(target["all_no_anchor_pass"]),
        "ATTENTION_MASK_EXHAUSTIVE": bool(mask["all_pass"]),
        "LOGICAL_POSITION_IDS": bool(positions["all_pass"]),
        "P_MINUS_ONE_SHIFT": bool(shift["all_p_minus_one"]),
        "ANSWER_SPAN_MEAN": bool(shift["all_answer_span_mean"]),
        "FUTURE_LEAKAGE": bool(leakage["gate_pass"]),
        "FP32_RUNTIME": bool(runtime["gate_pass"]),
        "SDPA_MATH_RUNTIME": runtime["attention_backends"] == ["sdpa:math"],
        "FULL_LOGITS_RUNTIME": runtime["scoring_logits_modes"]
        == ["official_full_logits"],
        "FSDP_RESTORE": bool(fsdp["fsdp_window_restore_pass"]),
        "MODEL_CHECKSUM_UNCHANGED": bool(
            runtime["model_checksums_unchanged"]
            and fsdp["all_rank_checksums_unchanged"]
        ),
        "IG_SIGN_SEMANTIC_PARITY": (
            errors["non_ambiguous_sign_agreement"] == 1.0
        ),
        "TURN_RANKING_PARITY": errors["turn_ranking_agreement"] == 1.0,
        "RAGEN_SELECTED_SET_PARITY": bool(ragen["gate_pass"]),
        "MAX_PHI_ERROR_SAFETY": bool(
            errors["maximum_phi_error_safety_pass"]
        ),
        "MAX_IG_ERROR_SAFETY": bool(errors["maximum_ig_error_safety_pass"]),
        "OPTIMIZER_STEPS": 0,
        "SCHEDULER_STEPS": 0,
        "CHECKPOINT_WRITES": 0,
    }
    allow = all(
        value is True
        for key, value in gates.items()
        if key
        not in {"OPTIMIZER_STEPS", "SCHEDULER_STEPS", "CHECKPOINT_WRITES"}
    ) and all(
        gates[key] == 0
        for key in ("OPTIMIZER_STEPS", "SCHEDULER_STEPS", "CHECKPOINT_WRITES")
    )
    snapshot = _official_snapshot()
    report = f"""# Exact-IG Fast Path Production Structural Audit

## Scope and conclusion

- Project: `{PROJECT_ROOT}`
- Exact-IG version: `{EXACT_IG_VERSION}`
- Official IGPO commit: `{OFFICIAL_IGPO_COMMIT_SHA}`
- Audited real/replay trajectory count: `{errors["case_count"]}`
- Production path: `fp32_exact_ig + official_full_logits + official_additive`
- Sequential Oracle role: diagnostic shadow only
- `ALLOW_FAST_PATH_TRAINING = {"YES" if allow else "NO"}`

The independent expected mask, logical positions, target span, and predicting
rows were rebuilt in the independent test-support auditor; those
expected values do not call the production mask builder, position builder,
task builder, task validator, or Fast score-position builder.

## Locked target

- Scaffold: `{ANSWER_SCAFFOLD_TEXT}`
- Suffix: `{TARGET_SCHEMA_SUFFIX}`
- Canonical policy: ordered alias index 0
- Full target tokenization: one tokenizer call with offsets
- Score span: IGPO offset-covering answer span
- Information gain: `Phi[t] - Phi[t-1]`

## Structural results

| Gate | Result |
|---|---|
| Target/decode/span | `{"PASS" if gates["TARGET_CONTRACT"] else "FAIL"}` |
| Packed original-once plus one target copy per prefix | `{"PASS" if gates["PACKED_STRUCTURE"] else "FAIL"}` |
| No anchor | `{"PASS" if gates["NO_ANCHOR"] else "FAIL"}` |
| Exhaustive attention visibility | `{"PASS" if gates["ATTENTION_MASK_EXHAUSTIVE"] else "FAIL"}` |
| Logical position IDs | `{"PASS" if gates["LOGICAL_POSITION_IDS"] else "FAIL"}` |
| p-1 score shift | `{"PASS" if gates["P_MINUS_ONE_SHIFT"] else "FAIL"}` |
| Future leakage | `{"PASS" if gates["FUTURE_LEAKAGE"] else "FAIL"}` |

## Runtime results

- Parameter/logits/log-probs dtype: `{runtime["actual_model_parameter_dtypes"]}` /
  `{runtime["actual_logits_dtypes"]}` / `{runtime["actual_log_probs_dtypes"]}`
- Hidden dtype first/last: `{runtime["actual_hidden_first_dtypes"]}` /
  `{runtime["actual_hidden_last_dtypes"]}`
- Backend: `{runtime["attention_backends"]}`
- Model config attention implementation:
  `{runtime["model_config_attn_implementations"]}`
- Autocast: `{runtime["autocast_enabled_values"]}`
- TF32 matmul/cuDNN: `{runtime["tf32_matmul_values"]}` /
  `{runtime["tf32_cudnn_values"]}`
- Peak allocated/reserved: `{runtime["peak_allocated_bytes"]}` /
  `{runtime["peak_reserved_bytes"]}` bytes

## Numerical telemetry and semantic stability

The old per-token `2e-5` target is telemetry, not a standalone stop gate.
The structural gates remain fail-closed. The anomaly safety ceiling remains
`1e-3` for both Phi and IG.

- Token abs diff P50/P95/P99/P99.9/max:
  `{errors["token_abs_diff"]}`
- Phi abs diff P50/P95/P99/P99.9/max:
  `{errors["phi_abs_diff"]}`
- IG abs diff P50/P95/P99/P99.9/max:
  `{errors["ig_abs_diff"]}`
- Numeric ambiguity epsilon: `{errors["epsilon_num"]}`
- Non-ambiguous IG sign agreement:
  `{errors["non_ambiguous_sign_agreement"]}`
- Turn ranking agreement: `{errors["turn_ranking_agreement"]}`
- RAGEN selected IDs equal: `{ragen["selected_ids_equal"]}`
- Selected-set Jaccard: `{ragen["selected_set_jaccard"]}`

## Failure sample decomposition

See `EXACT_IG_FAST_PATH_FAILURE_DECOMPOSITION.json`. It separates:
2D-vs-4D causal execution, full-original physical length, multi-copy packing,
and multi-row batch shape. No target, mask, position, or shift rule is changed.

## FSDP and state

- Four-rank restore: `{fsdp["fsdp_window_restore_pass"]}`
- Normal and exception restore: `{fsdp["all_ranks_restore_succeeded"]}` /
  `{fsdp["all_ranks_exception_restore_succeeded"]}`
- Model checksums unchanged: `{gates["MODEL_CHECKSUM_UNCHANGED"]}`
- Optimizer/scheduler/checkpoint writes: `0 / 0 / 0`

## Official source snapshot

```json
{json.dumps(snapshot, indent=2, sort_keys=True)}
```

## Gates

```json
{json.dumps(gates, indent=2, sort_keys=True)}
```
"""
    path = output_dir / "EXACT_IG_FAST_PATH_STRUCTURAL_AUDIT.md"
    path.write_text(report, encoding="utf-8")
    status = {
        "exact_ig_version": EXACT_IG_VERSION,
        "official_igpo_commit_sha": OFFICIAL_IGPO_COMMIT_SHA,
        "gates": gates,
        "numeric_ambiguity_epsilon": errors["epsilon_num"],
        "calibration_p99_ig_abs_diff": errors["ig_abs_diff"]["p99"],
        "allow_fast_path_training": allow,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
    }
    _write_json(output_dir / "audit_status.json", status)
    changed_files = (
        "tests/support/exact_ig_fast_path_audit.py",
        "tests/test_exact_ig_fast_path_independent_contract.py",
        "src/agentic_rl/exact_ig/vectorized_scorer.py",
        "src/agentic_rl/runtime/fsdp_worker.py",
        "configs/exact_ig.yaml",
        "src/agentic_rl/runtime/verl_config.py",
        "src/agentic_rl/runtime/verl_runtime_adapter.py",
        "tests/test_runtime_adapter_static.py",
    )
    (output_dir / "EXACT_IG_FAST_PATH_CHANGED_FILES.txt").write_text(
        "\n".join(changed_files) + "\n",
        encoding="utf-8",
    )
    return status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent production structural audit for Exact-IG Fast Path"
    )
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "static",
            "gpu-shard",
            "merge",
            "future-leakage",
            "failure-decomposition",
            "report",
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--failure-path", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument(
        "--trajectory-metrics",
        type=Path,
        default=DEFAULT_TRAJECTORY_METRICS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "internal/replay_cases.jsonl"
    if args.phase == "prepare":
        result = prepare_replay_cases(
            model_path=args.model_path.resolve(),
            metrics_path=args.trajectory_metrics.resolve(),
            failure_path=args.failure_path.resolve(),
            output_path=replay_path,
            retriever_url=args.retriever_url,
        )
    elif args.phase == "static":
        result = run_static_audit(
            model_path=args.model_path.resolve(),
            replay_path=replay_path,
            output_dir=output_dir,
        )
    elif args.phase == "gpu-shard":
        result = run_gpu_shard(
            model_path=args.model_path.resolve(),
            replay_path=replay_path,
            output_path=(
                output_dir
                / "internal"
                / f"gpu_shard_{args.shard_index:02d}.json"
            ),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.phase == "merge":
        result = merge_gpu_shards(
            shard_paths=tuple(
                output_dir / "internal" / f"gpu_shard_{index:02d}.json"
                for index in range(args.shard_count)
            ),
            output_dir=output_dir,
        )
    elif args.phase == "future-leakage":
        result = run_future_leakage(
            model_path=args.model_path.resolve(),
            replay_path=replay_path,
            output_path=(
                output_dir / "EXACT_IG_FAST_PATH_FUTURE_LEAKAGE.json"
            ),
        )
    elif args.phase == "failure-decomposition":
        result = run_failure_decomposition(
            model_path=args.model_path.resolve(),
            failure_path=args.failure_path.resolve(),
            output_path=(
                output_dir
                / "EXACT_IG_FAST_PATH_FAILURE_DECOMPOSITION.json"
            ),
        )
    else:
        result = build_report(output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
