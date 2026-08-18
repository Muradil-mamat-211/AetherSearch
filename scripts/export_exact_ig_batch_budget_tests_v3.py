#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from agentic_rl.exact_ig.precision_policy import ExactIGPrecisionPolicy
from agentic_rl.exact_ig.task_builder import (
    ExactIGTaskBuilder,
    SequentialExactIGTask,
)
from agentic_rl.exact_ig.vectorized_scorer import (
    SingleFastTaskBudgetExceeded,
    VectorizedExactIGScorer,
    estimate_exact_ig_batch,
    pack_exact_ig_microbatches,
)


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        if not kwargs.get("return_offsets_mapping"):
            raise AssertionError("V3 target tokenization requires offsets")
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=256)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        use_cache=False,
    ):
        del attention_mask, position_ids, use_cache
        logits = torch.zeros(
            (*input_ids.shape, self.config.vocab_size),
            dtype=self.scale.dtype,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits + self.scale * 0.0)


def _policy() -> ExactIGPrecisionPolicy:
    return ExactIGPrecisionPolicy(
        mode="cpu_official_semantics",
        autocast_enabled=False,
        autocast_dtype="bfloat16",
        temperature=1.0,
        attention_implementation="eager",
        sdpa_backend=None,
    )


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = CharacterTokenizer()
    builder = ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=4096,
        maximum_position_id_exclusive=4096,
    )
    task = builder.build(
        prompt_global_id="p",
        trajectory_id="t",
        full_trajectory_input_ids=[11, 12, 13, 14],
        original_attention_mask=[1, 1, 1, 1],
        prefix_end_positions=[2, 4],
        canonical_answer="A",
    )
    estimate = estimate_exact_ig_batch(
        (task,),
        vocabulary_size=256,
        logits_element_size=4,
        structural_mask_element_size=4,
    )
    equal_budgets = {
        "max_records_per_forward": 1,
        "max_attention_cost_per_batch": estimate.padded_attention_cost,
        "max_extended_tokens_per_batch": estimate.padded_token_count,
        "max_full_logits_bytes": estimate.full_logits_estimated_bytes,
        "max_selected_logits_bytes": estimate.selected_logits_estimated_bytes,
        "vocabulary_size": 256,
        "logits_element_size": 4,
        "structural_mask_element_size": 4,
    }
    rows: list[dict] = []
    equal_accepted = bool(
        pack_exact_ig_microbatches((task,), **equal_budgets)
    )
    rows.append(
        {
            "case": "single_task_equal_to_all_budgets",
            "pass": equal_accepted,
            "estimate": estimate.__dict__,
        }
    )
    rows.append(
        {
            "case": "single_task_below_all_budgets",
            "pass": bool(
                pack_exact_ig_microbatches(
                    (task,),
                    **{
                        **equal_budgets,
                        "max_attention_cost_per_batch": (
                            estimate.padded_attention_cost + 1
                        ),
                        "max_extended_tokens_per_batch": (
                            estimate.padded_token_count + 1
                        ),
                        "max_full_logits_bytes": (
                            estimate.full_logits_estimated_bytes + 1
                        ),
                        "max_selected_logits_bytes": (
                            estimate.selected_logits_estimated_bytes + 1
                        ),
                    },
                )
            ),
        }
    )
    for field, estimate_field in (
        ("max_attention_cost_per_batch", "padded_attention_cost"),
        ("max_extended_tokens_per_batch", "padded_token_count"),
        ("max_full_logits_bytes", "full_logits_estimated_bytes"),
        ("max_selected_logits_bytes", "selected_logits_estimated_bytes"),
    ):
        rejected = False
        reason = None
        try:
            pack_exact_ig_microbatches(
                (task,),
                **{
                    **equal_budgets,
                    field: int(getattr(estimate, estimate_field)) - 1,
                },
            )
        except SingleFastTaskBudgetExceeded as error:
            rejected = field in error.reasons
            reason = list(error.reasons)
        rows.append(
            {
                "case": f"single_task_over_{field}",
                "pass": rejected,
                "reasons": reason,
            }
        )

    target_length = len(task.canonical_target.token_ids)
    largest_sequential = max(task.prefix_end_positions) + target_length
    fast_length = int(task.input_ids.size)
    sequential_budget = largest_sequential * largest_sequential
    scorer = VectorizedExactIGScorer(
        precision_policy=_policy(),
        padding_token_id=0,
        tokenizer=tokenizer,
    )
    fallback = scorer.score_many(
        TinyCausalModel(),
        (task,),
        torch.device("cpu"),
        max_records_per_forward=1,
        max_attention_cost_per_batch=sequential_budget,
        max_extended_tokens_per_batch=fast_length,
        max_full_logits_bytes=None,
        max_selected_logits_bytes=None,
    )["t"]
    rows.append(
        {
            "case": "single_fast_budget_exceeded_sequential_legal",
            "pass": (
                fallback.execution_path == "official_sequential_fallback"
                and fallback.runtime_metadata.get("fallback_reason")
                == "single_fast_task_budget_exceeded"
            ),
            "execution_path": fallback.execution_path,
            "fallback_reason": fallback.runtime_metadata.get("fallback_reason"),
        }
    )

    sequential_fail_closed = False
    try:
        scorer.score_many(
            TinyCausalModel(),
            (task,),
            torch.device("cpu"),
            max_records_per_forward=1,
            max_attention_cost_per_batch=1,
            max_extended_tokens_per_batch=None,
            max_full_logits_bytes=None,
            max_selected_logits_bytes=None,
        )
    except RuntimeError as error:
        sequential_fail_closed = "both Fast and Sequential" in str(error)
    rows.append(
        {
            "case": "fast_and_sequential_budget_exceeded",
            "pass": sequential_fail_closed,
            "policy": "fail_closed",
        }
    )

    context_limit = 4 + target_length
    context_fallback = ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=context_limit,
        maximum_position_id_exclusive=4096,
    ).build(
        prompt_global_id="p",
        trajectory_id="context-fallback",
        full_trajectory_input_ids=[11, 12, 13, 14],
        original_attention_mask=[1, 1, 1, 1],
        prefix_end_positions=[2, 4],
        canonical_answer="A",
    )
    rows.append(
        {
            "case": "fast_physical_context_exceeded_sequential_legal",
            "pass": isinstance(context_fallback, SequentialExactIGTask),
            "execution_path": "official_sequential_fallback",
        }
    )

    sequential_context_closed = False
    try:
        ExactIGTaskBuilder(
            tokenizer,
            maximum_extended_sequence_length=4,
            maximum_position_id_exclusive=4096,
        ).build(
            prompt_global_id="p",
            trajectory_id="context-fail",
            full_trajectory_input_ids=[11, 12],
            original_attention_mask=[1, 1],
            prefix_end_positions=[2],
            canonical_answer="A",
        )
    except ValueError as error:
        sequential_context_closed = "Sequential prefix exceeds" in str(error)
    rows.append(
        {
            "case": "sequential_context_exceeded",
            "pass": sequential_context_closed,
            "policy": "fail_closed",
        }
    )

    position_closed = False
    try:
        ExactIGTaskBuilder(
            tokenizer,
            maximum_extended_sequence_length=4096,
            maximum_position_id_exclusive=80,
        ).build(
            prompt_global_id="p",
            trajectory_id="position-fail",
            full_trajectory_input_ids=[11, 12],
            original_attention_mask=[1, 1],
            original_position_ids=[70, 71],
            prefix_end_positions=[2],
            canonical_answer="A",
        )
    except ValueError as error:
        position_closed = "logical position limit" in str(error)
    rows.append(
        {
            "case": "sequential_logical_position_exceeded",
            "pass": position_closed,
            "policy": "fail_closed",
        }
    )
    payload = {
        "schema": "exact_ig_batch_budget_tests_v3",
        "rows": rows,
        "all_pass": all(row["pass"] for row in rows),
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
    }
    _write(args.output, payload)
    print(json.dumps({"all_pass": payload["all_pass"], "cases": len(rows)}))
    raise SystemExit(0 if payload["all_pass"] else 2)


if __name__ == "__main__":
    main()
