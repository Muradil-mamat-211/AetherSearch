from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_rl.controller.dataset_view import (
    DeterministicNQHotpotLogicalView,
    _canonical_ground_truth,
)
from agentic_rl.exact_ig.target_schema import encode_exact_ig_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "exact_ig_official_alignment_v3_20260730"
)
MODEL_PATH = Path(
    "/root/autodl-tmp/search-r1-workspace/models/dpo_v2_final_model"
)
TRAIN_DATA = Path(
    "/root/autodl-tmp/search-r1-workspace/data/nq_hotpotqa_train/train.parquet"
)
LOGICAL_IDENTITY = (
    "b8bc8792a85e1172e52ceb5eaefb9c6065aa9c0dabf5fe4cb6004ddc4281710e"
)


def _token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canary-count", type=int, default=20)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Exact-IG V3 requires a fast tokenizer with offsets")
    view = DeterministicNQHotpotLogicalView(
        TRAIN_DATA,
        expected_identity_sha256=LOGICAL_IDENTITY,
    )

    counts = {
        "total_target_count": 0,
        "left_boundary_crossing_count": 0,
        "right_boundary_crossing_count": 0,
        "any_boundary_crossing_count": 0,
        "empty_answer_span_count": 0,
        "decode_mismatch_count": 0,
        "invalid_canonical_answer_count": 0,
        "other_tokenization_error_count": 0,
    }
    canaries: list[dict[str, Any]] = []
    non_boundary_canaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for logical_index in range(len(view)):
        source_index = view.source_index(logical_index)
        raw = view._frame.loc[source_index].to_dict()
        try:
            canonical_answer = _canonical_ground_truth(raw)
        except ValueError as error:
            counts["invalid_canonical_answer_count"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "logical_index": logical_index,
                        "source_index": source_index,
                        "error": repr(error),
                    }
                )
            continue
        counts["total_target_count"] += 1
        try:
            target = encode_exact_ig_target(tokenizer, canonical_answer)
        except ValueError as error:
            message = str(error)
            if "do not decode" in message:
                counts["decode_mismatch_count"] += 1
            elif "empty or invalid answer span" in message:
                counts["empty_answer_span_count"] += 1
            else:
                counts["other_tokenization_error_count"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "logical_index": logical_index,
                        "source_index": source_index,
                        "canonical_answer": canonical_answer,
                        "error": repr(error),
                    }
                )
            continue

        counts["left_boundary_crossing_count"] += int(
            target.left_boundary_crossing
        )
        counts["right_boundary_crossing_count"] += int(
            target.right_boundary_crossing
        )
        counts["any_boundary_crossing_count"] += int(
            target.boundary_crossing_any
        )
        entry = {
            "logical_index": logical_index,
            "source_index": source_index,
            "dataset_row_id": str(raw.get("id", "")),
            "data_source": str(raw.get("data_source", "")),
            "rendered_target": target.rendered_text,
            "canonical_answer": target.canonical_answer,
            "token_ids": list(target.token_ids),
            "token_decoded_pieces": [
                _token_piece(tokenizer, token_id)
                for token_id in target.token_ids
            ],
            "offset_mapping": [list(value) for value in target.offset_mapping],
            "answer_char_span": [
                target.answer_char_start,
                target.answer_char_end,
            ],
            "answer_token_span": [
                target.answer_token_start,
                target.answer_token_end,
            ],
            "answer_token_ids": list(target.answer_token_ids),
            "left_boundary_crossing": target.left_boundary_crossing,
            "right_boundary_crossing": target.right_boundary_crossing,
            "boundary_crossing_any": target.boundary_crossing_any,
            "full_target_token_ids_sha256": (
                target.full_target_token_ids_sha256
            ),
            "answer_span_token_ids_sha256": (
                target.answer_span_token_ids_sha256
            ),
        }
        if target.boundary_crossing_any and len(canaries) < args.canary_count:
            canaries.append(entry)
        elif (
            not target.boundary_crossing_any
            and len(non_boundary_canaries) < args.canary_count
        ):
            non_boundary_canaries.append(entry)

    if len(canaries) < args.canary_count:
        raise RuntimeError(
            f"Only {len(canaries)} boundary canaries found; "
            f"{args.canary_count} required"
        )
    total = counts["total_target_count"]
    report = {
        **counts,
        "boundary_crossing_rate": (
            counts["any_boundary_crossing_count"] / total if total else 0.0
        ),
        "logical_view_rows": len(view),
        "logical_view_identity_sha256": (
            view.identity.ordered_view_identity_sha256
        ),
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "canary_count": len(canaries),
        "error_samples": errors,
        "gate_pass": (
            counts["empty_answer_span_count"] == 0
            and counts["decode_mismatch_count"] == 0
            and counts["invalid_canonical_answer_count"] == 0
            and counts["other_tokenization_error_count"] == 0
            and total == len(view)
        ),
    }
    (output_dir / "EXACT_IG_BOUNDARY_CROSSING_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (
        output_dir / "EXACT_IG_TARGET_TOKENIZATION_AUDIT.jsonl"
    ).open("w", encoding="utf-8") as handle:
        for entry in (*canaries, *non_boundary_canaries):
            handle.write(
                json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
