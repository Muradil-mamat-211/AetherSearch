from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from agentic_rl.runtime.fixed_eval import (
    create_or_validate_eval_manifest,
    load_eval_rows,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_validation_manifest_preserves_every_source_row(tmp_path: Path) -> None:
    validation = tmp_path / "test.parquet"
    manifest_path = tmp_path / "full_manifest.json"
    frame = pd.DataFrame(
        {
            "id": ["n0", "t0", "h0", "n1"],
            "data_source": ["nq", "triviaqa", "hotpotqa", "nq"],
        }
    )
    frame.to_parquet(validation, index=False)

    manifest = create_or_validate_eval_manifest(
        validation_path=validation,
        manifest_path=manifest_path,
        manifest_mode="full_validation",
        expected_validation_sha256=_sha256(validation),
        expected_row_count=4,
        expected_source_counts={"hotpotqa": 1, "nq": 2, "triviaqa": 1},
    )

    assert manifest["manifest_mode"] == "full_validation"
    assert manifest["counts"] == {"hotpotqa": 1, "nq": 2, "triviaqa": 1}
    assert [row["source_index"] for row in manifest["rows"]] == [0, 1, 2, 3]
    assert [row["id"] for row in manifest["rows"]] == frame["id"].tolist()
    assert create_or_validate_eval_manifest(
        validation_path=validation,
        manifest_path=manifest_path,
        manifest_mode="full_validation",
        expected_validation_sha256=_sha256(validation),
        expected_row_count=4,
        expected_source_counts={"hotpotqa": 1, "nq": 2, "triviaqa": 1},
    ) == manifest


def test_full_validation_manifest_rejects_partial_expectation(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "test.parquet"
    pd.DataFrame(
        {"id": ["n0", "h0"], "data_source": ["nq", "hotpotqa"]}
    ).to_parquet(validation, index=False)

    with pytest.raises(RuntimeError, match="dataset counts changed"):
        create_or_validate_eval_manifest(
            validation_path=validation,
            manifest_path=tmp_path / "manifest.json",
            manifest_mode="full_validation",
            expected_row_count=2,
            expected_source_counts={"nq": 1},
        )


def test_fixed_eval_rejects_non_full_mode(tmp_path: Path) -> None:
    validation = tmp_path / "test.parquet"
    pd.DataFrame(
        {"id": ["n0"], "data_source": ["nq"]}
    ).to_parquet(validation, index=False)

    with pytest.raises(RuntimeError, match="requires full_validation"):
        create_or_validate_eval_manifest(
            validation_path=validation,
            manifest_path=tmp_path / "manifest.json",
            manifest_mode="partial",
        )


def test_eval_1400_jsonl_preserves_order_and_keeps_gold_out_of_prompt(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "eval_1400.jsonl"
    records = [
        {
            "id": "eval1400_0001",
            "question": "Where was the scientist born?",
            "source_dataset": "NQ",
            "answers": ["Secretville", "Secret Town"],
            "context": "Evaluator-only supporting evidence",
            "prompt": [{"role": "user", "content": "Leaked gold: Secretville"}],
        },
        {
            "id": "eval1400_0002",
            "question": "Who founded the company?",
            "source_dataset": "MuSiQue",
            "answers": ["Secret Founder"],
        },
    ]
    validation.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = create_or_validate_eval_manifest(
        validation_path=validation,
        manifest_path=tmp_path / "manifest.json",
        expected_validation_sha256=_sha256(validation),
        expected_row_count=2,
        expected_source_counts={"nq": 1, "musique": 1},
    )
    rows = load_eval_rows(manifest=manifest)

    assert [row["id"] for row in rows] == [record["id"] for record in records]
    assert [row["source_index"] for row in rows] == [0, 1]
    assert [row["data_source"] for row in rows] == ["nq", "musique"]
    assert rows[0]["gold_aliases"] == ("Secretville", "Secret Town")
    assert rows[0]["canonical_answer"] == "Secretville"
    assert rows[1]["gold_aliases"] == ("Secret Founder",)
    assert rows[0]["prompt_global_id"] == "eval:nq:eval1400_0001:0"

    expected_prompt = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
        "and it will return the top searched results between <information> and </information>. "
        "You can search as many times as your want. If you find no further external knowledge needed, "
        "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
        "For example, <answer> Beijing </answer>. Question: Where was the scientist born?\n"
    )
    assert rows[0]["prompt_messages"] == (
        {"role": "user", "content": expected_prompt},
    )
    assert "context" not in rows[0]
    for row, record in zip(rows, records):
        assert row["prompt_messages"][0]["content"].endswith(
            f"Question: {record['question']}\n"
        )
        for answer in record["answers"]:
            assert answer not in row["prompt_messages"][0]["content"]


def test_eval_1400_jsonl_rejects_gold_changes_after_manifest_creation(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "eval_1400.jsonl"
    record = {
        "id": "eval1400_0001",
        "question": "Where was the scientist born?",
        "source_dataset": "NQ",
        "answers": ["Secretville"],
    }
    validation.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = create_or_validate_eval_manifest(
        validation_path=validation,
        manifest_path=tmp_path / "manifest.json",
    )
    record["answers"] = ["Changed gold"]
    validation.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after manifest creation"):
        load_eval_rows(manifest=manifest)
