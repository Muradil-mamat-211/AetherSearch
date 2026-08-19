from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from agentic_rl.runtime.fixed_eval import create_or_validate_eval_manifest


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
