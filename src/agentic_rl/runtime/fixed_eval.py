from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from agentic_rl.controller.dataset_view import (
    _canonical_ground_truth,
    _ground_truth_aliases,
    _prompt_messages,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{int(row['source_index'])}\0{row['id']}\0"
                f"{row['data_source']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def create_or_validate_eval_manifest(
    *,
    validation_path: str | Path,
    manifest_path: str | Path,
    manifest_mode: str = "full_validation",
    expected_validation_sha256: str | None = None,
    expected_row_count: int | None = None,
    expected_source_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    source = Path(validation_path).resolve()
    destination = Path(manifest_path).resolve()
    source_sha256 = _sha256_file(source)
    mode = str(manifest_mode)
    if mode != "full_validation":
        raise RuntimeError("Fixed evaluation requires full_validation mode")
    if (
        expected_validation_sha256 is not None
        and source_sha256 != str(expected_validation_sha256)
    ):
        raise RuntimeError("Fixed-eval source parquet SHA-256 changed")

    frame = pd.read_parquet(source, columns=["id", "data_source"])
    frame = frame.reset_index(drop=True)
    source_counts = {
        str(key): int(value)
        for key, value in frame["data_source"]
        .astype(str)
        .str.lower()
        .value_counts()
        .sort_index()
        .items()
    }
    normalized_expected_counts = (
        {
            str(key).lower(): int(value)
            for key, value in expected_source_counts.items()
        }
        if expected_source_counts is not None
        else None
    )
    if expected_row_count is not None and len(frame) != int(expected_row_count):
        raise RuntimeError("Fixed-eval source parquet row count changed")
    if (
        normalized_expected_counts is not None
        and source_counts != normalized_expected_counts
    ):
        raise RuntimeError("Fixed-eval source dataset counts changed")

    selected = [
        {
            "source_index": int(source_index),
            "id": str(row.id),
            "data_source": str(row.data_source),
        }
        for source_index, row in enumerate(frame.itertuples(index=False))
    ]

    expected_manifest_sha256 = _manifest_digest(selected)
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload["validation_path"] != str(source):
            raise RuntimeError("Fixed-eval validation path changed")
        if payload["validation_sha256"] != source_sha256:
            raise RuntimeError("Fixed-eval source parquet changed")
        if str(payload.get("manifest_mode")) != mode:
            raise RuntimeError("Fixed-eval manifest mode changed")
        if len(payload["rows"]) != len(selected):
            raise RuntimeError("Fixed-eval cardinality changed")
        if _manifest_digest(payload["rows"]) != payload["manifest_sha256"]:
            raise RuntimeError("Fixed-eval manifest identity changed")
        if payload["manifest_sha256"] != expected_manifest_sha256:
            raise RuntimeError("Fixed-eval manifest rows changed")
        if payload["rows"] != selected:
            raise RuntimeError("Fixed-eval manifest does not match source parquet")
        return payload
    payload = {
        "schema_version": 2,
        "manifest_mode": mode,
        "validation_path": str(source),
        "validation_sha256": source_sha256,
        "counts": source_counts,
        "rows": selected,
        "manifest_sha256": expected_manifest_sha256,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def create_or_validate_eval_manifest_from_config(
    *,
    validation_path: str | Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return create_or_validate_eval_manifest(
        validation_path=validation_path,
        manifest_path=evaluation["manifest_path"],
        manifest_mode=str(evaluation.get("manifest_mode", "full_validation")),
        expected_validation_sha256=evaluation.get(
            "expected_validation_sha256"
        ),
        expected_row_count=(
            int(evaluation["expected_row_count"])
            if evaluation.get("expected_row_count") is not None
            else None
        ),
        expected_source_counts=evaluation.get("expected_source_counts"),
    )


def load_eval_rows(
    *,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    source = Path(str(manifest["validation_path"])).resolve()
    if _sha256_file(source) != str(manifest["validation_sha256"]):
        raise RuntimeError("Fixed-eval parquet changed after manifest creation")
    frame = pd.read_parquet(source)
    rows = []
    for eval_index, entry in enumerate(manifest["rows"]):
        source_index = int(entry["source_index"])
        raw = frame.loc[source_index].to_dict()
        if (
            str(raw["id"]) != str(entry["id"])
            or str(raw["data_source"]) != str(entry["data_source"])
        ):
            raise RuntimeError("Fixed-eval row identity mismatch")
        aliases = _ground_truth_aliases(raw)
        rows.append(
            {
                **raw,
                "logical_index": int(eval_index),
                "source_index": source_index,
                "prompt_global_id": (
                    f"eval:{raw['data_source']}:{raw['id']}:{source_index}"
                ),
                "prompt_messages": _prompt_messages(raw["prompt"]),
                "gold_aliases": aliases,
                "canonical_answer": _canonical_ground_truth(raw),
            }
        )
    return tuple(rows)
