from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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
    seed: int,
    nq_count: int,
    hotpotqa_count: int,
) -> dict[str, Any]:
    source = Path(validation_path).resolve()
    destination = Path(manifest_path).resolve()
    source_sha256 = _sha256_file(source)
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload["validation_path"] != str(source):
            raise RuntimeError("Fixed-eval validation path changed")
        if payload["validation_sha256"] != source_sha256:
            raise RuntimeError("Fixed-eval source parquet changed")
        if int(payload["seed"]) != int(seed):
            raise RuntimeError("Fixed-eval seed changed")
        if len(payload["rows"]) != int(nq_count) + int(hotpotqa_count):
            raise RuntimeError("Fixed-eval cardinality changed")
        if _manifest_digest(payload["rows"]) != payload["manifest_sha256"]:
            raise RuntimeError("Fixed-eval manifest identity changed")
        return payload

    frame = pd.read_parquet(
        source,
        columns=["id", "data_source"],
    )
    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    for data_source, count in (
        ("nq", int(nq_count)),
        ("hotpotqa", int(hotpotqa_count)),
    ):
        indices = frame.index[
            frame["data_source"].astype(str).str.lower() == data_source
        ].to_numpy(dtype=np.int64)
        if indices.size < count:
            raise RuntimeError(
                f"Fixed-eval source has only {indices.size} {data_source} rows"
            )
        sampled = np.sort(rng.choice(indices, size=count, replace=False))
        for source_index in sampled.tolist():
            row = frame.loc[int(source_index)]
            selected.append(
                {
                    "source_index": int(source_index),
                    "id": str(row["id"]),
                    "data_source": str(row["data_source"]),
                }
            )
    selected.sort(
        key=lambda row: (
            str(row["data_source"]),
            int(row["source_index"]),
            str(row["id"]),
        )
    )
    payload = {
        "schema_version": 1,
        "validation_path": str(source),
        "validation_sha256": source_sha256,
        "seed": int(seed),
        "counts": {
            "nq": int(nq_count),
            "hotpotqa": int(hotpotqa_count),
        },
        "rows": selected,
        "manifest_sha256": _manifest_digest(selected),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


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
