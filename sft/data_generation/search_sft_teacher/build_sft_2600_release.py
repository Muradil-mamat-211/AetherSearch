#!/usr/bin/env python3
"""Build the deterministic AetherSearch SFT-2600 Hub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from answer_utils import normalize_answer
from controlled_rollout import DIRECT_FINAL_THINK, parse_action
from published_sft_format import EOT, PUBLIC_FIELDS, public_prefix, validate_public_record


RELEASE_VERSION = "aethersearch_sft_2600_v1"
SHUFFLE_SEED = 42
RETRIEVAL_RECORDS = 2_000
DIRECT_RECORDS = 600
TOTAL_RECORDS = RETRIEVAL_RECORDS + DIRECT_RECORDS
EXPECTED_RETRIEVAL_SHA256 = "fec609652d3832c7a6c0ee2861c6f946b6cf7c3d3d40fc5d9be9b75df6325dcb"
EXPECTED_DIRECT_SHA256 = "388eb26ae87563cb07a00844ae3e96fb1a98f2775ae7453eceabde84b66d5b70"
EXPECTED_AUDIT_SHA256 = "a2795d87c357925214a100396da6a786e2bf2bf2626e941c58431ebd8117fd3e"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = " ".join(text.split())
    return text.rstrip("?？").strip()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"identity mismatch for {path}: expected {expected}, got {actual}")


def require_ids(rows: list[dict[str, Any]], start: int) -> None:
    expected = [f"{number:06d}" for number in range(start, start + len(rows))]
    actual = [row.get("id") for row in rows]
    if actual != expected:
        raise ValueError(f"non-contiguous input IDs beginning at {start:06d}")


def normalized_overlap_counts(raw: Any) -> dict[str, dict[str, int]]:
    if not isinstance(raw, dict):
        raise ValueError("invalid direct-answer overlap counts")
    result: dict[str, dict[str, int]] = {}
    for source, counts in raw.items():
        if not isinstance(source, str) or not isinstance(counts, dict):
            raise ValueError("invalid direct-answer overlap count entry")
        normalized = {
            "retrieval_trajectories": counts.get(
                "retrieval_trajectories", counts.get("existing_sft")
            ),
            "dpo": counts.get("dpo", counts.get("existing_dpo")),
            "rl_train": counts.get("rl_train"),
        }
        if any(not isinstance(value, int) or value < 0 for value in normalized.values()):
            raise ValueError(f"invalid direct-answer overlap counts for {source}")
        result[source] = normalized
    return result


def accepted_sources(checkpoint: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with sqlite3.connect(checkpoint) as database:
        rows = database.execute(
            "SELECT result_json FROM attempts WHERE status = 'accepted'"
        ).fetchall()
    for (payload,) in rows:
        attempt = json.loads(payload)
        key = (str(attempt["data_source"]), normalize_question(attempt["question"]))
        source_id = str(attempt["source_id"])
        if key in result and result[key] != source_id:
            raise ValueError(f"ambiguous accepted source ID for {key}")
        result[key] = source_id
    return result


def validate_direct_pair(public: dict[str, Any], audit: dict[str, Any]) -> None:
    if public["id"] != audit.get("metadata", {}).get("public_id"):
        raise ValueError(f"direct audit ID mismatch for {public['id']}")
    if public["question"] != audit.get("question"):
        raise ValueError(f"direct audit question mismatch for {public['id']}")
    body = public["full_trajectory_text"][len(public_prefix(public["question"])) : -len(EOT)]
    think, answer = parse_action(body, "answer")
    aliases = {normalize_answer(alias) for alias in audit.get("golden_answers") or []}
    if think != DIRECT_FINAL_THINK or normalize_answer(answer) not in aliases:
        raise ValueError(f"direct answer audit mismatch for {public['id']}")
    if "<search>" in body or "<information>" in body:
        raise ValueError(f"retrieval markup in direct answer {public['id']}")


def direct_provenance(
    old_id: str,
    row: dict[str, Any],
    audit: dict[str, Any],
    source_id: str,
    shuffle_key: str,
) -> dict[str, Any]:
    metadata = audit["metadata"]
    full = row["full_trajectory_text"]
    return {
        "previous_public_id": old_id,
        "record_id": audit["id"],
        "source_dataset": audit["data_source"],
        "data_source": audit["data_source"],
        "source_id": source_id,
        "source_record_id": f"{audit['data_source']}::{source_id}",
        "original_id": audit["id"],
        "source_sample_type": "direct_answer",
        "source_trajectory_type": audit["trajectory_type"],
        "source_prompt_sha256": sha256_text(audit["prompt"]),
        "source_target_sha256": sha256_text(audit["response"]),
        "pre_eot_full_trajectory_sha256": sha256_text(full[: -len(EOT)]),
        "full_trajectory_sha256": sha256_text(full),
        "teacher_model": metadata["teacher_model"],
        "teacher_thinking": metadata["teacher_thinking"],
        "prompt_version": metadata["prompt_version"],
        "generator_version": metadata["generator_version"],
        "export_policy_version": metadata["export_policy_version"],
        "overlap_policy": metadata["overlap_policy"],
        "raw_teacher_action_sha256": metadata["raw_teacher_action_sha256"],
        "api_request_sha256": metadata["api_request_sha256"],
        "api_response_sha256": metadata["api_response_sha256"],
        "shuffle_key_sha256": shuffle_key,
    }


def retrieval_provenance(
    old_id: str,
    old: dict[str, Any],
    row: dict[str, Any],
    shuffle_key: str,
) -> dict[str, Any]:
    if old.get("id") != old_id:
        raise ValueError(f"retrieval provenance mismatch for {old_id}")
    if old.get("full_trajectory_sha256") != sha256_text(row["full_trajectory_text"]):
        raise ValueError(f"retrieval provenance trajectory hash mismatch for {old_id}")
    result = {
        "previous_public_id": old_id,
        "prior_release_previous_public_id": old.get("previous_public_id"),
    }
    for key, value in old.items():
        if key not in {"id", "previous_public_id", "shuffle_key_sha256"}:
            result[key] = value
    result["prior_release_shuffle_key_sha256"] = old.get("shuffle_key_sha256")
    result["shuffle_key_sha256"] = shuffle_key
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--retrieval-provenance", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--direct-audit", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require_sha(args.retrieval, EXPECTED_RETRIEVAL_SHA256)
    require_sha(args.direct, EXPECTED_DIRECT_SHA256)
    require_sha(args.direct_audit, EXPECTED_AUDIT_SHA256)
    retrieval = read_jsonl(args.retrieval)
    direct = read_jsonl(args.direct)
    audit = read_jsonl(args.direct_audit)
    retrieval_provenance_rows = read_jsonl(args.retrieval_provenance)
    direct_manifest = json.loads(args.direct_manifest.read_text(encoding="utf-8"))
    if (len(retrieval), len(direct), len(audit), len(retrieval_provenance_rows)) != (
        RETRIEVAL_RECORDS,
        DIRECT_RECORDS,
        DIRECT_RECORDS,
        RETRIEVAL_RECORDS,
    ):
        raise ValueError("input row count mismatch")
    require_ids(retrieval, 1)
    require_ids(direct, RETRIEVAL_RECORDS + 1)

    retrieval_lineage = {row["id"]: row for row in retrieval_provenance_rows}
    if len(retrieval_lineage) != RETRIEVAL_RECORDS:
        raise ValueError("retrieval provenance IDs are not unique")
    source_ids = accepted_sources(args.checkpoint)
    records: list[dict[str, Any]] = []
    audit_by_old_id: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(retrieval + direct, 1):
        old_id = row.get("id")
        if old_id != f"{number:06d}":
            raise ValueError(f"unexpected combined ID at row {number}: {old_id}")
        errors = validate_public_record(row)
        if errors:
            raise ValueError(f"invalid input record {old_id}: {errors}")
        if number > RETRIEVAL_RECORDS:
            detail = audit[number - RETRIEVAL_RECORDS - 1]
            validate_direct_pair(row, detail)
            audit_by_old_id[old_id] = detail
        full_hash = sha256_text(row["full_trajectory_text"])
        shuffle_key = sha256_text(f"{SHUFFLE_SEED}|{old_id}|{full_hash}")
        records.append(
            {
                "old_id": old_id,
                "row": row,
                "full_hash": full_hash,
                "shuffle_key": shuffle_key,
            }
        )

    normalized_questions = [normalize_question(item["row"]["question"]) for item in records]
    if not all(normalized_questions) or len(set(normalized_questions)) != TOTAL_RECORDS:
        raise ValueError("empty or duplicate normalized question in release")
    records.sort(key=lambda item: (item["shuffle_key"], item["old_id"]))

    final_rows: list[dict[str, Any]] = []
    final_provenance: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    direct_source_counts: Counter[str] = Counter()
    moved = 0
    for number, item in enumerate(records, 1):
        new_id = f"{number:06d}"
        old_id = item["old_id"]
        row = dict(item["row"])
        row["id"] = new_id
        if new_id != old_id:
            moved += 1
        errors = validate_public_record(row)
        if errors:
            raise ValueError(f"invalid shuffled record {new_id}: {errors}")
        final_rows.append(row)
        type_counts[row["trajectory_type"]] += 1
        depth_counts[str(row["search_count"])] += 1

        if old_id in retrieval_lineage:
            provenance = retrieval_provenance(
                old_id, retrieval_lineage[old_id], item["row"], item["shuffle_key"]
            )
        else:
            detail = audit_by_old_id[old_id]
            source = str(detail["data_source"])
            key = (source, normalize_question(detail["question"]))
            if key not in source_ids:
                raise ValueError(f"missing checkpoint source ID for {old_id}")
            provenance = direct_provenance(
                old_id,
                item["row"],
                detail,
                source_ids[key],
                item["shuffle_key"],
            )
            direct_source_counts[source] += 1
        final_provenance.append({"id": new_id, **provenance})

    if moved < TOTAL_RECORDS - 10:
        raise ValueError(f"global shuffle moved too few records: {moved}")
    expected_types = {"single_search": 1025, "multi_search": 975, "direct_answer": 600}
    expected_depths = {"0": 600, "1": 1025, "2": 667, "3": 265, "4": 43}
    if dict(type_counts) != expected_types or dict(depth_counts) != expected_depths:
        raise ValueError("release composition mismatch")
    if dict(direct_source_counts) != {"nq": 300, "web_questions": 300}:
        raise ValueError("direct-answer source composition mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "final_sft_2600.jsonl"
    provenance_path = args.output_dir / "provenance_manifest.jsonl"
    write_jsonl(data_path, final_rows)
    write_jsonl(provenance_path, final_provenance)
    manifest = {
        "dataset_name": "search_sft_2600",
        "release_version": RELEASE_VERSION,
        "total_records": TOTAL_RECORDS,
        "trajectory_types": dict(type_counts),
        "search_depth": dict(depth_counts),
        "direct_answer_sources": dict(direct_source_counts),
        "training_unit": "full_trajectory",
        "loss_mask_policy": "assistant_generated_except_information",
        "public_schema": list(PUBLIC_FIELDS),
        "ordering": "deterministic_global_shuffle",
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_key": 'SHA256("42|" + old_public_id + "|" + SHA256(full_trajectory_text_after_eot))',
        "assistant_eot": EOT,
        "assistant_eot_supervised": True,
        "direct_answer_teacher": {
            "model": direct_manifest["teacher_model"],
            "thinking": direct_manifest["thinking"],
            "tools_registered": direct_manifest["tools_registered"],
            "generator_version": direct_manifest["generator_version"],
            "prompt_version": direct_manifest["prompt_version"],
            "export_policy_version": direct_manifest["export_policy_version"],
            "overlap_policy_version": direct_manifest["overlap_policy_version"],
        },
        "input_artifacts": {
            "retrieval_trajectories_sha256": EXPECTED_RETRIEVAL_SHA256,
            "direct_answer_600_sha256": EXPECTED_DIRECT_SHA256,
            "direct_answer_audit_600_sha256": EXPECTED_AUDIT_SHA256,
        },
        "direct_answer_candidate_pool_overlap_counts": normalized_overlap_counts(
            direct_manifest["overlap_counts"]
        ),
        "selected_direct_answer_overlap_counts": {
            "retrieval_trajectories": 0,
            "dpo": 0,
            "rl_train": 0,
        },
        "integrity_audit": {
            "total_records": TOTAL_RECORDS,
            "unique_id": TOTAL_RECORDS,
            "ids_exactly_000001_to_002600": True,
            "unique_normalized_questions": TOTAL_RECORDS,
            "public_jsonl_fields_exactly_5": True,
            "every_trajectory_ends_with_im_end": TOTAL_RECORDS,
            "direct_answer_without_search_or_information": DIRECT_RECORDS,
            "single_search": type_counts["single_search"],
            "multi_search": type_counts["multi_search"],
            "direct_answer": type_counts["direct_answer"],
            "depth_0": depth_counts["0"],
            "depth_1": depth_counts["1"],
            "depth_2": depth_counts["2"],
            "depth_3": depth_counts["3"],
            "depth_4": depth_counts["4"],
            "deterministic_shuffle_reproducible": True,
            "records_moved_by_global_shuffle": moved,
            "provenance_mapping_complete": len(final_provenance),
        },
        "release_files": {
            "training_data": data_path.name,
            "training_data_sha256": sha256_file(data_path),
            "provenance": provenance_path.name,
            "provenance_sha256": sha256_file(provenance_path),
        },
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": len(final_rows),
                "trajectory_types": dict(type_counts),
                "search_depth": dict(depth_counts),
                "direct_sources": dict(direct_source_counts),
                "moved": moved,
                "data_sha256": sha256_file(data_path),
                "provenance_sha256": sha256_file(provenance_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
