#!/usr/bin/env python3
"""Validate the DeepSeek direct-answer branch and its audit rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from answer_utils import normalize_answer
from controlled_rollout import DIRECT_FINAL_THINK, parse_action
from generate_direct_answer_sft import (EXPECTED_DPO_SHA256, EXPECTED_RETRIEVAL_SHA256,
                                        SOURCE_QUOTAS,
                                        checked_questions, normalize_question,
                                        read_jsonl, rl_exclusions, read_arrow,
                                        sha256_file)
from published_sft_format import EOT, PUBLIC_FIELDS, public_prefix, validate_public_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--retrieval-input", type=Path, required=True)
    parser.add_argument("--dpo-input", type=Path, required=True)
    parser.add_argument("--rl-train", type=Path, required=True)
    parser.add_argument("--nq-arrow", type=Path, required=True)
    args = parser.parse_args()
    public, audit = read_jsonl(args.input), read_jsonl(args.audit)
    if len(public) != 600 or len(audit) != 600:
        raise SystemExit(f"FAIL row_count public={len(public)} audit={len(audit)}")
    retrieval, _ = checked_questions(
        args.retrieval_input, EXPECTED_RETRIEVAL_SHA256, 2000
    )
    dpo, _ = checked_questions(args.dpo_input, EXPECTED_DPO_SHA256, 2126)
    nq = read_arrow(args.nq_arrow, "nq")
    rl, _ = rl_exclusions(args.rl_train, nq)
    exclusions = retrieval | dpo | rl
    seen_ids, seen_questions, sources = set(), set(), Counter()
    errors: list[str] = []
    for number, (row, detail) in enumerate(zip(public, audit), 1):
        if tuple(row) != PUBLIC_FIELDS:
            errors.append(f"{number}:fields")
        errors.extend(f"{number}:{error}" for error in validate_public_record(row))
        if row.get("id") != f"{2000 + number:06d}" or row.get("id") in seen_ids:
            errors.append(f"{number}:id")
        seen_ids.add(row.get("id"))
        normalized = normalize_question(row.get("question"))
        if not normalized or normalized in seen_questions or normalized in exclusions:
            errors.append(f"{number}:question_overlap")
        seen_questions.add(normalized)
        if row.get("question") != detail.get("question") or detail.get("metadata", {}).get("public_id") != row.get("id"):
            errors.append(f"{number}:audit_pairing")
        sources[detail.get("data_source")] += 1
        if row.get("trajectory_type") != "direct_answer" or row.get("search_count") != 0:
            errors.append(f"{number}:trajectory")
            continue
        full = row.get("full_trajectory_text", "")
        prefix = public_prefix(row["question"])
        body = full[len(prefix):-len(EOT)] if full.startswith(prefix) and full.endswith(EOT) else ""
        try:
            think, answer = parse_action(body, "answer")
        except Exception:
            errors.append(f"{number}:action")
            continue
        if think != DIRECT_FINAL_THINK or any(tag in body for tag in ("<search>", "<information>")):
            errors.append(f"{number}:direct_protocol")
        aliases = detail.get("golden_answers") or []
        if normalize_answer(answer) not in {normalize_answer(alias) for alias in aliases}:
            errors.append(f"{number}:gold_match")
        meta = detail.get("metadata") or {}
        if (meta.get("teacher_thinking") != "disabled" or meta.get("registered_tools") != []
                or meta.get("normalized_answer_match") is not True):
            errors.append(f"{number}:audit_policy")
    if dict(sources) != SOURCE_QUOTAS:
        errors.append(f"source_quotas:{dict(sources)}")
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors[:100], "error_count": len(errors)}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps({"status": "PASS", "records": len(public), "sources": dict(sources),
                      "search_count": 0, "retrieval_overlap": 0,
                      "dpo_overlap": 0, "rl_overlap": 0,
                      "sha256": sha256_file(args.input)}, sort_keys=True))


if __name__ == "__main__":
    main()
