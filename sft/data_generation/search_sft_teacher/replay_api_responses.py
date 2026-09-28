#!/usr/bin/env python3
"""Compare saved first-decision responses offline; no API, RAG or training export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controlled_rollout import (DEEPSEEK_GENERATOR_VERSION, SEARCH_EXTRACTION_POLICY,
                                PROMPT_VERSION, QUERY_POLICY_VERSION, TOOL_POLICY_VERSION, WORKSPACE, Rejected,
                                evidence_citations, normalize, parse_action)
from deepseek_client import extract_search_message
from test_api_20_questions import write_json
from token_budget import student_budget


def evaluate(row: dict[str, Any]) -> dict[str, Any]:
    result = {"number": row["number"], "question": row["question"], "data_source": row["data_source"],
              "before_status": row["status"], "before_reason": row.get("reason"),
              "retriever_executed": False, "format_valid": False}
    responses = row.get("responses")
    if not isinstance(responses, list) or len(responses) != 1:
        raise ValueError("replay_requires_one_saved_response_per_question")
    raw = responses[0]
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("replay_requires_one_response_choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("invalid_saved_message")
    try:
        if choice.get("finish_reason") not in {"stop", "tool_calls"}:
            raise Rejected("incomplete_model_output")
        search = extract_search_message(message, finish_reason=choice["finish_reason"],
                                        tool_choice=row["requests"][0]["tool_choice"], response_id=raw["id"])
        if search is not None:
            _, query = parse_action(search["action"], "search")
            result.update(search)
            result.update(status="search_requested", query=query, format_valid=True,
                          action_student_tokens=student_budget().action_tokens(search["action"]))
        else:
            if choice["finish_reason"] != "stop":
                raise Rejected("missing_final_answer")
            content = message.get("content")
            reason, answer = parse_action(content, "answer")
            evidence_citations(reason, 0)
            result.update(status="direct_answer", action=content, answer=answer, format_valid=True,
                          action_student_tokens=student_budget().action_tokens(content),
                          normalized_reference_match=normalize(answer) in {normalize(a) for a in row["golden_answers"]})
    except Rejected as exc:
        result.update(status="rejected", reason=str(exc))
    result["recovered"] = row["status"] == "rejected" and result["status"] != "rejected"
    return result


def run(input_path: Path, output_dir: Path | None = None, *, expected_questions: int = 100) -> dict[str, Any]:
    started = time.perf_counter()
    raw_bytes = input_path.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != expected_questions or len({normalize(row["question"]) for row in rows}) != expected_questions:
        raise ValueError("replay_question_count_or_uniqueness_mismatch")
    results = [evaluate(row) for row in rows]
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = WORKSPACE / "logs/search_sft_teacher" / f"controller_replay_{expected_questions}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    counts = lambda values: dict(Counter(values))
    report = {"test_scope": "offline_first_decision_replay", "input": str(input_path.resolve()),
              "input_sha256": hashlib.sha256(raw_bytes).hexdigest(), "questions": len(rows),
              "api_requests": 0, "retriever_executed": False, "training_data_created": False,
              "generator_version": DEEPSEEK_GENERATOR_VERSION, "tool_policy": TOOL_POLICY_VERSION,
              "search_extraction_policy": SEARCH_EXTRACTION_POLICY,
              "query_policy": QUERY_POLICY_VERSION, "current_prompt_version": PROMPT_VERSION,
              "fresh_prompt_test": False,
              "before_statuses": counts(row["status"] for row in rows),
              "after_statuses": counts(row["status"] for row in results),
              "before_rejection_reasons": counts(row["reason"] for row in rows if row["status"] == "rejected"),
              "after_rejection_reasons": counts(row["reason"] for row in results if row["status"] == "rejected"),
              "recovered_count": sum(row["recovered"] for row in results),
              "recovered_numbers": [row["number"] for row in results if row["recovered"]],
              "regressions": [row["number"] for row in results if row["before_status"] != "rejected" and row["status"] == "rejected"],
              "accepted_search_sources": counts(row["raw_action_source"] for row in results if row["status"] == "search_requested"),
              "source_statistics": {name: {"before": counts(row["status"] for row in rows if row["data_source"] == name),
                                           "after": counts(row["status"] for row in results if row["data_source"] == name),
                                           "skip_reasons": counts(row["reason"] for row in results if row["data_source"] == name and row["status"] == "rejected")}
                                    for name in sorted({row["data_source"] for row in rows})},
              "max_action_student_tokens": max((row.get("action_student_tokens", 0) for row in results), default=0),
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    with (output_dir / "replay_results.jsonl").open("x", encoding="utf-8") as stream:
        os.chmod(stream.name, 0o600)
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    write_json(output_dir / "summary.json", report)
    with (output_dir / "comparison.txt").open("x", encoding="utf-8") as stream:
        os.chmod(stream.name, 0o600)
        stream.write("OFFLINE FIRST-DECISION REPLAY ONLY; no new API requests, retrieval or training data.\n\n")
        for result in results:
            stream.write(f"{result['number']}. {result['question']}\nBefore: {result['before_status']} {result['before_reason']}\n")
            stream.write(f"After: {result['status']} {result.get('reason')}\nAction: {result.get('action', '')}\n")
            if "extraction" in result:
                stream.write("Extraction: " + json.dumps(result["extraction"], ensure_ascii=False) + "\n")
            stream.write("\n")
    print(json.dumps({"output_dir": str(output_dir.resolve()), **report}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-questions", type=int, default=100)
    args = parser.parse_args()
    run(args.input, args.output_dir, expected_questions=args.expected_questions)
