#!/usr/bin/env python3
"""Real first-decision API tests (20 by default), without training data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import secrets
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controlled_rollout import (DEEPSEEK_GENERATOR_VERSION, PROMPT, PROMPT_VERSION, SEARCH_ACTION_SOURCE, SEARCH_EXTRACTION_POLICY,
                                QUERY_POLICY_VERSION, TOOL, TOOL_POLICY_VERSION, WORKSPACE, Rejected,
                                evidence_citations, normalize, parse_action, response_items)
from deepseek_client import API_URL, DEFAULT_THINKING_MAX_TOKENS, APIError, DeepSeekClient
from deepseek_key import load_key_file
from deepseek_rollout import doctor
from token_budget import MAX_SEARCH_TURNS, student_budget


def without_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        result = {k: without_reasoning(v) for k, v in value.items() if k != "reasoning_content"}
        if "reasoning_content" in value:
            result["reasoning_content_omitted"] = True
            result["reasoning_content_chars"] = len(value["reasoning_content"] or "")
        return result
    if isinstance(value, list):
        return [without_reasoning(v) for v in value]
    return value


def question_history(root: Path) -> tuple[set[str], list[str]]:
    excluded, files = set(), []
    for path in sorted(root.glob("max_thinking_*/input_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for row in manifest["questions"]:
            excluded.add(normalize(row["question"]))
        files.append(str(path.resolve()))
    for path in sorted(root.glob("exact_search_live_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        excluded.add(normalize(summary["question"]))
        files.append(str(path.resolve()))
    return excluded, files


def select_questions(source: Path, *, num_questions: int = 20, seed: int = 42,
                     excluded: set[str] | None = None) -> list[dict[str, Any]]:
    if not 1 <= num_questions <= 1000:
        raise ValueError("num_questions must be between 1 and 1000")
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in ("nq", "triviaqa", "web_questions")}
    seen = {"who is barack hussein obama ii"} | (excluded or set())
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            original = json.loads(line)
            question, answers = original.get("question"), original.get("golden_answers")
            name = original.get("data_source")
            if name not in groups or not isinstance(question, str) or not isinstance(answers, list) or not answers:
                continue
            key = normalize(question)
            if key in seen:
                continue
            seen.add(key)
            groups[name].append({"id": original["id"], "question": question, "golden_answers": answers,
                                 "data_source": name, "split": original.get("split")})
    rng = random.Random(seed)
    selected = []
    if num_questions == 20 and excluded is None:
        # Preserve the original regression batch for prompt-version comparisons.
        selected = [{"id": "obama-regression", "question": "Who is Barack Hussein Obama II?",
                     "golden_answers": [], "data_source": "manual_regression", "split": "diagnostic"}]
        counts = {"nq": 7, "triviaqa": 6, "web_questions": 6}
    else:
        if sum(map(len, groups.values())) < num_questions:
            raise ValueError("not_enough_unseen_questions")
        counts = dict.fromkeys(groups, 0)
        for _ in range(num_questions):
            available = [name for name in groups if counts[name] < len(groups[name])]
            name = min(available, key=lambda name: counts[name])
            counts[name] += 1
    for name, count in counts.items():
        if len(groups[name]) < count:
            raise ValueError(f"not_enough_questions:{name}")
        selected.extend(rng.sample(groups[name], count))
    return selected


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def protocol_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    messages = [choice["message"] for result in results for response in result["responses"]
                for choice in response.get("choices", []) if isinstance(choice.get("message"), dict)]
    tool_messages = [message for message in messages if message.get("tool_calls")]
    invalid_arguments = 0
    for message in tool_messages:
        for call in message["tool_calls"]:
            try:
                arguments = json.loads(call["function"]["arguments"])
                action = arguments.get("action") if isinstance(arguments, dict) else None
                valid = isinstance(action, str) and re.fullmatch(TOOL["parameters"]["properties"]["action"]["pattern"], action)
            except (KeyError, TypeError, ValueError):
                valid = False
            invalid_arguments += not bool(valid)
    return {"rejection_reasons": dict(Counter(r["reason"] for r in results if r["status"] == "rejected")),
            "raw_tool_call_responses": len(tool_messages),
            "mixed_tool_output_responses": sum(m.get("content") not in (None, "") for m in tool_messages),
            "invalid_raw_action_schema": invalid_arguments,
            "special_decline_tag_outputs": sum("<abstain>" in json.dumps(m) for m in messages)}


def run(source: Path, output_dir: Path | None = None, *, num_questions: int = 20,
        seed: int = 42, exclude_history: bool = False) -> None:
    excluded, history_files = question_history(WORKSPACE / "logs/search_sft_teacher") if exclude_history else (None, [])
    questions = select_questions(source, num_questions=num_questions, seed=seed, excluded=excluded)
    selected_keys = {normalize(row["question"]) for row in questions}
    if len(questions) != num_questions or len(selected_keys) != num_questions or selected_keys & (excluded or set()):
        raise ValueError("question_selection_not_unique_or_disjoint")
    budget = student_budget()
    health = doctor()
    key = load_key_file()
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = WORKSPACE / "logs/search_sft_teacher" / f"max_thinking_{num_questions}_{stamp}_{secrets.token_hex(3)}"
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    settings = {"test_scope": "first_decision_only", "training_data_created": False,
                "model": "deepseek-flash", "thinking": {"type": "enabled"}, "reasoning_effort": "max",
                "max_tokens": DEFAULT_THINKING_MAX_TOKENS, "api_endpoint": API_URL,
                "strict": True, "tool_choice": "auto", "num_questions": num_questions, "seed": seed,
                "exclude_history": exclude_history, "excluded_question_count": len(excluded or set()),
                "exclusion_history_files": history_files, "question_overlap_with_history": 0 if exclude_history else None,
                "prompt_version": PROMPT_VERSION, "generator_version": DEEPSEEK_GENERATOR_VERSION,
                "tool_policy": TOOL_POLICY_VERSION, "search_action_source": SEARCH_ACTION_SOURCE,
                "search_extraction_policy": SEARCH_EXTRACTION_POLICY,
                "query_policy": QUERY_POLICY_VERSION,
                "source_file": str(source.resolve()), "preflight": health,
                "student_500_token_budget_checked": True,
                "rollout_budget": budget.specification(), "max_searches": MAX_SEARCH_TURNS}
    write_json(output_dir / "input_manifest.json", {"settings": settings, "questions": questions})
    results = []
    total_started = time.perf_counter()
    print(json.dumps({"event": "test_started", "output_dir": str(output_dir), "settings": settings}), flush=True)
    with (output_dir / "results.jsonl").open("x", encoding="utf-8") as journal:
        os.chmod(output_dir / "results.jsonl", 0o600)
        for number, row in enumerate(questions, 1):
            client = DeepSeekClient("deepseek-flash", key, thinking="enabled", reasoning_effort="max",
                                    timeout=120, retries=0, max_requests=2)
            real_post = client._post
            requests, responses = [], []

            def traced_post(payload):
                requests.append(without_reasoning(copy.deepcopy(payload)))
                raw, attempts = real_post(payload)
                responses.append(without_reasoning(raw))
                return raw, attempts

            client._post = traced_post
            started = time.perf_counter()
            result = {"number": number, **row, "retriever_executed": False, "format_valid": False}
            fatal = False
            try:
                raw = client.create([{"role": "user", "content": PROMPT.format(question=row["question"], max_searches=MAX_SEARCH_TURNS)}], "auto")
                call, text = response_items(raw)
                if call is not None:
                    action = json.loads(call["arguments"])["action"]
                    _, query = parse_action(action, "search")
                    result.update(status="search_requested", action=action, query=query, format_valid=True,
                                  action_student_tokens=budget.action_tokens(action),
                                  raw_action_source=client.calls[-1]["raw_action_source"],
                                  native_tool_call=client.calls[-1]["native_tool_call"],
                                  extraction=client.calls[-1]["extraction"],
                                  continuation="requires_real_rag" if health["ready"] else "blocked_rag_unavailable")
                else:
                    reason, answer = parse_action(text, "answer")
                    evidence_citations(reason, 0)
                    correct = normalize(answer) in {normalize(a) for a in row["golden_answers"]} if row["golden_answers"] else None
                    result.update(status="direct_answer", action=text, answer=answer, answer_chars=len(answer),
                                  action_student_tokens=budget.action_tokens(text), answer_student_tokens=budget.count(answer),
                                  format_valid=True, normalized_reference_match=correct)
            except Rejected as exc:
                result.update(status="rejected", reason=str(exc))
            except APIError as exc:
                result.update(status="api_error", reason=str(exc))
                fatal = str(exc) in {"deepseek_http_400", "deepseek_http_401", "deepseek_http_402", "deepseek_http_403"}
            result.update(seconds=round(time.perf_counter() - started, 3), http_requests=client.request_count,
                          answer_repairs=sum(c.get("discarded_reason") == "answer_action_token_budget_exceeded" for c in client.calls),
                          api_receipts=client.calls, requests=requests, responses=responses)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            results.append(result)
            print(json.dumps({"event": "question_complete", "number": number, "question": row["question"],
                              "status": result["status"], "answer": result.get("answer"), "query": result.get("query"),
                              "seconds": result["seconds"], "http_requests": result["http_requests"]}, ensure_ascii=False), flush=True)
            if fatal:
                break
    elapsed = time.perf_counter() - total_started
    receipts = [c for r in results for c in r["api_receipts"]]
    tokens = {k: sum(c.get("usage", {}).get(k, 0) for c in receipts) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    report = {"settings": settings, "completed_questions": len(results), "statuses": dict(Counter(r["status"] for r in results)),
              "format_valid": sum(r["format_valid"] for r in results), "elapsed_seconds": round(elapsed, 3),
              "average_seconds_per_question": round(elapsed / len(results), 3),
              "http_requests": sum(r["http_requests"] for r in results), "tokens": tokens,
              "answer_repairs": sum(r["answer_repairs"] for r in results),
              "direct_answers_with_references": sum(r.get("normalized_reference_match") is not None for r in results),
              "direct_normalized_reference_matches": sum(r.get("normalized_reference_match") is True for r in results),
              "source_counts": dict(Counter(r["data_source"] for r in results)),
              "retriever_executed": False, "full_rollout_test_completed": False, "training_data_created": False,
              "prompt_version": PROMPT_VERSION, "search_extraction_policy": SEARCH_EXTRACTION_POLICY,
              "query_policy": QUERY_POLICY_VERSION,
              "accepted_search_sources": dict(Counter(r["raw_action_source"] for r in results if r["status"] == "search_requested"))}
    action_counts = [r["action_student_tokens"] for r in results if "action_student_tokens" in r]
    report.update(max_action_student_tokens=max(action_counts, default=0),
                  average_action_student_tokens=sum(action_counts) / len(action_counts) if action_counts else 0,
                  student_budget_failures=sum("token_budget_exceeded" in r.get("reason", "") for r in results))
    report.update(protocol_statistics(results))
    write_json(output_dir / "summary.json", report)
    with (output_dir / "review_packet.txt").open("x", encoding="utf-8") as review:
        review.write("FIRST-DECISION API TEST ONLY; no retrieved evidence or training samples are produced.\n\n")
        for result in results:
            review.write(f"{result['number']}. {result['question']}\nSource: {result['data_source']}\nStatus: {result['status']}\n")
            review.write(f"Action: {result.get('action', '')}\nReference match: {result.get('normalized_reference_match')}\n")
            review.write(f"Action student tokens (includes EOS reserve): {result.get('action_student_tokens')}\n")
            review.write(f"Continuation: {result.get('continuation', '')}\nFailure: {result.get('reason', '')}\n\n")
            for response in result["responses"]:
                for choice in response.get("choices", []):
                    review.write("Public API message: " + json.dumps(choice.get("message"), ensure_ascii=False) + "\n")
            review.write("\n")
    print(json.dumps({"event": "test_complete", "output_dir": str(output_dir), **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORKSPACE / "data/search_sft/search_sft_hybrid_v1_real_500.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-history", action="store_true", help="Exclude every question in prior diagnostic manifests")
    args = parser.parse_args()
    run(args.source, args.output_dir, num_questions=args.num_questions, seed=args.seed,
        exclude_history=args.exclude_history)
