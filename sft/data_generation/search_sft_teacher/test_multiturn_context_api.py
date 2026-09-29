#!/usr/bin/env python3
"""Off-policy next-action API diagnostics on published SFT trajectory prefixes."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controlled_rollout import (DEEPSEEK_GENERATOR_VERSION, INSTRUCTIONS, PROMPT, PROMPT_VERSION,
                                QUERY_POLICY_VERSION, SEARCH_EXTRACTION_POLICY, TOOL, TOOL_POLICY_VERSION, WORKSPACE,
                                Rejected, contains_alias, evidence_citations, normalize,
                                parse_action, response_items)
from deepseek_client import API_URL, DEFAULT_THINKING_MAX_TOKENS, APIError, DeepSeekClient, extract_search_message
from deepseek_key import load_key_file
from test_api_20_questions import protocol_statistics, without_reasoning, write_json
from token_budget import BudgetError, INFO_PATTERN, MAX_SEARCH_TURNS, student_budget


REPOSITORY = "muradil211/AetherSearch_SFT"
REVISION = "969de77a3ac5a7a40382e0cdae3d534b6fdbb8d8"
DATA_FILE = "final_sft_2600.jsonl"
TEST_VERSION = "published_sft_prefix_next_action_v3_neutral_queries"
PAIR = re.compile(r"(?P<action><think>[^<>]+</think><search>[^<>\r\n]{1,300}</search>)"
                  r"(?P<information><information>.*?</information>)", re.S)


def download_snapshot(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    base = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
    fetched = {}
    for name in ("checksums.sha256", "dataset_manifest.json", DATA_FILE):
        with urllib.request.urlopen(base + name, timeout=120) as response:
            raw = response.read(32_000_001)
        if len(raw) > 32_000_000:
            raise ValueError("dataset_snapshot_exceeds_size_limit")
        with (output_dir / name).open("xb") as stream:
            os.chmod(stream.name, 0o600)
            stream.write(raw)
        fetched[name] = raw
    expected = {}
    for line in fetched["checksums.sha256"].decode().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip("*")] = digest
    actual = hashlib.sha256(fetched[DATA_FILE]).hexdigest()
    if expected.get(DATA_FILE) != actual:
        raise ValueError("published_dataset_checksum_mismatch")
    return output_dir / DATA_FILE, {"repository": REPOSITORY, "revision": REVISION,
                                    "file": DATA_FILE, "sha256": actual,
                                    "published_checksum_verified": True}


def parse_trajectory(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    text = row["full_trajectory_text"]
    marker = "<|im_start|>assistant\n"
    if not isinstance(text, str) or text.count(marker) != 1 or not text.endswith("<|im_end|>"):
        raise ValueError("invalid_source_chat_template")
    prefix, body = text.split(marker)
    if not prefix.endswith(row["question"] + "<|im_end|>\n"):
        raise ValueError("source_question_mismatch")
    body = body.removesuffix("<|im_end|>")
    pairs, position = [], 0
    while match := PAIR.match(body, position):
        pairs.append(match.groupdict())
        position = match.end()
    final = body[position:]
    _, answer = parse_action(final, "answer")
    if len(pairs) != row["search_count"] or not 2 <= len(pairs) <= 4:
        raise ValueError("source_search_count_mismatch")
    return pairs, answer


def source_observation(information: str, *, source_id: str, turn: int) -> dict[str, Any]:
    match = INFO_PATTERN.fullmatch(information)
    if not match:
        raise ValueError("invalid_source_information")
    docs = [{"doc_id": f"historical:{source_id}:turn:{turn}:doc:{n}", "title": title, "text": text}
            for n, (title, text) in enumerate(zip(match.groups()[::2], match.groups()[1::2]), 1)]
    visible = student_budget().information(docs)
    return {"original_information": information, "original_tokens": student_budget().count(information),
            **visible}


def select_contexts(source: Path, *, count: int = 100, seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {turn: [] for turn in (2, 3, 4)}
    skips, source_depths = Counter(), Counter()
    source_errors = []
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("trajectory_type") != "multi_search":
            continue
        source_depths[row["search_count"]] += 1
        try:
            pairs, reference = parse_trajectory(row)
        except (ValueError, Rejected) as exc:
            skips[f"source:{exc}"] += 1
            source_errors.append({"id": row["id"], "reason": str(exc)})
            continue
        history = [{"role": "user", "content": PROMPT.format(question=row["question"], max_searches=MAX_SEARCH_TURNS)}]
        observations, queries, invalid_prefix = [], [], None
        for prior_turn, pair in enumerate(pairs[:-1], 1):
            target = prior_turn + 1
            try:
                if invalid_prefix:
                    raise ValueError(invalid_prefix)
                _, query = parse_action(pair["action"], "search")
                if normalize(query) in {normalize(q) for q in queries}:
                    raise Rejected("duplicate_query_in_source_prefix")
                observation = source_observation(pair["information"], source_id=row["id"], turn=prior_turn)
                call_id = f"historical_{row['id']}_turn_{prior_turn}"
                history.extend([
                    {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [{
                        "id": call_id, "type": "function", "function": {
                            "name": "retrieve", "arguments": json.dumps({"action": pair["action"]}, ensure_ascii=False)}}]},
                    {"role": "tool", "tool_call_id": call_id, "content": observation["information"]},
                ])
                queries.append(query)
                observations.append(observation)
                groups[target].append({"source_id": row["id"], "question": row["question"],
                                       "target_search_turn": target, "source_search_count": row["search_count"],
                                       "reference_answer": reference, "reference_type": "held_out_dataset_final_answer",
                                       "history": copy.deepcopy(history), "previous_queries": list(queries),
                                       "observations": copy.deepcopy(observations)})
            except (ValueError, Rejected, BudgetError) as exc:
                invalid_prefix = str(exc)
                skips[f"turn_{target}:{exc}"] += 1
    rng, selected, seen_questions = random.Random(seed), [], set()
    for pool in groups.values():
        rng.shuffle(pool)
    # Prefer scarce deeper states while taking at most one prefix per question.
    quotas = {2: count - 2 * (count // 3), 3: count // 3, 4: count // 3}
    for turn in (4, 3, 2):
        for candidate in groups[turn]:
            key = normalize(candidate["question"])
            if key not in seen_questions and quotas[turn] > 0:
                selected.append(candidate)
                seen_questions.add(key)
                quotas[turn] -= 1
    remaining = [candidate for turn in (4, 3, 2) for candidate in groups[turn]]
    rng.shuffle(remaining)
    for candidate in remaining:
        key = normalize(candidate["question"])
        if len(selected) >= count:
            break
        if key not in seen_questions:
            selected.append(candidate)
            seen_questions.add(key)
    if len(selected) != count:
        raise ValueError(f"not_enough_unique_valid_contexts:{len(selected)}/{count};skips={dict(skips)}")
    rng.shuffle(selected)
    for number, case in enumerate(selected, 1):
        case["number"] = number
    return selected, {"source_multi_search_depths": dict(source_depths),
                      "eligible_states_by_target_turn": {str(k): len(v) for k, v in groups.items()},
                      "selected_states_by_target_turn": dict(Counter(c["target_search_turn"] for c in selected)),
                      "unique_selected_questions": len(seen_questions), "selection_skip_reasons": dict(skips),
                      "source_parse_errors": source_errors}


def prepare(count: int, seed: int, output_dir: Path | None) -> Path:
    if not 1 <= count <= 100:
        raise ValueError("This diagnostic is bounded to 1..100 contexts")
    student_budget()
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = WORKSPACE / "logs/search_sft_teacher" / f"multiturn_context_{count}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    source, snapshot = download_snapshot(output_dir)
    cases, selection = select_contexts(source, count=count, seed=seed)
    manifest = {"test_version": TEST_VERSION, "snapshot": snapshot, "selection": selection,
                "seed": seed, "count": count, "teacher_prompt_version": PROMPT_VERSION,
                "generator_version": DEEPSEEK_GENERATOR_VERSION, "tool_policy": TOOL_POLICY_VERSION,
                "search_extraction_policy": SEARCH_EXTRACTION_POLICY, "rollout_budget": student_budget().specification(),
                "query_policy": QUERY_POLICY_VERSION,
                "test_scope": "off_policy_historical_prefix_next_action", "retriever_executed": False,
                "training_data_created": False, "historical_tool_calls_reconstructed": True,
                "historical_private_reasoning_available": False, "contexts": cases}
    write_json(output_dir / "input_manifest.json", manifest)
    print(json.dumps({"event": "contexts_prepared", "output_dir": str(output_dir), "snapshot": snapshot,
                      "selection": selection}, ensure_ascii=False), flush=True)
    return output_dir


def evaluate_next_action(case: dict[str, Any], raw: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Same next-action checks for live diagnostics and credential-free replay."""
    result = {k: case[k] for k in ("number", "source_id", "question", "target_search_turn", "source_search_count", "reference_answer")}
    result.update(format_valid=False, retriever_executed=False, training_data_created=False)
    try:
        call, final = response_items(raw)
        if call is not None:
            action = json.loads(call["arguments"])["action"]
            _, query = parse_action(action, "search")
            if normalize(query) in {normalize(q) for q in case["previous_queries"]}:
                raise Rejected("duplicate_query")
            result.update(status="search_requested", action=action, query=query, format_valid=True,
                          action_student_tokens=student_budget().action_tokens(action),
                          extraction=receipt["extraction"], raw_action_source=receipt["raw_action_source"],
                          continuation="requires_new_real_retrieval_not_executed")
        else:
            reason, answer = parse_action(final, "answer")
            refs = evidence_citations(reason, len(case["observations"]))
            supported = any(contains_alias(case["observations"][turn - 1]["visible_documents"][doc - 1]["title"] + "\n" +
                                          case["observations"][turn - 1]["visible_documents"][doc - 1]["text"], [answer])
                            for turn, doc in refs)
            result.update(action=final, answer=answer, citations=refs, answer_in_cited_visible_evidence=supported,
                          normalized_reference_match=normalize(answer) == normalize(case["reference_answer"]))
            if not supported:
                raise Rejected("answer_not_in_cited_evidence")
            result.update(status="final_answer", format_valid=True,
                          action_student_tokens=student_budget().action_tokens(final))
    except Rejected as exc:
        result.update(status="rejected", reason=str(exc))
    return result


def evaluate_case(case: dict[str, Any], key: str) -> dict[str, Any]:
    client = DeepSeekClient("deepseek-flash", key, thinking="enabled", reasoning_effort="max",
                            max_tokens=DEFAULT_THINKING_MAX_TOKENS, timeout=120, retries=0, max_requests=2)
    real_post, requests, responses = client._post, [], []

    def traced(payload):
        requests.append(without_reasoning(copy.deepcopy(payload)))
        raw, attempts = real_post(payload)
        responses.append(without_reasoning(raw))
        return raw, attempts

    client._post = traced
    started = time.perf_counter()
    result = {k: case[k] for k in ("number", "source_id", "question", "target_search_turn", "source_search_count", "reference_answer")}
    result.update(format_valid=False, retriever_executed=False, training_data_created=False)
    try:
        raw = client.create(case["history"], "auto")
        result = evaluate_next_action(case, raw, client.calls[-1])
    except Rejected as exc:
        result.update(status="rejected", reason=str(exc))
    except APIError as exc:
        result.update(status="api_error", reason=str(exc))
    result.update(seconds=round(time.perf_counter() - started, 3), requests=requests, responses=responses,
                  api_receipts=client.calls, http_requests=client.request_count,
                  answer_repairs=sum(c.get("discarded_reason") == "answer_action_token_budget_exceeded" for c in client.calls))
    return result


def execute(output_dir: Path, workers: int, *, resume: bool = False) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    manifest = json.loads((output_dir / "input_manifest.json").read_text())
    expected = {"teacher_prompt_version": PROMPT_VERSION, "generator_version": DEEPSEEK_GENERATOR_VERSION,
                "tool_policy": TOOL_POLICY_VERSION, "search_extraction_policy": SEARCH_EXTRACTION_POLICY,
                "query_policy": QUERY_POLICY_VERSION,
                "rollout_budget": student_budget().specification(), "test_version": TEST_VERSION}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("diagnostic_configuration_changed")
    if len(manifest["contexts"]) != manifest["count"]:
        raise ValueError("invalid_context_manifest")
    results_path = output_dir / "results.jsonl"
    if results_path.exists() and not resume:
        raise FileExistsError("Existing results are not overwritten or automatically replayed")
    started, results, fatal = time.perf_counter(), [], False
    with results_path.open("a+" if resume else "x", encoding="utf-8") as journal, ThreadPoolExecutor(max_workers=workers) as executor:
        fcntl.flock(journal.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.chmod(results_path, 0o600)
        if resume:
            journal.seek(0)
            results = [json.loads(line) for line in journal if line.strip()]
            numbers = {r["number"] for r in results}
            cases_by_number = {c["number"]: c for c in manifest["contexts"]}
            if len(numbers) != len(results) or any(r["number"] not in cases_by_number or
                    any(r[k] != cases_by_number[r["number"]][k] for k in ("source_id", "question", "target_search_turn")) for r in results):
                raise ValueError("invalid_existing_diagnostic_results")
            journal.seek(0, os.SEEK_END)
            if len(results) == manifest["count"] and (output_dir / "summary.json").exists():
                report = json.loads((output_dir / "summary.json").read_text())
                print(json.dumps({"event": "already_complete_no_api_calls", "output_dir": str(output_dir), **report}), flush=True)
                return report
        finished = {r["number"] for r in results}
        key = load_key_file()
        print(json.dumps({"event": "api_test_started", "contexts": manifest["count"], "already_completed": len(finished),
                          "workers": workers, "thinking": "enabled", "reasoning_effort": "max",
                          "max_tokens": DEFAULT_THINKING_MAX_TOKENS}), flush=True)
        iterator, pending = iter(c for c in manifest["contexts"] if c["number"] not in finished), {}

        def submit():
            if (case := next(iterator, None)) is not None:
                pending[executor.submit(evaluate_case, case, key)] = case["number"]

        for _ in range(workers):
            submit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                results.append(result)
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"event": "context_complete", "completed": len(results), "number": result["number"],
                                  "target_search_turn": result["target_search_turn"], "status": result["status"],
                                  "reason": result.get("reason"), "seconds": result["seconds"]}), flush=True)
                fatal = fatal or result.get("reason") in {"deepseek_http_400", "deepseek_http_401", "deepseek_http_402", "deepseek_http_403"}
            if not fatal:
                for _ in done:
                    submit()
    results.sort(key=lambda row: row["number"])
    report = {"test_version": TEST_VERSION, "snapshot": manifest["snapshot"], "selection": manifest["selection"],
              "requested_contexts": manifest["count"], "completed_contexts": len(results),
              "statuses": dict(Counter(r["status"] for r in results)), "fatal_api_error": fatal,
              "elapsed_seconds": round(time.perf_counter() - started, 3),
              "http_requests": sum(r["http_requests"] for r in results),
              "answer_repairs": sum(r["answer_repairs"] for r in results),
              "tokens": {name: sum(c.get("usage", {}).get(name, 0) for r in results for c in r["api_receipts"])
                         for name in ("prompt_tokens", "completion_tokens", "total_tokens")},
              "by_target_search_turn": {str(turn): dict(Counter(r["status"] for r in results if r["target_search_turn"] == turn))
                                        for turn in (2, 3, 4)},
              "final_reference_matches": sum(r.get("normalized_reference_match") is True and r["status"] == "final_answer" for r in results),
              "accepted_search_sources": dict(Counter(r["raw_action_source"] for r in results if r["status"] == "search_requested")),
              "max_action_student_tokens": max((r.get("action_student_tokens", 0) for r in results), default=0),
              "max_information_student_tokens": max(o["information_tokens"] for c in manifest["contexts"] for o in c["observations"]),
              "truncated_history_observations": sum(o["information_truncated"] for c in manifest["contexts"] for o in c["observations"]),
              "test_scope": "off_policy_historical_prefix_next_action", "new_retriever_calls": 0,
              "training_data_created": False, "semantic_support_independently_verified": False,
              **expected}
    report.update(protocol_statistics(results))
    suffix = "" if not (output_dir / "summary.json").exists() else "_resume_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary_path, review_path = output_dir / f"summary{suffix}.json", output_dir / f"review_packet{suffix}.txt"
    write_json(summary_path, report)
    with review_path.open("x", encoding="utf-8") as review:
        os.chmod(review.name, 0o600)
        review.write("HISTORICAL PREFIX NEXT-ACTION TEST; no new retrieval or training data.\n"
                     "API requests and public responses are complete. Private reasoning is omitted and counted.\n\n")
        for result in results:
            review.write(f"Case {result['number']}; source {result['source_id']}; before search {result['target_search_turn']}\n")
            review.write(f"Question: {result['question']}\nStatus: {result['status']}; reason: {result.get('reason')}\n")
            review.write("Action: " + result.get("action", "") + "\n")
            review.write("REQUESTS:\n" + json.dumps(result["requests"], ensure_ascii=False, indent=2) + "\n")
            review.write("PUBLIC RESPONSES:\n" + json.dumps(result["responses"], ensure_ascii=False, indent=2) + "\n\n")
    print(json.dumps({"event": "test_complete", "output_dir": str(output_dir), "summary_path": str(summary_path),
                      "review_packet_path": str(review_path), **report}, ensure_ascii=False, indent=2), flush=True)
    return report


def audit_saved_run(output_dir: Path) -> dict[str, Any]:
    """Validate actual sent prefixes and accepted actions without any network access."""
    manifest = json.loads((output_dir / "input_manifest.json").read_text())
    source = output_dir / DATA_FILE
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["snapshot"]["sha256"]:
        raise ValueError("source_snapshot_changed")
    selected, selection = select_contexts(source, count=manifest["count"], seed=manifest["seed"])
    # JSON manifests stringify numeric histogram keys.
    if selected != manifest["contexts"] or json.loads(json.dumps(selection)) != manifest["selection"]:
        raise ValueError("prefix_selection_not_reproducible")
    rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]
    if len(rows) != manifest["count"] or len({r["number"] for r in rows}) != manifest["count"]:
        raise ValueError("results_count_or_uniqueness_mismatch")
    cases = {case["number"]: case for case in selected}
    function = {k: v for k, v in TOOL.items() if k != "type"}
    budget = student_budget()
    for row in rows:
        case = cases[row["number"]]
        if any(row[k] != case[k] for k in ("source_id", "question", "target_search_turn", "reference_answer")):
            raise ValueError("result_context_mismatch")
        payload = row["requests"][0]
        if (payload["messages"] != [{"role": "system", "content": INSTRUCTIONS}] + without_reasoning(case["history"])
                or payload["tools"] != [{"type": "function", "function": function}]
                or payload["model"] != "deepseek-flash" or payload["thinking"] != {"type": "enabled"}
                or payload["reasoning_effort"] != "max" or payload["max_tokens"] != DEFAULT_THINKING_MAX_TOKENS
                or payload["tool_choice"] != "auto" or payload["stream"] is not False):
            raise ValueError("actual_request_not_equal_to_expected_prefix_and_policy")
        if "<answer>" in json.dumps(case["history"]) or len(case["history"]) != 1 + 2 * (case["target_search_turn"] - 1):
            raise ValueError("future_action_or_final_answer_in_history")
        for observation in case["observations"]:
            if budget.count(observation["information"]) > 500 or not INFO_PATTERN.fullmatch(observation["information"]):
                raise ValueError("history_observation_budget_or_top3_failure")
        if row["status"] == "search_requested":
            _, query = parse_action(row["action"], "search")
            raw = row["responses"][-1]
            choice = raw["choices"][0]
            extracted = extract_search_message(choice["message"], finish_reason=choice["finish_reason"],
                                              tool_choice=payload["tool_choice"], response_id=raw["id"])
            if extracted is None or extracted["action"] != row["action"] or extracted["extraction"] != row["extraction"]:
                raise ValueError("search_extraction_not_equal_to_original_response")
            if normalize(query) in {normalize(q) for q in case["previous_queries"]}:
                raise ValueError("accepted_duplicate_query")
        elif row["status"] == "final_answer":
            parse_action(row["action"], "answer")
            if row.get("answer_in_cited_visible_evidence") is not True:
                raise ValueError("accepted_final_without_visible_support")
        if row["retriever_executed"] is not False or row["training_data_created"] is not False:
            raise ValueError("false_retrieval_or_training_claim")
    report = {"audit": "PASS", "completed_contexts": len(rows), "new_api_requests": 0,
              "checks": {name: "PASS" for name in (
                  "source_checksum", "deterministic_prefix_selection", "unique_questions",
                  "no_future_actions_observations_or_reference_injection", "actual_api_prompt_and_schema",
                  "enabled_max_thinking", "student_information_budget_and_top3", "student_action_budget",
                  "literal_search_receipt_provenance", "accepted_query_duplicate_filter",
                  "no_new_retrieval_or_training_data")},
              "average_request_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 3),
              "supported_final_answers": sum(r["status"] == "final_answer" for r in rows),
              "attempted_final_answers": sum(c.get("action_type") == "final" for r in rows for c in r["api_receipts"])}
    write_json(output_dir / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def reclassify_saved_run(input_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Re-evaluate public saved returns, not old checkpoint/prompt compatibility."""
    started = time.perf_counter()
    manifest_bytes = (input_dir / "input_manifest.json").read_bytes()
    results_bytes = (input_dir / "results.jsonl").read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = [json.loads(line) for line in results_bytes.decode("utf-8").splitlines()]
    if hashlib.sha256((input_dir / DATA_FILE).read_bytes()).hexdigest() != manifest["snapshot"]["sha256"]:
        raise ValueError("source_snapshot_changed")
    if manifest.get("rollout_budget") != student_budget().specification():
        raise ValueError("saved_tokenizer_or_budget_changed")
    cases = {case["number"]: case for case in manifest["contexts"]}
    if (len(cases) != len(manifest["contexts"]) or len(rows) != manifest["count"]
            or len(cases) != len(rows) or {row["number"] for row in rows} != set(cases)):
        raise ValueError("results_count_or_uniqueness_mismatch")
    results = []
    for row in sorted(rows, key=lambda r: r["number"]):
        case = cases[row["number"]]
        if any(row[k] != case[k] for k in ("source_id", "question", "target_search_turn", "reference_answer")):
            raise ValueError("result_context_mismatch")
        if len(row["responses"]) != 1 or len(row["requests"]) != 1:
            raise ValueError("replay_requires_one_saved_response_per_context")
        payload, raw = row["requests"][0], row["responses"][0]
        if (payload["messages"][1:] != without_reasoning(case["history"])
                or payload["tool_choice"] != "auto"):
            raise ValueError("actual_request_not_equal_to_saved_prefix")
        result = {k: case[k] for k in ("number", "source_id", "question", "target_search_turn", "source_search_count", "reference_answer")}
        result.update(format_valid=False, retriever_executed=False, training_data_created=False)
        try:
            choices = raw.get("choices")
            if (not isinstance(raw.get("id"), str) or not raw["id"]
                    or not isinstance(choices, list) or len(choices) != 1):
                raise Rejected("invalid_model_response")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise Rejected("invalid_model_response")
            if choice.get("finish_reason") not in {"stop", "tool_calls"}:
                raise Rejected("incomplete_model_output")
            search = extract_search_message(message, finish_reason=choice["finish_reason"],
                                            tool_choice=payload["tool_choice"], response_id=raw["id"])
            if search is not None:
                call = search["normalized_tool_call"]
                output = [{"type": "function_call", "name": "retrieve", "call_id": call["id"],
                           "arguments": call["function"]["arguments"]}]
            else:
                if choice["finish_reason"] != "stop":
                    raise Rejected("missing_final_answer")
                output = [{"type": "message", "content": [{"type": "output_text", "text": message.get("content")}]}]
            result = evaluate_next_action(case, {"status": "completed", "output": output}, search or {})
        except Rejected as exc:
            result.update(status="rejected", reason=str(exc))
        result.update(before_status=row["status"], before_reason=row.get("reason"))
        result["recovered"] = row["status"] == "rejected" and result["status"] != "rejected"
        results.append(result)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = WORKSPACE / "logs/search_sft_teacher" / f"candidate_query_replay_{len(rows)}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    counts = lambda values: dict(Counter(values))
    report = {"test_scope": "offline_historical_prefix_next_action_reclassification",
              "input_dir": str(input_dir.resolve()), "output_dir": str(output_dir.resolve()),
              "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "input_results_sha256": hashlib.sha256(results_bytes).hexdigest(),
              "original_prompt_version": manifest["teacher_prompt_version"],
              "current_prompt_version": PROMPT_VERSION, "query_policy": QUERY_POLICY_VERSION,
              "contexts": len(rows), "new_api_requests": 0, "new_retriever_calls": 0,
              "training_data_created": False, "fresh_prompt_test": False,
              "private_reasoning_checked": False, "semantic_support_independently_verified": False,
              "before_statuses": counts(row["status"] for row in rows),
              "after_statuses": counts(row["status"] for row in results),
              "before_rejection_reasons": counts(row["reason"] for row in rows if row["status"] == "rejected"),
              "after_rejection_reasons": counts(row["reason"] for row in results if row["status"] == "rejected"),
              "recovered_count": sum(row["recovered"] for row in results),
              "recovered_numbers": [row["number"] for row in results if row["recovered"]],
              "regressions": [row["number"] for row in results if row["before_status"] != "rejected" and row["status"] == "rejected"],
              "accepted_search_sources": counts(row["raw_action_source"] for row in results if row["status"] == "search_requested"),
              "max_action_student_tokens": max(row.get("action_student_tokens", 0) for row in results),
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    write_json(output_dir / "summary.json", report)
    with (output_dir / "reclassified_results.jsonl").open("x", encoding="utf-8") as stream:
        os.chmod(stream.name, 0o600)
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-contexts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true", help="Audit saved requests and responses without API calls")
    parser.add_argument("--reclassify-only", action="store_true", help="Apply current query checks to saved returns, without API calls or changing old artifacts")
    parser.add_argument("--resume", action="store_true", help="Append only unattempted contexts; do not repeat completed requests")
    args = parser.parse_args()
    if args.audit_only and args.reclassify_only:
        parser.error("Choose audit or reclassification, not both")
    if (args.audit_only or args.reclassify_only) and args.prepared_dir is None:
        parser.error("Offline audit/reclassification requires --prepared-dir")
    directory = args.prepared_dir or prepare(args.num_contexts, args.seed, args.output_dir)
    if args.reclassify_only:
        reclassify_saved_run(directory, args.output_dir)
    elif args.audit_only:
        audit_saved_run(directory)
    elif not args.prepare_only:
        execute(directory, args.workers, resume=args.resume)
