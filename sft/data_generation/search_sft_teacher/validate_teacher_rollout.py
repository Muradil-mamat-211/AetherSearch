#!/usr/bin/env python3
"""Validate real DeepSeek teacher trajectories, optionally against checkpoint audit."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from controlled_rollout import (CONTINUATION_POLICY_VERSION, DEEPSEEK_GENERATOR_VERSION, FINAL_SUMMARY_POLICY_VERSION,
                                DIRECT_FINAL_THINK, RETRIEVED_FINAL_THINK, INSTRUCTIONS, PROMPT, PROMPT_VERSION,
                                SEARCH_ACTION_SOURCE, SEARCH_EXTRACTION_POLICY,
                                QUERY_POLICY_VERSION, TOOL_POLICY_VERSION, Rejected,
                                canonical_answer, contains_alias, evidence_citations,
                                normalize, parse_action, support_status, training_final_action)
from token_budget import INFO_PATTERN, MAX_INFORMATION_TOKENS, MAX_SEARCH_TURNS, BudgetError, student_budget
from deepseek_client import (ANSWER_REPAIR_INSTRUCTION, API_URL, MAX_API_REQUEST_BYTES, SEARCH_EXHAUSTED_INSTRUCTION,
                             extract_search_message, history_manifest, message_manifest)
from published_sft_format import PUBLIC_FORMAT_VERSION, public_record, validate_public_record


REQUIRED = {"id", "data_source", "split", "trajectory_type", "question", "golden_answers",
            "prompt", "messages", "response", "events", "metadata"}
EXPECTED_METADATA = {"retriever": "hybrid_rag_v1", "corpus": "wiki18", "sparse_retriever": "bm25",
                     "dense_retriever": "e5-base-v2", "dense_index": "e5_Flat.index",
                     "fusion": "rrf", "topk": 3, "loss_mask_policy": "mask_information",
                     "teacher_provider": "deepseek", "action_origin": "model_rollout",
                     "generator_version": DEEPSEEK_GENERATOR_VERSION,
                     "search_policy": "adaptive", "prompt_version": PROMPT_VERSION,
                     "final_summary_policy": FINAL_SUMMARY_POLICY_VERSION,
                     "tool_policy": TOOL_POLICY_VERSION, "search_action_source": SEARCH_ACTION_SOURCE,
                     "search_extraction_policy": SEARCH_EXTRACTION_POLICY,
                     "query_policy": QUERY_POLICY_VERSION,
                     "continuation_policy": CONTINUATION_POLICY_VERSION,
                     "public_format": PUBLIC_FORMAT_VERSION,
                     "format_valid": True, "skip_reason": None}


def validate_record(row: Any, audit: dict[str, Any] | None = None,
                    require_approved: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict) or not REQUIRED.issubset(row):
        return ["missing_fields"]
    if not isinstance(row["question"], str) or not row["question"].strip():
        return ["invalid_question"]
    if not isinstance(row["id"], str) or not row["id"].startswith("teacher_"):
        errors.append("invalid_id")
    if row["trajectory_type"] != "teacher_hybrid_v1_real_rollout":
        errors.append("invalid_trajectory_type")
    if not isinstance(row["data_source"], str) or not row["data_source"] or row["split"] != "train":
        errors.append("invalid_training_source_or_split")
    answers = row["golden_answers"]
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not normalize(a) for a in answers):
        return errors + ["invalid_answers"]
    events, meta = row["events"], row["metadata"]
    if not isinstance(events, list) or not 1 <= len(events) <= 2 * MAX_SEARCH_TURNS + 1 or len(events) % 2 != 1 or any(not isinstance(e, dict) or not isinstance(e.get("text"), str) for e in events):
        return errors + ["invalid_events"]
    if not isinstance(meta, dict):
        return errors + ["invalid_metadata"]
    public_id = meta.get("public_id")
    if (not isinstance(public_id, str) or re.fullmatch(r"[0-9]{6}", public_id) is None
            or public_id == "000000"):
        errors.append("invalid_public_id")
    if not isinstance(meta.get("semantic_review_status"), str) or meta["semantic_review_status"] not in {"pending", "approved"}:
        errors.append("invalid_semantic_review_status")
    if meta.get("teacher_model") not in ("deepseek-flash", "deepseek-v4-pro"):
        errors.append("invalid_teacher_model")
    if any(answer != canonical_answer(answer) for answer in answers):
        errors.append("noncanonical_golden_answers")
    for key, expected in EXPECTED_METADATA.items():
        if key not in meta or meta[key] != expected or isinstance(expected, bool) and meta[key] is not expected:
            errors.append("metadata_" + key)
    count = (len(events) - 1) // 2
    if require_approved and count < 1:
        errors.append("zero_search_not_training_eligible")
    if type(meta.get("search_count")) is not int or meta["search_count"] != count:
        errors.append("search_count_mismatch")
    budget = meta.get("max_searches")
    if type(budget) is not int or not 1 <= budget <= MAX_SEARCH_TURNS or count > budget:
        return errors + ["invalid_search_budget"]
    tokenizer = student_budget()
    if meta.get("rollout_budget") != tokenizer.specification():
        errors.append("rollout_budget_mismatch")
    expected_action_counts = [tokenizer.action_tokens(e["text"]) for e in events[::2]]
    if meta.get("action_token_counts") != expected_action_counts:
        errors.append("action_token_counts_mismatch")
    if meta.get("information_token_counts") != [tokenizer.count(e["text"]) for e in events[1::2]]:
        errors.append("information_token_counts_mismatch")
    if meta.get("retrieval_used") is not bool(count):
        errors.append("metadata_retrieval_used")
    if meta.get("evidence_support_checked") is not bool(count):
        errors.append("metadata_evidence_support_checked")
    if meta.get("answer_source") != ("retrieved_evidence" if count else "prior_knowledge"):
        errors.append("metadata_answer_source")
    if "answer_in_evidence" not in meta or meta["answer_in_evidence"] is not (True if count else None):
        errors.append("metadata_answer_in_evidence")
    queries, information, parsed_docs = [], [], []
    for i, event in enumerate(events):
        role = "assistant" if i % 2 == 0 else "environment"
        if event.get("role") != role or event.get("train_on_tokens") is not (role == "assistant"):
            errors.append("role_or_loss_mask_mismatch")
        if role == "environment":
            if tokenizer.count(event["text"]) > MAX_INFORMATION_TOKENS:
                errors.append("information_token_budget_exceeded")
            match = INFO_PATTERN.fullmatch(event["text"])
            if not match or "<" in ("".join(match.groups()) if match else "") or ">" in ("".join(match.groups()) if match else ""):
                errors.append("invalid_information")
            else:
                pairs = list(zip(match.groups()[::2], match.groups()[1::2]))
                parsed_docs.append(pairs)
            information.append(event["text"])
        elif i < len(events) - 1:
            try:
                _, query = parse_action(event["text"], "search")
                if normalize(query) in {normalize(q) for q in queries}:
                    errors.append("duplicate_query")
                queries.append(query)
            except Rejected as exc:
                errors.append(str(exc))
    if queries != meta.get("search_queries"):
        errors.append("search_queries_mismatch")
    if row["response"] != "".join(e["text"] for e in events):
        errors.append("response_mismatch")
    prompt = [{"role": "user", "content": PROMPT.format(question=row["question"], max_searches=budget)}]
    if row["prompt"] != prompt:
        errors.append("prompt_mismatch")
    if row["messages"] != prompt + [{"role": e["role"], "content": e["text"]} for e in events]:
        errors.append("messages_mismatch")
    try:
        reason, answer = parse_action(events[-1]["text"], "answer")
        if reason != (RETRIEVED_FINAL_THINK if count else DIRECT_FINAL_THINK):
            errors.append("noncanonical_final_think")
        if normalize(answer) not in {normalize(a) for a in answers}:
            errors.append("answer_not_equal_to_gold")
        if answer != canonical_answer(answer):
            errors.append("noncanonical_answer")
        cited = evidence_citations(reason, count, required=False)
        supporting = ([parsed_docs[step - 1][number - 1] for step, number in cited]
                      if cited and len(parsed_docs) == count else
                      [doc for step in parsed_docs for doc in step] if not cited else [])
        if count and (len(parsed_docs) != count or not any(contains_alias("\n".join(doc), [answer]) for doc in supporting)):
            errors.append("answer_not_in_evidence")
    except Rejected as exc:
        errors.append(str(exc))
    support = support_status(row["question"], "\n".join(information), retrieval_used=bool(count))
    for key, expected in [("evidence_support_warning", support["warning"]),
                          ("question_key_entities", support["question_key_entities"]),
                          ("missing_question_entities", support["missing_question_entities"])]:
        if meta.get(key) != expected:
            errors.append("metadata_" + key)
    if type(meta.get("evidence_support_warning")) is not bool:
        errors.append("invalid_evidence_support_warning_type")
    for name in ("evidence_doc_ids", "evidence_titles", "evidence_rrf_scores", "evidence_source_branches"):
        value = meta.get(name)
        if not isinstance(value, list) or len(value) != count * 3:
            errors.append("metadata_" + name)
    try:
        public = public_record(row)
        if meta.get("public_full_trajectory_tokens") != student_budget().count(public["full_trajectory_text"]):
            errors.append("public_full_trajectory_tokens_mismatch")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid_public_trajectory")
    if audit is not None:
        errors.extend(validate_audit(row, audit, information, queries, parsed_docs))
    if require_approved:
        approval = audit.get("approval") if isinstance(audit, dict) else None
        if meta.get("semantic_review_status") != "approved" or not isinstance(approval, dict) or not all(isinstance(approval.get(k), str) and approval[k].strip() for k in ("reviewer", "note", "time")):
            errors.append("semantic_review_not_approved")
        elif meta.get("evidence_support_warning") and approval.get("evidence_warning_acknowledged") is not True:
            errors.append("evidence_warning_not_acknowledged")
    return list(dict.fromkeys(errors))


def validate_continuation(row: dict[str, Any], calls: list[dict[str, Any]]) -> list[str]:
    """Reconstruct the entire wire history, including discarded answer repairs."""
    errors, seen_ids, search_count = [], set(), 0
    expected = history_manifest([{"role": "system", "content": INSTRUCTIONS},
                                 {"role": "user", "content": PROMPT.format(question=row["question"],
                                                                         max_searches=row["metadata"]["max_searches"])}])
    settings = {k: calls[0].get(k) for k in ("thinking", "reasoning_effort", "max_tokens")}
    for index, call in enumerate(calls):
        repair = index > 0 and calls[index - 1].get("discarded_reason") == "answer_action_token_budget_exceeded"
        choice = "none" if repair or search_count >= row["metadata"]["max_searches"] else "auto"
        if choice == "none" and not repair:
            expected = expected + [message_manifest({"role": "user", "content": SEARCH_EXHAUSTED_INSTRUCTION})]
        if (call.get("continuation_policy") != CONTINUATION_POLICY_VERSION
                or call.get("request_messages") != expected or call.get("tool_choice") != choice
                or any(call.get(k) != v for k, v in settings.items())):
            errors.append("continuation_request_mismatch")
        replayed = [part["reasoning"]["characters"] for part in expected
                    if isinstance(part.get("reasoning"), dict)
                    and type(part["reasoning"].get("characters")) is int]
        if (call.get("replayed_reasoning_messages") != len(replayed)
                or call.get("replayed_reasoning_characters") != sum(replayed)):
            errors.append("reasoning_replay_budget_mismatch")
        if not isinstance(call.get("request_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", call["request_sha256"]):
            errors.append("missing_request_fingerprint")
        public = {"role": "assistant", "content": call.get("action")}
        if call.get("action_type") == "search":
            public.update(content="", tool_calls=[call.get("normalized_tool_call")])
            call_id = call.get("tool_call_id")
            if not isinstance(call_id, str) or call_id in seen_ids:
                errors.append("continuation_duplicate_tool_id")
            else:
                seen_ids.add(call_id)
        manifest = call.get("response_message")
        if not isinstance(manifest, dict):
            errors.append("missing_continuation_response")
            return errors
        actual = message_manifest(public)
        actual["reasoning"] = manifest.get("reasoning")
        reasoning = actual["reasoning"]
        if settings["thinking"] == "enabled":
            if (not isinstance(reasoning, dict) or set(reasoning) != {"sha256", "characters"}
                    or type(reasoning.get("characters")) is not int or reasoning["characters"] < 0
                    or not isinstance(reasoning.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", reasoning["sha256"])):
                errors.append("invalid_reasoning_replay_receipt")
        elif settings["thinking"] != "disabled" or reasoning is not None:
            errors.append("invalid_reasoning_replay_receipt")
        if actual != manifest:
            errors.append("continuation_response_mismatch")
        if call.get("action_type") == "search":
            event_index = 2 * search_count + 1
            if event_index >= len(row["events"]):
                errors.append("continuation_observation_mismatch")
                return errors
            expected = expected + [manifest, message_manifest({"role": "tool", "tool_call_id": call.get("tool_call_id"),
                                                               "content": row["events"][event_index]["text"]})]
            search_count += 1
        elif call.get("discarded_reason"):
            expected = expected + [manifest, message_manifest({"role": "user", "content": ANSWER_REPAIR_INSTRUCTION})]
    return errors


def validate_audit(row: dict[str, Any], audit: Any, information: list[str], queries: list[str],
                   parsed_docs: list[list[tuple[str, str]]]) -> list[str]:
    if not isinstance(audit, dict):
        return ["missing_verified_provenance"]
    if audit.get("provenance_checked") is not bool(queries):
        return ["missing_verified_provenance" if queries else "false_retrieval_provenance_claim"]
    calls = audit.get("api_calls")
    if not isinstance(calls, list) or not calls or any(not isinstance(c, dict) or not isinstance(c.get("action"), str) for c in calls):
        return ["missing_model_action_receipts"]
    errors = validate_continuation(row, calls)
    limit = row["metadata"].get("teacher_max_cumulative_reasoning_tokens")
    if type(limit) is not int or limit < 1:
        errors.append("invalid_cumulative_reasoning_limit")
    if "teacher_max_cumulative_prompt_tokens" in row["metadata"]:
        errors.append("legacy_prompt_limit_present")
    cumulative = prompt_cumulative = 0
    for call in calls:
        usage = call.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        prompt_charge, prompt_source = call.get("prompt_usage_units"), call.get("prompt_usage_source")
        prompt_reported = usage.get("prompt_tokens")
        if type(prompt_charge) is not int or prompt_charge < 0:
            errors.append("invalid_prompt_usage_receipt")
        else:
            prompt_cumulative += prompt_charge
            if (prompt_source == "provider_usage" and (type(prompt_reported) is not int or prompt_reported != prompt_charge)
                    or prompt_source == "utf8_request_byte_upper_bound" and prompt_charge != call.get("request_bytes")
                    or prompt_source not in {"provider_usage", "utf8_request_byte_upper_bound"}
                    or call.get("cumulative_prompt_usage_units") != prompt_cumulative
                    or "prompt_budget_limit" in call):
                errors.append("invalid_prompt_usage_receipt")
        charge, source = call.get("reasoning_budget_charge"), call.get("reasoning_budget_source")
        if type(charge) is not int or charge < 0:
            errors.append("invalid_reasoning_budget_receipt")
            continue
        cumulative += charge
        details = usage.get("completion_tokens_details") or {}
        reported = details.get("reasoning_tokens") if isinstance(details, dict) else None
        manifest = call.get("response_message") or {}
        reasoning = manifest.get("reasoning") if isinstance(manifest, dict) else None
        if source == "provider_usage":
            if type(reported) is not int or reported != charge:
                errors.append("invalid_reasoning_budget_receipt")
        elif source == "utf8_byte_upper_bound":
            chars = reasoning.get("characters") if isinstance(reasoning, dict) else None
            if (call.get("thinking") != "enabled" or type(chars) is not int
                    or not chars <= charge <= 4 * chars):
                errors.append("invalid_reasoning_budget_receipt")
        elif source != "thinking_disabled" or charge != 0 or call.get("thinking") != "disabled":
            errors.append("invalid_reasoning_budget_receipt")
        if (call.get("cumulative_reasoning_tokens") != cumulative
                or call.get("reasoning_budget_limit") != limit):
            errors.append("invalid_reasoning_budget_receipt")
        request_bytes = call.get("request_bytes")
        if type(request_bytes) is not int or not 0 < request_bytes <= MAX_API_REQUEST_BYTES:
            errors.append("invalid_request_byte_budget")
    if cumulative != row["metadata"].get("teacher_cumulative_reasoning_tokens") or type(limit) is int and cumulative > limit:
        errors.append("cumulative_reasoning_budget_mismatch")
    if prompt_cumulative != row["metadata"].get("teacher_cumulative_prompt_usage_units"):
        errors.append("cumulative_prompt_usage_mismatch")
    for call in calls:
        if (call.get("endpoint") != API_URL or call.get("tool_policy") != TOOL_POLICY_VERSION
                or call.get("strict_requested") is not True):
            errors.append("invalid_tool_policy_receipt")
        if call.get("query_policy") != QUERY_POLICY_VERSION:
            errors.append("invalid_query_policy_receipt")
        if call.get("action_type") == "search":
            if (call.get("action_source") != SEARCH_ACTION_SOURCE or call.get("tool_name") != "retrieve"
                    or not isinstance(call.get("tool_call_id"), str) or not call["tool_call_id"]):
                errors.append("invalid_search_action_source")
            try:
                if "assistant_content" not in call or "raw_tool_calls" not in call:
                    raise Rejected("missing_raw_search_receipt")
                extracted = extract_search_message({"content": call["assistant_content"], "tool_calls": call["raw_tool_calls"]},
                                                  finish_reason=call.get("finish_reason"),
                                                  tool_choice=call.get("tool_choice"), response_id=call.get("id"))
                if extracted is None or any(call.get(key) != value for key, value in extracted.items()):
                    errors.append("invalid_search_action_receipt")
                parse_action(call["action"], "search")
            except (ValueError, TypeError, Rejected):
                errors.append("invalid_search_action_receipt")
        elif call.get("action_type") != "final" or call.get("assistant_content") != call["action"]:
            errors.append("invalid_final_action_receipt")
    discarded = [i for i, c in enumerate(calls) if "discarded_reason" in c]
    if discarded:
        if discarded != [len(calls) - 2]:
            return ["invalid_answer_repair_audit"]
        original, corrected = calls[-2:]
        if (original.get("discarded_reason") != "answer_action_token_budget_exceeded" or original.get("action_type") != "final"
                or not isinstance(original.get("id"), str) or not original["id"]
                or corrected.get("repair_of_response_id") != original["id"]
                or corrected.get("id") == original["id"] or corrected.get("action_type") != "final"
                or corrected.get("tool_choice") != "none"
                or any("repair_of_response_id" in c for c in calls[:-1])):
            return ["invalid_answer_repair_audit"]
        try:
            parse_action(original["action"], "answer")
        except Rejected as exc:
            if str(exc) != "answer_action_token_budget_exceeded":
                return ["invalid_answer_repair_audit"]
        else:
            return ["invalid_answer_repair_audit"]
        calls = calls[:-2] + [corrected]
    elif any("repair_of_response_id" in c for c in calls):
        return ["invalid_answer_repair_audit"]
    if len(calls) != len(queries) + 1:
        return ["missing_model_action_receipts"]
    if [c.get("id") for c in calls] != audit.get("model_response_ids"):
        errors.append("model_response_receipts_mismatch")
    search_events = row["events"][:-1:2]
    for call, event in zip(calls[:-1], search_events):
        if call.get("action_type") != "search" or call["action"] != event["text"]:
            errors.append("search_action_not_equal_to_model_output")
    raw_final = calls[-1]["action"]
    try:
        raw_reason, raw_answer = parse_action(raw_final, "answer")
        cited = evidence_citations(raw_reason, len(queries), required=False)
        if cited and (len(parsed_docs) != len(queries) or not any(
                contains_alias("\n".join(parsed_docs[step - 1][number - 1]), [raw_answer])
                for step, number in cited)):
            errors.append("raw_final_citation_not_supported")
        canonical = training_final_action(raw_final, len(queries))
        if calls[-1].get("action_type") != "final" or raw_final != audit.get("raw_final_action") or canonical != row["events"][-1]["text"]:
            errors.append("final_action_not_equal_to_model_output")
    except Rejected:
        errors.append("invalid_raw_final_action")
    trace = audit.get("retrieval_trace")
    if not isinstance(trace, list) or len(trace) != len(information) or len(queries) != len(information):
        return ["audit_trace_mismatch"]
    all_docs = []
    for i, step in enumerate(trace):
        if not isinstance(step, dict) or step.get("query") != queries[i] or step.get("information") != information[i]:
            errors.append("audit_trace_mismatch")
            continue
        docs = step.get("documents")
        if not isinstance(docs, list) or len(docs) != 3 or any(not isinstance(d, dict) for d in docs):
            errors.append("audit_documents_invalid")
            continue
        try:
            if len({d["doc_id"] for d in docs}) != 3:
                errors.append("duplicate_evidence_docs")
            for number, doc in enumerate(docs, 1):
                bm25, dense = doc.get("bm25_rank"), doc.get("dense_rank")
                ranks = [r for r in (bm25, dense) if r is not None]
                branch = "both" if bm25 is not None and dense is not None else "bm25" if bm25 is not None else "dense"
                if not ranks or any(type(r) is not int or not 1 <= r <= 20 for r in ranks) or doc["source_branch"] != branch:
                    errors.append("invalid_branch_ranks")
                elif not isinstance(doc["rrf_score"], (int, float)) or not math.isclose(doc["rrf_score"], sum(1 / (60 + r) for r in ranks), rel_tol=1e-9):
                    errors.append("invalid_rrf_score")
            observation = student_budget().information(docs)
            if observation["information"] != information[i]:
                errors.append("evidence_not_equal_to_retrieval")
            if any(step.get(key) != value for key, value in observation.items()):
                errors.append("audit_visible_observation_mismatch")
            all_docs.extend(docs)
        except (KeyError, TypeError, AttributeError, BudgetError):
            errors.append("audit_documents_invalid")
    for key, field in [("evidence_doc_ids", "doc_id"), ("evidence_titles", "title"),
                       ("evidence_rrf_scores", "rrf_score"), ("evidence_source_branches", "source_branch")]:
        if row["metadata"].get(key) != [d.get(field) for d in all_docs]:
            errors.append("audit_" + key)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, help="Read-only checkpoint used to verify the retrieval audit")
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    if args.require_approved and not args.db:
        parser.error("--require-approved requires --db")
    db = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) if args.db else None
    failures, sources, answer_sources, ids = Counter(), Counter(), Counter(), set()
    count = warnings = 0
    try:
        with args.input.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "full_trajectory_text" in row:
                        errors = validate_public_record(row)
                        if row.get("search_count") == 0:
                            errors.append("zero_search_not_training_eligible")
                        if db:
                            uid = str(row.get("id", ""))
                            result = db.execute("SELECT status,record_json,audit_json FROM rollouts WHERE public_id=?",
                                                (uid,)).fetchone()
                            if result is None or result[1] is None:
                                errors.append("missing_checkpoint_record")
                            else:
                                status, record_json, audit_json = result
                                detailed = json.loads(record_json)
                                errors.extend(validate_record(detailed, json.loads(audit_json), args.require_approved))
                                if row != public_record(detailed):
                                    errors.append("public_export_not_equal_to_checkpoint")
                                if args.require_approved and status != "approved":
                                    errors.append("public_export_not_approved")
                                sources[str(detailed.get("data_source", "unknown"))] += 1
                                answer_sources[str(detailed["metadata"].get("answer_source", "unknown"))] += 1
                                warnings += int(detailed["metadata"].get("evidence_support_warning") is True)
                    else:
                        audit = None
                        if db and isinstance(row, dict):
                            result = db.execute("SELECT audit_json FROM rollouts WHERE uid=?", (str(row.get("id", ""))[8:],)).fetchone()
                            audit = json.loads(result[0]) if result else {}
                        errors = validate_record(row, audit, args.require_approved)
                    if isinstance(row, dict):
                        uid = row.get("id")
                        if isinstance(uid, str):
                            if uid in ids:
                                errors.append("duplicate_id")
                            ids.add(uid)
                        if "full_trajectory_text" not in row or not db:
                            sources[str(row.get("data_source", "unknown"))] += 1
                            if isinstance(row.get("metadata"), dict):
                                answer_sources[str(row["metadata"].get("answer_source", "unknown"))] += 1
                            warnings += int(isinstance(row.get("metadata"), dict) and row["metadata"].get("evidence_support_warning") is True)
                except (ValueError, TypeError):
                    errors = ["invalid_json_or_schema"]
                count += 1
                failures.update(errors)
                if errors:
                    print(json.dumps({"line": number, "errors": errors}))
    finally:
        if db:
            db.close()
    print(json.dumps({"rows": count, "passed": not failures and count > 0, "errors": dict(failures),
                      "data_sources": dict(sources), "evidence_support_warnings": warnings,
                      "answer_sources": dict(answer_sources),
                      "audit_checked": bool(args.db), "semantic_approval_required": args.require_approved}, sort_keys=True))
    raise SystemExit(0 if count and not failures else 1)


if __name__ == "__main__":
    main()
