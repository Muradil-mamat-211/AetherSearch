"""Offline fixtures only. No mock retriever is available from the production CLI."""

import copy
from dataclasses import replace
import io
import json
import sqlite3
import sys
import tempfile
import unittest
import urllib.error
from argparse import Namespace
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "search_sft_hybrid_v1"))

import deepseek_rollout as runner
from controlled_rollout import INSTRUCTIONS, PROMPT, Rejected, canonical_answer, parse_action, rollout
from deepseek_client import APIError, DeepSeekClient
from hybrid_retriever_v1 import RetrievedDoc, docs_to_jsonable, format_information
from published_sft_format import PUBLIC_FIELDS, public_record, validate_public_record
from validate_teacher_rollout import validate_record


QUESTION = "Which road project was linked to the Bendigo Street housing dispute?"
ANSWER = "East West Link"
KEY = "offline-test-key-not-a-credential"
ROW = {"id": "qa-1", "public_id": "500001", "question": QUESTION,
       "golden_answers": [ANSWER], "data_source": "nq", "split": "train"}
ACTION = "<think>I need evidence identifying the road project associated with the dispute.</think><search>Bendigo Street housing dispute road project</search>"
FINAL = "<think>Doc 2 identifies the project connected to the dispute.</think><answer>East West Link</answer>"
NORMALIZED_FINAL = "<think>The retrieved evidence now supports the answer.</think><answer>East West Link</answer>"
DIRECT_FINAL = "<think>Reliable prior knowledge is sufficient; no search is needed.</think><answer>East West Link</answer>"
NORMALIZED_DIRECT_FINAL = "<think>Reliable prior knowledge is sufficient to answer.</think><answer>East West Link</answer>"


def response(search=False, *, text=FINAL, name="retrieve", reasoning=None):
    message = {"role": "assistant", "content": None if search else text}
    if search:
        message["tool_calls"] = [{"id": "call1", "type": "function", "function": {
            "name": name, "arguments": json.dumps({"action": ACTION})}}]
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"id": "unit_response", "model": "deepseek-flash", "system_fingerprint": "unit_fingerprint",
            "choices": [{"finish_reason": "tool_calls" if search else "stop", "message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}}


def make_client(*, final=None, thinking="disabled", direct=False, **kwargs):
    client = DeepSeekClient("deepseek-flash", KEY, thinking=thinking, **kwargs)
    client.payloads = []

    def fake_post(payload):
        client.payloads.append(copy.deepcopy(payload))
        client.request_count += 1
        initial = payload["messages"][-1]["role"] == "user" and not any(m["role"] == "tool" for m in payload["messages"])
        text = final if final is not None else DIRECT_FINAL if direct else FINAL
        return response(initial and not direct, text=text, reasoning="offline hidden reasoning" if thinking == "enabled" else None), 1

    client._post = fake_post
    return client


class FixtureRetriever:
    def __init__(self):
        self.trace = []
        self.closed = False
        self.calls = 0
        self.fail_after = None

    def retrieve(self, query, topk):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("unit backend unavailable")
        docs = [
            RetrievedDoc("1", "Other road", "A different road project.", "bm25", 1, None, 1 / 61),
            RetrievedDoc("2", "Bendigo Street housing dispute", "The East West Link road project was linked to this dispute.", "both", 2, 1, 1 / 62 + 1 / 61),
            RetrievedDoc("3", "Melbourne", "A city.", "dense", None, 2, 1 / 62),
        ]
        self.trace.append({"query": query, "information": format_information(docs), "documents": docs_to_jsonable(docs)})
        return docs

    def close(self):
        self.closed = True


def fixture_record(*, direct=False):
    record, audit = rollout(ROW, make_client(direct=direct), FixtureRetriever(), 3)
    audit["provenance_checked"] = bool(record["metadata"]["search_count"])
    return record, audit


class ClientTests(unittest.TestCase):
    def test_invalid_numeric_limits_are_rejected_without_network(self):
        for kwargs in ({"timeout": float("nan")}, {"max_tokens": 100000}, {"max_requests": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DeepSeekClient("deepseek-flash", KEY, **kwargs)

    def test_model_whitelist_no_implicit_substitution(self):
        with self.assertRaises(ValueError):
            DeepSeekClient("made-up-model", KEY)

    def test_payload_only_exposes_retrieve_and_not_key_or_gold(self):
        client = make_client()
        client.create([{"role": "user", "content": QUESTION}], "auto")
        payload = client.payloads[0]
        self.assertEqual([t["function"]["name"] for t in payload["tools"]], ["retrieve"])
        self.assertNotIn(KEY, json.dumps(payload))
        self.assertNotIn(ANSWER, json.dumps(payload))
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertIn("Answer directly", payload["messages"][0]["content"])
        self.assertNotIn("Your first action must call retrieve", json.dumps(payload))

    def test_prompt_allows_direct_or_search_with_matching_budget(self):
        self.assertIn("Search is optional", PROMPT)
        self.assertIn("Do not search merely because a tool exists", INSTRUCTIONS)
        client = make_client()
        rollout(ROW, client, FixtureRetriever(), 2)
        self.assertIn("Search budget: at most 2 calls", client.payloads[0]["messages"][1]["content"])
        self.assertEqual(client.payloads[0]["tool_choice"], "auto")

    def test_thinking_mode_replays_hidden_reasoning_without_training_it(self):
        client = make_client(thinking="enabled")
        record, _ = rollout(ROW, client, FixtureRetriever(), 3)
        self.assertEqual(client.payloads[0]["tool_choice"], "auto")
        self.assertEqual(client.payloads[1]["messages"][2]["reasoning_content"], "offline hidden reasoning")
        self.assertNotIn("offline hidden reasoning", record["response"])

    def test_unknown_tool_is_not_dispatched(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", return_value=(response(True, name="web_search"), 1)):
            with self.assertRaisesRegex(Rejected, "unauthorized_tool"):
                client.create([], "required")

    def test_parallel_tools_rejected(self):
        raw = response(True)
        raw["choices"][0]["message"]["tool_calls"] *= 2
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", return_value=(raw, 1)):
            with self.assertRaisesRegex(Rejected, "parallel_tool"):
                client.create([], "required")

    def test_extra_text_and_tool_call_are_extracted_and_audited(self):
        raw = response(True)
        raw["choices"][0]["message"]["content"] = "Unexpected text"
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", return_value=(raw, 1)):
            result = client.create([], "required")
        self.assertEqual(json.loads(result["output"][0]["arguments"])["action"], ACTION)
        self.assertEqual(client.calls[0]["assistant_content"], "Unexpected text")
        self.assertEqual(result["assistant_message"]["content"], "")

    def test_extra_or_duplicate_tool_arguments_rejected(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        for arguments in ('{"action":"x","path":"/root"}', '{"action":"x","action":"y"}', '[]', '{"action":3}', 'not-json'):
            with self.subTest(arguments=arguments):
                raw = response(True)
                raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
                with patch.object(client, "_post", return_value=(raw, 1)):
                    with self.assertRaisesRegex(Rejected, "invalid_tool_arguments"):
                        client.create([], "required")

    def test_truncated_generation_is_not_accepted(self):
        raw = response()
        raw["choices"][0]["finish_reason"] = "length"
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", return_value=(raw, 1)):
            with self.assertRaisesRegex(Rejected, "incomplete_model_output"):
                client.create([], "auto")

    def test_missing_thinking_content_rejected(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled")
        with patch.object(client, "_post", return_value=(response(True), 1)):
            with self.assertRaisesRegex(Rejected, "missing_reasoning"):
                client.create([], "required")

    def test_tool_call_after_budget_rejected(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", return_value=(response(True), 1)):
            with self.assertRaisesRegex(Rejected, "search_budget"):
                client.create([], "none")

    def test_auth_failure_does_not_retry_or_echo_server_body(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        error = urllib.error.HTTPError("https://api.deepseek.com/chat/completions", 401, "Unauthorized", {}, io.BytesIO(KEY.encode()))
        client.opener = SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(error))
        with self.assertRaisesRegex(APIError, "deepseek_http_401") as caught:
            client._post({"model": client.model})
        self.assertEqual(client.request_count, 1)
        self.assertNotIn(KEY, str(caught.exception))

    def test_transient_http_failure_retries_with_a_bound(self):
        client = DeepSeekClient("deepseek-flash", KEY, retries=1)
        attempts = []
        def open_request(request, timeout):
            attempts.append(request)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "Rate limited", {}, io.BytesIO())
            return io.BytesIO(json.dumps(response()).encode())
        client.opener = SimpleNamespace(open=open_request)
        with patch("deepseek_client.time.sleep"):
            raw, count = client._post({"model": client.model})
        self.assertEqual(count, 2)
        self.assertEqual(client.request_count, 2)
        self.assertEqual(raw["model"], client.model)

    def test_request_budget_includes_failed_http_attempts(self):
        client = DeepSeekClient("deepseek-flash", KEY, retries=2, max_requests=1)
        def unavailable(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {}, io.BytesIO())
        client.opener = SimpleNamespace(open=unavailable)
        with patch("deepseek_client.time.sleep"), self.assertRaisesRegex(APIError, "budget_exhausted"):
            client._post({"model": client.model})
        self.assertEqual(client.request_count, 1)

    def test_invalid_or_oversized_response_rejected(self):
        for raw in (b'[]', b'not-json', b'{"id":1,"id":2}', b'x' * 2_097_153):
            with self.subTest(size=len(raw)):
                client = DeepSeekClient("deepseek-flash", KEY)
                client.opener = SimpleNamespace(open=lambda *a, **k: io.BytesIO(raw))
                with self.assertRaises(APIError):
                    client._post({"model": client.model})


class RolloutTests(unittest.TestCase):
    def test_short_uppercase_words_and_real_acronyms(self):
        self.assertEqual(canonical_answer("EAST WEST LINK!!!"), "East West Link")
        self.assertEqual(canonical_answer("ROME."), "Rome")
        self.assertEqual(canonical_answer("NASA"), "NASA")
        self.assertEqual(canonical_answer("SQL"), "SQL")

    def test_complete_model_actions_and_real_tool_replay(self):
        client = make_client()
        record, audit = rollout(ROW, client, FixtureRetriever(), 3)
        self.assertEqual(record["events"][0]["text"], ACTION)
        self.assertEqual(record["events"][-1]["text"], NORMALIZED_FINAL)
        self.assertEqual(audit["raw_final_action"], FINAL)
        self.assertNotIn(ANSWER, json.dumps(client.payloads[0]))
        self.assertEqual(client.payloads[1]["messages"][-1]["role"], "tool")
        self.assertEqual(client.payloads[1]["messages"][-1]["content"], audit["retrieval_trace"][0]["information"])
        self.assertEqual(record["metadata"]["teacher_provider"], "deepseek")

    def test_candidate_query_is_accepted_without_gold_injection(self):
        client, retriever = make_client(), FixtureRetriever()
        raw = response(True)
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": ACTION.replace("Bendigo Street housing dispute road project", ANSWER)})
        with patch.object(client, "_post", side_effect=[(raw, 1), (response(text=FINAL), 1)]) as post:
            record, audit = rollout(ROW, client, retriever, 3)
        audit["provenance_checked"] = True
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(record["metadata"]["search_queries"], [ANSWER])
        self.assertEqual(validate_record(record, audit), [])
        self.assertNotIn(ANSWER, json.dumps(post.call_args_list[0].args[0]))

    def test_hidden_reference_changes_do_not_change_requests_or_retrieval_query(self):
        original, changed = make_client(), make_client()
        original_backend, changed_backend = FixtureRetriever(), FixtureRetriever()
        rollout(ROW, original, original_backend, 3)
        row = {**ROW, "golden_answers": ["Heldout label never sent"]}
        with self.assertRaisesRegex(Rejected, "answer_not_equal_to_gold"):
            rollout(row, changed, changed_backend, 3)
        self.assertEqual(original.payloads, changed.payloads)
        self.assertEqual(original_backend.trace, changed_backend.trace)
        self.assertNotIn(row["golden_answers"][0], json.dumps(changed.payloads))

    def test_candidate_query_does_not_waive_final_evidence_support(self):
        raw = response(True)
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": ACTION.replace("Bendigo Street housing dispute road project", ANSWER)})
        client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
        with patch.object(client, "_post", side_effect=[(raw, 1), (response(text=FINAL.replace("Doc 2", "Doc 1")), 1)]), \
             self.assertRaisesRegex(Rejected, "answer_not_in_evidence"):
            rollout(ROW, client, retriever, 3)
        self.assertEqual(retriever.trace[0]["query"], ANSWER)

    def test_query_inner_whitespace_is_forwarded_and_audited_exactly(self):
        query = " Bendigo Street  housing dispute road project "
        raw = response(True)
        action = ACTION.replace("Bendigo Street housing dispute road project", query)
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": action})
        client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
        with patch.object(client, "_post", side_effect=[(raw, 1), (response(), 1)]):
            record, audit = rollout(ROW, client, retriever, 3)
        audit["provenance_checked"] = True
        self.assertEqual(retriever.trace[0]["query"], query)
        self.assertEqual(record["metadata"]["search_queries"], [query])
        self.assertEqual(record["events"][0]["text"], action)
        self.assertEqual(validate_record(record, audit), [])
        audit["retrieval_trace"][0]["query"] = " ".join(query.split())
        self.assertIn("audit_trace_mismatch", validate_record(record, audit))

    def test_model_canonicalization_retains_rationale_and_raw_audit(self):
        text = FINAL.replace(ANSWER, '"EAST WEST LINK!!!"')
        record, audit = rollout(ROW, make_client(final=text), FixtureRetriever(), 3)
        self.assertTrue(record["events"][-1]["text"].endswith("<answer>East West Link</answer>"))
        self.assertEqual(audit["raw_final_action"], text)

    def test_false_doc_citation_is_not_corrected_by_controller(self):
        with self.assertRaisesRegex(Rejected, "answer_not_in_evidence"):
            rollout(ROW, make_client(final=FINAL.replace("Doc 2", "Doc 1")), FixtureRetriever(), 3)

    def test_published_final_summary_needs_no_citation_when_evidence_supports_answer(self):
        final = NORMALIZED_FINAL
        record, audit = rollout(ROW, make_client(final=final), FixtureRetriever(), 3)
        audit["provenance_checked"] = True
        self.assertEqual(record["events"][-1]["text"], final)
        self.assertEqual(validate_record(record, audit), [])

    def test_untagged_refusal_is_not_replaced_by_gold_answer(self):
        with self.assertRaisesRegex(Rejected, "invalid_answer_action"):
            rollout(ROW, make_client(final="I cannot determine the answer."), FixtureRetriever(), 3)

    def test_direct_answer_is_parsed_for_audit_without_retrieval(self):
        client, retriever = make_client(direct=True), FixtureRetriever()
        record, audit = rollout(ROW, client, retriever, 3)
        self.assertEqual(retriever.calls, 0)
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(client.payloads[0]["tool_choice"], "auto")
        self.assertEqual(record["events"], [{"role": "assistant", "text": NORMALIZED_DIRECT_FINAL, "train_on_tokens": True}])
        self.assertEqual(audit["raw_final_action"], DIRECT_FINAL)
        self.assertEqual(record["metadata"]["search_count"], 0)
        audit["provenance_checked"] = False
        self.assertIn("zero_search_not_training_eligible", validate_record(record, audit, True))

    def test_direct_wrong_answer_is_rejected_without_search(self):
        retriever = FixtureRetriever()
        with self.assertRaisesRegex(Rejected, "answer_not_equal_to_gold"):
            rollout(ROW, make_client(direct=True, final=DIRECT_FINAL.replace(ANSWER, "Wrong project")), retriever, 3)
        self.assertEqual(retriever.calls, 0)

    def test_direct_answer_cannot_invent_a_doc_citation(self):
        with self.assertRaisesRegex(Rejected, "citation_without_retrieval"):
            rollout(ROW, make_client(direct=True, final=FINAL), FixtureRetriever(), 3)

    def test_first_turn_untagged_refusal_is_not_forced_into_search(self):
        client, retriever = make_client(direct=True, final="I cannot determine the answer."), FixtureRetriever()
        with self.assertRaisesRegex(Rejected, "invalid_answer_action"):
            rollout(ROW, client, retriever, 3)
        self.assertEqual(retriever.calls, 0)
        self.assertEqual(client.payloads[0]["tool_choice"], "auto")

    def test_search_budget_uses_auto_then_none(self):
        client, retriever = make_client(), FixtureRetriever()
        rollout(ROW, client, retriever, 1)
        self.assertEqual([p["tool_choice"] for p in client.payloads], ["auto", "none"])
        self.assertEqual(retriever.calls, 1)

    def test_extra_search_after_budget_never_reaches_backend(self):
        client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
        with patch.object(client, "_post", return_value=(response(True), 1)), self.assertRaisesRegex(Rejected, "search_budget"):
            rollout(ROW, client, retriever, 1)
        self.assertEqual(retriever.calls, 1)

    def test_multiturn_queries_are_model_actions_not_templates(self):
        second_action = ACTION.replace("Bendigo Street housing dispute road project", "Bendigo Street dispute project connection").replace("I need evidence", "The first results are incomplete; I need evidence")
        second = response(True)
        second["choices"][0]["message"]["tool_calls"][0]["id"] = "call2"
        second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": second_action})
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", side_effect=[(response(True), 1), (second, 1), (response(), 1)]):
            record, audit = rollout(ROW, client, FixtureRetriever(), 3)
        audit["provenance_checked"] = True
        self.assertEqual(record["metadata"]["search_count"], 2)
        self.assertEqual(record["events"][2]["text"], second_action)
        self.assertEqual(validate_record(record, audit), [])

    def test_final_can_cite_an_earlier_retrieval_turn(self):
        second = response(True)
        second["choices"][0]["message"]["tool_calls"][0]["id"] = "call2"
        second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": ACTION.replace("road project</search>", "related infrastructure</search>")})
        class EarlierEvidenceRetriever(FixtureRetriever):
            def retrieve(self, query, topk):
                docs = super().retrieve(query, topk)
                if self.calls == 2:
                    docs[1].text = "The dispute involved local housing."
                return docs
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", side_effect=[(response(True), 1), (second, 1), (response(text=FINAL.replace("Doc 2", "Turn 1 Doc 2")), 1)]):
            record, audit = rollout(ROW, client, EarlierEvidenceRetriever(), 3)
        audit["provenance_checked"] = True
        self.assertEqual(validate_record(record, audit), [])
        modified = copy.deepcopy(record)
        modified["events"][-1]["text"] = modified["events"][-1]["text"].replace(
            "The retrieved evidence now supports the answer.", "Turn 1 Doc 2")
        self.assertIn("noncanonical_final_think", validate_record(modified))

    def test_raw_false_citation_cannot_be_hidden_by_normalized_training_think(self):
        record, audit = fixture_record()
        tampered = copy.deepcopy(audit)
        wrong = FINAL.replace("Doc 2", "Doc 1")
        tampered["raw_final_action"] = wrong
        tampered["api_calls"][-1]["action"] = wrong
        tampered["api_calls"][-1]["assistant_content"] = wrong
        self.assertIn("raw_final_citation_not_supported", validate_record(record, tampered))

    def test_duplicate_query_is_rejected(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        repeated = response(True)
        repeated["choices"][0]["message"]["tool_calls"][0]["id"] = "call2"
        with patch.object(client, "_post", side_effect=[(response(True), 1), (repeated, 1)]), self.assertRaisesRegex(Rejected, "duplicate_query"):
            rollout(ROW, client, FixtureRetriever(), 3)

    def test_search_query_rejects_urls_and_newlines(self):
        for query in ("https://example.com", "word\nother", "file:///root/secret"):
            with self.subTest(query=query), self.assertRaises(Rejected):
                parse_action(f"<think>Need evidence.</think><search>{query}</search>", "search")


class ValidatorTests(unittest.TestCase):
    def test_valid_candidate_and_approval_gate(self):
        record, audit = fixture_record()
        self.assertEqual(validate_record(record, audit), [])
        self.assertIn("semantic_review_not_approved", validate_record(record, audit, True))

    def test_direct_record_passes_but_still_requires_semantic_review(self):
        record, audit = fixture_record(direct=True)
        self.assertEqual(validate_record(record, audit), [])
        self.assertFalse(audit["provenance_checked"])
        self.assertIn("semantic_review_not_approved", validate_record(record, audit, True))

    def test_direct_record_cannot_claim_retrieval_or_fake_information(self):
        mutations = [
            lambda r, a: r["metadata"].update(answer_in_evidence=True),
            lambda r, a: r["metadata"].update(answer_source="retrieved_evidence"),
            lambda r, a: r["metadata"].update(retrieval_used=True),
            lambda r, a: r["metadata"].update(evidence_support_checked=True),
            lambda r, a: r["metadata"].update(evidence_doc_ids=["invented"]),
            lambda r, a: a.update(provenance_checked=True),
            lambda r, a: a.update(retrieval_trace=[{"information": "invented"}]),
            lambda r, a: a.update(api_calls=[]),
            lambda r, a: r["events"][0].update(train_on_tokens=False),
            lambda r, a: r["metadata"].update(max_searches=True),
            lambda r, a: r["metadata"].update(search_count=0.0),
        ]
        for mutate in mutations:
            record, audit = fixture_record(direct=True)
            mutate(record, audit)
            self.assertTrue(validate_record(record, audit))

    def test_masks_response_messages_metadata_and_audit_mutations(self):
        mutations = [
            lambda r, a: r["events"][1].update(train_on_tokens=True),
            lambda r, a: r["events"][0].update(train_on_tokens=False),
            lambda r, a: r.update(response="different"),
            lambda r, a: r["messages"].pop(),
            lambda r, a: r["metadata"].update(search_count=2),
            lambda r, a: r["metadata"].update(answer_in_evidence=False),
            lambda r, a: r["metadata"].update(format_valid=False),
            lambda r, a: r["metadata"].update(teacher_provider="unknown"),
            lambda r, a: r["metadata"].update(evidence_source_branches=[]),
            lambda r, a: r["events"][1].update(text="<information></information>"),
            lambda r, a: a.update(provenance_checked=False),
            lambda r, a: a["retrieval_trace"][0]["documents"][1].update(text="Invented text"),
            lambda r, a: a["retrieval_trace"][0]["documents"][1].update(rrf_score=100),
            lambda r, a: a["retrieval_trace"][0]["documents"][1].update(source_branch="dense"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record, audit = fixture_record()
                mutate(record, audit)
                self.assertTrue(validate_record(record, audit))

    def test_invalid_search_does_not_crash_audit_validator(self):
        record, audit = fixture_record()
        record["events"][0]["text"] = "<think>broken</think>"
        self.assertTrue(validate_record(record, audit))

    def test_malformed_metadata_status_returns_error(self):
        record, audit = fixture_record()
        record["metadata"]["semantic_review_status"] = []
        self.assertIn("invalid_semantic_review_status", validate_record(record, audit))

    def test_action_cannot_be_replaced_by_controller_template(self):
        record, audit = fixture_record()
        record["events"][0]["text"] = ACTION.replace("I need evidence", "Fixed template: I need evidence")
        record["response"] = "".join(e["text"] for e in record["events"])
        record["messages"][1]["content"] = record["events"][0]["text"]
        self.assertIn("search_action_not_equal_to_model_output", validate_record(record, audit))

    def test_no_api_receipts_means_no_proven_model_rollout(self):
        record, audit = fixture_record()
        audit.pop("api_calls")
        self.assertIn("missing_model_action_receipts", validate_record(record, audit))

    def test_schema_errors_return_failures_not_exceptions(self):
        for row in (None, [], {}, {"question": 3}):
            self.assertTrue(validate_record(row))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = self.root / "questions.jsonl"
        self.questions.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
        self.args = Namespace(questions=self.questions, db=self.root / "checkpoint.sqlite", model="deepseek-flash",
                              thinking="disabled", max_tokens=1600, max_searches=3, max_examples=10,
                              retry_ids=None, max_api_requests=100, api_retries=2, timeout=90,
                              export_candidates=True, output=self.root / "candidates.jsonl",
                              ids=None, reviewer=None, review_note=None, ack_evidence_warning=False)
        self.client, self.retriever = make_client(), FixtureRetriever()
        self.addCleanup(patch.stopall)
        self.real_doctor = runner.doctor
        patch.object(runner, "doctor", return_value={"ready": True}).start()
        patch.object(runner, "VerifiedHybridRetriever", return_value=self.retriever).start()
        patch.object(runner, "make_client", return_value=self.client).start()
        patch("sys.stdout", io.StringIO()).start()

    def records(self):
        with sqlite3.connect(self.args.db) as db:
            return db.execute("SELECT uid,status,record_json,attempts FROM rollouts ORDER BY uid").fetchall()

    def test_missing_rag_stops_before_api_or_checkpoint(self):
        with patch.object(runner, "doctor", return_value={"ready": False}), self.assertRaises(RuntimeError):
            runner.run(self.args)
        self.assertEqual(self.client.payloads, [])
        self.assertFalse(self.args.db.exists())

    def test_resume_skips_valid_rows_and_does_not_extend_window(self):
        runner.run(self.args)
        first = self.records()
        runner.run(self.args)
        self.assertEqual(len(self.client.payloads), 2)
        self.assertEqual(first, self.records())
        self.assertEqual(runner.VerifiedHybridRetriever.call_count, 1)

    def test_direct_run_is_checkpointed_as_skip_and_cannot_be_approved_or_exported(self):
        self.client = make_client(direct=True)
        with patch.object(runner, "make_client", return_value=self.client):
            runner.run(self.args)
            runner.run(self.args)
        self.assertEqual(len(self.client.payloads), 1)
        self.assertEqual(runner.VerifiedHybridRetriever.call_count, 0)
        self.assertEqual(self.retriever.calls, 0)
        self.args.ids = "nq:train:qa-1"
        self.args.reviewer, self.args.review_note = "unit-test-reviewer", "Zero-search rows are ineligible"
        with self.assertRaisesRegex(ValueError, "approval_only_allows_pending_ids"):
            runner.approve(self.args)
        with self.assertRaisesRegex(ValueError, "no_rows_in_requested_export_status"):
            runner.export(self.args)
        self.assertFalse(self.args.output.exists())
        with runner.checkpoint(self.args.db, readonly=True) as db:
            audit = json.loads(db.execute("SELECT audit_json FROM rollouts").fetchone()[0])
            stats = runner.summary(db)
        self.assertEqual(audit["reason"], "zero_search_not_training_eligible")
        self.assertEqual(stats["skip_reasons"], {"zero_search_not_training_eligible": 1})
        self.assertEqual(stats["source_level_skip_statistics"]["nq"],
                         {"zero_search_not_training_eligible": 1})
        self.assertEqual(stats["search_count_distribution"], {})

    def test_legacy_approved_direct_answer_cannot_be_exported(self):
        runner.run(self.args)
        direct, _ = fixture_record(direct=True)
        with sqlite3.connect(self.args.db) as db:
            db.execute("UPDATE rollouts SET status='approved',record_json=?",
                       (json.dumps(direct),))
        self.args.export_candidates = False
        with self.assertRaisesRegex(ValueError, "zero_search_not_training_eligible"):
            runner.export(self.args)
        self.assertFalse(self.args.output.exists())

    def test_mixed_direct_and_search_run_loads_once_without_trace_leakage(self):
        rows = [{**ROW, "id": str(n), "question": QUESTION + (f" Direct case {n}." if n != 2 else " Search case.")} for n in (1, 2, 3)]
        self.questions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        original = self.client._post
        def choose_action(payload):
            if "Direct case" in payload["messages"][1]["content"]:
                self.client.payloads.append(copy.deepcopy(payload))
                self.client.request_count += 1
                return response(text=DIRECT_FINAL), 1
            return original(payload)
        self.client._post = choose_action
        runner.run(self.args)
        self.assertEqual(runner.VerifiedHybridRetriever.call_count, 1)
        self.assertEqual(self.retriever.calls, 1)
        self.assertEqual(len(self.client.payloads), 4)
        self.assertTrue(all(p["tool_choice"] == "auto" for p in self.client.payloads))
        with runner.checkpoint(self.args.db, readonly=True) as db:
            stats = runner.summary(db)
            for record_json, audit_json in db.execute("SELECT record_json,audit_json FROM rollouts WHERE record_json IS NOT NULL"):
                record, audit = json.loads(record_json), json.loads(audit_json)
                self.assertEqual(validate_record(record, audit), [])
                self.assertGreaterEqual(record["metadata"]["search_count"], 1)
        self.assertEqual(stats["answer_sources"], {"retrieved_evidence": 1})
        self.assertEqual(stats["skip_reasons"], {"zero_search_not_training_eligible": 2})

    def test_concurrent_questions_microbatch_retrieval_and_preserve_audits(self):
        rows = [ROW, {**ROW, "id": "qa-2", "question": QUESTION + " Second case."}]
        self.questions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.args.concurrency = 2
        self.args.retrieval_batch_queries = 2
        self.args.retrieval_batch_wait_ms = 20.0
        barrier = Barrier(2)
        clients = []

        def client_factory(_args, *, request_budget):
            client = make_client()
            post = client._post

            def synchronized_post(payload):
                if not any(message["role"] == "tool" for message in payload["messages"]):
                    barrier.wait(timeout=3)
                request_budget.reserve()
                raw, attempts = post(payload)
                if raw["choices"][0]["finish_reason"] == "tool_calls" and "Second case" in payload["messages"][1]["content"]:
                    raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({
                        "action": ACTION.replace("road project</search>", "road project second case</search>")})
                return raw, attempts

            client._post = synchronized_post
            clients.append(client)
            return client

        class BatchFixtureRetriever(FixtureRetriever):
            def __init__(self):
                super().__init__()
                self.batch_calls = []

            def retrieve_batch(self, queries, topk):
                self.batch_calls.append(list(queries))
                return [self.retrieve(query, topk) for query in queries]

            def retrieve(self, query, topk):
                return [replace(doc, title=f"{query} | {doc.title}")
                        for doc in super().retrieve(query, topk)]

        retriever = BatchFixtureRetriever()
        with patch.object(runner, "make_client", side_effect=client_factory), \
             patch.object(runner, "VerifiedHybridRetriever", return_value=retriever):
            runner.run(self.args)
        self.assertEqual(len(clients), 2)
        self.assertEqual(len(retriever.batch_calls), 1)
        self.assertEqual(len(retriever.batch_calls[0]), 2)
        self.assertEqual(retriever.calls, 2)
        self.assertTrue(retriever.closed)
        with runner.checkpoint(self.args.db, readonly=True) as db:
            self.assertEqual(runner.summary(db)["statuses"], {"needs_semantic_review": 2})
            for record_json, audit_json in db.execute("SELECT record_json,audit_json FROM rollouts"):
                record, audit = json.loads(record_json), json.loads(audit_json)
                self.assertEqual(validate_record(record, audit), [])
                self.assertTrue(all(title.startswith(record["metadata"]["search_queries"][0] + " | ")
                                    for title in record["metadata"]["evidence_titles"]))

    def test_concurrent_api_budget_failure_stops_without_synthetic_records(self):
        rows = [ROW, {**ROW, "id": "qa-2", "question": QUESTION + " Second case."}]
        self.questions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.args.concurrency = 2
        self.args.retrieval_batch_queries = 2
        self.args.retrieval_batch_wait_ms = 5.0
        self.args.max_api_requests = 1
        clients = []

        def client_factory(_args, *, request_budget):
            client = make_client()
            post = client._post

            def counted_post(payload):
                request_budget.reserve()
                return post(payload)

            client._post = counted_post
            clients.append(client)
            return client

        with patch.object(runner, "make_client", side_effect=client_factory), \
             self.assertRaisesRegex(RuntimeError, "no_fallback"):
            runner.run(self.args)
        self.assertLessEqual(sum(len(client.payloads) for client in clients), 1)
        with runner.checkpoint(self.args.db, readonly=True) as db:
            stats = runner.summary(db)
            self.assertEqual(stats["statuses"], {"error": 2})
            self.assertEqual(db.execute("SELECT COUNT(*) FROM rollouts WHERE record_json IS NOT NULL").fetchone()[0], 0)

    def test_lazy_backend_load_failure_is_checkpointed_without_fallback(self):
        with patch.object(runner, "VerifiedHybridRetriever", side_effect=RuntimeError("backend load failed")):
            with self.assertRaisesRegex(RuntimeError, "no_fallback"):
                runner.run(self.args)
        self.assertEqual(self.records()[0][1], "error")
        self.assertEqual(len(self.client.payloads), 1)

    def test_policy_version_or_prompt_change_cannot_mix_checkpoints(self):
        runner.run(self.args)
        for name, value in (("VERSION", "legacy_version"), ("INSTRUCTIONS", "Different policy"),
                            ("PROMPT_VERSION", "legacy_prompt"), ("SCHEMA", "legacy_schema"),
                            ("TOOL_POLICY_VERSION", "legacy_tool_policy"),
                            ("SEARCH_ACTION_SOURCE", "legacy_source"),
                            ("SEARCH_EXTRACTION_POLICY", "legacy_extraction"),
                            ("QUERY_POLICY_VERSION", "legacy_query_policy"),
                            ("CONTINUATION_POLICY_VERSION", "legacy_continuation")):
            with patch.object(runner, name, value), self.assertRaisesRegex(ValueError, "config_changed"):
                runner.run(self.args)

    def test_previous_protocol_checkpoint_is_not_resumed_exported_or_rewritten(self):
        runner.run(self.args)
        with sqlite3.connect(self.args.db) as db:
            config = json.loads(db.execute("SELECT config_json FROM run_config WHERE id=1").fetchone()[0])
            config.update(schema="deepseek_teacher_checkpoint_v10_extracted_search",
                          generator="controlled_deepseek_teacher_v10_extracted_search",
                          prompt_version="adaptive_search_v9_search_final",
                          tool_policy="hybrid_only_strict_v7_extracted_search")
            config.pop("query_policy", None)
            legacy_config = json.dumps(config, sort_keys=True)
            db.execute("UPDATE run_config SET config_json=? WHERE id=1", (legacy_config,))
            uid, record_json = db.execute("SELECT uid,record_json FROM rollouts").fetchone()
            record = json.loads(record_json)
            record["metadata"].update(generator_version=config["generator"],
                                      prompt_version=config["prompt_version"], tool_policy=config["tool_policy"])
            legacy_record = json.dumps(record)
            db.execute("UPDATE rollouts SET record_json=? WHERE uid=?", (legacy_record, uid))
        request_count = len(self.client.payloads)
        with self.assertRaisesRegex(ValueError, "config_changed"):
            runner.run(self.args)
        with self.assertRaisesRegex(ValueError, "export_validation_failed"):
            runner.export(self.args)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(len(self.client.payloads), request_count)
        with runner.checkpoint(self.args.db, readonly=True) as db:
            self.assertEqual(db.execute("SELECT config_json FROM run_config").fetchone()[0], legacy_config)
            self.assertEqual(db.execute("SELECT record_json FROM rollouts").fetchone()[0], legacy_record)
            self.assertEqual(runner.summary(db)["stored_candidates"], 1)

    def test_immediate_predecessor_prompt_checkpoint_is_not_reused(self):
        runner.run(self.args)
        with sqlite3.connect(self.args.db) as db:
            config = json.loads(db.execute("SELECT config_json FROM run_config WHERE id=1").fetchone()[0])
            config.update(schema="deepseek_teacher_checkpoint_v11_candidate_queries",
                          generator="controlled_deepseek_teacher_v11_candidate_queries",
                          prompt_version="adaptive_search_v10_candidate_queries",
                          tool_policy="hybrid_only_strict_v8_candidate_queries")
            old_config = json.dumps(config, sort_keys=True)
            db.execute("UPDATE run_config SET config_json=? WHERE id=1", (old_config,))
        prior_requests = len(self.client.payloads)
        with self.assertRaisesRegex(ValueError, "config_changed"):
            runner.run(self.args)
        self.assertEqual(len(self.client.payloads), prior_requests)
        with runner.checkpoint(self.args.db, readonly=True) as db:
            self.assertEqual(db.execute("SELECT config_json FROM run_config").fetchone()[0], old_config)

    def test_retry_refuses_good_rows_without_api_calls(self):
        runner.run(self.args)
        self.args.retry_ids = "nq:train:qa-1"
        with self.assertRaisesRegex(ValueError, "failed_ids"):
            runner.run(self.args)
        self.assertEqual(len(self.client.payloads), 2)

    def test_retry_only_failed_row_preserves_accepted_row(self):
        row2 = {**ROW, "id": "qa-2", "question": QUESTION + " Identify the linked project."}
        self.questions.write_text(json.dumps(ROW) + "\n" + json.dumps(row2) + "\n", encoding="utf-8")
        original = self.client._post
        def invalid_second(payload):
            raw, n = original(payload)
            if len(self.client.payloads) == 4:
                raw = response(text=FINAL.replace(ANSWER, "Incorrect project"))
            return raw, n
        self.client._post = invalid_second
        runner.run(self.args)
        before = self.records()
        self.assertEqual([r[1] for r in before], ["needs_semantic_review", "rejected"])
        self.args.retry_ids = "nq:train:qa-2"
        self.client._post = original
        runner.run(self.args)
        after = self.records()
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[1][1], "needs_semantic_review")
        self.assertEqual(after[1][3], 2)
        self.assertEqual(len(self.client.payloads), 6)
        with runner.checkpoint(self.args.db, readonly=True) as db:
            history = db.execute("SELECT status FROM attempt_history WHERE uid='nq:train:qa-2' ORDER BY attempt").fetchall()
            self.assertEqual(history, [("rejected",), ("needs_semantic_review",)])
            self.assertEqual(runner.summary(db)["historical_attempt_skip_reasons"]["answer_not_equal_to_gold"], 1)

    def test_input_or_model_changes_cannot_mix_checkpoints(self):
        runner.run(self.args)
        self.args.model = "deepseek-v4-pro"
        with self.assertRaisesRegex(ValueError, "config_changed"):
            runner.run(self.args)

    def test_input_content_change_is_detected(self):
        runner.run(self.args)
        self.questions.write_text(json.dumps({**ROW, "golden_answers": ["Changed answer"]}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "config_changed"):
            runner.run(self.args)

    def test_infrastructure_failure_checkpoints_and_stops(self):
        row2 = {**ROW, "id": "qa-2", "question": QUESTION + " Identify the linked project."}
        self.questions.write_text(json.dumps(ROW) + "\n" + json.dumps(row2) + "\n", encoding="utf-8")
        self.retriever.fail_after = 1
        with self.assertRaisesRegex(RuntimeError, "no_fallback"):
            runner.run(self.args)
        before = self.records()
        self.assertEqual([r[1] for r in before], ["needs_semantic_review", "error"])
        self.retriever.fail_after = None
        runner.run(self.args)
        self.assertEqual(self.records()[0], before[0])
        self.assertEqual(self.records()[1][1], "needs_semantic_review")

    def test_candidate_export_never_overwrites_existing_file(self):
        runner.run(self.args)
        runner.export(self.args)
        value = self.args.output.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.export(self.args)
        self.assertEqual(self.args.output.read_bytes(), value)

    def test_pending_rows_cannot_be_exported_for_training(self):
        runner.run(self.args)
        self.args.export_candidates = False
        with self.assertRaisesRegex(ValueError, "no_rows"):
            runner.export(self.args)
        self.assertFalse(self.args.output.exists())

    def test_manual_approval_and_independent_validation(self):
        runner.run(self.args)
        self.args.ids = "nq:train:qa-1"
        self.args.reviewer, self.args.review_note = "unit-test-reviewer", "Offline fixture only"
        self.args.ack_evidence_warning = True
        runner.approve(self.args)
        self.args.export_candidates = False
        runner.export(self.args)
        row = json.loads(self.args.output.read_text())
        with sqlite3.connect(self.args.db) as db:
            detailed = json.loads(db.execute("SELECT record_json FROM rollouts").fetchone()[0])
            audit = json.loads(db.execute("SELECT audit_json FROM rollouts").fetchone()[0])
        self.assertEqual(tuple(row), PUBLIC_FIELDS)
        self.assertEqual(row["id"], "500001")
        self.assertEqual(detailed["metadata"]["public_id"], row["id"])
        self.assertEqual(row, public_record(detailed))
        self.assertEqual(validate_public_record(row), [])
        self.assertEqual(validate_record(detailed, audit, require_approved=True), [])
        import validate_teacher_rollout
        with patch.object(sys, "argv", ["validator", "--input", str(self.args.output),
                                        "--db", str(self.args.db), "--require-approved"]):
            with self.assertRaises(SystemExit) as outcome:
                validate_teacher_rollout.main()
        self.assertEqual(outcome.exception.code, 0)

    def test_source_level_skip_statistics(self):
        row2 = {**ROW, "id": "qa-2", "data_source": "triviaqa", "question": QUESTION + " Identify the linked project.", "golden_answers": ["Different project"]}
        self.questions.write_text(json.dumps(ROW) + "\n" + json.dumps(row2) + "\n", encoding="utf-8")
        runner.run(self.args)
        with runner.checkpoint(self.args.db, readonly=True) as db:
            stats = runner.summary(db)
        self.assertEqual(stats["statuses"], {"needs_semantic_review": 1, "rejected": 1})
        self.assertEqual(stats["source_level_skip_statistics"]["triviaqa"]["answer_not_equal_to_gold"], 1)

    def test_eval_split_is_not_sent_to_model(self):
        self.questions.write_text(json.dumps({**ROW, "split": "test"}) + "\n", encoding="utf-8")
        runner.run(self.args)
        self.assertEqual(self.client.payloads, [])
        self.assertEqual(self.records()[0][1], "rejected")

    def test_answer_string_in_original_question_is_not_label_leakage(self):
        self.questions.write_text(json.dumps({**ROW, "question": QUESTION + " East West Link?"}) + "\n", encoding="utf-8")
        runner.run(self.args)
        self.assertEqual(len(self.client.payloads), 2)
        self.assertEqual(self.records()[0][1], "needs_semantic_review")

    def test_independent_cli_validates_export_against_checkpoint(self):
        import validate_teacher_rollout
        runner.run(self.args)
        runner.export(self.args)
        with patch.object(sys, "argv", ["validator", "--input", str(self.args.output), "--db", str(self.args.db)]):
            with self.assertRaises(SystemExit) as caught:
                validate_teacher_rollout.main()
        self.assertEqual(caught.exception.code, 0)

    def test_review_packet_contains_actual_stored_actions(self):
        runner.run(self.args)
        self.args.output = self.root / "review.txt"
        runner.review_packet(self.args)
        text = self.args.output.read_text()
        self.assertIn(ACTION, text)
        self.assertIn(NORMALIZED_FINAL, text)
        self.assertIn("Raw teacher FINAL (audit only):\n" + FINAL, text)
        self.assertIn("train_on_tokens=False", text)

    def test_same_checkpoint_cannot_have_two_writers(self):
        with runner.checkpoint(self.args.db):
            with self.assertRaisesRegex(RuntimeError, "already_in_use"):
                with runner.checkpoint(self.args.db):
                    pass

    def test_invalid_input_and_duplicate_questions_fail_before_api(self):
        for text in ('not-json\n', '[1]\n', json.dumps(ROW) + '\n' + json.dumps({**ROW, "id":"another"}) + '\n'):
            self.questions.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.run(self.args)
        self.assertEqual(self.client.payloads, [])

    def test_numeric_zero_id_is_preserved(self):
        self.questions.write_text(json.dumps({**ROW, "id":0}) + "\n", encoding="utf-8")
        rows, _ = runner.load_candidates(self.questions, 1)
        self.assertEqual(rows[0]["id"], "nq:train:0")
        self.assertEqual(rows[0]["public_id"], "500001")

    def test_public_ids_are_stable_when_input_window_grows(self):
        rows = [{**ROW, "id": f"qa-{number}", "question": QUESTION + f" Case {number}."}
                for number in range(1, 4)]
        self.questions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        small, _ = runner.load_candidates(self.questions, 1)
        large, _ = runner.load_candidates(self.questions, 3)
        self.assertEqual(small[0], large[0])
        self.assertEqual([row["public_id"] for row in large], ["500001", "500002", "500003"])
        custom, _ = runner.load_candidates(self.questions, 3, 700001)
        self.assertEqual([row["public_id"] for row in custom], ["700001", "700002", "700003"])
        with self.assertRaisesRegex(ValueError, "public_id_range_invalid"):
            runner.load_candidates(self.questions, 3, 999999)

    def test_public_id_range_cannot_change_on_checkpoint_resume(self):
        runner.run(self.args)
        self.args.public_id_start = 700001
        with self.assertRaisesRegex(ValueError, "checkpoint_config_changed"):
            runner.run(self.args)
        self.assertEqual(len(self.records()), 1)

    def test_two_public_ids_export_and_validate_against_distinct_checkpoint_rows(self):
        rows = [ROW, {**ROW, "id": "qa-2", "question": QUESTION + " Second case."}]
        self.questions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        runner.run(self.args)
        self.args.ids = "nq:train:qa-1,nq:train:qa-2"
        self.args.reviewer, self.args.review_note = "unit-test-reviewer", "Two fixture records"
        self.args.ack_evidence_warning = True
        runner.approve(self.args)
        self.args.export_candidates = False
        runner.export(self.args)
        exported = [json.loads(line) for line in self.args.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in exported], ["500001", "500002"])
        import validate_teacher_rollout
        with patch.object(sys, "argv", ["validator", "--input", str(self.args.output),
                                        "--db", str(self.args.db), "--require-approved"]):
            with self.assertRaises(SystemExit) as result:
                validate_teacher_rollout.main()
        self.assertEqual(result.exception.code, 0)

    def test_invalid_public_id_is_rejected_before_model_request(self):
        for value in (None, "teacher_qa-1", "000000", "50001", "500001 "):
            client = make_client()
            with self.subTest(value=value), self.assertRaisesRegex(Rejected, "invalid_public_id"):
                rollout({**ROW, "public_id": value}, client, FixtureRetriever(), 3)
            self.assertEqual(client.payloads, [])

    def test_retriever_probe_is_fail_closed_and_never_calls_teacher(self):
        model = self.root / "e5-probe-fixture"
        model.mkdir()
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (model / name).write_text("fixture", encoding="utf-8")
        assets = {name: True for name in ("uncompressed_wiki18_corpus", "bm25_index",
                  "faiss_flat_index", "e5_model", "retriever_env", "dense_server_listening")}
        with patch.object(runner, "resource_status", return_value=assets), \
             patch.object(runner, "E5_MODEL", model), \
             patch.object(runner, "dense_process_verified", return_value=True), \
             patch.object(runner, "load_key_file", return_value=KEY), \
             patch.object(runner, "probe_hybrid_retriever") as probe:
            self.assertTrue(self.real_doctor()["ready"])
            probe.assert_called_once_with()
            probe.side_effect = RuntimeError("broken_bm25_or_dense")
            health = self.real_doctor()
            self.assertFalse(health["ready"])
            self.assertFalse(health["checks"]["hybrid_retrieval_probe"])
            self.assertIn("broken_bm25_or_dense", health["retriever_probe_error"])
        self.assertEqual(self.client.payloads, [])


class ProvenanceTests(unittest.TestCase):
    def test_hybrid_probe_queries_once_and_closes_backend(self):
        backend = FixtureRetriever()
        with patch.object(runner, "VerifiedHybridRetriever", return_value=backend):
            runner.probe_hybrid_retriever()
        self.assertEqual(backend.calls, 1)
        self.assertTrue(backend.closed)
        self.assertEqual(backend.trace[0]["query"], "United States")

        backend = FixtureRetriever()
        backend.retrieve = lambda *_args, **_kwargs: []
        with patch.object(runner, "VerifiedHybridRetriever", return_value=backend), \
             self.assertRaisesRegex(RuntimeError, "hybrid_probe_invalid_top3"):
            runner.probe_hybrid_retriever()
        self.assertTrue(backend.closed)

    def test_dense_candidates_must_match_corpus_backed_table(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE docs(doc_id TEXT PRIMARY KEY,title TEXT,text TEXT)")
            db.execute("INSERT INTO docs VALUES ('1','Known','Real passage')")
            branch = SimpleNamespace(search=lambda *a: [{"doc_id":"1", "title":"Known", "text":"Real passage"}])
            checked = runner.CheckedDenseBranch(branch, db)
            self.assertEqual(len(checked.search("query", 1)), 1)
            branch.search = lambda *a: [{"doc_id":"1", "title":"Known", "text":"Fake passage"}]
            with self.assertRaisesRegex(RuntimeError, "does_not_match"):
                checked.search("query", 1)

    def test_dense_missing_or_duplicate_candidates_stop(self):
        with sqlite3.connect(":memory:") as db:
            branch = SimpleNamespace(search=lambda *a: [])
            with self.assertRaisesRegex(RuntimeError, "candidate_count"):
                runner.CheckedDenseBranch(branch, db).search("query", 20)

    def test_bm25_branch_cannot_silently_return_too_few_candidates(self):
        branch = SimpleNamespace(search=lambda *a: [], conn=None)
        with self.assertRaisesRegex(RuntimeError, "bm25_candidate_count"):
            runner.CheckedBM25Branch(branch).search("query", 20)


if __name__ == "__main__":
    unittest.main()
