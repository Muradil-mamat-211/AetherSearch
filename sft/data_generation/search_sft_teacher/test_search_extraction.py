"""Literal extraction, replay provenance and offline comparison boundaries."""

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controlled_rollout import Rejected, extract_search_action, rollout
from deepseek_client import DeepSeekClient, extract_search_message
from replay_api_responses import evaluate, run
from test_deepseek_rollout import ACTION, DIRECT_FINAL, FINAL, KEY, ROW, FixtureRetriever, fixture_record, response
from test_search_protocol import raw_search
from token_budget import student_budget
from validate_teacher_rollout import validate_record


class ExtractionTests(unittest.TestCase):
    def test_selected_span_and_inner_text_are_literal(self):
        action = "<think>  brief  basis\n</think><search>query  with   spaces</search>"
        raw = "prefix\n<action>" + action + "</action>\n"
        selected = extract_search_action(raw)
        start, end = selected["span"]
        self.assertEqual(raw[start:end], action)
        self.assertEqual(selected["action"], action)
        self.assertEqual(selected["discarded_prefix"], "prefix\n<action>")
        self.assertEqual(selected["discarded_suffix"], "</action>\n")
        self.assertEqual(selected["match_count"], 1)

    def test_identical_blocks_select_first_without_duplication(self):
        selected = extract_search_action(" " + ACTION + "\n" + ACTION)
        self.assertEqual(selected["span"], [1, 1 + len(ACTION)])
        self.assertEqual(selected["match_count"], 2)
        self.assertEqual(selected["action"], ACTION)
        self.assertEqual(selected["discarded_suffix"], "\n" + ACTION)

    def test_distinct_blocks_reject_before_backend_dispatch(self):
        other = ACTION.replace("road project</search>", "new focused query</search>")
        for raw in (raw_search(ACTION + other), raw_search(ACTION, other), response(text=ACTION + other)):
            client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
            client._post = Mock(return_value=(raw, 1))
            with self.assertRaisesRegex(Rejected, "ambiguous_search_action"):
                rollout(ROW, client, retriever, 5)
            self.assertEqual(retriever.calls, 0)
            self.assertEqual(client._post.call_count, 1)

    def test_mixed_search_and_final_is_not_salvaged(self):
        for raw in (raw_search(ACTION + FINAL), raw_search(ACTION, FINAL), response(text=ACTION + FINAL)):
            client = DeepSeekClient("deepseek-flash", KEY)
            client._post = Mock(return_value=(raw, 1))
            with self.assertRaisesRegex(Rejected, "mixed_search_final_output"):
                client.create([], "auto")

    def test_invalid_inner_format_and_query_are_never_repaired(self):
        bad = [ACTION.replace("</think><search>", "</think> <search>"),
               ACTION.replace("</think><search>", "</think>\n<search>"),
               "<search>query</search>", "<think>basis</think>"]
        bad.extend(f"<think>x</think><search>{query}</search>"
                   for query in ("", " ", "q" * 301, "q\nq", "q\rq", "https://example.test"))
        for action in bad:
            with self.subTest(action=action), self.assertRaises(Rejected):
                extract_search_action("<action>" + action + "</action>")

    def test_real_student_budget_checks_extracted_action_not_discarded_text(self):
        raw = "noise " * 1600 + ACTION + "</action>"
        self.assertGreater(student_budget().action_tokens(raw), 500)
        self.assertEqual(extract_search_action(raw)["action"], ACTION)
        overlong = "<think>" + "decision " * 800 + "</think><search>query</search>"
        with self.assertRaisesRegex(Rejected, "search_action_token_budget_exceeded"):
            extract_search_action("prefix" + overlong + "suffix")

    def test_invalid_source_values_fail_closed(self):
        for value in (None, 123, [], "noise " * 3000):
            with self.subTest(value_type=type(value)), self.assertRaises(Rejected):
                extract_search_action(value)

    def test_malformed_native_calls_cannot_fall_back_to_content(self):
        variants = []
        raw = raw_search(ACTION, ACTION)
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "browser"
        variants.append(raw)
        raw = raw_search(ACTION, ACTION)
        raw["choices"][0]["message"]["tool_calls"] *= 2
        variants.append(raw)
        for arguments in ("not-json", '{"action":"x","action":"y"}', json.dumps({"action": ACTION, "extra": 1})):
            raw = raw_search(ACTION, ACTION)
            raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
            variants.append(raw)
        for raw in variants:
            client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
            client._post = Mock(return_value=(raw, 1))
            with self.assertRaises(Rejected):
                rollout(ROW, client, retriever, 5)
            self.assertEqual(retriever.calls, 0)

    def test_content_search_is_prohibited_after_budget_exhaustion(self):
        for raw in (raw_search(ACTION), response(text=ACTION)):
            client = DeepSeekClient("deepseek-flash", KEY)
            client._post = Mock(return_value=(raw, 1))
            with self.assertRaisesRegex(Rejected, "tool_call_after_search_budget"):
                client.create([], "none")
        client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
        next_action = ACTION.replace("road project</search>", "unresolved historical fact</search>")
        client._post = Mock(side_effect=[(response(text=ACTION), 1), (response(text=next_action), 1)])
        with self.assertRaisesRegex(Rejected, "tool_call_after_search_budget"):
            rollout(ROW, client, retriever, 1)
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(client._post.call_args_list[-1].args[0]["tool_choice"], "none")

    def test_fallback_history_matches_real_observation_and_hides_private_reasoning(self):
        for native in (False, True):
            raw = raw_search("<action>" + ACTION + "</action>", ACTION) if native else response(text=ACTION)
            raw["choices"][0]["message"]["reasoning_content"] = "private fixture reasoning"
            client, retriever = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", reasoning_effort="max"), FixtureRetriever()
            client._post = Mock(side_effect=[(raw, 1), (response(text=FINAL, reasoning="private fixture reasoning"), 1)])
            record, audit = rollout(ROW, client, retriever, 5)
            audit["provenance_checked"] = True
            self.assertEqual(validate_record(record, audit), [])
            self.assertEqual(audit["api_calls"][0]["native_tool_call"], native)
            history = client._post.call_args_list[-1].args[0]["messages"]
            assistant, tool = history[-2:]
            self.assertEqual(assistant["content"], "")
            self.assertEqual(len(assistant["tool_calls"]), 1)
            self.assertEqual(tool["tool_call_id"], assistant["tool_calls"][0]["id"])
            self.assertEqual(tool["content"], record["events"][1]["text"])
            self.assertEqual(assistant["reasoning_content"], "private fixture reasoning")
            self.assertNotIn("private fixture reasoning", json.dumps(record))
            self.assertNotIn("private fixture reasoning", json.dumps(audit))
            self.assertEqual(record["events"][0]["text"], ACTION)

    def test_audit_rejects_extraction_or_envelope_tampering(self):
        baseline_record, baseline_audit = fixture_record()
        mutations = [
            ("raw_action", ACTION + "suffix"), ("raw_action_source", "assistant.content"),
            ("native_tool_call", False), ("tool_call_id", "different"), ("normalized_tool_call", {}),
            ("raw_tool_calls", []), ("finish_reason", "stop"), ("tool_choice", None),
            ("id", None), ("extraction", {}),
        ]
        for key, value in mutations:
            audit = copy.deepcopy(baseline_audit)
            audit["api_calls"][0][key] = value
            with self.subTest(key=key):
                self.assertIn("invalid_search_action_receipt", validate_record(baseline_record, audit))
        for key, value in (("span", [1, len(ACTION)]), ("discarded_prefix", "wrapper"),
                           ("discarded_suffix", "suffix"), ("match_count", 2), ("policy", "old_policy")):
            audit = copy.deepcopy(baseline_audit)
            audit["api_calls"][0]["extraction"][key] = value
            with self.subTest(extraction_field=key):
                self.assertIn("invalid_search_action_receipt", validate_record(baseline_record, audit))

    def test_raw_tool_call_json_tampering_is_detected(self):
        record, audit = fixture_record()
        audit["api_calls"][0]["raw_tool_calls"][0]["function"]["arguments"] = json.dumps({"action": ACTION + "</action>"})
        self.assertIn("invalid_search_action_receipt", validate_record(record, audit))

    def test_shared_resolver_rejects_bad_call_container_types(self):
        for calls in ({}, "", False, [None]):
            with self.subTest(calls=calls), self.assertRaises(Rejected):
                extract_search_message({"content": ACTION, "tool_calls": calls},
                                       finish_reason="stop", tool_choice="auto", response_id="fixture-id")


class OfflineReplayTests(unittest.TestCase):
    def row(self, raw, *, number=1, status="rejected", reason="invalid_search_action"):
        return {"number": number, **ROW, "question": ROW["question"] + f" Case {number}.",
                "status": status, "reason": reason, "requests": [{"tool_choice": "auto"}], "responses": [raw]}

    def test_replay_allows_candidate_queries_and_preserves_direct_answer(self):
        row = self.row(raw_search(ACTION + "</action>"))
        result = evaluate(row)
        self.assertTrue(result["recovered"])
        self.assertEqual(result["action"], ACTION)
        row = self.row(response(text=ACTION.replace("road project</search>", "East West Link</search>")))
        self.assertEqual(evaluate(row)["status"], "search_requested")
        row = self.row(response(text=DIRECT_FINAL), status="direct_answer", reason=None)
        result = evaluate(row)
        self.assertEqual(result["status"], "direct_answer")
        self.assertTrue(result["normalized_reference_match"])
        self.assertFalse(result["recovered"])

    def test_offline_replay_has_no_api_key_or_retriever_calls_and_no_input_changes(self):
        rows = [self.row(raw_search(ACTION + "</action>")),
                self.row(response(text=DIRECT_FINAL), number=2, status="direct_answer", reason=None)]
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "saved.jsonl", Path(tmp) / "new_results"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            before = source.read_bytes()
            with patch.object(DeepSeekClient, "__init__", side_effect=AssertionError("No API client in replay")), \
                 patch("deepseek_key.load_key_file", side_effect=AssertionError("No key access")), \
                 patch("deepseek_rollout.VerifiedHybridRetriever", side_effect=AssertionError("No RAG")), \
                 redirect_stdout(io.StringIO()):
                report = run(source, output, expected_questions=2)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(report["input_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(report["api_requests"], 0)
            self.assertFalse(report["training_data_created"])
            self.assertFalse(report["retriever_executed"])
            self.assertEqual(report["recovered_count"], 1)
            self.assertEqual(report["regressions"], [])
            self.assertEqual(report["after_statuses"], {"search_requested": 1, "direct_answer": 1})
            self.assertEqual(len((output / "replay_results.jsonl").read_text().splitlines()), 2)
            saved = (output / "summary.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run(source, output, expected_questions=2)
            self.assertEqual((output / "summary.json").read_bytes(), saved)

    def test_invalid_count_or_duplicate_question_writes_nothing(self):
        row = self.row(raw_search(ACTION))
        for rows in ([row], [row, row]):
            with tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp) / "saved.jsonl", Path(tmp) / "results"
                source.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "count_or_uniqueness"):
                    run(source, output, expected_questions=2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
