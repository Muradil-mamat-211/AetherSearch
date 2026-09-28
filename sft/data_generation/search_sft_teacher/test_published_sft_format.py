"""Public SFT compatibility and multi-turn reasoning budget contracts."""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from controlled_rollout import Rejected, rollout
from deepseek_client import DeepSeekClient, MAX_API_REQUEST_BYTES
from published_sft_format import (ASSISTANT_START, EOT, MAX_FULL_TRAJECTORY_TOKENS,
                                  PUBLIC_FIELDS, public_prefix, public_record, validate_public_record)
from test_deepseek_rollout import (ACTION, DIRECT_FINAL, FINAL, KEY, ROW,
                                   FixtureRetriever, fixture_record, make_client, response)
from token_budget import student_budget
from validate_teacher_rollout import validate_record


class PublishedFormatTests(unittest.TestCase):
    def test_matches_the_published_five_field_qwen_chat_unit(self):
        record, audit = fixture_record()
        public = public_record(record)
        self.assertEqual(tuple(public), PUBLIC_FIELDS)
        self.assertEqual(public["id"], "500001")
        self.assertEqual(public["trajectory_type"], "single_search")
        self.assertEqual(public["search_count"], 1)
        self.assertEqual(public["full_trajectory_text"], public_prefix(record["question"]) + record["response"] + EOT)
        self.assertEqual(public["full_trajectory_text"].count(EOT), 3)
        self.assertEqual(public["full_trajectory_text"].count(ASSISTANT_START), 1)
        self.assertEqual(student_budget().count(public["full_trajectory_text"]),
                         record["metadata"]["public_full_trajectory_tokens"])
        self.assertEqual(validate_public_record(public), [])
        self.assertEqual(validate_record(record, audit), [])
        self.assertNotIn("reasoning_content", public["full_trajectory_text"])

    def test_multi_search_and_direct_answer_have_distinct_internal_types(self):
        second = response(True)
        second["choices"][0]["message"]["tool_calls"][0]["id"] = "call2"
        second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({
            "action": ACTION.replace("road project</search>", "road project second pass</search>")})
        client = DeepSeekClient("deepseek-flash", KEY)
        with patch.object(client, "_post", side_effect=[(response(True), 1), (second, 1), (response(), 1)]):
            multi, audit = rollout(ROW, client, FixtureRetriever(), 3)
        audit["provenance_checked"] = True
        self.assertEqual(validate_record(multi, audit), [])
        public = public_record(multi)
        self.assertEqual((public["trajectory_type"], public["search_count"]), ("multi_search", 2))
        body = public["full_trajectory_text"][len(public_prefix(public["question"])):-len(EOT)]
        self.assertEqual(body.count("<information>"), 2)
        self.assertTrue(body.endswith("<think>The retrieved evidence now supports the answer.</think>"
                                      "<answer>East West Link</answer>"))
        self.assertEqual(validate_public_record(public), [])
        direct, _ = rollout(ROW, make_client(direct=True), FixtureRetriever(), 3)
        simple = public_record(direct)
        self.assertEqual((simple["trajectory_type"], simple["search_count"]), ("direct_answer", 0))
        self.assertIn("<think>Reliable prior knowledge is sufficient to answer.</think>",
                      simple["full_trajectory_text"])
        self.assertNotIn("<information>", simple["full_trajectory_text"][len(public_prefix(simple["question"])):])
        self.assertEqual(validate_public_record(simple), [])

    def test_public_validator_rejects_template_eot_depth_and_evidence_changes(self):
        record, _ = fixture_record()
        public = public_record(record)
        mutations = (
            lambda r: r.update(id="teacher_qa-1"),
            lambda r: r.update(id="000000"),
            lambda r: r.update(trajectory_type="multi_search"),
            lambda r: r.update(search_count=0),
            lambda r: r.update(full_trajectory_text=r["full_trajectory_text"] + EOT),
            lambda r: r.update(full_trajectory_text=r["full_trajectory_text"].replace(
                record["events"][1]["text"], "<context>wrong observation</context>")),
            lambda r: r.update(full_trajectory_text=r["full_trajectory_text"].replace("<answer>East West Link</answer>", "<answer>East West Link</answer></answer>")),
            lambda r: r.update(extra="not_public"),
        )
        for mutate in mutations:
            changed = copy.deepcopy(public)
            mutate(changed)
            self.assertTrue(validate_public_record(changed))

    def test_public_format_accepts_historical_observation_markup_and_uncited_summary(self):
        record, audit = fixture_record()
        public = public_record(record)
        body = record["response"].replace("<information>", "<information><poem>Historical corpus markup</poem>\n", 1)
        body = body.replace(FINAL, "<think>The retrieved evidence supports the answer.</think><answer>East West Link</answer>")
        public["full_trajectory_text"] = public_prefix(record["question"]) + body + EOT
        self.assertEqual(validate_public_record(public), [])
        record["events"][1]["text"] = record["events"][1]["text"].replace(
            "<information>", "<information><poem>Historical corpus markup</poem>\n", 1)
        self.assertTrue(validate_record(record, audit))

    def test_full_trajectory_has_a_student_token_cap(self):
        record, _ = fixture_record()
        record["response"] += " other" * MAX_FULL_TRAJECTORY_TOKENS
        with self.assertRaisesRegex(ValueError, "public_trajectory_invalid"):
            public_record(record)


class ReasoningBudgetTests(unittest.TestCase):
    def test_reasoning_usage_is_audited_but_hidden_text_is_not_exported(self):
        client = make_client(thinking="enabled")
        record, audit = rollout(ROW, client, FixtureRetriever(), 3)
        audit["provenance_checked"] = True
        self.assertEqual(validate_record(record, audit), [])
        self.assertGreater(record["metadata"]["teacher_cumulative_reasoning_tokens"], 0)
        self.assertNotIn("offline hidden reasoning", public_record(record)["full_trajectory_text"])
        self.assertEqual(record["metadata"]["teacher_cumulative_prompt_usage_units"], 200)
        self.assertEqual(audit["api_calls"][1]["replayed_reasoning_messages"], 1)
        tampered = copy.deepcopy(audit)
        tampered["api_calls"][0]["reasoning_budget_charge"] += 1
        self.assertIn("invalid_reasoning_budget_receipt", validate_record(record, tampered))
        tampered = copy.deepcopy(audit)
        tampered["api_calls"][1]["replayed_reasoning_messages"] = 0
        self.assertIn("reasoning_replay_budget_mismatch", validate_record(record, tampered))
        tampered = copy.deepcopy(audit)
        tampered["api_calls"][1]["prompt_usage_units"] += 1
        self.assertIn("invalid_prompt_usage_receipt", validate_record(record, tampered))
        tampered = copy.deepcopy(audit)
        tampered["api_calls"][0]["usage"] = []
        self.assertIn("invalid_prompt_usage_receipt", validate_record(record, tampered))

    def test_replayed_reasoning_is_metered_without_a_cumulative_prompt_rejection(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled")
        first = response(True, reasoning="private one")
        second = response(reasoning="private two")
        client._post = Mock(side_effect=[(first, 1), (second, 1)])
        retriever = FixtureRetriever()
        record, audit = rollout(ROW, client, retriever, 3)
        audit["provenance_checked"] = True
        self.assertEqual(validate_record(record, audit), [])
        self.assertEqual(retriever.calls, 1)
        self.assertEqual([c["cumulative_prompt_usage_units"] for c in client.calls], [100, 200])
        self.assertEqual(client.calls[1]["replayed_reasoning_messages"], 1)
        self.assertNotIn("prompt_budget_limit", client.calls[1])

    def test_provider_reported_reasoning_accumulates_across_search_turns(self):
        first = response(True, reasoning="private one")
        second = response(True, reasoning="private two")
        second["id"] = "second"
        second["choices"][0]["message"]["tool_calls"][0]["id"] = "call2"
        second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({
            "action": ACTION.replace("road project</search>", "road project followup</search>")})
        for raw in (first, second):
            raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": 7}
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", max_cumulative_reasoning_tokens=10)
        client._post = Mock(side_effect=[(first, 1), (second, 1)])
        retriever = FixtureRetriever()
        with self.assertRaisesRegex(Rejected, "cumulative_reasoning_budget_exceeded"):
            rollout(ROW, client, retriever, 3)
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(client._post.call_count, 2)
        self.assertEqual([c["cumulative_reasoning_tokens"] for c in client.calls], [7, 14])
        self.assertEqual([c["reasoning_budget_source"] for c in client.calls], ["provider_usage", "provider_usage"])

    def test_missing_provider_reasoning_count_uses_conservative_byte_charge(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", max_cumulative_reasoning_tokens=5)
        raw = response(True, reasoning="你好")
        client._post = Mock(return_value=(raw, 1))
        with self.assertRaisesRegex(Rejected, "cumulative_reasoning_budget_exceeded"):
            client.create([{"role": "user", "content": "Test"}], "auto")
        self.assertEqual(client.calls[0]["reasoning_budget_charge"], len("你好".encode("utf-8")))
        self.assertEqual(client.calls[0]["reasoning_budget_source"], "utf8_byte_upper_bound")

    def test_request_byte_cap_rejects_before_second_http_call(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", max_cumulative_reasoning_tokens=2_000_000)
        client._post = Mock(return_value=(response(True, reasoning="r" * MAX_API_REQUEST_BYTES), 1))
        with self.assertRaisesRegex(Rejected, "api_request_byte_budget_exceeded"):
            rollout(ROW, client, FixtureRetriever(), 2)
        self.assertEqual(client._post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
