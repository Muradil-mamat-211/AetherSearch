"""Short-answer correction preserves model authorship, evidence and API budgets."""

import copy
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controlled_rollout import INSTRUCTIONS, PROMPT, Rejected, parse_action, rollout
from token_budget import student_budget
from deepseek_client import ANSWER_REPAIR_INSTRUCTION, APIError, DeepSeekClient
from test_deepseek_rollout import ANSWER, DIRECT_FINAL, NORMALIZED_DIRECT_FINAL, FINAL, NORMALIZED_FINAL, KEY, ROW, FixtureRetriever, response
from validate_teacher_rollout import validate_record


LONG_DIRECT = '<think>Prior knowledge is sufficient.</think><answer>' + 'A detailed biography. ' * 200 + '</answer>'
LONG_SEARCHED = '<think>Doc 2 supports the answer.</think><answer>' + 'A lengthy explanation. ' * 200 + '</answer>'


def scripted_client(*outputs, thinking="disabled"):
    client = DeepSeekClient("deepseek-flash", KEY, thinking=thinking)
    raws = []
    for i, output in enumerate(outputs):
        raw = response(output == "search", text=output, reasoning="fixture private reasoning" if thinking == "enabled" else None)
        raw["id"] = f"short-answer-response-{i}"
        raws.append((raw, 1))
    client._post = Mock(side_effect=raws)
    return client


class ShortAnswerTests(unittest.TestCase):
    def test_prompts_request_brief_actions_without_student_token_estimation(self):
        for prompt in (INSTRUCTIONS, ANSWER_REPAIR_INSTRUCTION):
            self.assertNotIn("500 student-model tokens", prompt)
            self.assertNotIn("500 student-token limit", prompt)
            self.assertIn("Keep the think summary brief and the search query concise and focused.", prompt)
            self.assertIn("identifying role or definition", prompt)
            self.assertNotIn("160 characters", prompt)

    def test_answers_over_160_characters_are_allowed_within_the_token_budget(self):
        action = f'<think>Prior knowledge.</think><answer>{"A" * 200}</answer>'
        self.assertLess(student_budget().action_tokens(action), 500)
        parse_action(action, "answer")
        with self.assertRaisesRegex(Rejected, "answer_action_token_budget_exceeded"):
            parse_action(LONG_DIRECT, "answer")

    def test_short_answer_uses_one_request(self):
        client, retriever = scripted_client(DIRECT_FINAL), FixtureRetriever()
        record, audit = rollout(ROW, client, retriever, 3)
        audit["provenance_checked"] = False
        self.assertEqual(client._post.call_count, 1)
        self.assertEqual(retriever.calls, 0)
        self.assertEqual(validate_record(record, audit), [])

    def test_direct_correction_keeps_raw_receipts_but_trains_only_the_model_revision(self):
        client, retriever = scripted_client(LONG_DIRECT, DIRECT_FINAL), FixtureRetriever()
        record, audit = rollout(ROW, client, retriever, 3)
        audit["provenance_checked"] = False
        self.assertEqual(retriever.calls, 0)
        self.assertEqual(client._post.call_count, 2)
        requests = [call.args[0] for call in client._post.call_args_list]
        self.assertEqual([p["tool_choice"] for p in requests], ["auto", "none"])
        self.assertEqual(requests[1]["messages"][-2], {"role": "assistant", "content": LONG_DIRECT})
        self.assertEqual(requests[1]["messages"][-1]["content"], ANSWER_REPAIR_INSTRUCTION)
        self.assertNotIn(ANSWER, requests[1]["messages"][-1]["content"])
        self.assertNotIn("Search budget exhausted", requests[1]["messages"][-1]["content"])
        self.assertEqual(record["response"], NORMALIZED_DIRECT_FINAL)
        self.assertEqual(record["events"], [{"role": "assistant", "text": NORMALIZED_DIRECT_FINAL, "train_on_tokens": True}])
        self.assertEqual(audit["raw_final_action"], DIRECT_FINAL)
        self.assertEqual(audit["api_calls"][0]["action"], LONG_DIRECT)
        self.assertEqual(audit["api_calls"][0]["discarded_reason"], "answer_action_token_budget_exceeded")
        self.assertEqual(audit["api_calls"][1]["repair_of_response_id"], audit["api_calls"][0]["id"])
        self.assertEqual(audit["model_response_ids"], [audit["api_calls"][1]["id"]])
        self.assertEqual(validate_record(record, audit), [])

    def test_searched_correction_reuses_evidence_without_retrieval(self):
        client, retriever = scripted_client("search", LONG_SEARCHED, FINAL), FixtureRetriever()
        record, audit = rollout(ROW, client, retriever, 1)
        audit["provenance_checked"] = True
        requests = [call.args[0] for call in client._post.call_args_list]
        self.assertEqual([p["tool_choice"] for p in requests], ["auto", "none", "none"])
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(record["metadata"]["search_count"], 1)
        self.assertEqual(record["events"][-1]["text"], NORMALIZED_FINAL)
        self.assertEqual(audit["raw_final_action"], FINAL)
        self.assertEqual(len(record["events"]), 3)
        self.assertEqual([m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"],
                         [audit["retrieval_trace"][0]["information"]])
        self.assertEqual(validate_record(record, audit), [])

    def test_second_long_answer_stops_without_a_third_request(self):
        client, retriever = scripted_client(LONG_DIRECT, LONG_DIRECT), FixtureRetriever()
        with self.assertRaisesRegex(Rejected, "answer_action_token_budget_exceeded"):
            rollout(ROW, client, retriever, 3)
        self.assertEqual(client._post.call_count, 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(retriever.calls, 0)

    def test_tool_call_during_format_correction_never_reaches_backend(self):
        client, retriever = scripted_client(LONG_DIRECT, "search"), FixtureRetriever()
        with self.assertRaisesRegex(Rejected, "tool_call_after_search_budget"):
            rollout(ROW, client, retriever, 3)
        self.assertEqual(client._post.call_count, 2)
        self.assertEqual(retriever.calls, 0)

    def test_wrong_shortened_answer_is_not_replaced_with_gold(self):
        client = scripted_client(LONG_DIRECT, DIRECT_FINAL.replace(ANSWER, "Wrong project"))
        with self.assertRaisesRegex(Rejected, "answer_not_equal_to_gold"):
            rollout(ROW, client, FixtureRetriever(), 3)

    def test_false_citation_after_shortening_is_rejected(self):
        client = scripted_client("search", LONG_SEARCHED, FINAL.replace("Doc 2", "Doc 1"))
        with self.assertRaisesRegex(Rejected, "answer_not_in_evidence"):
            rollout(ROW, client, FixtureRetriever(), 3)

    def test_untagged_refusal_after_correction_is_rejected_without_gold_substitution(self):
        client = scripted_client(LONG_DIRECT, "I cannot determine the answer.")
        with self.assertRaisesRegex(Rejected, "invalid_answer_action"):
            rollout(ROW, client, FixtureRetriever(), 3)
        self.assertEqual(client._post.call_count, 2)

    def test_other_format_errors_do_not_trigger_length_correction(self):
        client = scripted_client("An untagged answer")
        with self.assertRaisesRegex(Rejected, "invalid_answer_action"):
            rollout(ROW, client, FixtureRetriever(), 3)
        self.assertEqual(client._post.call_count, 1)

    def test_thinking_content_is_replayed_for_correction_without_entering_training(self):
        client = scripted_client(LONG_DIRECT, DIRECT_FINAL, thinking="enabled")
        record, _ = rollout(ROW, client, FixtureRetriever(), 3)
        request = client._post.call_args_list[-1].args[0]
        self.assertEqual(request["messages"][-2]["reasoning_content"], "fixture private reasoning")
        self.assertNotIn("fixture private reasoning", record["response"])

    def test_correction_respects_http_attempt_budget(self):
        client = DeepSeekClient("deepseek-flash", KEY, max_requests=1, retries=0)
        raw = response(text=LONG_DIRECT)
        opener = Mock(return_value=io.BytesIO(json.dumps(raw).encode()))
        client.opener = SimpleNamespace(open=opener)
        with self.assertRaisesRegex(APIError, "api_request_budget_exhausted"):
            client.create([{"role": "user", "content": ROW["question"]}], "auto")
        self.assertEqual(client.request_count, 1)
        self.assertEqual(opener.call_count, 1)

    def test_discarded_receipts_cannot_hide_bad_or_extra_actions(self):
        client = scripted_client(LONG_DIRECT, DIRECT_FINAL)
        record, audit = rollout(ROW, client, FixtureRetriever(), 3)
        audit["provenance_checked"] = False
        for mutation in (
                lambda a: a["api_calls"][0].update(action=DIRECT_FINAL),
                lambda a: a["api_calls"][0].update(discarded_reason="wrong_answer"),
                lambda a: a["api_calls"][1].update(repair_of_response_id="unrelated"),
                lambda a: a["api_calls"][1].update(tool_choice="auto"),
                lambda a: a["api_calls"].insert(0, copy.deepcopy(a["api_calls"][0])),
                lambda a: a["api_calls"].pop(0)):
            changed = copy.deepcopy(audit)
            mutation(changed)
            self.assertIn("invalid_answer_repair_audit", validate_record(record, changed))


if __name__ == "__main__":
    unittest.main()
