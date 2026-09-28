"""FINAL receipts remain exact; citations and unsupported answers cannot be trained."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controlled_rollout import Rejected, parse_action, rollout
from deepseek_client import DeepSeekClient
from test_deepseek_rollout import ANSWER, DIRECT_FINAL, NORMALIZED_DIRECT_FINAL, FINAL, KEY, ROW, FixtureRetriever, fixture_record, response
from token_budget import student_budget
from validate_teacher_rollout import validate_record


INVALID_FINALS = [
    " " + DIRECT_FINAL, "\n" + DIRECT_FINAL, "\t" + DIRECT_FINAL,
    DIRECT_FINAL + " ", DIRECT_FINAL + "\n", DIRECT_FINAL + "\t",
    DIRECT_FINAL.replace("</think><answer>", "</think> <answer>"),
    DIRECT_FINAL.replace("</think><answer>", "</think>\n<answer>"),
    "<action>" + DIRECT_FINAL + "</action>", DIRECT_FINAL + "</action>",
    DIRECT_FINAL + "</invoke>", DIRECT_FINAL + "extra text",
]


class FinalProtocolTests(unittest.TestCase):
    def test_exact_final_and_prior_knowledge_direct_answer_remain_valid(self):
        self.assertEqual(parse_action("<think>x</think><answer>y</answer>", "answer"), ("x", "y"))
        record, audit = fixture_record(direct=True)
        self.assertEqual(record["events"][-1]["text"], NORMALIZED_DIRECT_FINAL)
        self.assertEqual(audit["raw_final_action"], DIRECT_FINAL)
        self.assertEqual(validate_record(record, audit), [])

    def test_final_whitespace_and_wrappers_reject_without_format_cleanup_or_repair(self):
        for action in INVALID_FINALS:
            with self.subTest(action=action):
                with self.assertRaisesRegex(Rejected, "invalid_answer_action"):
                    parse_action(action, "answer")
                client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
                client._post = Mock(return_value=(response(text=action), 1))
                with self.assertRaisesRegex(Rejected, "invalid_answer_action"):
                    rollout(ROW, client, retriever, 5)
                self.assertEqual(client._post.call_count, 1)
                self.assertEqual(client.calls[0]["action"], action)
                self.assertNotIn("discarded_reason", client.calls[0])
                self.assertEqual(retriever.calls, 0)

    def test_controller_cannot_trim_final_when_api_adapter_is_bypassed(self):
        for action in INVALID_FINALS:
            with self.subTest(action=action):
                result = {"status": "completed", "output": [{"type": "message",
                          "content": [{"type": "output_text", "text": action}]}]}
                client = SimpleNamespace(model="fixture-teacher", create=Mock(return_value=result))
                with self.assertRaisesRegex(Rejected, "invalid_answer_action"):
                    rollout(ROW, client, FixtureRetriever(), 5)

    def test_citations_in_answer_reject_for_direct_and_searched_candidates(self):
        for citation in ("Turn 1 Doc 2", "Doc 2", "turn 1 doc 2", "Turn\n1\tDoc\n2"):
            for searched in (False, True):
                with self.subTest(citation=citation, searched=searched):
                    action = (FINAL if searched else DIRECT_FINAL).replace(
                        "</answer>", " (" + citation + ")</answer>")
                    with self.assertRaisesRegex(Rejected, "citation_in_answer"):
                        parse_action(action, "answer")
                    client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
                    outputs = ([(response(True), 1)] if searched else []) + [(response(text=action), 1)]
                    client._post = Mock(side_effect=outputs)
                    with self.assertRaisesRegex(Rejected, "citation_in_answer"):
                        rollout(ROW, client, retriever, 5)
                    self.assertEqual(client._post.call_count, 2 if searched else 1)
                    self.assertEqual(retriever.calls, int(searched))

    def test_supported_citation_remains_only_in_final_think(self):
        action = FINAL.replace("Doc 2", "Turn 1 Doc 2")
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(side_effect=[(response(True), 1), (response(text=action), 1)])
        record, audit = rollout(ROW, client, FixtureRetriever(), 1)
        audit["provenance_checked"] = True
        self.assertEqual(parse_action(record["events"][-1]["text"], "answer")[1], ANSWER)
        self.assertEqual(audit["raw_final_action"], action)
        self.assertEqual(validate_record(record, audit), [])

    def test_validator_rejects_citations_in_event_and_raw_receipt(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                record, audit = fixture_record(direct=direct)
                action = record["events"][-1]["text"].replace("</answer>", " (Turn 1 Doc 2)</answer>")
                record["events"][-1]["text"] = action
                record["messages"][-1]["content"] = action
                record["response"] = "".join(event["text"] for event in record["events"])
                record["metadata"]["action_token_counts"][-1] = student_budget().action_tokens(action)
                audit["raw_final_action"] = action
                audit["api_calls"][-1].update(action=action, assistant_content=action)
                errors = validate_record(record, audit)
                self.assertIn("citation_in_answer", errors)
                self.assertIn("invalid_raw_final_action", errors)

    def test_raw_receipt_whitespace_cannot_be_hidden_by_trimmed_final_audit(self):
        for action in INVALID_FINALS:
            with self.subTest(action=action):
                record, audit = fixture_record(direct=True)
                audit["api_calls"][-1].update(action=action, assistant_content=action)
                self.assertIn("invalid_raw_final_action", validate_record(record, audit))

    def test_training_final_whitespace_cannot_be_hidden_by_a_valid_receipt(self):
        record, audit = fixture_record(direct=True)
        changed = copy.deepcopy(record)
        changed["events"][-1]["text"] = " " + NORMALIZED_DIRECT_FINAL
        changed["response"] = changed["events"][-1]["text"]
        changed["messages"][-1]["content"] = changed["response"]
        self.assertIn("invalid_answer_action", validate_record(changed, audit))
        self.assertIn("final_action_not_equal_to_model_output", validate_record(changed, audit))

    def test_exhausted_budget_with_no_visible_support_rejects_instead_of_fabricating(self):
        class InsufficientRetriever(FixtureRetriever):
            def retrieve(self, query, topk):
                docs = super().retrieve(query, topk)
                docs[1].text = "The dispute concerned housing; the road project's name is not stated."
                return docs

        client, retriever = DeepSeekClient("deepseek-flash", KEY), InsufficientRetriever()
        client._post = Mock(side_effect=[(response(True), 1), (response(text=FINAL), 1)])
        with self.assertRaisesRegex(Rejected, "answer_not_in_evidence"):
            rollout(ROW, client, retriever, 1)
        self.assertEqual([call.args[0]["tool_choice"] for call in client._post.call_args_list], ["auto", "none"])
        self.assertEqual(client.calls[-1]["action"], FINAL)
        self.assertEqual(retriever.calls, 1)

    def test_exhausted_budget_refusals_wrong_answers_and_special_tags_reject(self):
        cases = [("I cannot determine a reliable answer.", "invalid_answer_action"),
                 ("<abstain>insufficient_evidence</abstain>", "invalid_answer_action"),
                 (FINAL.replace(ANSWER, "Unknown"), "answer_not_equal_to_gold")]
        for action, error in cases:
            with self.subTest(action=action):
                client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
                client._post = Mock(side_effect=[(response(True), 1), (response(text=action), 1)])
                with self.assertRaisesRegex(Rejected, error):
                    rollout(ROW, client, retriever, 1)
                self.assertEqual(client._post.call_count, 2)
                self.assertEqual(client.calls[-1]["action"], action)
                self.assertEqual(retriever.calls, 1)

    def test_no_special_decline_protocol_in_search_or_final(self):
        for action in ("<abstain>insufficient_evidence</abstain>",
                       "<think>Evidence is insufficient.</think><abstain>insufficient_evidence</abstain>"):
            for kind in ("search", "answer"):
                with self.subTest(action=action, kind=kind), self.assertRaisesRegex(Rejected, "invalid_.*_action"):
                    parse_action(action, kind)


if __name__ == "__main__":
    unittest.main()
