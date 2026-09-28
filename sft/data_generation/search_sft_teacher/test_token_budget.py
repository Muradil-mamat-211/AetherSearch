"""Regression checks for RL budgets, visible evidence and conversation growth."""

import copy
import json
import unittest
from unittest.mock import Mock

from controlled_rollout import Rejected, parse_action, rollout
from deepseek_client import DeepSeekClient
from hybrid_retriever_v1 import RetrievedDoc, docs_to_jsonable
from test_deepseek_rollout import ANSWER, FINAL, KEY, ROW, FixtureRetriever, response
from token_budget import BudgetError, MAX_ACTION_TOKENS, student_budget
from validate_teacher_rollout import validate_record


class TokenBudgetTests(unittest.TestCase):
    def test_exact_complete_action_boundary_includes_eos(self):
        budget = student_budget()
        for repetitions in range(450, 501):
            text = '<think>Prior knowledge.</think><answer>' + 'word ' * repetitions + '</answer>'
            if budget.action_tokens(text) == MAX_ACTION_TOKENS:
                parse_action(text, "answer")
                with self.assertRaisesRegex(Rejected, "answer_action_token_budget_exceeded"):
                    parse_action(text.replace('</answer>', ' word</answer>'), "answer")
                break
        else:
            self.fail("Could not construct a boundary action")

    def test_overlong_think_counts_even_when_answer_is_short(self):
        text = '<think>' + 'reason ' * 600 + '</think><answer>Yes</answer>'
        with self.assertRaisesRegex(Rejected, "answer_action_token_budget_exceeded"):
            parse_action(text, "answer")

    def test_overlong_search_never_reaches_retrieval(self):
        raw = response(True)
        action = '<think>' + 'reason ' * 600 + '</think><search>Bendigo Street dispute</search>'
        raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps({"action": action})
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(return_value=(raw, 1))
        retriever = FixtureRetriever()
        with self.assertRaisesRegex(Rejected, "search_action_token_budget_exceeded"):
            rollout(ROW, client, retriever, 5)
        self.assertEqual(retriever.calls, 0)

    def test_all_three_docs_and_tags_count_toward_one_information_budget(self):
        docs = [RetrievedDoc(str(i), f"Title {i}", "short text" if i < 3 else "background " * 700,
                             "bm25", i, None, 1 / (60 + i)) for i in range(1, 4)]
        budget = student_budget()
        observation = budget.information(docs_to_jsonable(docs))
        self.assertTrue(observation["information_truncated"])
        self.assertEqual(observation["information_injected_tokens"], 500)
        self.assertLessEqual(budget.count(observation["information"]), 500)
        self.assertEqual(len(observation["visible_documents"]), 3)
        self.assertTrue(observation["information"].endswith('</information>'))
        self.assertEqual(budget.decode(observation["information_token_ids"]), observation["information"])

    def test_missing_third_doc_after_prefix_clip_is_rejected(self):
        docs = docs_to_jsonable(FixtureRetriever().retrieve("test", 3))
        docs[0]["text"] = "background " * 700
        with self.assertRaisesRegex(BudgetError, "information_missing_top3_after_truncation"):
            student_budget().information(docs)

    def test_answer_outside_visible_prefix_cannot_support_final_action(self):
        class LateAnswerRetriever:
            def retrieve(self, query, topk):
                return [RetrievedDoc(str(i), f"Title {i}", "irrelevant" if i < 3 else "background " * 700 + ANSWER,
                                     "bm25", i, None, 1 / (60 + i)) for i in range(1, 4)]
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(side_effect=[(response(True), 1), (response(text=FINAL.replace("Doc 2", "Doc 3")), 1)])
        with self.assertRaisesRegex(Rejected, "answer_not_in_evidence"):
            rollout(ROW, client, LateAnswerRetriever(), 5)
        tool = client._post.call_args_list[-1].args[0]["messages"][-1]
        self.assertEqual(tool["role"], "tool")
        self.assertNotIn(ANSWER, tool["content"])
        self.assertLessEqual(student_budget().count(tool["content"]), 500)

    def test_five_searches_preserve_all_pairs_then_allow_final_answer(self):
        outputs = []
        for i in range(5):
            raw = response(True)
            raw["id"] = f"response-{i}"
            call = raw["choices"][0]["message"]["tool_calls"][0]
            call["id"] = f"call-{i}"
            call["function"]["arguments"] = json.dumps({"action": f"<think>I need the project's name.</think><search>Bendigo Street dispute detail {i}</search>"})
            outputs.append((raw, 1))
        final = response(text=FINAL.replace("Doc 2", "Turn 5 Doc 2"))
        final["id"] = "response-final"
        outputs.append((final, 1))
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(side_effect=outputs)
        retriever = FixtureRetriever()
        record, audit = rollout(ROW, client, retriever, 5)
        audit["provenance_checked"] = True
        self.assertEqual(validate_record(record, audit), [])
        self.assertEqual(record["metadata"]["search_count"], 5)
        self.assertEqual(len(record["events"]), 11)
        self.assertEqual([c.args[0]["tool_choice"] for c in client._post.call_args_list], ["auto"] * 5 + ["none"])
        history = client._post.call_args_list[-1].args[0]["messages"]
        self.assertEqual([m["role"] for m in history], ["system", "user"] + ["assistant", "tool"] * 5 + ["user"])
        for i in range(5):
            assistant, tool = history[2 + 2 * i:4 + 2 * i]
            self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])
            self.assertEqual(tool["content"], record["events"][2 * i + 1]["text"])
        self.assertEqual(retriever.calls, 5)

    def test_validator_recomputes_counts_and_visible_observation(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(side_effect=[(response(True), 1), (response(text=FINAL), 1)])
        record, audit = rollout(ROW, client, FixtureRetriever(), 5)
        audit["provenance_checked"] = True
        changed = copy.deepcopy(record)
        changed["metadata"]["action_token_counts"][0] += 1
        self.assertIn("action_token_counts_mismatch", validate_record(changed, audit))
        changed_audit = copy.deepcopy(audit)
        changed_audit["retrieval_trace"][0]["visible_documents"][0]["text"] = "invented"
        self.assertIn("audit_visible_observation_mismatch", validate_record(record, changed_audit))


if __name__ == "__main__":
    unittest.main()
