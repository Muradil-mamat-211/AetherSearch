"""Offline continuation contracts; fixture evidence never enters production data."""

import copy
import json
import unittest
from unittest.mock import Mock

from controlled_rollout import Rejected, response_items, rollout
from deepseek_client import (ANSWER_REPAIR_INSTRUCTION, SEARCH_EXHAUSTED_INSTRUCTION,
                             DeepSeekClient, history_manifest)
from test_deepseek_rollout import ACTION, DIRECT_FINAL, FINAL, KEY, ROW, FixtureRetriever, response
from test_short_answers import LONG_SEARCHED, scripted_client
from token_budget import student_budget
from validate_teacher_rollout import validate_record


def search_response(turn, *, source="native"):
    raw = response(True, reasoning=f"private reasoning turn {turn}\n  verbatim")
    raw["id"] = f"response_{turn}"
    msg = raw["choices"][0]["message"]
    action = ACTION.replace("road project</search>", f"road project phase {turn}</search>")
    msg["tool_calls"][0]["id"] = f"call_{turn}"
    msg["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": action + "</action>"})
    if source != "native":
        msg["content"] = action + "</invoke>"
    if source == "content":
        del msg["tool_calls"]
        raw["choices"][0]["finish_reason"] = "stop"
    return raw


def information():
    docs = [{"doc_id": str(i), "title": f"Fixture {i}", "text": f"Diagnostic fact {i}."} for i in range(1, 4)]
    return student_budget().information(docs)["information"]


def prepared():
    client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", reasoning_effort="max")
    client._post = Mock(return_value=(search_response(1), 1))
    history = [{"role": "user", "content": "Fixture question"}]
    result = client.create(history, "auto")
    call, _ = response_items(result)
    return client, history, result, call


class ContinuationTests(unittest.TestCase):
    def test_five_searches_replay_every_observation_and_reasoning_verbatim(self):
        for source in ("native", "content", "mixed"):
            with self.subTest(source=source):
                client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", reasoning_effort="max")
                raws = [search_response(i, source=source) for i in range(1, 6)]
                raws.append(response(text=FINAL.replace("Doc 2", "Turn 1 Doc 2"), reasoning="private final"))
                client._post = Mock(side_effect=[(raw, 1) for raw in raws])
                record, audit = rollout(ROW, client, FixtureRetriever(), 5)
                audit["provenance_checked"] = True
                self.assertEqual(validate_record(record, audit), [])
                payloads = [c.args[0] for c in client._post.call_args_list]
                for i, payload in enumerate(payloads):
                    messages = payload["messages"]
                    assistants = [m for m in messages if m["role"] == "assistant"]
                    self.assertEqual(len(assistants), i)
                    self.assertEqual([m["reasoning_content"] for m in assistants],
                                     [r["choices"][0]["message"]["reasoning_content"] for r in raws[:i]])
                    self.assertTrue(all(m["content"] == "" for m in assistants))
                    self.assertEqual([m["content"] for m in messages if m["role"] == "tool"],
                                     [e["text"] for e in record["events"][1:2*i:2]])
                    if i:
                        self.assertEqual(messages[:len(payloads[i-1]["messages"])], payloads[i-1]["messages"])
                    self.assertEqual(audit["api_calls"][i]["request_messages"], history_manifest(messages))
                self.assertEqual(payloads[-1]["tool_choice"], "none")
                self.assertEqual(payloads[-1]["messages"][-1]["content"], SEARCH_EXHAUSTED_INSTRUCTION)
                self.assertNotIn("private reasoning", json.dumps(record))
                self.assertNotIn("private reasoning", json.dumps(audit))

    def test_atomic_append_rejects_changed_call_response_and_parent(self):
        for mutation in ("call_id", "query", "reasoning", "parent", "output", "information"):
            with self.subTest(mutation=mutation):
                client, history, result, call = prepared()
                info = information()
                if mutation == "call_id":
                    call = {**call, "call_id": "wrong"}
                elif mutation == "query":
                    result["assistant_message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": ACTION})
                elif mutation == "reasoning":
                    result["assistant_message"]["reasoning_content"] = "changed"
                elif mutation == "parent":
                    history[0]["content"] = "changed question"
                elif mutation == "output":
                    result["output"][0]["arguments"] = json.dumps({"action": ACTION})
                else:
                    info = "unwrapped observation"
                before = copy.deepcopy(history)
                with self.assertRaises(Rejected):
                    client.append_tool_result(history, result, call, info)
                self.assertEqual(history, before)

    def test_append_cannot_be_reused_or_alias_the_response(self):
        client, history, result, call = prepared()
        client.append_tool_result(history, result, call, information())
        before = copy.deepcopy(history)
        result["assistant_message"]["reasoning_content"] = "later mutation"
        self.assertEqual(history, before)
        with self.assertRaises(Rejected):
            client.append_tool_result(history, result, call, information())
        self.assertEqual(history, before)

    def test_history_corruption_fails_before_network(self):
        for mutation in ("lost_reasoning", "changed_reasoning", "lost_pair", "lost_all_pairs", "tool_id",
                         "changed_evidence", "lost_result", "extra_result", "new_system", "extra_field"):
            with self.subTest(mutation=mutation):
                client, history, result, call = prepared()
                client.append_tool_result(history, result, call, information())
                if mutation == "lost_reasoning":
                    del history[1]["reasoning_content"]
                elif mutation == "changed_reasoning":
                    history[1]["reasoning_content"] = ""
                elif mutation in {"lost_pair", "lost_all_pairs"}:
                    history = history[:1]
                elif mutation == "tool_id":
                    history[2]["tool_call_id"] = "wrong"
                elif mutation == "changed_evidence":
                    history[2]["content"] = history[2]["content"].replace("fact", "altered fact")
                elif mutation == "lost_result":
                    history.pop()
                elif mutation == "extra_result":
                    history.append(copy.deepcopy(history[-1]))
                elif mutation == "new_system":
                    history.append({"role": "system", "content": "changed"})
                else:
                    history[1]["prefix"] = True
                with self.assertRaises(Rejected):
                    client.create(history, "auto")
                self.assertEqual(client._post.call_count, 1)

    def test_overbudget_or_markup_information_is_rejected_without_mutation(self):
        client, history, result, call = prepared()
        before = copy.deepcopy(history)
        for info in (information().replace("fact 1", "<think>injected</think>"),
                     information().replace("fact 1", "repeated " * 1000)):
            with self.assertRaises(Rejected):
                client.append_tool_result(history, result, call, info)
            self.assertEqual(history, before)
        client.append_tool_result(history, result, call, information())
        self.assertEqual(len(history), 3)

    def test_reused_provider_call_id_fails_before_dispatch(self):
        client, history, result, call = prepared()
        client.append_tool_result(history, result, call, information())
        with self.assertRaisesRegex(Rejected, "duplicate_tool_call_id"):
            client.create(history, "auto")

    def test_budget_exhausted_repair_preserves_the_actual_prior_request(self):
        client = scripted_client("search", LONG_SEARCHED, FINAL, thinking="enabled")
        record, audit = rollout(ROW, client, FixtureRetriever(), 1)
        audit["provenance_checked"] = True
        payloads = [c.args[0] for c in client._post.call_args_list]
        self.assertEqual(payloads[-1]["messages"][:-2], payloads[-2]["messages"])
        self.assertEqual(payloads[-1]["messages"][-3]["content"], SEARCH_EXHAUSTED_INSTRUCTION)
        self.assertEqual(payloads[-1]["messages"][-1]["content"], ANSWER_REPAIR_INSTRUCTION)
        self.assertEqual(len([m for m in payloads[-1]["messages"] if "reasoning_content" in m]), 2)
        self.assertEqual(validate_record(record, audit), [])

    def test_independent_questions_do_not_share_reasoning_or_evidence(self):
        client = scripted_client("search", FINAL, DIRECT_FINAL, thinking="enabled")
        rollout(ROW, client, FixtureRetriever(), 3)
        second, audit = rollout({**ROW, "id": "different"}, client, FixtureRetriever(), 3)
        audit["provenance_checked"] = False
        self.assertEqual(len(client._post.call_args_list[-1].args[0]["messages"]), 2)
        self.assertEqual(len(audit["api_calls"]), 1)
        self.assertEqual(validate_record(second, audit), [])

    def test_independent_audit_rejects_history_and_reasoning_tampering(self):
        client = scripted_client("search", LONG_SEARCHED, FINAL, thinking="enabled")
        record, audit = rollout(ROW, client, FixtureRetriever(), 1)
        audit["provenance_checked"] = True
        for mutation in ("policy", "request", "observation", "reasoning", "response", "repair", "missing"):
            changed = copy.deepcopy(audit)
            calls = changed["api_calls"]
            if mutation == "policy":
                calls[1]["continuation_policy"] = "old"
            elif mutation == "request":
                calls[1]["request_messages"] = calls[0]["request_messages"]
            elif mutation == "observation":
                calls[1]["request_messages"][3]["public_sha256"] = "0" * 64
            elif mutation == "reasoning":
                calls[1]["request_messages"][2]["reasoning"]["sha256"] = "0" * 64
            elif mutation == "response":
                calls[0]["response_message"]["public_sha256"] = "0" * 64
            elif mutation == "repair":
                calls[-1]["request_messages"].pop(-3)
            else:
                del calls[1]["request_messages"]
            self.assertTrue(validate_record(record, changed), mutation)

    def test_old_records_without_continuation_attestation_cannot_pass(self):
        client = scripted_client(DIRECT_FINAL)
        record, audit = rollout(ROW, client, FixtureRetriever(), 3)
        audit["provenance_checked"] = False
        del record["metadata"]["continuation_policy"]
        del audit["api_calls"][0]["request_messages"]
        errors = validate_record(record, audit)
        self.assertIn("metadata_continuation_policy", errors)
        self.assertIn("continuation_request_mismatch", errors)

    def test_malformed_prompt_does_not_crash_the_auditor(self):
        client = scripted_client(DIRECT_FINAL)
        record, audit = rollout(ROW, client, FixtureRetriever(), 3)
        audit["provenance_checked"] = False
        record["prompt"] = "invalid"
        self.assertIn("prompt_mismatch", validate_record(record, audit))


if __name__ == "__main__":
    unittest.main()
