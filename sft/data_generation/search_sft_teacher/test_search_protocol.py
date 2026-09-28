"""Strict action bodies, extracted search receipts and teacher-facing prompts."""

import copy
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import controlled_rollout as controller
from controlled_rollout import (ACTION_GUIDANCE, DEEPSEEK_GENERATOR_VERSION, INSTRUCTIONS,
                                OPENAI_GENERATOR_VERSION, PROMPT, PROMPT_VERSION, QUERY_POLICY_VERSION, SEARCH_ACTION_SOURCE, SEARCH_ACTION_TEMPLATE, TOOL,
                                SEARCH_EXTRACTION_POLICY, TOOL_POLICY_VERSION, Rejected, extract_search_action, parse_action, rollout)
from deepseek_client import API_URL, ANSWER_REPAIR_INSTRUCTION, DeepSeekClient
from deepseek_rollout import SCHEMA, dense_command_matches
from test_deepseek_rollout import ACTION, FINAL, KEY, ROW, FixtureRetriever, fixture_record, response
from token_budget import MAX_ACTION_TOKENS, MAX_INFORMATION_TOKENS
from test_api_20_questions import protocol_statistics
from validate_teacher_rollout import validate_record


EXACT_PATTERN = r"^<think>[^<>]+</think><search>[^<>\r\n]{1,300}</search>$"
VALID_ACTION = "<think>x</think><search>query</search>"
INVALID_WHITESPACE = [
    " " + VALID_ACTION, "\t" + VALID_ACTION, "\n" + VALID_ACTION,
    VALID_ACTION + " ", VALID_ACTION + "\t", VALID_ACTION + "\n",
    VALID_ACTION.replace('</think><search>', '</think> <search>'),
    VALID_ACTION.replace('</think><search>', '</think>\n<search>'),
    VALID_ACTION.replace('</think><search>', '</think>\t<search>'),
    "\u00a0" + VALID_ACTION,
]


def raw_search(action: str, content=None):
    raw = response(True)
    message = raw["choices"][0]["message"]
    message["content"] = content
    message["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": action})
    return raw


class SearchProtocolTests(unittest.TestCase):
    def test_minimal_valid_action_matches_parser_and_schema(self):
        self.assertEqual(TOOL["parameters"]["properties"]["action"]["pattern"], EXACT_PATTERN)
        self.assertIsNotNone(re.fullmatch(EXACT_PATTERN, VALID_ACTION))
        self.assertEqual(parse_action(VALID_ACTION, "search"), ("x", "query"))

    def test_schema_stays_strict_but_outer_whitespace_is_extracted_by_adapter(self):
        for action in INVALID_WHITESPACE:
            with self.subTest(action=action):
                self.assertIsNone(re.fullmatch(EXACT_PATTERN, action))
                with self.assertRaisesRegex(Rejected, "invalid_search_action"):
                    parse_action(action, "search")
                client = DeepSeekClient("deepseek-flash", KEY)
                client._post = Mock(return_value=(raw_search(action), 1))
                if '</think><search>' not in action:
                    with self.assertRaisesRegex(Rejected, "invalid_search_action"):
                        client.create([], "auto")
                else:
                    result = client.create([], "auto")
                    self.assertEqual(json.loads(result["output"][0]["arguments"])["action"], VALID_ACTION)
                    self.assertEqual(client.calls[0]["raw_action"], action)

    def test_controller_cannot_trim_action_even_when_client_adapter_is_bypassed(self):
        for action in INVALID_WHITESPACE:
            with self.subTest(action=action):
                normalized = {"status": "completed", "output": [{"type": "function_call",
                              "name": "retrieve", "call_id": "unit-call",
                              "arguments": json.dumps({"action": action})}]}
                client = SimpleNamespace(model="unit-teacher", create=Mock(return_value=normalized))
                retriever = FixtureRetriever()
                with self.assertRaisesRegex(Rejected, "invalid_search_action"):
                    rollout(ROW, client, retriever, 5)
                self.assertEqual(retriever.calls, 0)

    def test_query_character_bounds_and_raw_whitespace_preservation(self):
        for query in ("q", "q" * 300, "query  with   spaces"):
            action = f"<think>x</think><search>{query}</search>"
            with self.subTest(length=len(query)):
                self.assertIsNotNone(re.fullmatch(EXACT_PATTERN, action))
                self.assertEqual(parse_action(action, "search"), ("x", query))
        for query in ("", "q" * 301, "a" + " " * 300 + "b", "q\nq", "q\rq", "q\r\nq"):
            action = f"<think>x</think><search>{query}</search>"
            with self.subTest(query=query):
                self.assertIsNone(re.fullmatch(EXACT_PATTERN, action))
                with self.assertRaises(Rejected):
                    parse_action(action, "search")

    def test_tool_content_null_and_empty_are_accepted(self):
        for content in (None, ""):
            client = DeepSeekClient("deepseek-flash", KEY)
            client._post = Mock(return_value=(raw_search(ACTION, content), 1))
            result = client.create([], "auto")
            self.assertEqual(json.loads(result["output"][0]["arguments"])["action"], ACTION)
            self.assertEqual(client.calls[0]["assistant_content"], content)

    def test_duplicate_or_decorative_content_is_recorded_not_trained(self):
        for content in (ACTION, "Unexpected text", " ", "\n", "\t", "\u200b"):
            with self.subTest(content=content):
                client = DeepSeekClient("deepseek-flash", KEY)
                client._post = Mock(return_value=(raw_search(ACTION, content), 1))
                result = client.create([], "auto")
                self.assertEqual(client.calls[0]["assistant_content"], content)
                self.assertEqual(result["assistant_message"]["content"], "")
                self.assertEqual(json.loads(result["output"][0]["arguments"])["action"], ACTION)

    def test_nontext_tool_content_is_rejected_before_retrieval(self):
        for content in (0, False, [], {}):
            with self.subTest(content=content):
                client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
                client._post = Mock(return_value=(raw_search(ACTION, content), 1))
                with self.assertRaisesRegex(Rejected, "invalid_model_output"):
                    rollout(ROW, client, retriever, 5)
                self.assertEqual(retriever.calls, 0)
                self.assertEqual(client._post.call_count, 1)

    def test_search_event_and_replayed_argument_are_exact_original_action(self):
        action = ACTION.replace('I need evidence', 'I  need evidence').replace('Bendigo Street', 'Bendigo   Street')
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(side_effect=[(raw_search(action), 1), (response(text=FINAL), 1)])
        record, audit = rollout(ROW, client, FixtureRetriever(), 5)
        audit["provenance_checked"] = True
        self.assertEqual(validate_record(record, audit), [])
        self.assertEqual(record["events"][0]["text"].encode(), action.encode())
        receipt = audit["api_calls"][0]
        self.assertEqual(receipt["action"], action)
        self.assertEqual(json.loads(receipt["function_arguments"]), {"action": action})
        self.assertEqual(receipt["action_source"], SEARCH_ACTION_SOURCE)
        replay = client._post.call_args_list[1].args[0]["messages"][2]
        self.assertEqual(replay["content"], "")
        self.assertEqual(json.loads(replay["tool_calls"][0]["function"]["arguments"])["action"], action)

    def test_prompts_expose_behavior_and_visible_evidence_without_token_numbers(self):
        for prompt in (INSTRUCTIONS, ACTION_GUIDANCE, ANSWER_REPAIR_INSTRUCTION):
            with self.subTest(prompt=prompt):
                self.assertIn("Keep the think summary brief and the search query concise and focused.", prompt)
                self.assertIn("Tool observations may be truncated.", prompt)
                self.assertIn("do not infer omitted text", prompt)
                self.assertNotIn("500", prompt)
                self.assertNotIn("student-model tokens", prompt)
                self.assertNotIn("student-token", prompt)
                self.assertNotIn("abstain", prompt.casefold())
        self.assertIn("Assistant content must be empty.", INSTRUCTIONS)

    def test_user_prompt_contains_only_optional_search_budget_and_question(self):
        prompt = PROMPT.format(question="Who is Barack Hussein Obama II?", max_searches=5)
        self.assertEqual(prompt, "Search is optional. Search budget: at most 5 calls.\n"
                         "Question: Who is Barack Hussein Obama II?")
        for markup in ("<answer>", "<abstain>", "<action>", "</action>"):
            self.assertNotIn(markup, PROMPT)
        for engineering_detail in ("500", "EOS reserve", "student tokenizer", "student-model tokens"):
            self.assertNotIn(engineering_detail, PROMPT)
        self.assertNotIn("A search action is", PROMPT)
        self.assertNotIn("retrieve.arguments.action", PROMPT)
        self.assertNotIn(SEARCH_ACTION_TEMPLATE, PROMPT)

    def test_system_prompt_defines_only_search_and_final_branches(self):
        self.assertNotIn("The retrieved evidence now supports the answer.", INSTRUCTIONS)
        self.assertNotIn("The retrieved evidence now supports the answer.", PROMPT)
        for instruction in ("Choose SEARCH or FINAL for each turn.",
                            "SEARCH: Emit exactly one native retrieve function call.",
                            "FINAL: Emit no tool call.",
                            "complete search action only in the JSON string field retrieve.arguments.action.",
                            "stop and wait for its tool result.",
                            "not private reasoning or the API reasoning_content.",
                            "specific unresolved factual gap that a new focused query is likely to resolve.",
                            "reliable prior knowledge was sufficient",
                            "Cite Turn N Doc M only in the final think summary, never inside answer",
                            "the controller rejects the candidate"):
            self.assertIn(instruction, INSTRUCTIONS)
        self.assertNotIn("output channel", INSTRUCTIONS.casefold())
        self.assertEqual(re.findall(r"^([A-Z]+):", INSTRUCTIONS, re.M), ["SEARCH", "FINAL"])
        self.assertIn("print argument JSON in assistant content", INSTRUCTIONS)
        self.assertNotIn("Search argument example:", INSTRUCTIONS)
        self.assertNotIn('{"action":', INSTRUCTIONS)

    def test_literal_field_contract_is_mechanical_without_description_duplication(self):
        for requirement in ("The string must begin with the literal <think> tag.",
                            "The string must end with the literal </search> tag.",
                            "</think> must be immediately followed by <search>.",
                            "before <think>, between </think> and <search>, or after </search>.",
                            "The JSON field name action is not a markup tag. Never output <action> or </action>.",
                            "Emit exactly one <think>...</think> block followed by exactly one <search>...</search> block.",
                            "The query must be a single line with 1-300 characters."):
            self.assertIn(requirement, INSTRUCTIONS)
        parameter_description = TOOL["parameters"]["properties"]["action"]["description"]
        for prompt in (INSTRUCTIONS, parameter_description):
            self.assertIn(SEARCH_ACTION_TEMPLATE, prompt)
            self.assertNotIn("Do not also emit ordinary text", prompt)
        self.assertEqual(TOOL["description"], "Search local wiki18 with BM25 + E5 FAISS FlatIP + RRF and return the top-3 passages. Not live web search.")
        self.assertEqual(parameter_description, "Exactly " + SEARCH_ACTION_TEMPLATE + ". No leading, trailing, or inter-tag whitespace.")
        self.assertIn("No leading, trailing, or inter-tag whitespace.", parameter_description)
        for description in (TOOL["description"], parameter_description):
            self.assertNotIn("assistant content", description)
            self.assertNotIn("<action>", description)
            self.assertNotIn("500", description)

    def test_literal_field_contract_is_sent_unchanged_in_actual_api_payload(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", reasoning_effort="max")
        raw = raw_search(ACTION, "")
        raw["choices"][0]["message"]["reasoning_content"] = "fixture private reasoning"
        client._post = Mock(return_value=(raw, 1))
        client.create([{"role": "user", "content": PROMPT.format(question=ROW["question"], max_searches=5)}], "auto")
        payload = client._post.call_args.args[0]
        self.assertIn("Never output <action> or </action>.", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["content"], "Search is optional. Search budget: at most 5 calls.\nQuestion: " + ROW["question"])
        self.assertEqual(payload["tools"][0]["function"]["description"], TOOL["description"])
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["properties"]["action"]["description"], "Exactly " + SEARCH_ACTION_TEMPLATE + ". No leading, trailing, or inter-tag whitespace.")
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["properties"]["action"]["pattern"], EXACT_PATTERN)

    def test_observed_suffixes_are_extracted_without_another_model_request(self):
        for suffix in ("</action>\\n", "</action>\n", "</action>\\n</invoke>\\n", "</search>", "</retrieve>\n", "</search>\n</invoke>\n"):
            for thinking in ("disabled", "enabled"):
                with self.subTest(suffix=suffix, thinking=thinking):
                    action = ACTION + suffix
                    client, retriever = DeepSeekClient("deepseek-flash", KEY, thinking=thinking), FixtureRetriever()
                    raw = raw_search(action, "")
                    if thinking == "enabled":
                        raw["choices"][0]["message"]["reasoning_content"] = "fixture private reasoning"
                    final = response(text=FINAL, reasoning="fixture private reasoning" if thinking == "enabled" else None)
                    client._post = Mock(side_effect=[(raw, 1), (final, 1)])
                    record, audit = rollout(ROW, client, retriever, 5)
                    audit["provenance_checked"] = True
                    self.assertEqual(validate_record(record, audit), [])
                    self.assertEqual(record["events"][0]["text"], ACTION)
                    self.assertEqual(audit["api_calls"][0]["raw_action"], action)
                    self.assertEqual(audit["api_calls"][0]["extraction"]["discarded_suffix"], suffix)
                    self.assertEqual(client._post.call_count, 2)
                    self.assertEqual(retriever.calls, 1)

    def test_action_markup_wrappers_are_extracted_but_intertag_markup_is_not_repaired(self):
        for action in ("<action>" + ACTION + "</action>", "<action>" + ACTION,
                       ACTION + "</action>", ACTION.replace("<search>", "<action><search>")):
            with self.subTest(action=action):
                self.assertIsNone(re.fullmatch(EXACT_PATTERN, action))
                with self.assertRaisesRegex(Rejected, "invalid_search_action"):
                    parse_action(action, "search")
                client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
                client._post = Mock(return_value=(raw_search(action), 1))
                if '</think><search>' in action:
                    result = client.create([], "auto")
                    self.assertEqual(json.loads(result["output"][0]["arguments"])["action"], ACTION)
                else:
                    with self.assertRaisesRegex(Rejected, "invalid_search_action"):
                        client.create([], "auto")
                self.assertEqual(retriever.calls, 0)

    def test_quote_wrapped_action_is_a_literal_substring(self):
        client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
        client._post = Mock(return_value=(raw_search('"' + ACTION + '"'), 1))
        result = client.create([], "auto")
        self.assertEqual(json.loads(result["output"][0]["arguments"])["action"], ACTION)
        self.assertEqual(client.calls[0]["extraction"]["span"], [1, len(ACTION) + 1])
        self.assertEqual(client._post.call_count, 1)
        self.assertEqual(retriever.calls, 0)

    def test_content_search_is_extracted_with_explicit_controller_envelope(self):
        for content in (json.dumps({"action": ACTION}), ACTION):
            with self.subTest(content=content):
                client, retriever = DeepSeekClient("deepseek-flash", KEY), FixtureRetriever()
                client._post = Mock(side_effect=[(response(text=content), 1), (response(text=FINAL), 1)])
                record, audit = rollout(ROW, client, retriever, 5)
                audit["provenance_checked"] = True
                self.assertEqual(validate_record(record, audit), [])
                self.assertEqual(record["events"][0]["text"], ACTION)
                self.assertFalse(audit["api_calls"][0]["native_tool_call"])
                self.assertEqual(audit["api_calls"][0]["raw_action_source"], "assistant.content")
                self.assertEqual(client._post.call_args_list[1].args[0]["messages"][2]["content"], "")
                self.assertEqual(client._post.call_count, 2)
                self.assertEqual(retriever.calls, 1)

    def test_extraction_never_assembles_missing_think_or_search_tags(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(return_value=(response(text="<thinking>Need evidence.</thinking><search>query</search>"), 1))
        with self.assertRaisesRegex(Rejected, "invalid_search_action"):
            rollout(ROW, client, FixtureRetriever(), 5)

    def test_diagnostic_counts_distinguish_primary_rejection_from_overlapping_defects(self):
        mixed = raw_search(ACTION + "</action>\\n", ACTION)
        malformed = raw_search(ACTION)
        malformed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "not-json"
        stats = protocol_statistics([
            {"status": "search_requested", "responses": [raw_search(ACTION)]},
            {"status": "rejected", "reason": "unauthorized_or_mixed_tool_output", "responses": [mixed]},
            {"status": "rejected", "reason": "invalid_tool_arguments", "responses": [malformed]},
        ])
        self.assertEqual(stats["raw_tool_call_responses"], 3)
        self.assertEqual(stats["mixed_tool_output_responses"], 1)
        self.assertEqual(stats["invalid_raw_action_schema"], 2)
        self.assertEqual(sum(stats["rejection_reasons"].values()), 2)
        self.assertEqual(stats["special_decline_tag_outputs"], 0)

    def test_actual_payload_uses_current_schema_and_prompt_in_both_thinking_modes(self):
        for thinking in ("disabled", "enabled"):
            client = DeepSeekClient("deepseek-flash", KEY, thinking=thinking,
                                    reasoning_effort="max" if thinking == "enabled" else None)
            raw = raw_search(ACTION)
            if thinking == "enabled":
                raw["choices"][0]["message"]["reasoning_content"] = "fixture private reasoning"
            client._post = Mock(return_value=(raw, 1))
            prompt = PROMPT.format(question=ROW["question"], max_searches=5)
            client.create([{"role": "user", "content": prompt}], "auto")
            payload = client._post.call_args.args[0]
            self.assertEqual(payload["messages"], [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}])
            self.assertEqual(payload["tools"][0]["function"]["parameters"], TOOL["parameters"])
            self.assertIs(payload["tools"][0]["function"]["strict"], True)
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertEqual(payload["tools"][0]["function"]["parameters"]["properties"]["action"]["pattern"], EXACT_PATTERN)

    def test_budget_exhaustion_prompt_has_no_decline_action_or_student_token_number(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(return_value=(response(text=FINAL), 1))
        client.create([], "none")
        payload = client._post.call_args.args[0]
        self.assertEqual(payload["tool_choice"], "none")
        prompt = payload["messages"][-1]["content"]
        self.assertIn("Search budget exhausted", prompt)
        self.assertIn("Use FINAL without a tool call", prompt)
        self.assertIn("The controller rejects unsupported answers", prompt)
        self.assertNotIn("abstain", prompt.casefold())
        self.assertNotIn("500", prompt)

    def test_serialized_http_request_matches_current_schema_and_prompts(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", reasoning_effort="max", retries=0)
        raw = raw_search(ACTION)
        raw["choices"][0]["message"]["reasoning_content"] = "fixture private reasoning"
        client.opener = SimpleNamespace(open=Mock(return_value=io.BytesIO(json.dumps(raw).encode())))
        prompt = PROMPT.format(question=ROW["question"], max_searches=5)
        client.create([{"role": "user", "content": prompt}], "auto")
        request = client.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, API_URL)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(payload["messages"], [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}])
        self.assertEqual(payload["tools"], [{"type": "function", "function": {k: v for k, v in TOOL.items() if k != "type"}}])
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("500", prompt)
        self.assertNotIn(KEY, request.data.decode())

    def test_controller_hard_budget_constants_remain_500(self):
        self.assertEqual(MAX_ACTION_TOKENS, 500)
        self.assertEqual(MAX_INFORMATION_TOKENS, 500)

    def test_audit_rejects_action_not_equal_to_selected_raw_receipt_block(self):
        for action in INVALID_WHITESPACE:
            record, audit = fixture_record()
            receipt = audit["api_calls"][0]
            receipt["action"] = action
            receipt["function_arguments"] = json.dumps({"action": action})
            errors = validate_record(record, audit)
            self.assertIn("invalid_search_action_receipt", errors)
            self.assertIn("search_action_not_equal_to_model_output", errors)

    def test_audit_reconstructs_duplicate_content_and_rejects_conflicts(self):
        for content in (ACTION, " ", "\n"):
            record, audit = fixture_record()
            audit["api_calls"][0]["assistant_content"] = content
            self.assertEqual(validate_record(record, audit), [])
        record, audit = fixture_record()
        audit["api_calls"][0]["assistant_content"] = ACTION.replace("road project</search>", "different subject</search>")
        self.assertIn("invalid_search_action_receipt", validate_record(record, audit))

    def test_audit_rejects_missing_or_altered_authoritative_arguments(self):
        for arguments in (None, '{}', '[]', 'not-json', json.dumps({"action": "different"}),
                          json.dumps({"action": ACTION, "extra": True}),
                          '{"action":"x","action":"y"}'):
            record, audit = fixture_record()
            audit["api_calls"][0]["function_arguments"] = arguments
            self.assertIn("invalid_search_action_receipt", validate_record(record, audit))

    def test_audit_requires_current_policy_and_authoritative_source(self):
        for field, value, error in (("tool_policy", "legacy_policy", "invalid_tool_policy_receipt"),
                                    ("query_policy", "legacy_policy", "invalid_query_policy_receipt"),
                                    ("endpoint", "https://example.invalid", "invalid_tool_policy_receipt"),
                                    ("strict_requested", False, "invalid_tool_policy_receipt"),
                                    ("action_source", "assistant.content", "invalid_search_action_source"),
                                    ("tool_name", "other_tool", "invalid_search_action_source")):
            record, audit = fixture_record()
            audit["api_calls"][0][field] = value
            self.assertIn(error, validate_record(record, audit))

    def test_audit_rejects_missing_query_policy(self):
        record, audit = fixture_record()
        audit["api_calls"][0].pop("query_policy")
        self.assertIn("invalid_query_policy_receipt", validate_record(record, audit))

    def test_prompt_is_neutral_about_candidate_answers_in_queries(self):
        self.assertNotIn("candidate answer", INSTRUCTIONS.casefold())
        self.assertNotIn("hypothesis", INSTRUCTIONS.casefold())
        self.assertIn("Use the question, prior knowledge, and visible evidence to target the missing fact", INSTRUCTIONS)
        self.assertIn("consider contradictory evidence", INSTRUCTIONS)
        self.assertIn("shortest complete answer to the actual question", INSTRUCTIONS)
        self.assertIn("Include all requested items", INSTRUCTIONS)
        for text in (INSTRUCTIONS, PROMPT, ANSWER_REPAIR_INSTRUCTION,
                     TOOL["description"], TOOL["parameters"]["properties"]["action"]["description"]):
            for banned in ("never add a proposed answer", "guessed answer", "golden answer"):
                self.assertNotIn(banned, text.casefold())

    def test_versions_and_record_metadata_are_current(self):
        self.assertEqual(PROMPT_VERSION, "adaptive_search_v12_faiss_flat")
        self.assertEqual(DEEPSEEK_GENERATOR_VERSION, "controlled_deepseek_teacher_v22_faiss_flat")
        self.assertEqual(TOOL_POLICY_VERSION, "hybrid_only_strict_v10_faiss_flat")
        self.assertEqual(SCHEMA, "deepseek_teacher_checkpoint_v22_faiss_flat")
        record, audit = fixture_record()
        self.assertEqual(record["metadata"]["dense_index"], "e5_Flat.index")
        for key, value in (("prompt_version", PROMPT_VERSION), ("generator_version", DEEPSEEK_GENERATOR_VERSION),
                            ("tool_policy", TOOL_POLICY_VERSION), ("search_action_source", SEARCH_ACTION_SOURCE),
                            ("search_extraction_policy", SEARCH_EXTRACTION_POLICY), ("query_policy", QUERY_POLICY_VERSION)):
            self.assertEqual(record["metadata"][key], value)
            changed = copy.deepcopy(record)
            changed["metadata"][key] = "legacy_value"
            self.assertIn("metadata_" + key, validate_record(changed, audit))

    def test_flat_index_preflight_rejects_hnsw_wrong_size_and_wrong_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            flat = Path(tmp) / "e5_Flat.index"
            hnsw = Path(tmp) / "e5_HNSW64.index"
            flat.write_bytes(b"IxFI")
            hnsw.write_bytes(b"IxFI")
            with patch.object(controller, "FAISS_FLAT_BYTES", 4), patch.object(controller, "FAISS_INDEX", flat):
                self.assertTrue(controller.faiss_flat_index_ready())
                flat.write_bytes(b"IxF2")
                self.assertFalse(controller.faiss_flat_index_ready())
                flat.write_bytes(b"IxFIx")
                self.assertFalse(controller.faiss_flat_index_ready())
            with patch.object(controller, "FAISS_FLAT_BYTES", 4), patch.object(controller, "FAISS_INDEX", hnsw):
                self.assertFalse(controller.faiss_flat_index_ready())

    def test_dense_server_command_requires_flat_index_and_gpu(self):
        base = ["python", "search_r1/search/retrieval_server.py",
                "--index_path", str(controller.FAISS_INDEX),
                "--corpus_path", str(controller.CORPUS),
                "--retriever_name", "e5",
                "--retriever_model", str(controller.E5_MODEL)]
        cwd = controller.WORKSPACE / "code/Search-R1"
        self.assertTrue(dense_command_matches(base + ["--faiss_gpu"], cwd))
        self.assertFalse(dense_command_matches(base, cwd))
        hnsw = base.copy()
        hnsw[hnsw.index("--index_path") + 1] = "/data/e5_HNSW64.index"
        self.assertFalse(dense_command_matches(hnsw + ["--faiss_gpu"], cwd))

    def test_readme_matches_the_actual_schema_and_versioned_protocol(self):
        readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
        self.assertIn("aethersearch_full_trajectory_v2_numeric_ids", readme)
        for literal in (EXACT_PATTERN, PROMPT_VERSION, DEEPSEEK_GENERATOR_VERSION, TOOL_POLICY_VERSION, SCHEMA,
                        SEARCH_ACTION_SOURCE, "strictly byte-for-byte", "without trimming or whitespace normalization",
                        "Keep the think summary brief and the search query concise and focused.", PROMPT,
                        TOOL["description"], TOOL["parameters"]["properties"]["action"]["description"],
                        "<think>brief basis</think><answer>minimal answer</answer>",
                        OPENAI_GENERATOR_VERSION, QUERY_POLICY_VERSION):
            self.assertIn(literal, readme)
        self.assertNotIn("apart from outer whitespace", readme)
        self.assertNotIn("output channel", readme.casefold())


if __name__ == "__main__":
    unittest.main()
