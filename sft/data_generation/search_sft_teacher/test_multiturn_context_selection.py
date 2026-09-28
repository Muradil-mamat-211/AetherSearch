"""Offline prefix isolation tests for the paid continuation diagnostic."""

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controlled_rollout import PROMPT, PROMPT_VERSION
from test_deepseek_rollout import ACTION, FINAL, KEY, ROW, FixtureRetriever, response
from test_multiturn_context_api import (DATA_FILE, audit_saved_run, evaluate_case, execute, parse_trajectory,
                                       reclassify_saved_run, select_contexts, source_observation)
from deepseek_client import DeepSeekClient
from hybrid_retriever_v1 import format_information
from token_budget import BudgetError, student_budget


def source_row(number, depth):
    info = format_information(FixtureRetriever().retrieve("fixture", 3))
    actions = [ACTION.replace("road project</search>", f"road project stage {turn}</search>") for turn in range(1, depth + 1)]
    question = ROW["question"] + f" Case {number}?"
    body = "".join(action + info for action in actions) + FINAL
    return {"id": str(number), "question": question, "search_count": depth, "trajectory_type": "multi_search",
            "full_trajectory_text": "<|im_start|>system\nHistorical system.<|im_end|>\n<|im_start|>user\nQuestion: " +
                                    question + "<|im_end|>\n<|im_start|>assistant\n" + body + "<|im_end|>"}


def selected_cases():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        source.write_text("".join(json.dumps(source_row(n, depth)) + "\n" for n, depth in enumerate((2, 3, 4), 1)))
        return select_contexts(source, count=3, seed=42)


class PrefixSelectionTests(unittest.TestCase):
    def test_balanced_distinct_questions_and_exact_cut_before_target(self):
        cases, selection = selected_cases()
        self.assertEqual(selection["unique_selected_questions"], 3)
        self.assertEqual(selection["selected_states_by_target_turn"], {2: 1, 3: 1, 4: 1})
        for case in cases:
            prior = case["target_search_turn"] - 1
            self.assertEqual(len(case["history"]), 1 + 2 * prior)
            self.assertEqual(case["history"][0]["content"], PROMPT.format(question=case["question"], max_searches=5))
            pairs, _ = parse_trajectory(source_row(int(case["source_id"]), case["source_search_count"]))
            rendered = json.dumps(case["history"], ensure_ascii=False)
            self.assertNotIn(pairs[prior]["action"], rendered)
            self.assertNotIn("<answer>", rendered)
            self.assertNotIn("Historical system", rendered)
            for turn in range(prior):
                assistant, tool = case["history"][1 + 2 * turn:3 + 2 * turn]
                self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"])["action"], pairs[turn]["action"])
                self.assertEqual(tool["tool_call_id"], assistant["tool_calls"][0]["id"])
                self.assertEqual(assistant["reasoning_content"], "")
                self.assertLessEqual(case["observations"][turn]["information_tokens"], 500)

    def test_parser_detects_bad_template_and_search_count(self):
        for mutate in (lambda row: row.update(search_count=4),
                       lambda row: row.update(question="Wrong question?"),
                       lambda row: row.update(full_trajectory_text=row["full_trajectory_text"][:-1])):
            row = source_row(1, 2)
            mutate(row)
            with self.assertRaises(ValueError):
                parse_trajectory(row)

    def test_observation_cannot_invent_lost_doc3_after_prefix_truncation(self):
        info = '<information>\nDoc 1(Title: "Long") ' + "word " * 1000 + '\nDoc 2(Title: "Two") text\nDoc 3(Title: "Three") text\n</information>'
        with self.assertRaisesRegex(BudgetError, "information_missing_top3_after_truncation"):
            source_observation(info, source_id="fixture", turn=1)

    def test_no_unique_context_replacement_or_duplicate_questions(self):
        row = source_row(1, 4)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "not_enough_unique_valid_contexts"):
                select_contexts(source, count=2)

    def test_reference_overlap_in_question_or_prior_query_does_not_filter_prefix(self):
        row = source_row(1, 2)
        row["full_trajectory_text"] = row["full_trajectory_text"].replace("road project stage 1</search>", "East West Link road project</search>")
        old_question = row["question"]
        row["question"] = old_question + " East West Link?"
        row["full_trajectory_text"] = row["full_trajectory_text"].replace(old_question, row["question"])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            source.write_text(json.dumps(row) + "\n")
            cases, _ = select_contexts(source, count=1)
        self.assertIn(cases[0]["reference_answer"], cases[0]["question"])
        self.assertIn(cases[0]["reference_answer"], cases[0]["previous_queries"][0])

    def test_old_prepared_manifest_is_rejected_before_key_or_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input_manifest.json").write_text(json.dumps({"teacher_prompt_version": "old_prompt"}))
            with patch("test_multiturn_context_api.load_key_file", side_effect=AssertionError("No key")), \
                 patch.object(DeepSeekClient, "__init__", side_effect=AssertionError("No API")), \
                 self.assertRaisesRegex(ValueError, "diagnostic_configuration_changed"):
                execute(root, 1, resume=True)

    def test_reclassification_is_offline_and_preserves_old_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, destination = Path(tmp) / "old", Path(tmp) / "new"
            root.mkdir()
            source = root / DATA_FILE
            source.write_text(json.dumps(source_row(1, 2)) + "\n")
            cases, _ = select_contexts(source, count=1)
            manifest = {"contexts": cases, "count": 1, "teacher_prompt_version": "historical_prompt",
                        "rollout_budget": student_budget().specification(),
                        "snapshot": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
            (root / "input_manifest.json").write_text(json.dumps(manifest))
            raw = response(True, reasoning="private fixture reasoning")
            raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": "<think>Verify candidate.</think><search>East West Link</search>"})
            with patch.object(DeepSeekClient, "_post", return_value=(raw, 1)):
                result = evaluate_case(cases[0], KEY)
            result.update(status="rejected", reason="historical_reference_overlap")
            path = root / "results.jsonl"
            path.write_text(json.dumps(result) + "\n")
            originals = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch("test_multiturn_context_api.load_key_file", side_effect=AssertionError("No key")), \
                 patch.object(DeepSeekClient, "__init__", side_effect=AssertionError("No API")), \
                 redirect_stdout(io.StringIO()):
                report = reclassify_saved_run(root, destination)
                with self.assertRaises(FileExistsError):
                    reclassify_saved_run(root, destination)
            self.assertEqual(report["recovered_count"], 1)
            self.assertEqual(report["after_statuses"], {"search_requested": 1})
            self.assertEqual(report["new_api_requests"], 0)
            self.assertFalse(report["fresh_prompt_test"])
            self.assertEqual(report["original_prompt_version"], "historical_prompt")
            self.assertEqual(report["current_prompt_version"], PROMPT_VERSION)
            self.assertEqual(originals, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_offline_audit_reconstructs_json_histogram_keys_and_actual_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / DATA_FILE
            source.write_text("".join(json.dumps(source_row(n, depth)) + "\n" for n, depth in enumerate((2, 3, 4), 1)))
            cases, selection = select_contexts(source, count=3, seed=42)
            manifest = {"snapshot": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                        "contexts": cases, "selection": selection, "count": 3, "seed": 42}
            (root / "input_manifest.json").write_text(json.dumps(manifest))
            results = []
            for case in cases:
                raw = response(search=True, reasoning="private fixture reasoning")
                with patch.object(DeepSeekClient, "_post", return_value=(raw, 1)):
                    results.append(evaluate_case(case, KEY))
            (root / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results))
            with patch("deepseek_key.load_key_file", side_effect=AssertionError("No credential in audit")), \
                 patch.object(DeepSeekClient, "__init__", side_effect=AssertionError("No API client in audit")), \
                 redirect_stdout(io.StringIO()):
                report = audit_saved_run(root)
            self.assertEqual(report["audit"], "PASS")
            self.assertEqual(report["new_api_requests"], 0)
            self.assertEqual(report["completed_contexts"], 3)


class ContinuationTests(unittest.TestCase):
    def test_actual_client_search_extraction_is_reused_without_retrieval(self):
        case = selected_cases()[0][0]
        raw = response(search=True, reasoning="private fixture reasoning")
        next_action = ACTION.replace("road project</search>", "new factual relationship</search>")
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": next_action + "</action>"})
        with patch.object(DeepSeekClient, "_post", return_value=(raw, 1)), \
             patch("deepseek_rollout.VerifiedHybridRetriever", side_effect=AssertionError("No new RAG")):
            result = evaluate_case(case, KEY)
        self.assertEqual(result["status"], "search_requested")
        self.assertEqual(result["action"], next_action)
        self.assertFalse(result["retriever_executed"])
        self.assertNotIn("private fixture reasoning", json.dumps(result))
        self.assertEqual(result["requests"][0]["thinking"], {"type": "enabled"})
        self.assertEqual(result["requests"][0]["reasoning_effort"], "max")
        self.assertEqual(result["requests"][0]["max_tokens"], 16384)

    def test_final_answer_must_have_visible_cited_support(self):
        case = selected_cases()[0][0]
        with patch.object(DeepSeekClient, "_post", return_value=(response(text=FINAL, reasoning="private"), 1)):
            result = evaluate_case(case, KEY)
        self.assertEqual(result["status"], "final_answer")
        self.assertTrue(result["normalized_reference_match"])
        self.assertTrue(result["answer_in_cited_visible_evidence"])
        wrong = FINAL.replace("East West Link</answer>", "Unsupported role</answer>")
        with patch.object(DeepSeekClient, "_post", return_value=(response(text=wrong, reasoning="private"), 1)):
            result = evaluate_case(case, KEY)
        self.assertEqual(result["reason"], "answer_not_in_cited_evidence")

    def test_candidate_query_is_allowed_but_repetition_still_rejects(self):
        case = selected_cases()[0][0]
        for query, reason in ((case["previous_queries"][0], "duplicate_query"),
                              (case["reference_answer"], None)):
            raw = response(search=True, reasoning="private")
            raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"action": f"<think>Need evidence.</think><search>{query}</search>"})
            with patch.object(DeepSeekClient, "_post", return_value=(raw, 1)):
                result = evaluate_case(copy.deepcopy(case), KEY)
            if reason is None:
                self.assertEqual(result["status"], "search_requested")
                self.assertEqual(result["query"], case["reference_answer"])
            else:
                self.assertEqual(result["reason"], reason)


if __name__ == "__main__":
    unittest.main()
