"""Deterministic diagnostic batches with disjoint historical questions."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controlled_rollout import normalize
from test_api_20_questions import question_history, select_questions


class QuestionSelectionTests(unittest.TestCase):
    def source(self, folder, *, nq=70, triviaqa=70, web_questions=9):
        rows = []
        for name, count in (("nq", nq), ("triviaqa", triviaqa), ("web_questions", web_questions)):
            rows.extend({"id": f"{name}-{n}", "question": f"Who discovered specimen {name}-{n}?",
                         "golden_answers": [f"Researcher {name}-{n}"], "data_source": name,
                         "split": "train", "context": "Must not be sent to the teacher."}
                        for n in range(count))
        path = Path(folder) / "qa.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_default_keeps_twenty_question_regression_and_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source(folder)
            first = select_questions(source)
            self.assertEqual(first, select_questions(source, num_questions=20, seed=42))
            self.assertEqual(first[0]["id"], "obama-regression")
            self.assertEqual(Counter(row["data_source"] for row in first),
                             {"manual_regression": 1, "nq": 7, "triviaqa": 6, "web_questions": 6})

    def test_additional_hundred_are_unique_disjoint_and_balanced_with_sparse_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source(folder)
            excluded = {normalize("Who discovered specimen nq-0?"), normalize("Who discovered specimen triviaqa-0?")}
            batch = select_questions(source, num_questions=100, seed=44, excluded=excluded)
            keys = {normalize(row["question"]) for row in batch}
            self.assertEqual(len(batch), 100)
            self.assertEqual(len(keys), 100)
            self.assertFalse(keys & excluded)
            self.assertEqual(Counter(row["data_source"] for row in batch), {"nq": 46, "triviaqa": 45, "web_questions": 9})
            self.assertEqual(batch, select_questions(source, num_questions=100, seed=44, excluded=excluded))
            self.assertNotEqual(batch, select_questions(source, num_questions=100, seed=45, excluded=excluded))
            for row in batch:
                self.assertEqual(set(row), {"id", "question", "golden_answers", "data_source", "split"})

    def test_insufficient_questions_and_invalid_counts_fail_before_api(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source(folder, nq=2, triviaqa=2, web_questions=2)
            for count in (0, -1, 1001):
                with self.assertRaisesRegex(ValueError, "num_questions"):
                    select_questions(source, num_questions=count)
            with self.assertRaisesRegex(ValueError, "not_enough_unseen_questions"):
                select_questions(source, num_questions=100, excluded=set())

    def test_history_includes_batches_and_single_question_tests_not_enum_probes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            batch = root / "max_thinking_20_fixture"
            batch.mkdir()
            (batch / "input_manifest.json").write_text(json.dumps({"questions": [{"question": "Who discovered X?"}]}))
            single = root / "exact_search_live_fixture"
            single.mkdir()
            (single / "summary.json").write_text(json.dumps({"question": "What is Y?"}))
            probe = root / "strict_enum_probe_fixture"
            probe.mkdir()
            (probe / "summary.json").write_text(json.dumps({"question": "Ignore this probe"}))
            keys, files = question_history(root)
            self.assertEqual(keys, {normalize("Who discovered X?"), normalize("What is Y?")})
            self.assertEqual(len(files), 2)

    def test_malformed_history_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "max_thinking_20_fixture"
            batch.mkdir()
            (batch / "input_manifest.json").write_text("invalid json")
            with self.assertRaises(ValueError):
                question_history(Path(folder))


if __name__ == "__main__":
    unittest.main()
