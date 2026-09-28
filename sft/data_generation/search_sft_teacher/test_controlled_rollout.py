import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "search_sft_hybrid_v1"))

from controlled_rollout import Rejected, ResponsesClient, rollout, run


QUESTION = "Which road project was linked to the Bendigo Street housing dispute?"
ANSWER = "East West Link"


class FakeClient:
    model = "test-teacher"

    def __init__(self, query="Bendigo Street housing dispute road project", answer=ANSWER):
        self.query = query
        self.answer = answer
        self.calls = []

    def create(self, history, choice):
        self.calls.append((json.dumps(history), choice))
        if len(self.calls) == 1:
            action = f"<think>I need the road project's name.</think><search>{self.query}</search>"
            return {
                "id": "resp_1", "status": "completed",
                "output": [{"type": "function_call", "name": "retrieve", "call_id": "call_1", "arguments": json.dumps({"action": action})}],
            }
        return {
            "id": "resp_2", "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": f"<think>Doc 2 names the project.</think><answer>{self.answer}</answer>"}]}],
        }


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, topk):
        self.queries.append((query, topk))
        return [
            SimpleNamespace(doc_id="1", title="Other road", text="A different road.", source_branch="bm25", bm25_rank=1, dense_rank=None, rrf_score=0.01),
            SimpleNamespace(doc_id="2", title="Bendigo Street housing dispute", text="The East West Link road project was discussed.", source_branch="both", bm25_rank=2, dense_rank=1, rrf_score=0.03),
            SimpleNamespace(doc_id="3", title="Melbourne", text="A city.", source_branch="dense", bm25_rank=None, dense_rank=2, rrf_score=0.01),
        ]

    def close(self):
        pass


class ControlledRolloutTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "qa-1", "question": QUESTION, "golden_answers": [ANSWER], "data_source": "hotpotqa", "split": "train"}

    def test_real_action_sequence_and_gold_isolation(self):
        client, retriever = FakeClient(), FakeRetriever()
        record, audit = rollout(self.row, client, retriever, max_searches=2)
        self.assertEqual([e["role"] for e in record["events"]], ["assistant", "environment", "assistant"])
        self.assertEqual([e["train_on_tokens"] for e in record["events"]], [True, False, True])
        self.assertEqual(record["response"], "".join(e["text"] for e in record["events"]))
        self.assertEqual(record["metadata"]["semantic_review_status"], "pending")
        self.assertEqual(len(audit["retrieval_trace"][0]["documents"]), 3)
        self.assertEqual(retriever.queries, [(client.query, 3)])
        self.assertNotIn(ANSWER, client.calls[0][0])
        self.assertIn("<information>", client.calls[1][0])
        self.assertEqual([choice for _, choice in client.calls], ["auto", "auto"])

    def test_model_candidate_answer_in_search_is_allowed(self):
        retriever = FakeRetriever()
        record, _ = rollout(self.row, FakeClient(query="East West Link Bendigo"), retriever, 2)
        self.assertEqual(retriever.queries, [("East West Link Bendigo", 3)])
        self.assertEqual(record["metadata"]["search_queries"], ["East West Link Bendigo"])

    def test_unsupported_final_answer_is_rejected(self):
        with self.assertRaisesRegex(Rejected, "answer_not_equal_to_gold"):
            rollout(self.row, FakeClient(answer="Unrelated road"), FakeRetriever(), 2)

    def test_model_request_has_only_local_retrieve_tool(self):
        class CapturedResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'{"id":"r","status":"completed","output":[]}'

        captured = []

        def fake_urlopen(request, timeout):
            captured.append(json.loads(request.data))
            return CapturedResponse()

        with patch("controlled_rollout.urllib.request.urlopen", fake_urlopen):
            ResponsesClient("test-teacher", "test-key").create([{"role": "user", "content": QUESTION}], "required")
        payload = captured[0]
        self.assertEqual([tool["name"] for tool in payload["tools"]], ["retrieve"])
        self.assertFalse(payload["store"])
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["tool_choice"], "required")
        self.assertNotIn("web_search", json.dumps(payload))

    def test_checkpoint_resumes_without_repeating_or_mixing_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            questions = root / "questions.jsonl"
            questions.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            args = Namespace(model="test-teacher", questions=questions, db=root / "pilot.sqlite", max_examples=1, max_searches=2)
            fake_client = FakeClient()
            with patch("controlled_rollout.resource_status", return_value={"ready": True}), patch("controlled_rollout.ResponsesClient", return_value=fake_client), patch("hybrid_retriever_v1.HybridRetrieverV1", return_value=FakeRetriever()), patch.dict("controlled_rollout.os.environ", {"OPENAI_API_KEY": "test-key"}):
                run(args)
                run(args)
                self.assertEqual(len(fake_client.calls), 2)
                args.model = "different-teacher"
                with self.assertRaisesRegex(SystemExit, "checkpoint belongs"):
                    run(args)


if __name__ == "__main__":
    unittest.main()
