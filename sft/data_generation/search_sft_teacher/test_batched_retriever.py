"""Offline batching and global-budget tests; no retriever service or API calls."""

import io
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "search_sft_hybrid_v1"))

from batched_retriever import BatchedRetrieverCoordinator
from deepseek_client import APIError, DeepSeekClient, SharedRequestBudget
from hybrid_retriever_v1 import HybridRetrieverV1
from test_deepseek_rollout import FixtureRetriever, KEY


class RecordingBackend:
    def __init__(self):
        self.calls = []
        self.fixture = FixtureRetriever()
        self.trace = []

    def retrieve(self, query, topk):
        self.calls.append([query])
        return self.fixture.retrieve(query, topk)

    def retrieve_batch(self, queries, topk):
        self.calls.append(list(queries))
        self.trace.extend({"query": query} for query in queries)
        return [self.fixture.retrieve(query, topk) for query in queries]


class BatchedRetrieverTests(unittest.TestCase):
    def test_independent_queries_share_one_backend_batch(self):
        backend = RecordingBackend()
        coordinator = BatchedRetrieverCoordinator(backend, max_batch_queries=2, batch_wait_ms=5)
        first = coordinator.submit("first", 3)
        second = coordinator.submit("second", 3)
        self.assertEqual(coordinator.dispatch(), 2)
        self.assertEqual(backend.calls, [["first", "second"]])
        self.assertEqual(backend.trace, [])
        self.assertEqual(first.result()[1].doc_id, "2")
        self.assertEqual(second.result()[1].doc_id, "2")
        self.assertEqual(coordinator.max_observed_batch, 2)

    def test_backend_failure_unblocks_every_waiter(self):
        class FailingBackend(RecordingBackend):
            def retrieve_batch(self, queries, topk):
                raise RuntimeError("backend_unavailable")

        coordinator = BatchedRetrieverCoordinator(FailingBackend(), max_batch_queries=2)
        futures = [coordinator.submit(query, 3) for query in ("first", "second")]
        coordinator.dispatch()
        for future in futures:
            with self.assertRaisesRegex(RuntimeError, "backend_unavailable"):
                future.result()

    def test_cancellation_unblocks_pending_request(self):
        coordinator = BatchedRetrieverCoordinator(RecordingBackend())
        future = coordinator.submit("first", 3)
        coordinator.fail_pending()
        with self.assertRaisesRegex(RuntimeError, "retrieval_cancelled"):
            future.result()

    def test_global_budget_is_atomic_across_workers(self):
        budget = SharedRequestBudget(3)

        def attempt(_):
            try:
                budget.reserve()
                return True
            except APIError:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(20)))
        self.assertEqual(sum(results), 3)
        self.assertEqual(budget.count, 3)

    def test_real_client_transport_charges_shared_budget_before_http(self):
        budget = SharedRequestBudget(1)
        client = DeepSeekClient("deepseek-flash", KEY, retries=0, request_budget=budget)
        calls = []

        def open_request(*_args, **_kwargs):
            calls.append(True)
            return io.BytesIO(b"{}")

        client.opener = SimpleNamespace(open=open_request)
        self.assertEqual(client._post({}), ({}, 1))
        with self.assertRaisesRegex(APIError, "api_request_budget_exhausted"):
            client._post({})
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.count, 1)

    def test_batched_rrf_matches_individual_retrieval(self):
        hybrid = HybridRetrieverV1.__new__(HybridRetrieverV1)
        hybrid.candidate_topn = 20
        hybrid.rrf_k = 60

        def sparse(query, _topn):
            return [{"doc_id": f"{query}:{i}", "title": f"{query} title {i}",
                     "text": f"{query} text {i}", "bm25_rank": i} for i in range(1, 21)]

        def dense(query, _topn):
            return [{"doc_id": f"{query}:{i}", "title": f"{query} title {i}",
                     "text": f"{query} text {i}", "dense_rank": rank}
                    for rank, i in enumerate(range(20, 0, -1), 1)]

        hybrid.bm25 = SimpleNamespace(search=sparse)
        hybrid.dense = SimpleNamespace(search=dense,
                                       search_batch=lambda queries, topn: [dense(query, topn) for query in queries])
        queries = ["alpha", "beta"]
        self.assertEqual(hybrid.retrieve_batch(queries, 3), [hybrid.retrieve(query, 3) for query in queries])


if __name__ == "__main__":
    unittest.main()
