"""Future-based retrieval gateway for independent, concurrent teacher trajectories."""

from __future__ import annotations

import queue
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass
class _Request:
    query: str
    topk: int
    future: Future[Any]


class BatchedRetrieverClient:
    def __init__(self, coordinator: "BatchedRetrieverCoordinator") -> None:
        self.coordinator = coordinator
        self.trace: list[dict[str, Any]] = []

    def retrieve(self, query: str, topk: int) -> Any:
        from hybrid_retriever_v1 import docs_to_jsonable, format_information

        docs = self.coordinator.submit(query, topk).result(timeout=150)
        self.trace.append({"query": query, "documents": docs_to_jsonable(docs),
                           "information": format_information(docs)})
        return docs


class BatchedRetrieverCoordinator:
    """Only the main thread calls dispatch, so backend SQLite stays thread-local."""

    def __init__(self, backend: Any, *, max_batch_queries: int = 8, batch_wait_ms: float = 5.0) -> None:
        if not 1 <= max_batch_queries <= 8 or not 0 <= batch_wait_ms <= 20:
            raise ValueError("invalid_retriever_batch_limits")
        self.backend = backend
        self.max_batch_queries = max_batch_queries
        self.batch_wait_seconds = batch_wait_ms / 1000
        self.requests: queue.Queue[_Request] = queue.Queue()
        self.batch_count = 0
        self.query_count = 0
        self.max_observed_batch = 0

    def client(self) -> BatchedRetrieverClient:
        return BatchedRetrieverClient(self)

    def submit(self, query: str, topk: int) -> Future[Any]:
        if not isinstance(query, str) or not 1 <= len(query) <= 300 or topk != 3:
            raise ValueError("invalid_batched_retrieval_request")
        future: Future[Any] = Future()
        self.requests.put(_Request(query, topk, future))
        return future

    def dispatch(self, *, idle_wait: float = 0.01) -> int:
        try:
            first = self.requests.get(timeout=idle_wait)
        except queue.Empty:
            return 0
        batch = [first]
        deadline = time.monotonic() + self.batch_wait_seconds
        while len(batch) < self.max_batch_queries:
            try:
                batch.append(self.requests.get(timeout=max(0, deadline - time.monotonic())))
            except queue.Empty:
                break
        self.batch_count += 1
        self.query_count += len(batch)
        self.max_observed_batch = max(self.max_observed_batch, len(batch))
        try:
            if len(batch) == 1:
                results = [self.backend.retrieve(batch[0].query, 3)]
            else:
                results = self.backend.retrieve_batch([item.query for item in batch], 3)
            if len(results) != len(batch):
                raise RuntimeError("retrieval_batch_result_count_mismatch")
            for item, docs in zip(batch, results, strict=True):
                item.future.set_result(docs)
        except BaseException as exc:
            for item in batch:
                item.future.set_exception(exc if isinstance(exc, Exception) else RuntimeError("retrieval_interrupted"))
            if not isinstance(exc, Exception):
                raise
        finally:
            trace = getattr(self.backend, "trace", None)
            if isinstance(trace, list):
                trace.clear()
        return len(batch)

    def fail_pending(self) -> None:
        while True:
            try:
                item = self.requests.get_nowait()
            except queue.Empty:
                break
            item.future.set_exception(RuntimeError("retrieval_cancelled"))
