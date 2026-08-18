from __future__ import annotations

import asyncio
import json
import time
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from .protocol import RetrievalDocument, RetrievalResponse, parse_retrieval_payload


@dataclass
class HybridRetrieverClient:
    service_url: str
    timeout_seconds: float = 180.0
    default_top_k: int = 3

    def retrieve(
        self,
        queries: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        normalized = [str(query).strip() for query in queries]
        if not normalized or any(not query for query in normalized):
            raise ValueError("Retriever queries must be non-empty")
        body = json.dumps(
            {
                "queries": normalized,
                "topk": int(top_k or self.default_top_k),
                "return_scores": True,
                "mode": "hybrid",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.service_url.rstrip("/") + "/retrieve",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return parse_retrieval_payload(
            payload,
            expected_query_count=len(normalized),
        )


@dataclass(frozen=True)
class AsyncRetrievalResult:
    request_id: str
    trajectory_id: str
    turn_id: int
    query: str
    documents: tuple[RetrievalDocument, ...]
    latency_seconds: float
    batch_query_count: int


@dataclass
class _PendingRetrieval:
    query: str
    trajectory_id: str
    turn_id: int
    request_id: str
    submitted_at: float
    future: asyncio.Future[AsyncRetrievalResult]


class AsyncHybridRetrieverClient:
    """Bounded, pooled and micro-batched async client for the GPU0 service."""

    def __init__(
        self,
        service_url: str,
        *,
        timeout_seconds: float = 180.0,
        default_top_k: int = 3,
        maximum_concurrency: int = 64,
        maximum_batch_queries: int = 256,
        batch_wait_ms: float = 5.0,
        network_retries: int = 2,
    ) -> None:
        if maximum_concurrency <= 0 or maximum_batch_queries <= 0:
            raise ValueError("Retriever concurrency and batch size must be positive")
        if not 2.0 <= float(batch_wait_ms) <= 5.0:
            raise ValueError("Retriever micro-batch window must be in [2, 5] ms")
        if network_retries < 0:
            raise ValueError("network_retries must be non-negative")
        self.service_url = str(service_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.default_top_k = int(default_top_k)
        self.maximum_concurrency = int(maximum_concurrency)
        self.maximum_batch_queries = int(maximum_batch_queries)
        self.batch_wait_seconds = float(batch_wait_ms) / 1000.0
        self.network_retries = int(network_retries)
        self._queue: asyncio.Queue[_PendingRetrieval | None] = asyncio.Queue(
            maxsize=self.maximum_batch_queries * 4
        )
        self._session: Any | None = None
        self._batch_worker: asyncio.Task[None] | None = None
        self._semaphore = asyncio.Semaphore(self.maximum_concurrency)
        self._closed = False
        self._batch_count = 0
        self._query_count = 0

    async def __aenter__(self) -> "AsyncHybridRetrieverClient":
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Retriever client is already closed")
        if self._session is not None:
            return
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(
            limit=self.maximum_concurrency,
            limit_per_host=self.maximum_concurrency,
            keepalive_timeout=60.0,
            enable_cleanup_closed=True,
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            raise_for_status=False,
        )
        self._batch_worker = asyncio.create_task(
            self._batch_loop(),
            name="async-hybrid-retriever-batcher",
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._batch_worker is not None:
            await self._queue.put(None)
            await self._batch_worker
            self._batch_worker = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def health(self) -> dict[str, Any]:
        await self.start()
        assert self._session is not None
        async with self._session.get(self.service_url + "/health") as response:
            if response.status != 200:
                raise RuntimeError(f"Retriever health returned HTTP {response.status}")
            payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Retriever health payload is not an object")
        return payload

    async def retrieve_one(
        self,
        query: str,
        trajectory_id: str,
        turn_id: int,
    ) -> AsyncRetrievalResult:
        normalized = str(query).strip()
        if not normalized:
            raise ValueError("Retriever query must be non-empty")
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AsyncRetrievalResult] = loop.create_future()
        request_id = (
            f"{trajectory_id}:{int(turn_id)}:"
            f"{uuid.uuid4().hex[:12]}"
        )
        await self._queue.put(
            _PendingRetrieval(
                query=normalized,
                trajectory_id=str(trajectory_id),
                turn_id=int(turn_id),
                request_id=request_id,
                submitted_at=time.perf_counter(),
                future=future,
            )
        )
        return await future

    def stats(self) -> dict[str, int | float]:
        return {
            "worker_batches": self._batch_count,
            "queries": self._query_count,
            "queue_size": self._queue.qsize(),
            "maximum_concurrency": self.maximum_concurrency,
            "maximum_batch_queries": self.maximum_batch_queries,
            "batch_wait_ms": self.batch_wait_seconds * 1000.0,
        }

    async def _batch_loop(self) -> None:
        deferred: _PendingRetrieval | None = None
        while True:
            first = deferred if deferred is not None else await self._queue.get()
            deferred = None
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_wait_seconds
            stop_after_batch = False
            while len(batch) < self.maximum_batch_queries:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if item is None:
                    self._queue.task_done()
                    stop_after_batch = True
                    break
                batch.append(item)
            try:
                await self._execute_batch(batch)
            except BaseException as exc:
                for pending in batch:
                    if not pending.future.done():
                        pending.future.set_exception(exc)
            finally:
                for _ in batch:
                    self._queue.task_done()
            if stop_after_batch:
                return

    async def _execute_batch(self, batch: Sequence[_PendingRetrieval]) -> None:
        import aiohttp

        if self._session is None:
            raise RuntimeError("Retriever session was not initialized")
        body = {
            "queries": [pending.query for pending in batch],
            "topk": self.default_top_k,
            "return_scores": True,
            "mode": "hybrid",
        }
        last_error: BaseException | None = None
        payload: Any = None
        for attempt in range(self.network_retries + 1):
            try:
                async with self._semaphore:
                    async with self._session.post(
                        self.service_url + "/retrieve",
                        json=body,
                    ) as response:
                        if response.status >= 500:
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message=await response.text(),
                                headers=response.headers,
                            )
                        if response.status >= 400:
                            message = await response.text()
                            raise RuntimeError(
                                f"Retriever request rejected with HTTP "
                                f"{response.status}: {message}"
                            )
                        payload = await response.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt >= self.network_retries:
                    raise
                await asyncio.sleep(0.05 * (2**attempt))
        if payload is None:
            raise RuntimeError("Retriever returned no payload") from last_error
        parsed = parse_retrieval_payload(
            payload,
            expected_query_count=len(batch),
        )
        finished = time.perf_counter()
        self._batch_count += 1
        self._query_count += len(batch)
        for pending, documents in zip(
            batch,
            parsed.documents_by_query,
            strict=True,
        ):
            if not pending.future.done():
                pending.future.set_result(
                    AsyncRetrievalResult(
                        request_id=pending.request_id,
                        trajectory_id=pending.trajectory_id,
                        turn_id=pending.turn_id,
                        query=pending.query,
                        documents=documents,
                        latency_seconds=finished - pending.submitted_at,
                        batch_query_count=len(batch),
                    )
                )
