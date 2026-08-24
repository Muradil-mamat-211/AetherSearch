import asyncio
from typing import Any

from agentic_rl.retriever.client import AsyncHybridRetrieverClient


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.request_info = None
        self.history = ()
        self.headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    async def json(self):
        return self.payload

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def post(self, _url: str, *, json: dict[str, Any]):
        self.requests.append(json)
        return _FakeResponse(
            {
                "result": [
                    [
                        {
                            "document": {
                                "id": f"doc-{index}",
                                "contents": f"result for {query}",
                            },
                            "score": 1.0,
                        }
                    ]
                    for index, query in enumerate(json["queries"])
                ]
            }
        )

    async def close(self) -> None:
        return None


def test_async_retriever_batches_concurrent_queries_and_records_metadata() -> None:
    async def run() -> None:
        client = AsyncHybridRetrieverClient(
            "http://127.0.0.1:8000",
            timeout_seconds=30.0,
            default_top_k=3,
            maximum_concurrency=8,
            batch_wait_ms=5.0,
            maximum_batch_queries=8,
            network_retries=2,
        )
        fake = _FakeSession()
        client._session = fake
        client._batch_worker = asyncio.create_task(client._batch_loop())
        results = await asyncio.gather(
            client.retrieve_one("query one", "trajectory-a", 0),
            client.retrieve_one("query two", "trajectory-b", 1),
        )
        assert len(fake.requests) == 1
        assert fake.requests[0]["queries"] == ["query one", "query two"]
        assert all(result.batch_query_count == 2 for result in results)
        assert results[0].trajectory_id == "trajectory-a"
        assert results[1].documents[0].document_id == "doc-1"
        await client.close()

    asyncio.run(run())


def test_async_retriever_rejects_empty_query() -> None:
    async def run() -> None:
        client = AsyncHybridRetrieverClient(
            "http://127.0.0.1:8000",
            timeout_seconds=30.0,
            default_top_k=3,
            maximum_concurrency=8,
            maximum_batch_queries=8,
            batch_wait_ms=5.0,
            network_retries=2,
        )
        try:
            await client.retrieve_one(" ", "trajectory", 0)
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:
            raise AssertionError("empty query was accepted")

    asyncio.run(run())
