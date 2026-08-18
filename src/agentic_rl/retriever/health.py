from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrieverHealth:
    ready: bool
    faiss_gpu_enabled: bool
    index_type: str
    corpus_passages: int
    index_ntotal: int
    embedding_dimension: int
    raw: dict[str, Any]

    def assert_expected(self) -> None:
        if not self.ready:
            raise RuntimeError("Retriever health endpoint is not ready")
        if not self.faiss_gpu_enabled:
            raise RuntimeError("FAISS GPU index is required")
        if self.index_type != "GpuIndexFlatIP":
            raise RuntimeError(
                f"Expected GpuIndexFlatIP, received {self.index_type}"
            )
        if self.corpus_passages != self.index_ntotal:
            raise RuntimeError("Retriever corpus and dense index row counts differ")


def query_health(service_url: str, timeout_seconds: float = 30.0) -> RetrieverHealth:
    with urllib.request.urlopen(
        service_url.rstrip("/") + "/health",
        timeout=timeout_seconds,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    health = RetrieverHealth(
        ready=bool(payload.get("ready")),
        faiss_gpu_enabled=bool(payload.get("faiss_gpu_enabled")),
        index_type=str(payload.get("index_type", "")),
        corpus_passages=int(payload.get("corpus_passages", -1)),
        index_ntotal=int(payload.get("index_ntotal", -1)),
        embedding_dimension=int(payload.get("embedding_dimension", -1)),
        raw=dict(payload),
    )
    health.assert_expected()
    return health
