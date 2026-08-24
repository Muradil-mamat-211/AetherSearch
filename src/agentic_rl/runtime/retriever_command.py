"""Build the external Retriever command from resolved configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .environment import retriever_runtime_options


def build_retriever_command(
    config: Mapping[str, Any],
    *,
    log_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return the exact server argv without starting a process."""

    paths = config["paths"]
    retriever = config["retriever"]
    runtime = retriever_runtime_options(config)
    service = urlparse(str(retriever["service_url"]))
    if service.scheme not in {"http", "https"} or not service.hostname or not service.port:
        raise ValueError("retriever.service_url must include scheme, host, and port")
    resolved_log_root = Path(
        str(log_root) if log_root is not None else str(paths["runtime_root"])
    )

    command = [
        str(paths["retriever_python"]),
        str(retriever["server_source"]),
        "--bm25-index-path",
        str(retriever["bm25_index_path"]),
        "--dense-index-path",
        str(retriever["dense_index_path"]),
        "--corpus-path",
        str(retriever["corpus_path"]),
        "--retriever-name",
        str(retriever["dense_encoder_name"]),
        "--retriever-model",
        str(retriever["dense_encoder_path"]),
        "--topk",
        str(retriever["top_k"]),
        "--bm25-topn",
        str(retriever["bm25_top_n"]),
        "--dense-topn",
        str(retriever["dense_top_n"]),
        "--alpha",
        str(retriever["fusion_alpha"]),
        "--query-max-length",
        str(runtime["query_max_length"]),
        "--dense-query-batch-size",
        str(runtime["dense_query_batch_size"]),
        "--bm25-workers",
        str(runtime["bm25_workers"]),
        "--request-batch-wait-ms",
        str(runtime["request_batch_wait_ms"]),
        "--request-batch-max-queries",
        str(runtime["request_batch_max_queries"]),
        "--request-wait-timeout-seconds",
        str(runtime["request_wait_timeout_seconds"]),
        "--faiss-gpu-device",
        str(runtime["faiss_gpu_device"]),
        "--faiss-temp-memory-mb",
        str(runtime["faiss_temp_memory_mb"]),
        "--faiss-add-batch-size",
        str(runtime["faiss_add_batch_size"]),
        "--dense-device",
        str(runtime["dense_device"]),
        "--host",
        str(service.hostname),
        "--port",
        str(service.port),
        "--log-file",
        str(resolved_log_root / "logs" / "retriever.log"),
    ]
    for enabled, flag in (
        (runtime["retrieval_use_fp16"], "--retrieval-use-fp16"),
        (runtime["faiss_gpu"], "--faiss-gpu"),
        (runtime["require_faiss_gpu"], "--require-faiss-gpu"),
        (runtime["faiss_gpu_stream_flat"], "--faiss-gpu-stream-flat"),
        (runtime["faiss_gpu_use_fp16"], "--faiss-gpu-use-fp16"),
    ):
        if bool(enabled):
            command.append(flag)
    return tuple(command)
