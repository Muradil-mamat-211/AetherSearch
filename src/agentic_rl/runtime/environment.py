"""Runtime-owned process environment resolution.

The shell supervisor and Ray adapters intentionally share this small helper so
thread/process policy is selected once by the runtime profile.  No reference
machine values are embedded in Python or shell launchers.
"""

from __future__ import annotations

from typing import Any, Mapping


class RuntimeEnvironmentError(RuntimeError):
    """Raised when an active runtime profile omits required environment data."""


def _environment_root(config: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise RuntimeEnvironmentError("runtime must be a mapping")
    environment = runtime.get("environment", {})
    if not isinstance(environment, Mapping):
        raise RuntimeEnvironmentError("runtime.environment must be a mapping")
    return environment


def runtime_environment(config: Mapping[str, Any], scope: str) -> dict[str, str]:
    """Resolve one named environment scope as string-valued assignments."""

    environment = _environment_root(config)
    values = environment.get(scope)
    if not isinstance(values, Mapping) or not values:
        raise RuntimeEnvironmentError(
            f"runtime.environment.{scope} must be a non-empty mapping"
        )
    result = {str(key): str(value) for key, value in values.items()}
    missing = sorted(key for key, value in result.items() if value == "")
    if missing:
        raise RuntimeEnvironmentError(
            "runtime.environment.%s contains empty values: %s"
            % (scope, ", ".join(missing))
        )
    return result


def runtime_process_environment(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve launcher process variables (for example vLLM spawn mode)."""

    return runtime_environment(config, "process")


def runtime_retriever_environment(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve retriever thread environment from the runtime profile."""

    return runtime_environment(config, "retriever")


RETRIEVER_RUNTIME_OPTION_KEYS = (
    "query_max_length",
    "dense_query_batch_size",
    "bm25_workers",
    "request_batch_wait_ms",
    "request_batch_max_queries",
    "request_wait_timeout_seconds",
    "client_batch_wait_ms",
    "client_max_concurrency",
    "client_max_batch_queries",
    "client_request_timeout_seconds",
    "client_network_retries",
    "health_timeout_seconds",
    "retrieval_use_fp16",
    "faiss_gpu",
    "require_faiss_gpu",
    "faiss_gpu_stream_flat",
    "faiss_gpu_device",
    "faiss_gpu_use_fp16",
    "faiss_temp_memory_mb",
    "faiss_add_batch_size",
    "dense_device",
)


def retriever_runtime_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete backend option set owned by runtime.retriever."""

    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise RuntimeEnvironmentError("runtime must be a mapping")
    values = runtime.get("retriever", {})
    if not isinstance(values, Mapping):
        raise RuntimeEnvironmentError("runtime.retriever must be a mapping")
    missing = [key for key in RETRIEVER_RUNTIME_OPTION_KEYS if key not in values]
    if missing:
        raise RuntimeEnvironmentError(
            "runtime.retriever is missing: " + ", ".join(missing)
        )
    return {key: values[key] for key in RETRIEVER_RUNTIME_OPTION_KEYS}
