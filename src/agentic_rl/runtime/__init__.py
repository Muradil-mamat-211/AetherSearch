"""Concrete veRL/Ray/vLLM/FSDP2 runtime bindings.

Heavy runtime dependencies are imported only when a runtime adapter is
requested. Configuration inspection remains usable on CPU-only login nodes.
"""

from __future__ import annotations

from typing import Any


__all__ = ["VerlAttemptRuntimeAdapter", "create_runtime_adapter"]


def create_runtime_adapter(*args: Any, **kwargs: Any) -> Any:
    from .verl_runtime_adapter import create_runtime_adapter as factory

    return factory(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "VerlAttemptRuntimeAdapter":
        from .verl_runtime_adapter import VerlAttemptRuntimeAdapter

        return VerlAttemptRuntimeAdapter
    raise AttributeError(name)
