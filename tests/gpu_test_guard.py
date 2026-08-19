"""Explicit guard for tests that import GPU/backend runtime stacks."""

from __future__ import annotations

import os

import pytest


def skip_if_no_gpu() -> None:
    """Skip backend/runtime modules when the test host is intentionally no-GPU."""

    if os.environ.get("AETHERSEARCH_NO_GPU", "") == "1":
        pytest.skip("NOT RUN - NO GPU AVAILABLE", allow_module_level=True)
