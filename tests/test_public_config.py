from __future__ import annotations

import pytest

from agentic_rl.config import ConfigError, _expand_environment


def test_public_config_expands_environment_recursively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AETHERSEARCH_TEST_ROOT", "/tmp/aethersearch")
    value = {
        "path": "${AETHERSEARCH_TEST_ROOT}/model",
        "nested": ["${AETHERSEARCH_TEST_ROOT}/data", 3],
    }
    assert _expand_environment(value) == {
        "path": "/tmp/aethersearch/model",
        "nested": ["/tmp/aethersearch/data", 3],
    }


def test_public_config_reports_missing_environment_variable() -> None:
    with pytest.raises(
        ConfigError,
        match="Missing configuration environment variables: AETHERSEARCH_MISSING",
    ):
        _expand_environment("${AETHERSEARCH_MISSING}/model")
