from __future__ import annotations

from pathlib import Path
from collections import namedtuple

import pytest

from agentic_rl.runtime.resource_guard import (
    BYTES_PER_GIB,
    validate_runtime_resource_budget,
    validate_checkpoint_runtime_budget,
)


def _config(*, ram: int = 360, object_store: int = 48) -> dict:
    return {
        "hardware": {
            "expected_host_ram_gb": ram,
            "expected_cpu_cores": 48,
            "ray_object_store_gb": object_store,
            "memory_safety_reserve_gb": 64,
        }
    }


def _snapshot(*, memory_gib: int = 360, cpu: float = 48.0) -> dict:
    return {
        "memory_limit_bytes": memory_gib * BYTES_PER_GIB,
        "memory_current_bytes": 200 * BYTES_PER_GIB,
        "cpu_quota_cores": cpu,
        "memory_events": {"max": 0},
    }


def test_resource_guard_accepts_actual_360_gib_48_cpu_contract() -> None:
    result = validate_runtime_resource_budget(_config(), snapshot=_snapshot())
    assert result["status"] == "PASS"
    assert result["headroom_after_object_store_and_reserve_gib"] == 248.0


def test_resource_guard_rejects_inherited_450_gib_contract() -> None:
    with pytest.raises(RuntimeError, match="does not match the cgroup limit"):
        validate_runtime_resource_budget(
            _config(ram=450),
            snapshot=_snapshot(),
        )


def test_resource_guard_rejects_cpu_above_cgroup_quota() -> None:
    with pytest.raises(RuntimeError, match="CPU count exceeds"):
        validate_runtime_resource_budget(
            _config(),
            snapshot=_snapshot(cpu=47.5),
        )


def test_resource_guard_rejects_unsafe_object_store_reserve_sum() -> None:
    with pytest.raises(RuntimeError, match="object store plus safety reserve"):
        validate_runtime_resource_budget(
            _config(object_store=300),
            snapshot=_snapshot(),
        )


def test_formal_launchers_do_not_disable_ray_memory_monitor() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "_run_runtime_job.sh").read_text(
        encoding="utf-8"
    )
    profile = (root / "configs" / "runtime" / "verl_fsdp2_vllm_4x48_reference.yaml").read_text(
        encoding="utf-8"
    )
    assert "RAY_memory_monitor_refresh_ms=0" not in launcher
    assert "memory_monitor_refresh_ms" in launcher
    assert "memory_usage_threshold" in launcher
    assert "memory_monitor_refresh_ms: 1000" in profile
    assert "memory_usage_threshold: 0.80" in profile


def _checkpoint_snapshot(*, current_gib: int = 200) -> dict:
    return {
        "memory_limit_bytes": 360 * BYTES_PER_GIB,
        "memory_current_bytes": current_gib * BYTES_PER_GIB,
        "memory_events": {},
    }


def test_checkpoint_budget_accepts_headroom_and_reports_artifact_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "checkpoints" / "resume").mkdir(parents=True)
    (tmp_path / "checkpoints" / "models").mkdir(parents=True)
    usage = namedtuple("Usage", "total used free")(
        1200 * BYTES_PER_GIB,
        1120 * BYTES_PER_GIB,
        80 * BYTES_PER_GIB,
    )
    monkeypatch.setattr(
        "agentic_rl.runtime.resource_guard.shutil.disk_usage",
        lambda _path: usage,
    )
    result = validate_checkpoint_runtime_budget(
        tmp_path,
        snapshot=_checkpoint_snapshot(),
    )
    assert result["status"] == "PASS"
    assert result["memory_headroom_required_gib"] == 24.0
    assert result["estimated_checkpoint_write_bytes"] > 0


def test_checkpoint_budget_blocks_when_memory_headroom_is_low(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="insufficient cgroup headroom"):
        validate_checkpoint_runtime_budget(
            tmp_path,
            snapshot=_checkpoint_snapshot(current_gib=337),
        )


def test_checkpoint_budget_blocks_projected_checkpoint_peak(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="projected checkpoint peak"):
        validate_checkpoint_runtime_budget(
            tmp_path,
            snapshot=_checkpoint_snapshot(current_gib=320),
        )


def test_checkpoint_budget_blocks_when_disk_headroom_is_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = namedtuple("Usage", "total used free")(
        1200 * BYTES_PER_GIB,
        1160 * BYTES_PER_GIB,
        40 * BYTES_PER_GIB,
    )
    monkeypatch.setattr(
        "agentic_rl.runtime.resource_guard.shutil.disk_usage",
        lambda _path: usage,
    )
    with pytest.raises(RuntimeError, match="insufficient disk"):
        validate_checkpoint_runtime_budget(
            tmp_path,
            snapshot=_checkpoint_snapshot(),
        )
