from __future__ import annotations

from pathlib import Path
from collections import namedtuple

import pytest

from agentic_rl.config import ConfigError, validate_resources
from agentic_rl.runtime.resource_guard import (
    BYTES_PER_GIB,
    validate_runtime_resource_budget,
    validate_checkpoint_runtime_budget,
)
from agentic_rl.runtime.resource_plan import build_runtime_cpu_resource_plan
from agentic_rl.topology import TopologyPlan


def _config(*, ram: int = 360, object_store: int = 48) -> dict:
    return {
        "hardware": {
            "expected_host_ram_gb": ram,
            "expected_cpu_cores": 48,
            "total_physical_gpus": 4,
            "gpu_memory_gb": 48,
            "cpu_reserved_for_os": 4,
        },
        "topology": {
            "cluster_mode": "local",
            "nnodes": 1,
            "roles": {
                "retriever": {"physical_gpu": 0},
                "eval": {"colocate_with": "retriever"},
                "rl": {"physical_gpus_by_node": [[1, 2, 3]]},
            },
            "ray": {"placement_strategy": "STRICT_PACK"},
            "rollout": {"data_parallel_size": 3, "tensor_parallel_size": 1},
        },
        "rollout": {"data_parallel_size": 3, "tensor_parallel_size": 1},
        "runtime": {
            "ray": {
                "object_store_gb": object_store,
                "memory_safety_reserve_gb": 64,
                "retriever_pool_cpus": 8,
                "rl_engine_cpus_per_gpu": 3,
                "vllm_http_server_cpus": 1,
                "control_actor_cpus": {
                    "prompt_sampler": 2,
                    "candidate_pool": 2,
                    "metrics": 1,
                    "outcome_worker": 1,
                    "checkpoint_commit": 1,
                    "exact_ig_task_builder": 2,
                },
                "outcome_worker_count": 4,
                "exact_ig_task_builder_count": 2,
                "agent_loop_worker_count": 12,
                "memory_monitor_refresh_ms": 1000,
                "memory_usage_threshold": 0.8,
                "object_spilling_directory": "runtime/ray_spill",
            }
        },
        "formal_schedule": {
            "maximum_extended_sequence_length": 16384,
            "maximum_position_id_exclusive": 32768,
        },
    }


def _snapshot(*, memory_gib: int = 360, cpu: float = 48.0) -> dict:
    return {
        "memory_limit_bytes": memory_gib * BYTES_PER_GIB,
        "memory_current_bytes": 200 * BYTES_PER_GIB,
        "cpu_quota_cores": cpu,
        "gpu_count": 4,
        "gpu_memory_gib": 48.0,
        "memory_events": {"max": 0},
    }


def test_resource_guard_accepts_actual_360_gib_48_cpu_contract() -> None:
    result = validate_runtime_resource_budget(_config(), snapshot=_snapshot())
    assert result["status"] == "PASS"
    assert result["headroom_after_object_store_and_reserve_gib"] == 248.0
    assert result["configured_runtime_required_cpu_cores"] == 38.0


def test_reference_actor_cpu_plan_and_exact_capacity_boundary() -> None:
    config = _config()
    ray = config["runtime"]["ray"]
    plan = build_runtime_cpu_resource_plan(
        config["hardware"],
        ray,
        learner_world_size=3,
        formal_schedule=config["formal_schedule"],
    )
    actors = plan.control_actors
    assert actors.prompt_sampler_cpus == 2
    assert actors.candidate_pool_cpus == 2
    assert actors.metrics_cpus == 1
    assert actors.checkpoint_commit_cpus == 1
    assert actors.outcome_worker_count == 4
    assert actors.exact_ig_task_builder_count == 2
    assert plan.agent_loop_worker_count == 12
    assert actors.controller_cpu_workers_compatibility == 5
    assert actors.total_cpus == 14
    assert plan.vllm_http_server_cpus == 3
    assert plan.total_cpus == 38

    config["hardware"]["expected_cpu_cores"] = 38
    topology = TopologyPlan.from_config(config)
    validate_resources(config, topology)
    config["hardware"]["expected_cpu_cores"] = 37
    with pytest.raises(ConfigError, match="CPU budget"):
        validate_resources(config, topology)


def test_actor_count_and_cpu_mutations_scale_the_same_plan() -> None:
    config = _config()
    ray = config["runtime"]["ray"]

    def required() -> float:
        return build_runtime_cpu_resource_plan(
            config["hardware"],
            ray,
            learner_world_size=3,
            formal_schedule=config["formal_schedule"],
        ).total_cpus

    baseline = required()
    ray["outcome_worker_count"] += 1
    assert required() == baseline + 1
    ray["outcome_worker_count"] -= 1
    ray["exact_ig_task_builder_count"] += 1
    assert required() == baseline + 2
    ray["exact_ig_task_builder_count"] -= 1
    ray["control_actor_cpus"]["outcome_worker"] = 2
    assert required() == baseline + 4


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
