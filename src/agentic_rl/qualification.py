"""Explicit release/reference qualification checks.

Qualification is a claim about a particular published experiment.  It is not
part of the portable topology validator and therefore cannot make an
otherwise valid user topology look like a malformed algorithm configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from agentic_rl.config import runtime_section
from agentic_rl.topology import TopologyPlan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PROFILE = PROJECT_ROOT / "configs" / "qualification" / "official_4x48gb_v1.yaml"


class QualificationError(ValueError):
    """Raised when a config does not match a named release qualification."""


def _read_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise QualificationError(f"Qualification profile does not exist: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise QualificationError("Qualification profile root must be a mapping")
    return value


def _require(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise QualificationError(
            f"Reference qualification mismatch for {field}: "
            f"expected={expected!r} got={actual!r}"
        )


def qualification_mode(config: Mapping[str, Any]) -> str:
    value = config.get("qualification", {})
    if not isinstance(value, Mapping):
        raise QualificationError("qualification must be a mapping")
    mode = str(value.get("mode", "portable")).lower()
    if mode not in {"portable", "reference", "formal"}:
        raise QualificationError(
            "qualification.mode must be portable, reference, or formal"
        )
    return mode


def qualification_profile_path(config: Mapping[str, Any]) -> Path:
    value = config.get("qualification", {})
    if not isinstance(value, Mapping):
        raise QualificationError("qualification must be a mapping")
    raw = value.get("profile", DEFAULT_REFERENCE_PROFILE)
    return Path(str(raw)).expanduser().resolve()


def validate_reference_qualification(
    config: Mapping[str, Any],
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require the exact official 4x48GB static configuration contract."""

    profile = _read_profile(profile_path or qualification_profile_path(config))
    _require(profile.get("profile"), "official_4x48gb_v1", "profile")
    plan = TopologyPlan.from_config(config)
    expected_topology = profile.get("topology", {})
    if not isinstance(expected_topology, Mapping):
        raise QualificationError("profile.topology must be a mapping")
    _require(plan.nnodes, int(expected_topology["nnodes"]), "topology.nnodes")
    _require(
        plan.retriever_physical_gpu,
        int(expected_topology["retriever_gpu"]),
        "topology.retriever_gpu",
    )
    _require(
        list(plan.rl_physical_gpus),
        [int(value) for value in expected_topology["rl_gpus"]],
        "topology.rl_gpus",
    )
    _require(
        plan.learner_world_size,
        int(expected_topology["learner_world_size"]),
        "topology.learner_world_size",
    )
    _require(
        plan.rollout_data_parallel_size,
        int(expected_topology["rollout_dp"]),
        "topology.rollout_dp",
    )
    _require(
        plan.rollout_tensor_parallel_size,
        int(expected_topology["rollout_tp"]),
        "topology.rollout_tp",
    )

    hardware = config["hardware"]
    expected_hardware = profile.get("hardware", {})
    if not isinstance(expected_hardware, Mapping):
        raise QualificationError("profile.hardware must be a mapping")
    for field in (
        "total_physical_gpus",
        "gpu_memory_gb",
        "expected_cpu_cores",
        "expected_host_ram_gb",
    ):
        _require(
            int(hardware[field]),
            int(expected_hardware[field]),
            f"hardware.{field}",
        )
    expected_ray = profile.get("ray", {})
    if not isinstance(expected_ray, Mapping):
        raise QualificationError("profile.ray must be a mapping")
    ray = config["ray"]
    for field in (
        "placement_strategy",
        "cluster_mode",
    ):
        _require(str(ray[field]).lower() if field == "cluster_mode" else str(ray[field]).upper(),
                 str(expected_ray[field]).lower() if field == "cluster_mode" else str(expected_ray[field]).upper(),
                 f"ray.{field}")
    training = config.get("training", {})
    expected_training = profile.get("training", {})
    _require(
        int(training.get("total_successful_updates", config["formal_schedule"].get("total_successful_updates", -1))),
        int(expected_training["total_successful_updates"]),
        "training.total_successful_updates",
    )
    backend = profile.get("backend", {})
    if isinstance(backend, Mapping):
        _require(str(config["learner"]["strategy"]), str(backend["learner_strategy"]), "learner.strategy")
        _require(int(config["formal_schedule"]["learner_micro_batch_size"]), int(backend["learner_micro_batch_size"]), "formal_schedule.learner_micro_batch_size")
        rollout = runtime_section(config, "rollout")
        _require(float(rollout["gpu_memory_utilization"]), float(backend["gpu_memory_utilization"]), "rollout.gpu_memory_utilization")
    return {
        "status": "PASS",
        "profile": str(profile_path or qualification_profile_path(config)),
        "name": str(profile["profile"]),
        "topology": plan.as_dict(),
    }
