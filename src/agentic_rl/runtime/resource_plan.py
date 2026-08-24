"""Config-owned CPU accounting for the active Ray runtime.

The runtime profile, validation guard, and actor construction all consume the
same immutable plan.  This module deliberately accepts already-resolved
hardware and Ray mappings so it remains below :mod:`agentic_rl.config` and
does not create a configuration import cycle.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class RuntimeResourcePlanError(ValueError):
    """Raised when a runtime resource declaration is incomplete or invalid."""


CONTROL_ACTOR_CPU_KEYS = (
    "prompt_sampler",
    "candidate_pool",
    "metrics",
    "outcome_worker",
    "checkpoint_commit",
    "exact_ig_task_builder",
)


def _positive_cpu(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeResourcePlanError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeResourcePlanError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise RuntimeResourcePlanError(f"{field} must be positive")
    return parsed


def _positive_count(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeResourcePlanError(f"{field} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeResourcePlanError(f"{field} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeResourcePlanError(f"{field} must be an integer")
    parsed = int(numeric)
    if parsed < 1:
        raise RuntimeResourcePlanError(f"{field} must be positive")
    return parsed


@dataclass(frozen=True)
class ControlActorResourcePlan:
    prompt_sampler_cpus: float
    candidate_pool_cpus: float
    metrics_cpus: float
    outcome_worker_cpus: float
    outcome_worker_count: int
    checkpoint_commit_cpus: float
    exact_ig_task_builder_cpus: float
    exact_ig_task_builder_count: int

    @property
    def controller_cpu_workers_compatibility(self) -> float:
        """Historical aggregate represented by the former value ``5``.

        It was the sum of the sampler, candidate-pool, and metrics actors.  It
        remains derived for resolved snapshots and is never an input to active
        resource validation.
        """

        return (
            self.prompt_sampler_cpus
            + self.candidate_pool_cpus
            + self.metrics_cpus
        )

    @property
    def project_control_actor_cpus(self) -> float:
        return (
            self.prompt_sampler_cpus
            + self.candidate_pool_cpus
            + self.metrics_cpus
            + self.outcome_worker_count * self.outcome_worker_cpus
            + self.checkpoint_commit_cpus
            + self.exact_ig_task_builder_count
            * self.exact_ig_task_builder_cpus
        )

    @property
    def total_cpus(self) -> float:
        return self.project_control_actor_cpus

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "controller_cpu_workers_compatibility": (
                    self.controller_cpu_workers_compatibility
                ),
                "project_control_actor_cpus": self.project_control_actor_cpus,
                "total_cpus": self.total_cpus,
            }
        )
        return result


@dataclass(frozen=True)
class RuntimeCpuResourcePlan:
    os_reserved_cpus: float
    retriever_pool_cpus: float
    learner_world_size: int
    rl_engine_cpus_per_gpu: float
    vllm_http_server_cpus_per_replica: float
    agent_loop_worker_count: int
    control_actors: ControlActorResourcePlan

    @property
    def learner_engine_cpus(self) -> float:
        return self.learner_world_size * self.rl_engine_cpus_per_gpu

    @property
    def vllm_http_server_cpus(self) -> float:
        return self.learner_world_size * self.vllm_http_server_cpus_per_replica

    @property
    def total_cpus(self) -> float:
        return (
            self.os_reserved_cpus
            + self.retriever_pool_cpus
            + self.learner_engine_cpus
            + self.vllm_http_server_cpus
            + self.control_actors.total_cpus
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "os_reserved_cpus": self.os_reserved_cpus,
            "retriever_pool_cpus": self.retriever_pool_cpus,
            "learner_world_size": self.learner_world_size,
            "rl_engine_cpus_per_gpu": self.rl_engine_cpus_per_gpu,
            "learner_engine_cpus": self.learner_engine_cpus,
            "vllm_http_server_cpus_per_replica": (
                self.vllm_http_server_cpus_per_replica
            ),
            "vllm_http_server_cpus": self.vllm_http_server_cpus,
            # veRL's upstream AgentLoopWorker actor does not declare an actor
            # creation CPU reservation. Its configured count is recorded but
            # is not invented as a project-owned num_cpus policy.
            "agent_loop_worker_count": self.agent_loop_worker_count,
            "control_actors": self.control_actors.as_dict(),
            "total_cpus": self.total_cpus,
        }


def build_control_actor_resource_plan(
    ray_config: Mapping[str, Any],
    formal_schedule: Mapping[str, Any] | None = None,
) -> ControlActorResourcePlan:
    allocations = ray_config.get("control_actor_cpus")
    if not isinstance(allocations, Mapping):
        raise RuntimeResourcePlanError(
            "runtime.ray.control_actor_cpus must be a mapping"
        )
    missing = [key for key in CONTROL_ACTOR_CPU_KEYS if key not in allocations]
    if missing:
        raise RuntimeResourcePlanError(
            "runtime.ray.control_actor_cpus is missing: " + ", ".join(missing)
        )
    unexpected = sorted(set(allocations) - set(CONTROL_ACTOR_CPU_KEYS))
    if unexpected:
        raise RuntimeResourcePlanError(
            "runtime.ray.control_actor_cpus has unknown actors: "
            + ", ".join(unexpected)
        )

    schedule = formal_schedule or {}
    maximum_extended = schedule.get("maximum_extended_sequence_length")
    maximum_position = schedule.get("maximum_position_id_exclusive")
    if (maximum_extended is None) != (maximum_position is None):
        raise RuntimeResourcePlanError(
            "Exact-IG task-builder bounds must either both be configured or both be absent"
        )
    configured_builder_count = _positive_count(
        ray_config.get("exact_ig_task_builder_count"),
        "runtime.ray.exact_ig_task_builder_count",
    )
    effective_builder_count = (
        configured_builder_count
        if maximum_extended is not None and maximum_position is not None
        else 0
    )

    return ControlActorResourcePlan(
        prompt_sampler_cpus=_positive_cpu(
            allocations["prompt_sampler"],
            "runtime.ray.control_actor_cpus.prompt_sampler",
        ),
        candidate_pool_cpus=_positive_cpu(
            allocations["candidate_pool"],
            "runtime.ray.control_actor_cpus.candidate_pool",
        ),
        metrics_cpus=_positive_cpu(
            allocations["metrics"],
            "runtime.ray.control_actor_cpus.metrics",
        ),
        outcome_worker_cpus=_positive_cpu(
            allocations["outcome_worker"],
            "runtime.ray.control_actor_cpus.outcome_worker",
        ),
        outcome_worker_count=_positive_count(
            ray_config.get("outcome_worker_count"),
            "runtime.ray.outcome_worker_count",
        ),
        checkpoint_commit_cpus=_positive_cpu(
            allocations["checkpoint_commit"],
            "runtime.ray.control_actor_cpus.checkpoint_commit",
        ),
        exact_ig_task_builder_cpus=_positive_cpu(
            allocations["exact_ig_task_builder"],
            "runtime.ray.control_actor_cpus.exact_ig_task_builder",
        ),
        exact_ig_task_builder_count=effective_builder_count,
    )


def build_runtime_cpu_resource_plan(
    hardware: Mapping[str, Any],
    ray_config: Mapping[str, Any],
    *,
    learner_world_size: int,
    formal_schedule: Mapping[str, Any] | None = None,
) -> RuntimeCpuResourcePlan:
    world_size = _positive_count(learner_world_size, "learner_world_size")
    engine_cpus = _positive_cpu(
        ray_config.get("rl_engine_cpus_per_gpu"),
        "runtime.ray.rl_engine_cpus_per_gpu",
    )
    server_cpus = _positive_cpu(
        ray_config.get("vllm_http_server_cpus"),
        "runtime.ray.vllm_http_server_cpus",
    )
    return RuntimeCpuResourcePlan(
        os_reserved_cpus=_positive_cpu(
            hardware.get("cpu_reserved_for_os"),
            "hardware.cpu_reserved_for_os",
        ),
        retriever_pool_cpus=_positive_cpu(
            ray_config.get("retriever_pool_cpus"),
            "runtime.ray.retriever_pool_cpus",
        ),
        learner_world_size=world_size,
        rl_engine_cpus_per_gpu=engine_cpus,
        vllm_http_server_cpus_per_replica=server_cpus,
        agent_loop_worker_count=_positive_count(
            ray_config.get("agent_loop_worker_count"),
            "runtime.ray.agent_loop_worker_count",
        ),
        control_actors=build_control_actor_resource_plan(
            ray_config,
            formal_schedule,
        ),
    )
