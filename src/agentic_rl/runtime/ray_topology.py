from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from agentic_rl.workers.ray_actors import (
    CandidatePoolActor,
    CheckpointCommitActor,
    ExactIGTaskBuilderActor,
    MetricsActor,
    OutcomeWorkerActor,
    PromptSamplerActor,
    ray_remote_class,
)
from agentic_rl.config import runtime_section
from agentic_rl.topology import TopologyPlan

from .environment import runtime_environment
from .resource_guard import validate_runtime_resource_budget
from .verl_config import build_verl_config, effective_rollout_topology


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    hex_method = getattr(value, "hex", None)
    if callable(hex_method):
        try:
            return str(hex_method())
        except (TypeError, ValueError):
            pass
    return str(value)


class FixedBundleRayResourcePool:
    """veRL-compatible pool with configured CPU capacity and one GPU per rank."""

    def __init__(
        self,
        *,
        topology: TopologyPlan,
        cpus_per_rank: int,
        placement_strategy: str,
        name: str,
    ) -> None:
        from verl.single_controller.ray.base import RayResourcePool

        bundles = topology.ray_bundles(cpus_per_rank)
        strategy = str(placement_strategy).upper()

        class _Pool(RayResourcePool):
            def __init__(inner_self) -> None:
                super().__init__(
                    process_on_nodes=[int(topology.rl_gpus_per_node)] * int(topology.nnodes),
                    use_gpu=True,
                    name_prefix=name,
                    max_colocate_count=1,
                )

            def get_placement_groups(
                inner_self,
                strategy: str = strategy,
                name: str | None = None,
                device_name: str = "cuda",
            ) -> list[Any]:
                if inner_self.pgs is not None:
                    return inner_self.pgs
                import ray
                from ray.util.placement_group import placement_group

                resource = "GPU" if device_name == "cuda" else device_name.upper()
                placement_bundles = [
                    {"CPU": bundle["CPU"], resource: bundle["GPU"]}
                    for bundle in bundles
                ]
                pg_name = name or f"{inner_self.name_prefix}rl"
                pg = placement_group(
                    bundles=placement_bundles,
                    strategy=strategy,
                    name=pg_name,
                )
                ray.get(pg.ready())
                inner_self.pgs = [pg]
                return inner_self.pgs

        self.pool = _Pool()


class RuntimeRayTopology:
    def __init__(self, project_config: Mapping[str, Any]) -> None:
        self.project_config = project_config
        self.topology = TopologyPlan.from_config(project_config)
        self.verl_config: Any | None = None
        self.resource_pool: Any | None = None
        self.worker_group: Any | None = None
        self.agent_loop_manager: Any | None = None
        self.control_actors: dict[str, Any] = {}

    def assert_rl_gpu_isolation(self) -> None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        expected = os.environ.get(
            "AGENTIC_RL_EXPECTED_RL_CUDA_VISIBLE_DEVICES",
            self.topology.rl_cuda_visible_devices,
        )
        if not expected:
            raise RuntimeError("TopologyPlan did not provide RL CUDA devices")
        if visible != expected:
            raise RuntimeError(
                f"RL runtime must be launched with CUDA_VISIBLE_DEVICES={expected}; "
                f"got {visible!r}"
            )

    def initialize_ray(self) -> dict[str, Any]:
        self.assert_rl_gpu_isolation()
        resource_budget = validate_runtime_resource_budget(self.project_config)
        import ray

        ray_config = runtime_section(self.project_config, "ray")
        os.environ.setdefault(
            "RAY_memory_monitor_refresh_ms",
            str(ray_config["memory_monitor_refresh_ms"]),
        )
        os.environ.setdefault(
            "RAY_memory_usage_threshold",
            str(ray_config["memory_usage_threshold"]),
        )

        if self.topology.cluster_mode == "existing":
            raise RuntimeError(
                "ray.cluster_mode=existing is parsed by TopologyPlan but the "
                "current launcher only supports cluster_mode=local"
            )
        if not ray.is_initialized():
            runtime_root = Path(
                str(self.project_config["paths"]["runtime_root"])
            ).resolve()
            spill_setting = Path(str(ray_config["object_spilling_directory"]))
            if spill_setting.is_absolute():
                spill = spill_setting
            elif spill_setting.parts and spill_setting.parts[0] == "runtime":
                spill = runtime_root.joinpath(*spill_setting.parts[1:])
            else:
                spill = runtime_root / spill_setting
            spill.mkdir(parents=True, exist_ok=True)
            object_store_gb = ray_config.get("object_store_gb")
            if object_store_gb is None:
                raise RuntimeError(
                    "runtime.ray.object_store_gb must be configured before Ray startup"
                )
            object_store_bytes = int(object_store_gb) * 1024**3
            runtime_kwargs: dict[str, Any] = {
                "include_dashboard": False,
                "runtime_env": {
                    "env_vars": runtime_environment(self.project_config, "driver")
                },
            }
            runtime_kwargs.update(
                {
                    "num_cpus": int(self.project_config["hardware"]["expected_cpu_cores"]),
                    "num_gpus": int(self.topology.learner_world_size),
                    "object_store_memory": object_store_bytes,
                }
            )
            runtime_kwargs["_system_config"] = {
                "object_spilling_config": json.dumps(
                    {
                        "type": "filesystem",
                        "params": {"directory_path": str(spill)},
                    }
                )
            }
            ray.init(**runtime_kwargs)
        resources = ray.cluster_resources()
        expected_gpus = int(self.topology.learner_world_size)
        if int(resources.get("GPU", 0)) < expected_gpus:
            raise RuntimeError(f"Ray must expose at least {expected_gpus} RL GPUs: {resources}")
        expected_cpus = int(self.project_config["hardware"]["expected_cpu_cores"])
        if int(resources.get("CPU", 0)) < expected_cpus:
            raise RuntimeError(
                "Ray CPU resources are below the configured contract: "
                f"expected={expected_cpus} resources={resources}"
            )
        return {
            "cluster_resources": dict(resources),
            "available_resources": dict(ray.available_resources()),
            "runtime_resource_budget": resource_budget,
        }

    def instantiate_control_actors(self) -> dict[str, Any]:
        import ray

        data = self.project_config["data"]
        paths = self.project_config["paths"]
        rollout = self.project_config["rollout"]
        runtime_root = Path(str(paths["runtime_root"])).resolve()
        metric_root = runtime_root / "metrics"
        metric_root.mkdir(parents=True, exist_ok=True)
        sampler_class = ray_remote_class(
            PromptSamplerActor,
            num_cpus=2,
            num_gpus=0,
        )
        candidate_class = ray_remote_class(
            CandidatePoolActor,
            num_cpus=2,
            num_gpus=0,
        )
        metrics_class = ray_remote_class(
            MetricsActor,
            num_cpus=1,
            num_gpus=0,
        )
        outcome_class = ray_remote_class(
            OutcomeWorkerActor,
            num_cpus=1,
            num_gpus=0,
        )
        checkpoint_class = ray_remote_class(
            CheckpointCommitActor,
            num_cpus=1,
            num_gpus=0,
        )
        self.control_actors = {
            "prompt_sampler": sampler_class.remote(
                int(data["expected_rows"]),
                int(data["shuffle_seed"]),
                {
                    "source_path": str(paths["train_data"]),
                    "selection_seed": int(data["selection_seed"]),
                    "expected_source_rows": int(data["source_rows"]),
                    "expected_logical_rows": int(data["expected_rows"]),
                    "expected_nq_rows": int(data["expected_source_counts"]["nq"]),
                    "expected_hotpotqa_rows": int(
                        data["expected_source_counts"]["hotpotqa"]
                    ),
                    "expected_identity_sha256": str(
                        data["ordered_view_identity_sha256"]
                    ),
                },
            ),
            "candidate_pool": candidate_class.remote(
                int(rollout["group_size"]),
                int(rollout["candidate_prompts_max"]),
            ),
            "metrics": metrics_class.remote(
                {
                    "attempt": str(metric_root / "attempt_metrics.jsonl"),
                    "update": str(metric_root / "update_metrics.jsonl"),
                    "channel": str(metric_root / "channel_metrics.jsonl"),
                    "prompt": str(metric_root / "prompt_metrics.jsonl"),
                    "trajectory": str(metric_root / "trajectory_metrics.jsonl"),
                    "turn": str(metric_root / "turn_metrics.jsonl"),
                    "behavior": str(metric_root / "behavior_metrics.jsonl"),
                    "system": str(metric_root / "system_metrics.jsonl"),
                    "checkpoint": str(metric_root / "checkpoint_metrics.jsonl"),
                    "eval": str(metric_root / "eval_metrics.jsonl"),
                }
            ),
            "checkpoint": checkpoint_class.remote(
                str(runtime_root / "checkpoint_events.jsonl")
            ),
        }
        ray_config = runtime_section(self.project_config, "ray")
        self.control_actors["outcome_workers"] = [
            outcome_class.remote()
            for _ in range(int(ray_config["outcome_worker_count"]))
        ]
        maximum_extended = self.project_config["formal_schedule"].get(
            "maximum_extended_sequence_length"
        )
        maximum_position = self.project_config["formal_schedule"].get(
            "maximum_position_id_exclusive"
        )
        if maximum_extended is not None and maximum_position is not None:
            task_builder_class = ray_remote_class(
                ExactIGTaskBuilderActor,
                num_cpus=2,
                num_gpus=0,
            )
            self.control_actors["exact_ig_task_builders"] = [
                task_builder_class.remote(
                    str(paths["actor_model"]),
                    maximum_extended_sequence_length=int(maximum_extended),
                    maximum_position_id_exclusive=int(maximum_position),
                )
                for _ in range(
                    int(
                        ray_config["exact_ig_task_builder_count"]
                    )
                )
            ]
        return self.control_actors

    def instantiate_gpu_workers(self, *, require_optimizer: bool) -> dict[str, Any]:
        import ray
        from verl.single_controller.ray.base import (
            RayClassWithInitArgs,
            RayWorkerGroup,
        )
        from .capped_vllm import StrictAgentLoopManager
        from .fsdp_worker import StrictOnPolicyFSDP2Worker

        self.verl_config = build_verl_config(
            self.project_config,
            require_optimizer=require_optimizer,
        )
        pool_wrapper = FixedBundleRayResourcePool(
            topology=self.topology,
            cpus_per_rank=int(
                runtime_section(self.project_config, "ray")[
                    "rl_engine_cpus_per_gpu"
                ]
            ),
            placement_strategy=self.topology.placement_strategy,
            name="strict_agentic_rl_",
        )
        self.resource_pool = pool_wrapper.pool
        worker_class = ray.remote(StrictOnPolicyFSDP2Worker)
        worker_init = RayClassWithInitArgs(
            cls=worker_class,
            config=self.verl_config.actor_rollout_ref,
            role="actor_rollout_ref",
        )
        self.worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=worker_init,
            bin_pack=self.topology.placement_strategy in {"PACK", "STRICT_PACK"},
            name_prefix="strict_fsdp2_",
            device_name="cuda",
            worker_env=runtime_environment(self.project_config, "worker"),
        )
        identities = self.worker_group.init_model()
        expected_world_size = int(self.topology.learner_world_size)
        if len(identities) != expected_world_size:
            raise RuntimeError(
                f"FSDP2 worker group did not initialize {expected_world_size} ranks"
            )
        self.agent_loop_manager = StrictAgentLoopManager(
            config=self.verl_config,
            worker_group=self.worker_group,
            rm_wg=None,
        )
        topology = effective_rollout_topology(self.verl_config)
        expected_effective = {
            "worker_world_size": expected_world_size,
            "per_replica_world_size": 1,
            "replica_count": expected_world_size,
            "aggregate_data_parallel_size": expected_world_size,
            "tensor_parallel_size": 1,
        }
        if topology != expected_effective:
            raise RuntimeError(f"Resolved rollout topology is wrong: {topology}")
        return {
            "fsdp_workers": identities,
            "rollout": self.agent_loop_manager.topology(),
            "effective": topology,
        }

    def runtime_tables(self) -> dict[str, Any]:
        import ray
        from ray._private import state as ray_state
        from ray.util import placement_group_table

        actor_source = "gcs_internal"
        placement_group_source = "gcs_internal"
        try:
            actor_state = ray_state.actors()
            actors = [
                {"actor_id": str(actor_id), **dict(metadata)}
                for actor_id, metadata in actor_state.items()
            ]
        except BaseException as exc:
            actor_source = f"owned_handles_fallback:{type(exc).__name__}"
            actors = []
            for name, handles in self.control_actors.items():
                iterable = handles if isinstance(handles, list) else [handles]
                for index, handle in enumerate(iterable):
                    actor_id = getattr(handle, "_actor_id", None)
                    actors.append(
                        {
                            "actor_id": (
                                actor_id.hex()
                                if actor_id is not None
                                else "unavailable"
                            ),
                            "name": name,
                            "index": index,
                            "state": "OWNED_HANDLE",
                        }
                    )
        try:
            placement_groups = placement_group_table()
        except BaseException as exc:
            placement_group_source = (
                f"owned_placement_groups_fallback:{type(exc).__name__}"
            )
            placement_groups = {}
            for index, group in enumerate(
                getattr(self.resource_pool, "pgs", None) or []
            ):
                group_id = getattr(group, "id", None)
                placement_groups[str(index)] = {
                    "placement_group_id": (
                        group_id.hex()
                        if group_id is not None
                        else "unavailable"
                    ),
                    "state": "OWNED_HANDLE",
                }
        return {
            "cluster_resources": dict(ray.cluster_resources()),
            "available_resources": dict(ray.available_resources()),
            "actor_table_source": actor_source,
            "actors": _json_safe(actors),
            "placement_group_table_source": placement_group_source,
            "placement_groups": _json_safe(placement_groups),
        }

    def shutdown(self) -> None:
        import ray

        if ray.is_initialized():
            ray.shutdown()
