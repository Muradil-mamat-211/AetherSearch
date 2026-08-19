"""Hardware topology planning and invariant validation.

The algorithm configuration describes *what* is trained.  This module is the
single source of truth for how the requested hardware is mapped to runtime
roles.  Reference-machine qualification is intentionally kept outside this
module; a portable configuration should be judged by invariants, not by the
shape of the machine used for the original experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class TopologyError(ValueError):
    """Raised when a hardware topology cannot be planned safely."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TopologyError(f"{field} must be a mapping")
    return value


def _positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TopologyError(f"{field} must be an integer") from exc
    if result < 1:
        raise TopologyError(f"{field} must be positive")
    return result


def _gpu_id(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TopologyError(f"{field} must be a GPU id") from exc
    if result < 0:
        raise TopologyError(f"{field} must be non-negative")
    return result


def _as_int_tuple(values: Any, field: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TopologyError(f"{field} must be a list of GPU ids")
    if not values:
        raise TopologyError(f"{field} must not be empty")
    return tuple(_gpu_id(value, f"{field}[{index}]") for index, value in enumerate(values))


@dataclass(frozen=True)
class TopologyPlan:
    """Resolved physical/logical topology consumed by all runtime adapters."""

    cluster_mode: str
    nnodes: int
    total_physical_gpus: int
    retriever_physical_gpu: int | None
    eval_physical_gpu: int | None
    rl_physical_gpus_by_node: tuple[tuple[int, ...], ...]
    rl_visible_gpus_by_node: tuple[tuple[int, ...], ...]
    learner_world_size: int
    rollout_data_parallel_size: int
    rollout_tensor_parallel_size: int
    placement_strategy: str
    ray_address: str

    @property
    def rl_physical_gpus(self) -> tuple[int, ...]:
        return tuple(gpu for node in self.rl_physical_gpus_by_node for gpu in node)

    @property
    def rl_visible_gpus(self) -> tuple[int, ...]:
        return tuple(gpu for node in self.rl_visible_gpus_by_node for gpu in node)

    @property
    def rl_gpus_per_node(self) -> int:
        return len(self.rl_physical_gpus_by_node[0])

    @property
    def retriever_cuda_visible_devices(self) -> str:
        if self.retriever_physical_gpu is None:
            return ""
        return str(self.retriever_physical_gpu)

    @property
    def eval_cuda_visible_devices(self) -> str:
        if self.eval_physical_gpu is None:
            return ""
        return str(self.eval_physical_gpu)

    @property
    def rl_cuda_visible_devices(self) -> str:
        return ",".join(str(gpu) for gpu in self.rl_physical_gpus_by_node[0])

    def ray_bundles(self, cpus_per_rank: int) -> list[dict[str, float]]:
        """Build one GPU worker bundle per FSDP rank."""

        cpu = _positive_int(cpus_per_rank, "ray.rl_engine_cpus_per_gpu")
        return [
            {"CPU": float(cpu), "GPU": 1.0}
            for _ in range(self.learner_world_size)
        ]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rl_physical_gpus_by_node"] = [
            list(node) for node in self.rl_physical_gpus_by_node
        ]
        result["rl_visible_gpus_by_node"] = [
            list(node) for node in self.rl_visible_gpus_by_node
        ]
        result["rl_physical_gpus"] = list(self.rl_physical_gpus)
        result["rl_visible_gpus"] = list(self.rl_visible_gpus)
        result["rl_gpus_per_node"] = self.rl_gpus_per_node
        result["retriever_cuda_visible_devices"] = self.retriever_cuda_visible_devices
        result["eval_cuda_visible_devices"] = self.eval_cuda_visible_devices
        result["rl_cuda_visible_devices"] = self.rl_cuda_visible_devices
        return result

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TopologyPlan":
        hardware = _mapping(config.get("hardware"), "hardware")
        ray = _mapping(config.get("ray", {}), "ray")
        rollout = _mapping(config.get("rollout"), "rollout")
        raw_topology = config.get("topology")

        if raw_topology is not None:
            topology = _mapping(raw_topology, "topology")
            roles = _mapping(topology.get("roles"), "topology.roles")
            retriever_role = _mapping(
                roles.get("retriever", {}), "topology.roles.retriever"
            )
            retriever = retriever_role.get(
                "physical_gpu", retriever_role.get("gpu")
            )
            retriever_gpu = None if retriever is None else _gpu_id(
                retriever, "topology.roles.retriever.physical_gpu"
            )

            eval_role = _mapping(roles.get("eval", {}), "topology.roles.eval")
            colocate_with = eval_role.get("colocate_with")
            if colocate_with is not None:
                if colocate_with != "retriever":
                    raise TopologyError(
                        "topology.roles.eval.colocate_with must name retriever"
                    )
                eval_gpu = retriever_gpu
            else:
                raw_eval = eval_role.get("physical_gpu", eval_role.get("gpu"))
                eval_gpu = None if raw_eval is None else _gpu_id(
                    raw_eval, "topology.roles.eval.physical_gpu"
                )

            rl_role = _mapping(roles.get("rl"), "topology.roles.rl")
            raw_by_node = rl_role.get("physical_gpus_by_node")
            if raw_by_node is None:
                raw_gpus = rl_role.get("physical_gpus", rl_role.get("gpus"))
                raw_by_node = [raw_gpus]
            if not isinstance(raw_by_node, (list, tuple)) or not raw_by_node:
                raise TopologyError(
                    "topology.roles.rl.physical_gpus_by_node must be non-empty"
                )
            by_node = tuple(
                _as_int_tuple(node, f"topology.roles.rl.physical_gpus_by_node[{i}]")
                for i, node in enumerate(raw_by_node)
            )
            nnodes = _positive_int(
                topology.get("nnodes", len(by_node)), "topology.nnodes"
            )
            if nnodes != len(by_node):
                raise TopologyError(
                    "topology.nnodes must equal the number of RL GPU nodes"
                )
            learner_topology = _mapping(topology.get("learner", {}), "topology.learner")
            explicit_world = learner_topology.get("world_size")
            world_size = sum(len(node) for node in by_node)
            if explicit_world is not None and int(explicit_world) != world_size:
                raise TopologyError(
                    "topology.learner.world_size does not match derived RL GPU count"
                )
            rollout_topology = _mapping(
                topology.get("rollout", {}), "topology.rollout"
            )
            rollout_dp = _positive_int(
                rollout_topology.get(
                    "data_parallel_size", rollout.get("data_parallel_size", world_size)
                ),
                "topology.rollout.data_parallel_size",
            )
            rollout_tp = _positive_int(
                rollout_topology.get(
                    "tensor_parallel_size", rollout.get("tensor_parallel_size", 1)
                ),
                "topology.rollout.tensor_parallel_size",
            )
            runtime_ray = _mapping(topology.get("ray", {}), "topology.ray")
            placement = runtime_ray.get(
                "placement_strategy", ray.get("placement_strategy", "STRICT_PACK")
            )
            cluster_mode = runtime_ray.get(
                "cluster_mode",
                topology.get("cluster_mode", ray.get("cluster_mode", "local")),
            )
            address = runtime_ray.get(
                "address", topology.get("address", ray.get("address", "auto"))
            )
        else:
            # Legacy recipes remain readable during the migration.  All fields
            # below are immediately normalized into the same plan.
            raw_physical = hardware.get("rl_physical_gpus")
            if raw_physical is None:
                raise TopologyError(
                    "Missing topology.roles.rl.physical_gpus_by_node or "
                    "hardware.rl_physical_gpus"
                )
            by_node = (_as_int_tuple(raw_physical, "hardware.rl_physical_gpus"),)
            nnodes = 1
            retriever_raw = hardware.get("retriever_physical_gpu")
            retriever_gpu = (
                None if retriever_raw is None else _gpu_id(
                    retriever_raw, "hardware.retriever_physical_gpu"
                )
            )
            eval_gpu = retriever_gpu
            world_size = sum(len(node) for node in by_node)
            legacy_world = hardware.get("rl_world_size")
            if legacy_world is not None and int(legacy_world) != world_size:
                raise TopologyError(
                    "hardware.rl_world_size does not match RL physical GPU count"
                )
            rollout_dp = _positive_int(
                rollout.get("data_parallel_size", world_size),
                "rollout.data_parallel_size",
            )
            rollout_tp = _positive_int(
                rollout.get("tensor_parallel_size", 1),
                "rollout.tensor_parallel_size",
            )
            placement = ray.get("placement_strategy", "STRICT_PACK")
            cluster_mode = ray.get("cluster_mode", "local")
            address = ray.get("address", "auto")

        total_raw = hardware.get("total_physical_gpus")
        largest_gpu = max(
            (*[gpu for node in by_node for gpu in node],)
            + (() if retriever_gpu is None else (retriever_gpu,))
            + (() if eval_gpu is None else (eval_gpu,)),
            default=-1,
        )
        total_gpus = (
            _positive_int(total_raw, "hardware.total_physical_gpus")
            if total_raw is not None
            else largest_gpu + 1
        )
        if largest_gpu >= total_gpus:
            raise TopologyError(
                "Topology references a GPU outside hardware.total_physical_gpus: "
                f"largest_id={largest_gpu} total={total_gpus}"
            )
        if len({gpu for node in by_node for gpu in node}) != world_size:
            raise TopologyError("RL physical GPU IDs must be unique")
        if retriever_gpu is not None and retriever_gpu in {
            gpu for node in by_node for gpu in node
        }:
            raise TopologyError("Retriever GPU cannot overlap an RL GPU")
        if eval_gpu is not None and eval_gpu in {
            gpu for node in by_node for gpu in node
        }:
            raise TopologyError("Eval GPU cannot overlap an RL GPU")
        if len({len(node) for node in by_node}) != 1:
            raise TopologyError(
                "RL GPU counts must be uniform across nodes for rank-per-GPU FSDP"
            )
        if world_size < 1:
            raise TopologyError("At least one RL GPU is required")
        rollout_parallel_size = rollout_dp * rollout_tp
        if world_size % rollout_parallel_size != 0:
            raise TopologyError(
                "Rollout tensor/data parallel sizes must divide the derived "
                f"learner world size: tp={rollout_tp} dp={rollout_dp} "
                f"world_size={world_size}"
            )

        cluster_mode = str(cluster_mode).lower()
        if cluster_mode not in {"local", "existing"}:
            raise TopologyError(
                "ray.cluster_mode must be 'local' or 'existing'"
            )
        placement = str(placement).upper()
        if placement not in {"PACK", "STRICT_PACK", "SPREAD", "STRICT_SPREAD"}:
            raise TopologyError(
                "ray.placement_strategy must be one of PACK, STRICT_PACK, "
                "SPREAD, STRICT_SPREAD"
            )
        address = str(address)
        visible_by_node = tuple(
            tuple(range(len(node))) for node in by_node
        )
        plan = cls(
            cluster_mode=cluster_mode,
            nnodes=nnodes,
            total_physical_gpus=total_gpus,
            retriever_physical_gpu=retriever_gpu,
            eval_physical_gpu=eval_gpu,
            rl_physical_gpus_by_node=by_node,
            rl_visible_gpus_by_node=visible_by_node,
            learner_world_size=world_size,
            rollout_data_parallel_size=rollout_dp,
            rollout_tensor_parallel_size=rollout_tp,
            placement_strategy=placement,
            ray_address=address,
        )
        if raw_topology is not None:
            compatibility = _mapping(
                _mapping(raw_topology, "topology").get("compatibility", {}),
                "topology.compatibility",
            )
            if bool(compatibility.get("validate_legacy_fields", False)):
                legacy_checks = (
                    (hardware.get("retriever_physical_gpu"), plan.retriever_physical_gpu, "hardware.retriever_physical_gpu"),
                    (hardware.get("rl_physical_gpus"), list(plan.rl_physical_gpus), "hardware.rl_physical_gpus"),
                    (hardware.get("rl_visible_gpus"), list(plan.rl_visible_gpus), "hardware.rl_visible_gpus"),
                    (hardware.get("rl_world_size"), plan.learner_world_size, "hardware.rl_world_size"),
                    (hardware.get("vllm_data_parallel_size"), plan.rollout_data_parallel_size, "hardware.vllm_data_parallel_size"),
                    (rollout.get("data_parallel_size"), plan.rollout_data_parallel_size, "rollout.data_parallel_size"),
                    (rollout.get("tensor_parallel_size"), plan.rollout_tensor_parallel_size, "rollout.tensor_parallel_size"),
                    (config.get("learner", {}).get("world_size"), plan.learner_world_size, "learner.world_size"),
                )
                for actual, expected, field in legacy_checks:
                    if actual is not None and actual != expected:
                        raise TopologyError(
                            f"Deprecated {field} disagrees with TopologyPlan: "
                            f"expected={expected!r} got={actual!r}"
                        )
        return plan


def materialize_topology(config: dict[str, Any]) -> TopologyPlan:
    """Derive compatibility fields in-place and return the resolved plan."""

    plan = TopologyPlan.from_config(config)
    hardware = config.setdefault("hardware", {})
    rollout = config.setdefault("rollout", {})
    learner = config.setdefault("learner", {})
    ray = config.setdefault("ray", {})
    hardware["retriever_physical_gpu"] = plan.retriever_physical_gpu
    hardware["rl_physical_gpus"] = list(plan.rl_physical_gpus)
    hardware["rl_visible_gpus"] = list(plan.rl_visible_gpus)
    hardware["rl_world_size"] = plan.learner_world_size
    hardware["vllm_data_parallel_size"] = plan.rollout_data_parallel_size
    rollout["data_parallel_size"] = plan.rollout_data_parallel_size
    rollout["tensor_parallel_size"] = plan.rollout_tensor_parallel_size
    learner["world_size"] = plan.learner_world_size
    ray["placement_strategy"] = plan.placement_strategy
    ray["cluster_mode"] = plan.cluster_mode
    ray["address"] = plan.ray_address
    return plan
