from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from agentic_rl.config import runtime_section
from agentic_rl.topology import TopologyPlan


@dataclass(frozen=True)
class ResourcePlan:
    retriever_physical_gpu: int
    rl_physical_gpus: tuple[int, ...]
    rl_visible_gpus: tuple[int, ...]
    rl_world_size: int
    vllm_data_parallel_size: int
    vllm_tensor_parallel_size: int
    retriever_cuda_visible_devices: str
    rl_cuda_visible_devices: str
    ray_object_store_gb: int

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rl_physical_gpus"] = list(self.rl_physical_gpus)
        value["rl_visible_gpus"] = list(self.rl_visible_gpus)
        return value


def build_resource_plan(config: Mapping[str, Any]) -> ResourcePlan:
    topology = TopologyPlan.from_config(config)
    hardware = config["hardware"]
    ray = runtime_section(config, "ray")
    plan = ResourcePlan(
        retriever_physical_gpu=int(topology.retriever_physical_gpu)
        if topology.retriever_physical_gpu is not None
        else -1,
        rl_physical_gpus=topology.rl_physical_gpus,
        rl_visible_gpus=topology.rl_visible_gpus,
        rl_world_size=topology.learner_world_size,
        vllm_data_parallel_size=topology.rollout_data_parallel_size,
        vllm_tensor_parallel_size=topology.rollout_tensor_parallel_size,
        retriever_cuda_visible_devices=topology.retriever_cuda_visible_devices,
        rl_cuda_visible_devices=topology.rl_cuda_visible_devices,
        ray_object_store_gb=int(ray["object_store_gb"])
        if "object_store_gb" in ray
        else int(hardware.get("ray_object_store_gb", 0)),
    )
    return plan
