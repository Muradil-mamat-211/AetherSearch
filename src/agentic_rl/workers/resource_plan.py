from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


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
    hardware = config["hardware"]
    rollout = config["rollout"]
    physical = tuple(int(value) for value in hardware["rl_physical_gpus"])
    visible = tuple(int(value) for value in hardware["rl_visible_gpus"])
    plan = ResourcePlan(
        retriever_physical_gpu=int(hardware["retriever_physical_gpu"]),
        rl_physical_gpus=physical,
        rl_visible_gpus=visible,
        rl_world_size=int(hardware["rl_world_size"]),
        vllm_data_parallel_size=int(rollout["data_parallel_size"]),
        vllm_tensor_parallel_size=int(rollout["tensor_parallel_size"]),
        retriever_cuda_visible_devices=str(hardware["retriever_physical_gpu"]),
        rl_cuda_visible_devices=",".join(str(value) for value in physical),
        ray_object_store_gb=int(config["ray"]["object_store_gb"])
        if "object_store_gb" in config["ray"]
        else int(hardware.get("ray_object_store_gb", 0)),
    )
    if plan.retriever_physical_gpu in plan.rl_physical_gpus:
        raise ValueError("Retriever GPU cannot be in the RL GPU set")
    if plan.rl_visible_gpus != tuple(range(plan.rl_world_size)):
        raise ValueError("RL visible GPU IDs must be contiguous local IDs")
    if len(plan.rl_physical_gpus) != plan.rl_world_size:
        raise ValueError("RL physical GPU count must equal world size")
    if plan.vllm_data_parallel_size != plan.rl_world_size:
        raise ValueError("vLLM DP must equal the configured RL world")
    if plan.vllm_tensor_parallel_size != 1:
        raise ValueError("vLLM TP must be 1")
    return plan
