#!/usr/bin/env python3
"""Read-only validation for the 3-rank/48-CPU resource profile."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rl.config import load_config
from agentic_rl.runtime.verl_config import build_verl_config, effective_rollout_topology
from agentic_rl.workers.resource_plan import build_resource_plan


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs" / "formal_resume_u20_3rank.yaml"
PROFILE = ROOT / "configs" / "formal_resume_u20_3rank_48cpu.yaml"


def _algorithm_snapshot(config: dict) -> dict:
    rollout = config["rollout"]
    schedule = config["formal_schedule"]
    return {
        "data": config["data"],
        "selection": config["selection"],
        "advantage": config["advantage"],
        "policy": config["policy"],
        "exact_ig": config["exact_ig"],
        "retriever": config["retriever"],
        "rollout_algorithm": {
            key: rollout[key]
            for key in (
                "engine",
                "group_size",
                "prompt_wave_size",
                "candidate_prompts_initial",
                "refill_prompts",
                "candidate_prompts_max",
                "temperature",
                "sampling_top_p",
                "top_p",
                "top_k",
                "max_search_turns",
                "max_new_tokens_per_turn",
                "max_information_tokens_per_turn",
            )
        },
        "optimizer_scheduler": {
            key: schedule[key]
            for key in (
                "optimizer_family",
                "learning_rate",
                "weight_decay",
                "optimizer_betas",
                "optimizer_epsilon",
                "scheduler",
                "warmup",
                "total_successful_updates",
            )
        },
    }


def validate() -> dict:
    parent = load_config(PARENT)
    config = load_config(PROFILE)
    if _algorithm_snapshot(parent) != _algorithm_snapshot(config):
        raise AssertionError("Algorithm or immutable data fields changed")

    plan = build_resource_plan(config)
    resolved = build_verl_config(config, require_optimizer=True)
    topology = effective_rollout_topology(resolved)

    ray = config["ray"]
    hardware = config["hardware"]
    static_control_cpu = (
        2  # PromptSamplerActor
        + 2  # CandidatePoolActor
        + 1  # MetricsActor
        + 1  # CheckpointCommitActor
        + int(ray["outcome_worker_count"])
        + 2 * int(ray["exact_ig_task_builder_count"])
    )
    owned_runtime_cpu_estimate = (
        int(hardware["cpu_reserved_for_os"])
        + int(ray["retriever_pool_cpus"])
        + int(hardware["rl_world_size"])
        * int(ray["rl_engine_cpus_per_gpu"])
        + static_control_cpu
        + int(ray["agent_loop_worker_count"])
    )
    if owned_runtime_cpu_estimate > int(hardware["expected_cpu_cores"]):
        raise AssertionError(
            "Conservative runtime CPU estimate exceeds expected CPU cores"
        )

    result = {
        "status": "PASS",
        "config": str(PROFILE),
        "parent_config": str(PARENT),
        "algorithm_unchanged": True,
        "resource_plan": plan.as_dict(),
        "cpu": {
            "expected_cpu_cores": int(hardware["expected_cpu_cores"]),
            "declared_budget": (
                int(hardware["cpu_reserved_for_os"])
                + int(ray["retriever_pool_cpus"])
                + int(hardware["rl_world_size"])
                * int(ray["rl_engine_cpus_per_gpu"])
                + int(ray["controller_cpu_workers"])
            ),
            "static_control_cpu": static_control_cpu,
            "agent_loop_worker_count": int(ray["agent_loop_worker_count"]),
            "conservative_owned_runtime_estimate": owned_runtime_cpu_estimate,
        },
        "gpu": {
            "world_size": int(hardware["rl_world_size"]),
            "physical_gpus": list(hardware["rl_physical_gpus"]),
            "gpu_memory_utilization": float(config["rollout"]["gpu_memory_utilization"]),
            "max_num_seqs": int(config["rollout"]["max_num_seqs"]),
        },
        "learner": {
            "micro_batch_size": int(config["formal_schedule"]["learner_micro_batch_size"]),
            "topology": topology,
            "agent_loop_workers": int(resolved.actor_rollout_ref.rollout.agent.num_workers),
        },
    }
    if topology["worker_world_size"] != 3 or topology["replica_count"] != 3:
        raise AssertionError(f"Unexpected resolved topology: {topology}")
    if result["learner"]["agent_loop_workers"] != 12:
        raise AssertionError("Agent loop worker count was not wired into veRL")
    return result


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
