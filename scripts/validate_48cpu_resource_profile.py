#!/usr/bin/env python3
"""Read-only validation for the 3-rank/48-CPU resource profile."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rl.config import (
    _load_config_tree,
    load_config,
    runtime_owned_section,
    runtime_section,
)
from agentic_rl.runtime.resource_plan import build_runtime_cpu_resource_plan
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
    # PARENT is intentionally an abstract experiment layer after hardware and
    # runtime ownership were split.  Compare its unmaterialized composition to
    # the profile source, then run the executable profile through load_config.
    parent = _load_config_tree(PARENT)
    profile_source = _load_config_tree(PROFILE)
    config = load_config(PROFILE)
    if _algorithm_snapshot(parent) != _algorithm_snapshot(profile_source):
        raise AssertionError("Algorithm or immutable data fields changed")

    plan = build_resource_plan(config)
    resolved = build_verl_config(config, require_optimizer=True)
    topology = effective_rollout_topology(resolved)

    ray = runtime_owned_section(config, "ray")
    runtime_rollout = runtime_section(config, "rollout")
    hardware = config["hardware"]
    cpu_plan = build_runtime_cpu_resource_plan(
        hardware,
        ray,
        learner_world_size=int(hardware["rl_world_size"]),
        formal_schedule=config["formal_schedule"],
    )
    if cpu_plan.total_cpus > int(hardware["expected_cpu_cores"]):
        raise AssertionError(
            "Runtime CPU resource plan exceeds expected CPU cores"
        )

    result = {
        "status": "PASS",
        "config": str(PROFILE),
        "parent_config": str(PARENT),
        "algorithm_unchanged": True,
        "resource_plan": plan.as_dict(),
        "cpu": {
            "expected_cpu_cores": int(hardware["expected_cpu_cores"]),
            "declared_budget": cpu_plan.total_cpus,
            "control_actor_plan": cpu_plan.control_actors.as_dict(),
            "agent_loop_worker_count": int(ray["agent_loop_worker_count"]),
            "runtime_required_cpus": cpu_plan.total_cpus,
        },
        "gpu": {
            "world_size": int(hardware["rl_world_size"]),
            "physical_gpus": list(hardware["rl_physical_gpus"]),
            "gpu_memory_utilization": float(runtime_rollout["gpu_memory_utilization"]),
            "max_num_seqs": int(runtime_rollout["max_num_seqs"]),
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
