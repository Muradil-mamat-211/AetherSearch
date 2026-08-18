from __future__ import annotations

import inspect
from pathlib import Path

from agentic_rl.config import load_config
from agentic_rl.runtime.verl_config import build_verl_config, effective_rollout_topology
from agentic_rl.workers.resource_plan import build_resource_plan


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/formal_resume_u20_3rank_48cpu.yaml"
PARENT = ROOT / "configs/formal_resume_u20_3rank.yaml"


def test_48cpu_profile_resolves_three_rank_topology_without_algorithm_changes() -> None:
    parent = load_config(PARENT)
    config = load_config(PROFILE)
    assert config["hardware"]["expected_cpu_cores"] == 48
    assert config["hardware"]["total_physical_gpus"] == 4
    assert config["hardware"]["rl_physical_gpus"] == [1, 2, 3]
    assert config["hardware"]["rl_visible_gpus"] == [0, 1, 2]
    assert config["hardware"]["rl_world_size"] == 3
    assert config["ray"]["agent_loop_worker_count"] == 12
    assert config["rollout"]["gpu_memory_utilization"] == 0.48
    assert config["rollout"]["max_num_seqs"] == 64
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6

    for key in ("data", "selection", "advantage", "policy", "exact_ig", "retriever"):
        assert config[key] == parent[key]
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
    ):
        assert config["rollout"][key] == parent["rollout"][key]


def test_48cpu_profile_wires_agent_workers_and_resource_plan() -> None:
    config = load_config(PROFILE)
    plan = build_resource_plan(config)
    assert plan.rl_world_size == 3
    assert plan.rl_physical_gpus == (1, 2, 3)
    assert plan.rl_visible_gpus == (0, 1, 2)
    resolved = build_verl_config(config, require_optimizer=True)
    assert resolved.actor_rollout_ref.rollout.agent.num_workers == 12
    assert effective_rollout_topology(resolved) == {
        "worker_world_size": 3,
        "per_replica_world_size": 1,
        "replica_count": 3,
        "aggregate_data_parallel_size": 3,
        "tensor_parallel_size": 1,
    }


def test_48cpu_validation_script_is_read_only_and_uses_new_profile() -> None:
    script = (ROOT / "scripts/validate_48cpu_resource_profile.py").read_text()
    assert "optimizer.step" not in script
    assert "scheduler.step" not in script
    assert "formal_resume_u20_3rank_48cpu.yaml" in script
    source = inspect.getsource(build_verl_config)
    assert "agent_loop_worker_count" in source
