#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from agentic_rl.advantage.a2tgpo import (
    SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
    _rebuild_sufficiency_novelty_local_ig,
)
from agentic_rl.config import DEFAULT_CONFIG, load_config
from agentic_rl.controller.update_controller import StrictAttemptController
from agentic_rl.outcome.workers import score_sufficiency_probe_completion
from agentic_rl.runtime.capped_vllm import (
    CappedVLLMHttpServerBase,
    StrictAgentLoopManager,
    _build_sufficiency_probe_sampling_params,
)
from agentic_rl.runtime.search_agent_loop import (
    compute_and_commit_passage_novelty,
    stable_passage_key,
)
from agentic_rl.runtime.stop_branching import (
    attach_sufficiency_probe_results,
    build_sufficiency_probe_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    advantage = config["advantage"]
    rebuild_source = inspect.getsource(_rebuild_sufficiency_novelty_local_ig)
    controller_source = inspect.getsource(StrictAttemptController.run_attempt)
    server_source = inspect.getsource(
        CappedVLLMHttpServerBase.generate_sufficiency_probe
    )
    manager_source = inspect.getsource(
        StrictAgentLoopManager.generate_sufficiency_probes
    )
    planning_source = inspect.getsource(build_sufficiency_probe_plan)
    attach_source = inspect.getsource(attach_sufficiency_probe_results)
    novelty_source = inspect.getsource(compute_and_commit_passage_novelty)
    passage_source = inspect.getsource(stable_passage_key)
    exact_source = inspect.getsource(score_sufficiency_probe_completion)
    controller_probe_before_step = (
        controller_source.index("prepare_stop_branches(selected_groups)")
        < controller_source.index("runtime.zero_grad()")
    )
    params = _build_sufficiency_probe_sampling_params(
        {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "n": 1,
        },
        max_tokens=int(
            advantage["sufficiency_probe"]["answer_max_new_tokens"]
        ),
    )

    checks = {
        "production_mode": (
            advantage["search_task_mode"]
            == SUFFICIENCY_NOVELTY_LOCAL_IG_MODE
        ),
        "exact_formula": (
            advantage["search_advantage_formula"]
            == "-1.0 if S else -1.0 if N else normalized_local_ig"
            and "if sufficient" in rebuild_source
            and "if no_new" in rebuild_source
            and "else local_ig_hat" in rebuild_source
        ),
        "no_external_ig_multiplier": (
            advantage["external_ig_multiplier"] is None
            and advantage["future_ig_accumulation"] is False
            and advantage["sqrt_n_rescale"] is False
        ),
        "no_outcome_or_sc_in_search": (
            '"search/z_o_actor_entry_count": 0' in rebuild_source
            and '"search/a_sc_actor_entry_count": 0' in rebuild_source
        ),
        "deterministic_single_probe": (
            params.n == 1
            and params.temperature == 0.0
            and params.top_p == 1.0
            and params.logprobs is None
            and params.prompt_logprobs is None
            and "generate_sufficiency_probe.remote" in manager_source
        ),
        "selected_only_before_optimizer": controller_probe_before_step,
        "prompt_affinity_and_local_waves": (
            "prompt_to_replica" in planning_source
            and '"prompt_affinity": True' in manager_source
            and '"cross_replica_depth_barrier": False' in manager_source
        ),
        "probe_detached_and_versioned": (
            '"detached": True' in attach_source
            and '"sufficiency_probe_policy_version"' in attach_source
            and "source_checksum" in server_source
        ),
        "alias_aware_exact_only": (
            "max_alias_exact_match" in exact_source
            and "partial_task_reward_shadow" in exact_source
        ),
        "passage_id_then_full_text_sha256": (
            "document.passage_id" in passage_source
            and "hashlib.sha256" in passage_source
            and "document.contents" in passage_source
        ),
        "novelty_before_commit": (
            novelty_source.index("current_passage_keys - seen_passage_keys")
            < novelty_source.index("seen_passage_keys.update")
        ),
        "answer_formula_unchanged": (
            advantage["answer_formula_terms"]
            == ["normalized_outcome", "centered_format_indicator"]
            and float(advantage["lambda_outcome"]) == 1.0
            and float(advantage["lambda_format"]) == 1.0
        ),
        "cpu_rebalance_total_unchanged": (
            int(config["ray"]["outcome_worker_count"]) == 24
            and int(config["ray"]["exact_ig_task_builder_count"]) == 20
            and 24 + 2 * 20 == 64
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "result": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "formula": advantage["search_advantage_formula"],
        "s_probe": advantage["sufficiency_probe"],
        "answer_terms": advantage["answer_formula_terms"],
        "runtime_tuning": {
            "max_num_seqs": config["rollout"]["max_num_seqs"],
            "gpu_memory_utilization": config["rollout"][
                "gpu_memory_utilization"
            ],
            "learner_micro_batch_size": config["runtime_smoke_schedule"][
                "learner_micro_batch_size"
            ],
            "outcome_worker_count": config["ray"]["outcome_worker_count"],
            "exact_ig_task_builder_count": config["ray"][
                "exact_ig_task_builder_count"
            ],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
