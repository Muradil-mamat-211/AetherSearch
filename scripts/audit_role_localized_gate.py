#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from agentic_rl.advantage.a2tgpo import (
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
    _rebuild_role_localized_gate,
    rebuild_search_advantages,
)
from agentic_rl.advantage.role_localized_gate import build_role_localized_trajectory_credits
from agentic_rl.config import load_config
from agentic_rl.policy.strict_onpolicy_loss import fixed_gate_turn_objective
from agentic_rl.runtime.fsdp_worker import StrictOnPolicyFSDP2Worker
from agentic_rl.runtime.learner_batch import prepare_selected_trajectories

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    advantage = config["advantage"]
    credit = inspect.getsource(build_role_localized_trajectory_credits)
    rebuild = inspect.getsource(_rebuild_role_localized_gate)
    dispatcher = inspect.getsource(rebuild_search_advantages)
    learner = inspect.getsource(prepare_selected_trajectories)
    worker = inspect.getsource(StrictOnPolicyFSDP2Worker.strict_backward_microbatch)
    gate = inspect.getsource(fixed_gate_turn_objective)
    checks = {
        "mode_selected": advantage["search_task_mode"]
        == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
        "dispatcher_is_explicit": "_rebuild_role_localized_gate" in dispatcher,
        "legacy_dispatchers_retained": all(name in dispatcher for name in (
            "_rebuild_sufficiency_novelty_local_ig",
            "_rebuild_sufficiency_novelty_cumulative_ig_probe_routed_outcome",
        )),
        "main_b_is_d_plus_route": "main_credit = float(d_ig_eff + route)" in credit,
        "budget_decision_minus_one": "ROLE_LOCALIZED_BRANCH_N_BUDGET" in credit and "-1.0" in credit,
        "invalid_and_s_decision_minus_half": "ROLE_LOCALIZED_BRANCH_N_INVALID" in credit and "-0.5" in credit,
        "soft_duplicate_raw_ig_gate": "float(raw_ig) <= 0.0" in credit,
        "a_sc_absent": "stop_continue_by_search_index={}" in rebuild,
        "answer_copy_asserted": "changed A_answer" in rebuild,
        "d_q_masks_provenance_backed": "decision_token_span" in learner and "query_token_span" in learner,
        "main_ratio_full_turn": "compute_turn_ratios" in worker and "torch.ones_like(row_turn_ids" in worker,
        "gate_ratios_segment_local": "row_decision_mask" in worker and "row_query_mask" in worker,
        "gate_reduction_event_mean": "event_denominator = float(max(search_turn_count, 1))" in worker,
        "main_reduction_unchanged": "main_trajectory_objective = token_objectives.mean()" in worker,
        "separate_surrogates": "fixed_gate_turn_objective" in worker,
        "fixed_dapo_bounds": "ADAPTIVE_CLIP_EPSILON_LOW" in gate and "ADAPTIVE_CLIP_EPSILON_HIGH" in gate,
        "formula_config": advantage["search_advantage_formula"]
        == "J_main + lambda_d*J_decision + lambda_q*J_query",
        "immutable_calibration": Path(
            advantage["role_localized_gate"]["calibration_manifest"]
        ).is_file(),
        "online_lambda_updates_disabled": advantage["role_localized_gate"]["online_lambda_updates"] is False,
    }
    failed = sorted(name for name, value in checks.items() if not value)
    files = [
        ROOT / "src/agentic_rl/advantage/role_localized_gate.py",
        ROOT / "src/agentic_rl/advantage/a2tgpo.py",
        ROOT / "src/agentic_rl/runtime/learner_batch.py",
        ROOT / "src/agentic_rl/runtime/fsdp_worker.py",
        ROOT / "src/agentic_rl/runtime/search_agent_loop.py",
        ROOT / "src/agentic_rl/runtime/stop_branching.py",
        ROOT / "tests/test_role_localized_gate.py",
        args.config.resolve(),
    ]
    payload = {
        "result": "PASS" if not failed else "FAIL",
        "mode": advantage["search_task_mode"],
        "checks": checks,
        "failed_checks": failed,
        "source_sha256": {str(path): digest(path) for path in files},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
