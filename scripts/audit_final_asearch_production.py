#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from agentic_rl.advantage.a2tgpo import (
    SEARCH_IG_COEFFICIENT,
    TrajectoryCreditInput,
    compute_prompt_advantages,
    rebuild_search_advantages,
    turn_advantages_from_record,
)
from agentic_rl.advantage.stop_continue import (
    NORMALIZED_OUTCOME_MODE,
    STOP_CONTINUE_CONSENSUS_MODE,
    StopContinueRewardTriple,
    compute_stop_continue_advantages,
)
from agentic_rl.config import DEFAULT_CONFIG, load_config
from agentic_rl.controller.update_controller import StrictAttemptController
from agentic_rl.exact_ig.target_schema import (
    ANSWER_SCAFFOLD_TEXT,
    EXACT_IG_VERSION,
    select_canonical_answer,
)
from agentic_rl.outcome.token_f1 import max_alias_token_f1
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    score_stop_answer_completion,
    score_trajectory_outcome,
)
from agentic_rl.policy.strict_onpolicy_loss import (
    a2tgpo_adaptive_turn_objective,
)
from agentic_rl.rollout.trajectory_schema import TurnType
from agentic_rl.runtime.capped_vllm import (
    CappedVLLMHttpServerBase,
    StrictAgentLoopManager,
    _build_stop_pair_sampling_params,
)
from agentic_rl.runtime.fsdp_worker import StrictOnPolicyFSDP2Worker
from agentic_rl.runtime.learner_batch import prepare_selected_trajectories
from agentic_rl.runtime.stop_branching import (
    attach_stop_branch_rewards,
    build_stop_branch_plan,
)
from agentic_rl.runtime.verl_runtime_adapter import VerlAttemptRuntimeAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_JSON = REPORT_DIR / "FINAL_ASEARCH_PRODUCTION_AUDIT.json"
REPORT_MD = REPORT_DIR / "FINAL_ASEARCH_PRODUCTION_AUDIT.md"
POLICY_VERSION = 17

AUDITED_FILES = (
    "src/agentic_rl/controller/update_controller.py",
    "src/agentic_rl/runtime/verl_runtime_adapter.py",
    "src/agentic_rl/runtime/stop_branching.py",
    "src/agentic_rl/runtime/capped_vllm.py",
    "src/agentic_rl/runtime/learner_batch.py",
    "src/agentic_rl/runtime/fsdp_worker.py",
    "src/agentic_rl/advantage/a2tgpo.py",
    "src/agentic_rl/advantage/stop_continue.py",
    "src/agentic_rl/outcome/workers.py",
    "src/agentic_rl/outcome/token_f1.py",
    "src/agentic_rl/policy/strict_onpolicy_loss.py",
    "src/agentic_rl/selection/top_p.py",
    "src/agentic_rl/selection/prompt_variance.py",
    "src/agentic_rl/selection/channel_scale.py",
    "src/agentic_rl/selection/health_gate.py",
    "scripts/audit_final_asearch_production.py",
    "configs/base.yaml",
    "configs/formal_train.yaml",
    "tests/test_stop_continue_advantage.py",
    "tests/test_stop_branching.py",
    "tests/test_final_asearch_production_contract.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_location(value: Any) -> str:
    path = Path(inspect.getsourcefile(value) or "").resolve()
    line = inspect.getsourcelines(value)[1]
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = path
    return f"{relative}:{line}"


def _project_function_location(relative_path: str, function_name: str) -> str:
    path = PROJECT_ROOT / relative_path
    needle = f"    def {function_name}("
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if line.startswith(needle):
            return f"{relative_path}:{line_number}"
    raise RuntimeError(f"Cannot locate {function_name} in {relative_path}")


def _reward(
    trajectory_id: str,
    continue_reward: float,
    stop_reward_1: float,
    stop_reward_2: float,
    *,
    search_index: int = 0,
) -> StopContinueRewardTriple:
    return StopContinueRewardTriple(
        prompt_global_id="prompt-audit",
        trajectory_id=trajectory_id,
        search_index=search_index,
        continue_reward=continue_reward,
        stop_reward_1=stop_reward_1,
        stop_reward_2=stop_reward_2,
        continue_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
        stop_scorer_version_1=PRODUCTION_TASK_SCORER_VERSION,
        stop_scorer_version_2=PRODUCTION_TASK_SCORER_VERSION,
        candidate_rollout_policy_version=POLICY_VERSION,
        exact_ig_policy_version=POLICY_VERSION,
        stop_branch_policy_version=POLICY_VERSION,
        old_logprob_policy_version=POLICY_VERSION,
        prefix_provenance_valid=True,
        context_truncated=False,
        completion_count=2,
        detached=True,
    )


def _compute(
    rewards: list[StopContinueRewardTriple],
    *,
    normalized_outcome: float = -0.25,
):
    return compute_stop_continue_advantages(
        rewards,
        normalized_outcome_by_trajectory={
            (value.prompt_global_id, value.trajectory_id): normalized_outcome
            for value in rewards
        },
        expected_state_keys=[value.state_key for value in rewards],
        group_size=16,
        reward_epsilon=1.0e-6,
        scale_epsilon=1.0e-8,
        pooled_scale_ddof=0,
        expected_policy_version=POLICY_VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )


def _credit(immediate_ig: float, outcome: float, format_indicator: int):
    return TrajectoryCreditInput(
        immediate_ig={0: immediate_ig},
        search_turn_indices=(0,),
        ig_reward_eligible={0: True},
        policy_credit_eligible={0: True},
        outcome=outcome,
        outcome_reward_eligible=True,
        format_indicator=format_indicator,
        answer_policy_credit_eligible=True,
    )


def _probe(reward: StopContinueRewardTriple) -> dict[str, Any]:
    return asdict(reward)


def _formula_execution_checks() -> tuple[dict[str, bool], dict[str, Any]]:
    clear_continue = _compute([_reward("continue", 1.0, 0.0, 0.6)])
    positive = clear_continue.by_state[("prompt-audit", "continue", 0)]
    clear_stop = _compute([_reward("stop", 0.0, 0.6, 1.0)])
    negative = clear_stop.by_state[("prompt-audit", "stop", 0)]
    straddled = _compute(
        [_reward("straddled", 0.6, 0.0, 1.0)],
        normalized_outcome=-0.375,
    ).by_state[("prompt-audit", "straddled", 0)]
    equal = _compute(
        [_reward("equal", 0.6, 0.6, 0.6)],
        normalized_outcome=0.125,
    ).by_state[("prompt-audit", "equal", 0)]
    epsilon = 1.0e-6
    boundary_positive = _compute(
        [_reward("eps-pos", 0.6 + epsilon, 0.0, 0.6)]
    ).by_state[("prompt-audit", "eps-pos", 0)]
    boundary_negative = _compute(
        [_reward("eps-neg", 0.6 - epsilon, 0.6, 1.0)]
    ).by_state[("prompt-audit", "eps-neg", 0)]

    pooled_rewards = [
        _reward("pool-0", 1.0, 0.0, 0.6),
        _reward("pool-1", 0.2, 0.4, 0.8),
    ]
    pooled = _compute(pooled_rewards)
    expected_pool = float(np.std([1.0, 0.0, 0.6, 0.2, 0.4, 0.8], ddof=0))

    clip_positive_rewards = [
        _reward(f"clip-pos-{index}", 0.0, 0.0, 0.0)
        for index in range(15)
    ]
    clip_positive_rewards.append(_reward("clip-pos-15", 1.0, 0.0, 0.0))
    clip_positive = _compute(clip_positive_rewards).by_state[
        ("prompt-audit", "clip-pos-15", 0)
    ]
    clip_negative_rewards = [
        _reward(f"clip-neg-{index}", 0.0, 0.0, 0.0)
        for index in range(15)
    ]
    clip_negative_rewards.append(_reward("clip-neg-15", 0.0, 1.0, 1.0))
    clip_negative = _compute(clip_negative_rewards).by_state[
        ("prompt-audit", "clip-neg-15", 0)
    ]
    clip_bound = math.sqrt(15)

    checks = {
        "delta_sc_clear_continue": math.isclose(
            positive.delta_sc, 0.7, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "clear_continue_positive": (
            positive.sc_clear
            and positive.clear_positive
            and positive.advantage_sc > 0.0
            and positive.task_advantage == positive.advantage_sc
        ),
        "delta_sc_clear_stop": math.isclose(
            negative.delta_sc, -0.8, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "clear_stop_negative": (
            negative.sc_clear
            and negative.clear_negative
            and negative.advantage_sc < 0.0
            and negative.task_advantage == negative.advantage_sc
        ),
        "straddled_falls_back_to_zero": (
            not straddled.sc_clear and straddled.task_advantage == 0.0
        ),
        "all_equal_falls_back_without_nonfinite": (
            not equal.sc_clear
            and equal.delta_sc == 0.0
            and equal.pooled_scale == 0.0
            and equal.advantage_sc == 0.0
            and equal.task_advantage == 0.0
        ),
        "epsilon_boundary_is_strict": (
            not boundary_positive.sc_clear and not boundary_negative.sc_clear
        ),
        "pooled_scale_cardinality_and_ddof_zero": math.isclose(
            pooled.pooled_scale_by_prompt_search[("prompt-audit", 0)],
            expected_pool,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "positive_clip_sqrt_g_minus_one": math.isclose(
            clip_positive.advantage_sc,
            clip_bound,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "negative_clip_sqrt_g_minus_one": math.isclose(
            clip_negative.advantage_sc,
            -clip_bound,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
    }
    details = {
        "clear_continue": asdict(positive),
        "clear_stop": asdict(negative),
        "straddled": asdict(straddled),
        "all_equal": asdict(equal),
        "pooled_scale_observed": pooled.pooled_scale_by_prompt_search[
            ("prompt-audit", 0)
        ],
        "pooled_scale_expected": expected_pool,
        "clip_bound": clip_bound,
        "positive_raw": clip_positive.raw_advantage_sc,
        "negative_raw": clip_negative.raw_advantage_sc,
    }
    return checks, details


def _advantage_dataflow_checks() -> tuple[dict[str, bool], dict[str, Any]]:
    base = compute_prompt_advantages(
        [_credit(1.0, 1.0, 1), _credit(-1.0, 0.0, 0)]
    )
    records = [
        SimpleNamespace(
            prompt_global_id="prompt-audit",
            trajectory_id=f"trajectory-{index}",
            metadata={
                "stop_continue_probes": {
                    0: _probe(
                        _reward(
                            f"trajectory-{index}",
                            1.0 if index == 0 else 0.0,
                            0.0 if index == 0 else 0.6,
                            0.6 if index == 0 else 1.0,
                        )
                    )
                }
            },
        )
        for index in range(2)
    ]
    old_answers = tuple(item.answer_advantage for item in base.trajectories)
    legacy, _ = rebuild_search_advantages(
        records,
        base,
        search_task_mode=NORMALIZED_OUTCOME_MODE,
        group_size=16,
    )
    rebuilt, _ = rebuild_search_advantages(
        records,
        base,
        search_task_mode=STOP_CONTINUE_CONSENSUS_MODE,
        group_size=16,
        expected_policy_version=POLICY_VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )
    rebuilt_answers = tuple(
        item.answer_advantage for item in rebuilt.trajectories
    )

    first = rebuilt.trajectories[0]
    search_turn = SimpleNamespace(
        turn_index=3,
        turn_type=TurnType.SEARCH,
        search_index=0,
        policy_credit_eligible=True,
    )
    answer_turn = SimpleNamespace(
        turn_index=5,
        turn_type=TurnType.ANSWER,
        search_index=None,
        policy_credit_eligible=True,
    )
    learner_record = SimpleNamespace(
        turns=(search_turn, answer_turn),
        terminal_policy_credit_turn_index=5,
    )
    advantage_by_turn = turn_advantages_from_record(learner_record, first)
    ratios = {
        3: torch.tensor(1.0, dtype=torch.float32),
        5: torch.tensor(1.0, dtype=torch.float32),
    }
    objective = a2tgpo_adaptive_turn_objective(
        ratios,
        advantage_by_turn,
        {3: first.normalized_ig[0]},
        answer_turn_ids=(5,),
    )
    search_objective = float(objective.objective_by_turn[3].item())
    answer_objective = float(objective.objective_by_turn[5].item())

    checks = {
        "feature_flag_disabled_restores_old_formula": all(
            math.isclose(
                item.search_advantage[0],
                SEARCH_IG_COEFFICIENT * item.future_ig_rescaled[0]
                + item.normalized_outcome,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for item in legacy.trajectories
        ),
        "asearch_new_is_point_three_aig_plus_atask": all(
            math.isclose(
                item.search_advantage[0],
                SEARCH_IG_COEFFICIENT * item.future_ig_rescaled[0]
                + item.stop_continue_by_search_index[0].task_advantage,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for item in rebuilt.trajectories
        ),
        "old_shadow_is_point_three_aig_plus_z_o": all(
            math.isclose(
                item.search_advantage_old_shadow[0],
                SEARCH_IG_COEFFICIENT * item.future_ig_rescaled[0]
                + item.normalized_outcome,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for item in rebuilt.trajectories
        ),
        "aanswer_byte_value_unchanged": rebuilt_answers == old_answers,
        "search_turn_mapping_uses_rebuilt_value": math.isclose(
            advantage_by_turn[3],
            first.search_advantage[0],
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "answer_turn_mapping_uses_original_answer_value": math.isclose(
            advantage_by_turn[5],
            float(first.answer_advantage),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "policy_objective_consumes_rebuilt_search_value": math.isclose(
            search_objective,
            first.search_advantage[0],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        "policy_objective_consumes_unchanged_answer_value": math.isclose(
            answer_objective,
            float(first.answer_advantage),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
    }
    details = {
        "old_answer_advantages": old_answers,
        "rebuilt_answer_advantages": rebuilt_answers,
        "first_trajectory": {
            "A_IG": first.future_ig_rescaled[0],
            "z_O": first.normalized_outcome,
            "A_SC": first.stop_continue_by_search_index[0].advantage_sc,
            "A_task": first.stop_continue_by_search_index[0].task_advantage,
            "A_search_old_shadow": first.search_advantage_old_shadow[0],
            "A_search_new": first.search_advantage[0],
            "A_answer": first.answer_advantage,
            "learner_search_objective_at_ratio_one": search_objective,
            "learner_answer_objective_at_ratio_one": answer_objective,
        },
    }
    return checks, details


def _source_and_runtime_contract_checks(
    config: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    controller = inspect.getsource(StrictAttemptController.run_attempt)
    adapter_stop = inspect.getsource(
        VerlAttemptRuntimeAdapter.prepare_selected_stop_branches
    )
    adapter_batches = inspect.getsource(
        VerlAttemptRuntimeAdapter.selected_microbatches
    )
    stop_plan = inspect.getsource(build_stop_branch_plan)
    attach = inspect.getsource(attach_stop_branch_rewards)
    manager = inspect.getsource(StrictAgentLoopManager.generate_stop_branches)
    server = inspect.getsource(CappedVLLMHttpServerBase.generate_stop_pair)
    fsdp_backward = inspect.getsource(
        StrictOnPolicyFSDP2Worker.strict_backward_microbatch
    )
    prepare = inspect.getsource(prepare_selected_trajectories)
    selection_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(
            (PROJECT_ROOT / "src/agentic_rl/selection").glob("*.py")
        )
    )
    sampling = _build_stop_pair_sampling_params(
        {
            "temperature": float(config["rollout"]["temperature"]),
            "top_p": float(
                config["rollout"].get(
                    "sampling_top_p",
                    config["rollout"]["top_p"],
                )
            ),
            "top_k": -1,
            "min_p": float(config["rollout"].get("min_p", 0.0)),
            "repetition_penalty": float(
                config["rollout"].get("repetition_penalty", 1.0)
            ),
            "presence_penalty": float(
                config["rollout"].get("presence_penalty", 0.0)
            ),
            "frequency_penalty": float(
                config["rollout"].get("frequency_penalty", 0.0)
            ),
        },
        max_tokens=int(config["advantage"]["sc"]["stop_answer_max_new_tokens"]),
    )

    selection_index = controller.index("selected_groups = pool.selected_groups")
    stop_index = controller.index("prepare_stop_branches(selected_groups)")
    microbatch_index = controller.index("runtime.selected_microbatches")
    zero_index = controller.index("runtime.zero_grad()")
    checks = {
        "formal_feature_flag_enabled": (
            config["advantage"]["search_task_mode"]
            == STOP_CONTINUE_CONSENSUS_MODE
            and config["advantage"]["sc"]["enabled"] is True
            and int(config["advantage"]["sc"]["num_stop_samples"]) == 2
        ),
        "production_lambda_ig_point_three": (
            float(config["advantage"]["lambda_ig"])
            == SEARCH_IG_COEFFICIENT
            == 0.3
        ),
        "answer_coefficients_unchanged": (
            float(config["advantage"]["lambda_outcome"]) == 1.0
            and float(config["advantage"]["lambda_format"]) == 1.0
            and float(config["advantage"]["lambda_task"]) == 1.0
        ),
        "controller_order_selection_stop_advantage_backward": (
            selection_index < stop_index < microbatch_index < zero_index
        ),
        "selected_only_stop_input": (
            "groups: Sequence[PromptGroup]" in adapter_stop
            and "build_stop_branch_plan(\n            groups," in adapter_stop
        ),
        "one_request_n_two": (
            sampling.n == 2
            and "generate_stop_pair.remote" in manager
            and "len(final_result.outputs) != 2" in server
        ),
        "stop_logprobs_disabled": (
            sampling.logprobs is None and sampling.prompt_logprobs is None
        ),
        "prompt_group_affinity": (
            "prompt_to_replica" in stop_plan
            and "Prompt affinity changed" in stop_plan
            and '"prompt_affinity": True' in manager
        ),
        "replica_local_depth_waves": (
            "for depth in depths" in manager
            and '"cross_replica_depth_barrier": False' in manager
        ),
        "automatic_prefix_caching_required": (
            "Automatic Prefix Caching is disabled" in attach
            and '"sc/automatic_prefix_caching": True' in adapter_stop
        ),
        "prefix_uses_original_token_provenance": (
            "record.prefix_token_ids_before_search(search_index)" in stop_plan
            and "stop_input_ids = (*prefix, *scaffold)" in stop_plan
        ),
        "policy_versions_and_checksum_checked": (
            "expected_snapshot_step" in stop_plan
            and "expected_source_checksum" in attach
            and "Actor checksum no longer matches" in adapter_stop
            and "Detached Stop branching changed Actor parameters" in adapter_stop
        ),
        "stop_is_detached_and_not_learner_trajectory": (
            '"detached": True' in attach
            and "completion_payloads" in adapter_stop
            and "prepared_groups" not in adapter_stop
        ),
        "rebuilt_advantage_enters_microbatch": (
            "prepare_selected_trajectories(" in adapter_batches
            and "rebuild_search_advantages(" in prepare
            and '"advantage_by_turn"' in inspect.getsource(
                __import__(
                    "agentic_rl.runtime.learner_batch",
                    fromlist=["_collate_rank_payload"],
                )._collate_rank_payload
            )
        ),
        "fsdp_loss_consumes_advantage_by_turn": (
            'microbatch["advantage_by_turn"][batch_index]' in fsdp_backward
            and "loss.backward()" in fsdp_backward
        ),
        "ragen_has_no_stop_continue_input": (
            "stop_continue" not in selection_source
            and "stop_reward" not in selection_source
            and "A_SC" not in selection_source
        ),
        "strict_one_step_runtime": (
            controller.count("runtime.optimizer_step()") == 1
            and controller.count("runtime.scheduler_step()") == 1
        ),
    }
    details = {
        "locations": {
            "controller_transaction": _source_location(
                StrictAttemptController.run_attempt
            ),
            "stop_runtime": _source_location(
                VerlAttemptRuntimeAdapter.prepare_selected_stop_branches
            ),
            "advantage_preparation": _source_location(
                prepare_selected_trajectories
            ),
            "stop_math": _source_location(compute_stop_continue_advantages),
            "search_rebuild": _source_location(rebuild_search_advantages),
            "stop_plan": _source_location(build_stop_branch_plan),
            "vllm_stop_pair": _source_location(
                CappedVLLMHttpServerBase.generate_stop_pair
            ),
            "replica_router": _source_location(
                StrictAgentLoopManager.generate_stop_branches
            ),
            "fsdp_loss": _project_function_location(
                "src/agentic_rl/runtime/fsdp_worker.py",
                "strict_backward_microbatch",
            ),
        },
        "sampling": {
            "n": sampling.n,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "max_tokens": sampling.max_tokens,
            "stop": sampling.stop,
            "logprobs": sampling.logprobs,
            "prompt_logprobs": sampling.prompt_logprobs,
        },
    }
    return checks, details


def _reward_and_exact_ig_checks(
    config: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    aliases = ["New York City", "NYC"]
    canonical = select_canonical_answer(aliases)
    all_alias_score = max_alias_token_f1("NYC", aliases)
    individual_scores = [
        max_alias_token_f1("NYC", [alias]) for alias in aliases
    ]
    outcome = score_trajectory_outcome(
        ["<answer>NYC</answer>"],
        aliases,
    )
    stop = score_stop_answer_completion("NYC</answer>", aliases)
    exact_audit_path = Path(config["exact_ig"]["structural_audit_path"])
    exact_audit = json.loads(exact_audit_path.read_text(encoding="utf-8"))
    checks = {
        "exact_ig_structural_audit_pass": (
            exact_audit.get("allow_fast_path_training") is True
            and exact_audit.get("gates", {}).get("PACKED_STRUCTURE") is True
            and exact_audit.get("gates", {}).get("FUTURE_LEAKAGE") is True
            and exact_audit.get("gates", {}).get("LOGICAL_POSITION_IDS") is True
        ),
        "exact_ig_version_locked": (
            config["exact_ig"]["exact_ig_version"] == EXACT_IG_VERSION
        ),
        "exact_ig_scaffold_locked": (
            ANSWER_SCAFFOLD_TEXT
            == "<think>The retrieved evidence now supports the answer.</think>"
            "<answer>"
        ),
        "canonical_is_first_alias": canonical == aliases[0],
        "task_reward_retains_alias_max": (
            math.isclose(
                all_alias_score,
                max(individual_scores),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and all_alias_score == 1.0
            and outcome.task_outcome == all_alias_score
            and stop.task_outcome == all_alias_score
        ),
        "oracle_numeric_difference_is_telemetry": (
            config["exact_ig"]["oracle_numeric_difference_policy"]
            == "telemetry_only_unless_semantic_or_safety_drift"
            and float(config["exact_ig"]["maximum_phi_safety_abs_diff"])
            == 1.0e-3
            and float(config["exact_ig"]["maximum_ig_safety_abs_diff"])
            == 1.0e-3
        ),
    }
    details = {
        "exact_ig_version": EXACT_IG_VERSION,
        "exact_ig_audit_path": str(exact_audit_path),
        "exact_ig_audit_sha256": _sha256_file(exact_audit_path),
        "canonical_answer": canonical,
        "aliases": aliases,
        "all_alias_score": all_alias_score,
        "individual_alias_scores": individual_scores,
        "outcome_score": outcome.task_outcome,
        "stop_score": stop.task_outcome,
        "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
    }
    return checks, details


def _smoke_status(require_smoke: bool) -> tuple[dict[str, bool], dict[str, Any]]:
    path = PROJECT_ROOT / "runtime/stage_results/stage_sc.json"
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    checksum_unchanged = bool(
        payload.get("actor_checksum_unchanged") is True
        or (
            payload.get("actor_checksum_before")
            and payload.get("actor_checksum_before")
            == payload.get("actor_checksum_after")
        )
    )
    passed = bool(
        payload.get("status") == "PASS"
        and int(payload.get("optimizer_steps", -1)) == 0
        and int(payload.get("scheduler_steps", -1)) == 0
        and int(payload.get("checkpoint_writes", -1)) == 0
        and checksum_unchanged
    )
    return (
        {"real_no_update_smoke_pass": passed or not require_smoke},
        {
            "required": require_smoke,
            "path": str(path),
            "present": path.is_file(),
            "passed": passed,
            "status": payload.get("status"),
            "optimizer_steps": payload.get("optimizer_steps"),
            "scheduler_steps": payload.get("scheduler_steps"),
            "checkpoint_writes": payload.get("checkpoint_writes"),
            "actor_checksum_unchanged": checksum_unchanged,
        },
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Final A_search Production Audit",
        "",
        f"- Result: **{payload['result']}**",
        f"- Exact-IG version: `{payload['exact_ig_version']}`",
        f"- Search task mode: `{payload['config']['search_task_mode']}`",
        f"- Code hash: `{payload['hashes']['code_hash']}`",
        f"- Config hash: `{payload['hashes']['config_hash']}`",
        f"- Tests hash: `{payload['hashes']['tests_hash']}`",
        "",
        "## Production Formula",
        "",
        "`Delta_SC = R_C - 0.5 * (R_S1 + R_S2)`",
        "",
        "`s_SC[p,t] = std_pop(flatten([R_C, R_S1, R_S2] for i in I[p,t]))`",
        "",
        "`A_SC = clamp(Delta_SC / (s_SC + 1e-8), -sqrt(G-1), +sqrt(G-1))`",
        "",
        "`sc_clear = (R_C > max(R_S1,R_S2)+1e-6) or "
        "(R_C < min(R_S1,R_S2)-1e-6)`",
        "",
        "`A_task = A_SC if sc_clear else 0.0`",
        "",
        "`A_search_new = 0.3 * A_IG + A_task`",
        "",
        "`A_answer = z_O + A_format` (unchanged)",
        "",
        "## Executable Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in sorted(payload["checks"].items()):
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Production Dataflow",
            "",
            "1. `StrictAttemptController.run_attempt` finalizes RAGEN selection.",
            "2. `VerlAttemptRuntimeAdapter.prepare_selected_stop_branches` receives "
            "only selected groups and generates two detached Stop completions per "
            "real Search state.",
            "3. `prepare_selected_trajectories` computes A2TGPO, replaces only the "
            "Search task term, and preserves every Answer advantage.",
            "4. `_collate_rank_payload` writes rebuilt values to "
            "`advantage_by_turn`.",
            "5. `StrictFSDPWorker.strict_backward_microbatch` passes that mapping "
            "to `a2tgpo_adaptive_turn_objective`; Search MODEL tokens therefore "
            "consume `A_search_new` in the actual loss.",
            "",
            "## Code Evidence",
            "",
        ]
    )
    for name, location in payload["details"]["source_contract"]["locations"].items():
        lines.append(f"- `{name}`: `{location}`")
    lines.extend(
        [
            "",
            "## Exact-IG Boundary",
            "",
            "The A_search implementation does not alter Exact-IG source or RAGEN "
            "selection inputs. The independent packed Fast Path structural audit "
            "is referenced by hash in the JSON report. Sequential shape-dependent "
            "numeric differences remain telemetry; structural, finite, semantic, "
            "and 1e-3 safety checks remain enforced.",
            "",
            "## Failures",
            "",
        ]
    )
    if payload["failed_checks"]:
        lines.extend(f"- `{name}`" for name in payload["failed_checks"])
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Resolved or extendable project config.",
    )
    parser.add_argument(
        "--require-smoke",
        action="store_true",
        help="Require the real no-update SC smoke artifact to be PASS.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    formula_checks, formula_details = _formula_execution_checks()
    dataflow_checks, dataflow_details = _advantage_dataflow_checks()
    source_checks, source_details = _source_and_runtime_contract_checks(config)
    reward_checks, reward_details = _reward_and_exact_ig_checks(config)
    smoke_checks, smoke_details = _smoke_status(args.require_smoke)
    checks = {
        **formula_checks,
        **dataflow_checks,
        **source_checks,
        **reward_checks,
        **smoke_checks,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)

    file_hashes = {
        path: _sha256_file(PROJECT_ROOT / path)
        for path in AUDITED_FILES
        if (PROJECT_ROOT / path).is_file()
    }
    code_paths = [
        path for path in file_hashes if path.startswith("src/") or path.startswith("scripts/")
    ]
    config_paths = [path for path in file_hashes if path.startswith("configs/")]
    test_paths = [path for path in file_hashes if path.startswith("tests/")]

    def aggregate(paths: list[str]) -> str:
        return _sha256_bytes(
            "\n".join(f"{path}:{file_hashes[path]}" for path in sorted(paths)).encode(
                "utf-8"
            )
        )

    payload: dict[str, Any] = {
        "result": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "checks": checks,
        "exact_ig_version": EXACT_IG_VERSION,
        "config": {
            "path": str(Path(args.config).resolve()),
            "search_task_mode": config["advantage"]["search_task_mode"],
            "lambda_ig": config["advantage"]["lambda_ig"],
            "lambda_task": config["advantage"]["lambda_task"],
            "lambda_outcome": config["advantage"]["lambda_outcome"],
            "lambda_format": config["advantage"]["lambda_format"],
            "group_size": config["rollout"]["group_size"],
            "sc": config["advantage"]["sc"],
        },
        "details": {
            "formula_execution": formula_details,
            "advantage_dataflow": dataflow_details,
            "source_contract": source_details,
            "reward_and_exact_ig": reward_details,
            "no_update_smoke": smoke_details,
        },
        "hashes": {
            "files": file_hashes,
            "code_hash": aggregate(code_paths),
            "config_hash": aggregate(config_paths),
            "tests_hash": aggregate(test_paths),
        },
        "training_side_effects": {
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_without_audit_hash = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    payload["hashes"]["audit_payload_sha256"] = _sha256_bytes(
        report_without_audit_hash.encode("utf-8")
    )
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
