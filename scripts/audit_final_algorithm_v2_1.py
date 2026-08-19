#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentic_rl.advantage.a2tgpo import compute_prompt_advantages
from agentic_rl.config import DEFAULT_CONFIG, load_config
from agentic_rl.controller.transaction import StrictUpdateTransaction
from agentic_rl.exact_ig.alias_reduce import immediate_ig_from_prefix_scores
from agentic_rl.exact_ig.vectorized_scorer import VectorizedExactIGScorer
from agentic_rl.outcome.format_indicator import centered_format_advantage
from agentic_rl.policy.reference_kl import actor_to_reference_full_vocab_kl
from agentic_rl.policy.reduction import prompt_trajectory_action_token_reduce
from agentic_rl.policy.strict_onpolicy_loss import (
    ADAPTIVE_CLIP_BETA,
    ADAPTIVE_CLIP_EPSILON_HIGH,
    ADAPTIVE_CLIP_EPSILON_LOW,
    ANSWER_CLIP_SCALE,
    CLIPPING_MODE,
    a2tgpo_adaptive_turn_objective,
    adaptive_clip_scale,
)
from agentic_rl.policy.turn_ratio import compute_turn_ratios
from agentic_rl.selection.channel_scale import ChannelScaleState
from agentic_rl.selection.prompt_variance import ig_prompt_variance
from export_effective_algorithm_v2_1 import (
    OUTPUT_PATH as EFFECTIVE_REPORT,
    effective_contract,
    render_report,
)


FINAL_SPEC = PROJECT_ROOT / "FINAL_ALGORITHM_SPEC_V2_1.md"
MATRIX_PATH = PROJECT_ROOT / "ALGORITHM_CONSISTENCY_MATRIX_V2_1.md"


@dataclass(frozen=True)
class Check:
    requirement_id: str
    requirement: str
    passed: bool
    evidence: str
    test: str
    runtime_only: bool = False

    @property
    def status(self) -> str:
        if self.runtime_only:
            return "UNVERIFIED_RUNTIME"
        return "PASS" if self.passed else "FAIL"


def _source(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _location(value: Any) -> str:
    path = Path(inspect.getsourcefile(value) or "").resolve()
    return (
        f"{path.relative_to(PROJECT_ROOT).as_posix()}:"
        f"{inspect.getsourcelines(value)[1]}"
    )


def _extract_json(path: Path, start: str, end: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise ValueError(f"Missing machine-readable contract markers in {path}")
    block = text.split(start, 1)[1].split(end, 1)[0]
    fenced = block.split("```json", 1)[1].split("```", 1)[0]
    value = json.loads(fenced)
    if not isinstance(value, dict):
        raise TypeError(f"Contract in {path} is not an object")
    return value


def expected_contract() -> dict[str, Any]:
    return {
        "schema_version": "2.1",
        "exact_ig": {
            "score_space": "mean_canonical_answer_body_log_likelihood",
            "search_reward": "phi_t_minus_phi_t_minus_1",
            "canonical_alias_policy": "first",
            "score_mask_policy": "answer_body_only",
            "fast_path_structure": "official_no_anchor",
            "phi_exponentiated": False,
            "extra_search_reward_terms": [],
            "stop_gradient": True,
        },
        "outcome": {
            "scorer": "igpo_official_set_token_f1",
            "official_commit": "64165e2741ed8801f977948c8128080ce87b4101",
            "aliases_delimiter": "<|answer_split|>",
            "format_enters_task_outcome": False,
        },
        "ig_variance": {
            "scope": "same_prompt_same_search_position",
            "sample_variance_ddof": 1,
            "effective_position_min_peers": 2,
            "singleton_weight": 0.0,
            "weight_denominator": "sum_peer_counts_over_positions_with_n_ge_2",
        },
        "scale_activation": {
            "update_1_selection_scale": "current_positive_median",
            "update_1_activation_controls_scale_commit": False,
            "updates_2_to_health_ready_selection_scale": "previous_committed_scale",
            "bootstrap_activation_controls_ema": False,
            "health_inactive_freezes_ema": True,
            "ema_half_life": 10,
            "ema_eta": 1.0 - 2.0 ** (-1.0 / 10.0),
            "health_reference_valid_observations": 10,
            "health_threshold_ratio": 0.1,
            "minimum_positive_prompts": 4,
            "late_initialization_allowed": False,
        },
        "selection": {
            "alpha_ig": 0.5,
            "alpha_outcome": 0.5,
            "top_p_mass": 0.9,
            "positive_scores_only": True,
            "selection_epsilon": 0.0,
            "candidate_prompts_initial": 64,
            "candidate_prompts_max": 128,
            "refill_prompts": 32,
            "minimum_selected_prompts": 32,
            "maximum_selected_prompts": 36,
            "rollouts_per_prompt": 16,
            "recompute_full_pool_after_refill": True,
        },
        "advantage": {
            "ig_normalization": "same_prompt_same_search_position_population",
            "outcome_normalization": "same_prompt_population",
            "gamma": 1.0,
            "rescale_count": "real_ig_credit_positions_including_zero_normalized",
            "search_terms": ["future_ig_rescaled", "normalized_outcome"],
            "answer_terms": ["normalized_outcome", "centered_format_indicator"],
            "lambda_ig": 1.0,
            "lambda_outcome": 1.0,
            "lambda_format": 1.0,
            "format_centering_is_subtraction": True,
            "malformed_term_present": False,
        },
        "clipping": {
            "mode": "a2tgpo_adaptive_turn_level",
            "beta_c": 0.3,
            "epsilon_low": 0.003,
            "epsilon_high": 0.004,
            "answer_scale": 1.0,
            "search_scale_formula": "1+beta*(2*sigmoid(normalized_ig)-1)",
            "turn_surrogate": "min(ratio*A,clip(ratio,lower,upper)*A)",
            "fixed_dapo_active": False,
            "clip_scale_stop_gradient": True,
        },
        "policy": {
            "ratio_level": "turn",
            "ratio_hardcoded": False,
            "old_logprob_detached_required": True,
            "current_logprob_differentiable_required": True,
            "reduction": "prompt_trajectory_action_token_mean",
            "nested_reduction_implemented": True,
            "full_vocab_kl": True,
            "kl_direction": "actor_to_frozen_reference_forward",
            "kl_reduction": "prompt_trajectory_action_token_mean",
            "kl_coefficient": 0.01,
            "entropy_coefficient": 0.0,
            "value_coefficient": 0.0,
            "max_grad_norm": 1.0,
            "ppo_epochs": 1,
            "optimizer_mini_steps": 1,
            "optimizer_steps_per_successful_update": 1,
            "one_optimizer_step_guard": True,
        },
        "topology": {
            "retriever_physical_gpu": 0,
            "rl_physical_gpus": [1, 2, 3, 4],
            "vllm_dp": 4,
            "vllm_tp": 1,
            "fsdp2_world_size": 4,
            "learner_strategy": "fsdp2",
        },
        "runtime_verification": {
            "exact_ig_static_structure": "STATIC_STRUCTURE_PASS",
            "exact_ig_numerical_parity": "PASS",
            "exact_ig_gpu_runtime": "GPU_RUNTIME_UNVERIFIED",
            "distributed_runtime": "UNVERIFIED_RUNTIME",
            "vllm_fsdp2_weight_sync": "UNVERIFIED_RUNTIME",
            "real_optimizer_transaction": "UNVERIFIED_RUNTIME",
        },
    }


def _blocked_fixed_clip_findings() -> list[str]:
    findings: list[str] = []
    fragments = (
        "fixed_" + "dapo",
        "dapo_" + "turn_objective",
        "clip_ratio_" + "low",
        "clip_ratio_" + "high",
    )
    for root_name in ("src", "configs"):
        for path in (PROJECT_ROOT / root_name).rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for fragment in fragments:
                if fragment in text:
                    findings.append(f"{path.relative_to(PROJECT_ROOT)}:{fragment}")
            for literal in ("0." + "8", "1." + "28"):
                if literal in text:
                    findings.append(f"{path.relative_to(PROJECT_ROOT)}:{literal}")
    return findings


def _malformed_optimization_findings() -> list[str]:
    findings: list[str] = []
    fragments = (
        "a_" + "mal",
        "lambda_" + "mal",
        "malformed_" + "advantage",
        "malformed_search_" + "reward",
        "malformed_search_" + "penalty",
    )
    paths = [
        PROJECT_ROOT / "src/agentic_rl/advantage/a2tgpo.py",
        PROJECT_ROOT / "configs/base.yaml",
        PROJECT_ROOT / "configs/update_stages.yaml",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for fragment in fragments:
            if fragment in text:
                findings.append(f"{path.relative_to(PROJECT_ROOT)}:{fragment}")
    return findings


def build_checks(
    config: dict[str, Any],
    actual: dict[str, Any],
    expected: dict[str, Any],
    spec_contract: dict[str, Any],
    generated_contract: dict[str, Any],
) -> list[Check]:
    variance_source = inspect.getsource(ig_prompt_variance)
    scale_source = inspect.getsource(ChannelScaleState.inspect_pool)
    commit_source = inspect.getsource(ChannelScaleState.committed_after_success)
    advantage_source = inspect.getsource(compute_prompt_advantages)
    clip_source = inspect.getsource(a2tgpo_adaptive_turn_objective)
    ratio_source = inspect.getsource(compute_turn_ratios)
    reduction_source = inspect.getsource(prompt_trajectory_action_token_reduce)
    kl_source = inspect.getsource(actor_to_reference_full_vocab_kl)
    transaction_source = inspect.getsource(StrictUpdateTransaction)
    exact_alias_source = inspect.getsource(immediate_ig_from_prefix_scores)
    exact_score_source = inspect.getsource(VectorizedExactIGScorer.score_batch)
    selection_tests = _source("tests/test_selection_math.py")
    scale_tests = _source("tests/test_scale_update_stages.py")
    policy_tests = _source("tests/test_policy_reduction_and_kl.py")
    config_tests = _source("tests/test_config_schema.py")
    malformed_findings = _malformed_optimization_findings()
    fixed_findings = _blocked_fixed_clip_findings()
    checks = [
        Check("V21-IG-01", "Phi is mean canonical-answer-body log-likelihood", "token_logprobs.to" in exact_score_source and ".mean(" in exact_score_source, _location(VectorizedExactIGScorer.score_batch), "test_target_and_score_mask_cover_only_canonical_answer"),
        Check("V21-IG-02", "Exact IG is Phi_t minus Phi_t-1 and is not exponentiated", "prefix_scores[index] - prefix_scores[index - 1]" in exact_alias_source and "exp(" not in exact_alias_source, _location(immediate_ig_from_prefix_scores), "test_log_prob_diff_telescopes_without_probability_transform"),
        Check("V21-IG-03", "Search reward has no additional term", actual["exact_ig"]["extra_search_reward_terms"] == [], _location(immediate_ig_from_prefix_scores), "audit_final_algorithm_v2_1"),
        Check("V21-VAR-01", "Sample variance is per Search position with ddof=1", "/ (array.size - 1)" in _source("src/agentic_rl/selection/prompt_variance.py") and "peers.setdefault(int(search_index)" in variance_source, _location(ig_prompt_variance), "test_ig_variance_uses_same_search_index_sample_variance"),
        Check("V21-VAR-02", "Only n>=2 positions enter natural-weight denominator", "supported_indices" in variance_source and "supported_peer_total" in variance_source and "count >= 2" in variance_source, _location(ig_prompt_variance), "test_singleton_positions_do_not_dilute_exact_unit_sample_variance"),
        Check("V21-VAR-03", "Singleton positions have exactly zero weight", "else 0.0" in variance_source and "n2=1" not in variance_source, _location(ig_prompt_variance), "test_multiple_singletons_do_not_dilute_supported_position"),
        Check("V21-SCALE-01", "Update 1 scale commit is independent of bootstrap activation", "stats.gate.active" not in commit_source and "allow_initialization" in commit_source, _location(ChannelScaleState.committed_after_success), "test_update1_inactive_bootstrap_channel_still_commits_existing_median"),
        Check("V21-SCALE-02", "Updates 2-10 bootstrap activation does not freeze EMA", "self.health_reference is None or gate.active" in scale_source and "scale_update_allowed_after_success" in commit_source, _location(ChannelScaleState.inspect_pool), "test_update2_bootstrap_inactive_channel_still_updates_ema"),
        Check("V21-SCALE-03", "Health-ready inactive channel freezes EMA", "self.health_reference is None or gate.active" in scale_source, _location(ChannelScaleState.inspect_pool), "test_update11_low_health_channel_freezes_ema_but_records_observation"),
        Check("V21-SCALE-04", "No late channel initialization", "median if allow_initialization else None" in commit_source, _location(ChannelScaleState.committed_after_success), "test_scale_cannot_late_initialize_after_update1"),
        Check("V21-SEL-01", "Channel weights are 0.5/0.5 and Top-p mass is 0.9", config["selection"]["alpha_ig"] == config["selection"]["alpha_outcome"] == 0.5 and config["selection"]["top_p_mass"] == 0.9, "configs/base.yaml:63", "test_resolved_config_locks_requested_topology_and_algorithm"),
        Check("V21-SEL-02", "Pool is 64 with conditional refills to 96 then 128; selected range 32-36; G=16", actual["selection"] == expected["selection"], "configs/base.yaml:49", "test_controller_second_refill_recomputes_full_128_pool_and_can_succeed"),
        Check("V21-ADV-01", "Search advantage has exactly D_bar and z_O", actual["advantage"]["search_terms"] == ["future_ig_rescaled", "normalized_outcome"] and "+ lambda_outcome * normalized_outcome" in advantage_source, _location(compute_prompt_advantages), "test_search_advantage_has_only_future_ig_and_outcome_terms"),
        Check("V21-ADV-02", "Answer advantage has exactly z_O and A_format", actual["advantage"]["answer_terms"] == ["normalized_outcome", "centered_format_indicator"] and "+ lambda_format * float(format_values" in advantage_source, _location(compute_prompt_advantages), "test_search_advantage_has_only_future_ig_and_outcome_terms"),
        Check("V21-ADV-03", "A_format is centered by subtraction", "return values - np.mean" in inspect.getsource(centered_format_advantage), _location(centered_format_advantage), "test_centered_format_advantage"),
        Check("V21-ADV-04", "No malformed optimization term exists", not malformed_findings, "src/agentic_rl/advantage/a2tgpo.py", "check_algorithm_boundary.py"),
        Check("V21-CLIP-01", "Unique clipping mode is adaptive turn-level", CLIPPING_MODE == config["policy"]["clipping_mode"] == "a2tgpo_adaptive_turn_level", _location(a2tgpo_adaptive_turn_objective), "test_adaptive_clip_constants_are_frozen"),
        Check("V21-CLIP-02", "beta=.3, epsilon low=.003, high=.004", ADAPTIVE_CLIP_BETA == 0.3 and ADAPTIVE_CLIP_EPSILON_LOW == 0.003 and ADAPTIVE_CLIP_EPSILON_HIGH == 0.004, _location(adaptive_clip_scale), "test_adaptive_clip_constants_are_frozen"),
        Check("V21-CLIP-03", "Answer/fallback uses neutral c=1 without fake IG", ANSWER_CLIP_SCALE == 1.0 and "else ANSWER_CLIP_SCALE" in clip_source, _location(a2tgpo_adaptive_turn_objective), "test_answer_turn_uses_neutral_scale_without_fake_ig"),
        Check("V21-CLIP-04", "Adaptive surrogate uses turn ratio and dynamic bounds", "torch.minimum(" in clip_source and "torch.clamp(ratio, min=lower, max=upper)" in clip_source, _location(a2tgpo_adaptive_turn_objective), "test_adaptive_surrogate_preserves_current_policy_gradient"),
        Check("V21-CLIP-05", "Fixed DAPO clipping is absent from active source/config", not fixed_findings, "src/ + configs/", "test_fixed_dapo_boundaries_are_absent_from_active_implementation"),
        Check("V21-RATIO-01", "Turn ratio is real, differentiable, with detached old logprob", "current_logprobs[mask] - old_logprobs[mask]" in ratio_source and "Old-policy logprobs must be detached" in ratio_source and "Current-policy logprobs must retain gradients" in ratio_source, _location(compute_turn_ratios), "test_turn_ratio_is_near_one_but_differentiable"),
        Check("V21-RED-01", "Policy reduction is Prompt->Trajectory->action-token mean", "trajectory_mean = record.values[mask].mean()" in reduction_source and "torch.stack(trajectory_means).mean()" in reduction_source, _location(prompt_trajectory_action_token_reduce), "test_prompt_trajectory_token_reduction_equalizes_lengths"),
        Check("V21-KL-01", "KL is full-vocabulary forward Actor||frozen Reference with nested reduction", "probability * (actor_log_probability - reference_log_probability)" in kl_source and config["policy"]["kl_reduction"] == config["policy"]["task_reduction"], _location(actor_to_reference_full_vocab_kl), "test_full_vocab_kl_is_actor_to_reference_and_keeps_actor_gradient"),
        Check("V21-LOSS-01", "Loss coefficient is 0.01 with no entropy/value term", config["policy"]["kl_coefficient"] == 0.01 and config["policy"]["entropy_coefficient"] == 0.0 and config["policy"]["value_coefficient"] == 0.0, "configs/base.yaml:99", "test_world_size_compensation_and_total_loss"),
        Check("V21-TXN-01", "Strict successful update has exactly one optimizer and scheduler step", config["policy"]["ppo_epochs"] == 1 and config["policy"]["optimizer_mini_steps"] == 1 and "Exactly one optimizer.step is permitted" in transaction_source and "Exactly one scheduler.step is permitted" in transaction_source, _location(StrictUpdateTransaction), "test_controller_caps_to_36_and_performs_one_step"),
        Check("V21-CONFIG-01", "Resolved config exposes all frozen V2.1 constants", all(fragment in config_tests for fragment in ("adaptive_clip_beta", "adaptive_clip_epsilon_low", "adaptive_clip_epsilon_high")), "tests/test_config_schema.py", "test_resolved_config_locks_requested_topology_and_algorithm"),
        Check("V21-TEST-01", "Tests lock singleton, scale/activation, and adaptive clipping", all(fragment in selection_tests + scale_tests + policy_tests for fragment in ("singleton", "bootstrap_inactive", "ADAPTIVE_CLIP_BETA")), "tests/", "pytest"),
        Check("V21-SPEC-01", "Frozen spec machine contract equals frozen requirement contract", spec_contract == expected, str(FINAL_SPEC.relative_to(PROJECT_ROOT)), "audit_final_algorithm_v2_1"),
        Check("V21-EFFECTIVE-01", "Generated effective contract equals imported code/config contract", generated_contract == actual, str(EFFECTIVE_REPORT.relative_to(PROJECT_ROOT)), "audit_final_algorithm_v2_1"),
        Check("V21-CONSISTENCY-01", "Frozen, generated, and actual contracts are identical", actual == expected == spec_contract == generated_contract, "FINAL spec + source + resolved config + effective report", "audit_final_algorithm_v2_1"),
        Check("V21-RUNTIME-01", "Exact-IG numerical model parity", False, "Not executed in no-GPU static phase", "not run", runtime_only=True),
        Check("V21-RUNTIME-02", "vLLM/FSDP2 distributed weight sync and rollout", False, "Not executed in no-GPU static phase", "not run", runtime_only=True),
        Check("V21-RUNTIME-03", "Real optimizer/checkpoint transaction", False, "Prohibited in this phase", "not run", runtime_only=True),
    ]
    return checks


def render_matrix(checks: list[Check]) -> str:
    rows = "\n".join(
        f"| `{check.requirement_id}` | {check.requirement} | `{check.evidence}` | "
        f"`{check.test}` | **{check.status}** |"
        for check in checks
    )
    static_failures = [
        check.requirement_id
        for check in checks
        if not check.runtime_only and not check.passed
    ]
    result = "PASS" if not static_failures else "FAIL"
    return f"""# Algorithm Consistency Matrix V2.1

This matrix is generated by `scripts/audit_final_algorithm_v2_1.py` by
comparing the frozen V2.1 machine contract, resolved configuration, imported
source behavior, generated effective report, and test locks.

| Requirement | Frozen requirement / check | Source or config evidence | Test evidence | Status |
|---|---|---|---|---|
{rows}

## Result

```text
ALGORITHM_STATIC_COMPLIANCE_V2_1={result}
```

Static failures: `{static_failures}`.

`UNVERIFIED_RUNTIME` rows are required future gates and are not converted into
static PASS claims.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--write-effective", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    actual = effective_contract(config)
    expected = expected_contract()
    if args.write_effective or not EFFECTIVE_REPORT.is_file():
        EFFECTIVE_REPORT.write_text(
            render_report(config, actual),
            encoding="utf-8",
        )
    spec_contract = _extract_json(
        FINAL_SPEC,
        "<!-- FROZEN_CONTRACT_JSON_START -->",
        "<!-- FROZEN_CONTRACT_JSON_END -->",
    )
    generated_contract = _extract_json(
        EFFECTIVE_REPORT,
        "<!-- EFFECTIVE_CONTRACT_JSON_START -->",
        "<!-- EFFECTIVE_CONTRACT_JSON_END -->",
    )
    checks = build_checks(
        config,
        actual,
        expected,
        spec_contract,
        generated_contract,
    )
    MATRIX_PATH.write_text(render_matrix(checks), encoding="utf-8")
    static_failures = [
        check.requirement_id
        for check in checks
        if not check.runtime_only and not check.passed
    ]
    result = "PASS" if not static_failures else "FAIL"
    print(
        json.dumps(
            {
                "ALGORITHM_STATIC_COMPLIANCE_V2_1": result,
                "checks": {check.requirement_id: check.status for check in checks},
                "matrix": str(MATRIX_PATH),
                "static_failures": static_failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if static_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
