from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from agentic_rl.advantage.stop_continue import (
    NORMALIZED_OUTCOME_MODE,
    STOP_CONTINUE_CONSENSUS_MODE,
    StopContinueAdvantage,
    compute_stop_continue_advantages,
    stop_continue_reward_from_mapping,
)


from agentic_rl.outcome.format_indicator import centered_format_advantage
from agentic_rl.advantage.role_localized_gate import (
    build_role_localized_trajectory_credits,
)
from agentic_rl.advantage.mica_ig import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
    compute_mica_search_advantage,
)
from agentic_rl.outcome.workers import SUFFICIENCY_EXACT_SCORER_VERSION
from agentic_rl.rollout.trajectory_schema import (
    TurnType,
    is_budget_exhausted_terminal_search,
)


SEARCH_IG_COEFFICIENT = 0.3
SUFFICIENCY_NOVELTY_LOCAL_IG_MODE = "sufficiency_novelty_local_ig"
SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE = (
    "sufficiency_novelty_cumulative_ig_probe_routed_outcome"
)
SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE = (
    "sufficiency_novelty_cumulative_ig_probe_routed_outcome_role_localized_gate"
)


@dataclass(frozen=True)
class TrajectoryCreditInput:
    """Channel-specific validity and credit inputs for one trajectory."""

    immediate_ig: Mapping[int, float]
    search_turn_indices: tuple[int, ...]
    ig_reward_eligible: Mapping[int, bool]
    policy_credit_eligible: Mapping[int, bool]
    outcome: float
    outcome_reward_eligible: bool
    format_indicator: int
    answer_policy_credit_eligible: bool
    trajectory_system_valid: bool = True

    def validate(self) -> None:
        indices = tuple(int(index) for index in self.search_turn_indices)
        if indices != tuple(sorted(set(indices))) or any(index < 0 for index in indices):
            raise ValueError("search_turn_indices must be sorted, unique, and non-negative")
        expected = set(indices)
        if set(map(int, self.ig_reward_eligible)) != expected:
            raise ValueError("ig_reward_eligible must cover every real Search position")
        if set(map(int, self.policy_credit_eligible)) != expected:
            raise ValueError(
                "policy_credit_eligible must cover every real Search position"
            )
        eligible = {
            int(index)
            for index, value in self.ig_reward_eligible.items()
            if bool(value)
        }
        if set(map(int, self.immediate_ig)) != eligible:
            raise ValueError(
                "Immediate IG must exist exactly for IG-reward-eligible Search turns"
            )
        if self.format_indicator not in {0, 1}:
            raise ValueError("format_indicator must be binary")
        if not self.trajectory_system_valid:
            if eligible or self.outcome_reward_eligible:
                raise ValueError(
                    "System-invalid trajectories cannot enter reward channels"
                )
            if self.answer_policy_credit_eligible or any(
                bool(value) for value in self.policy_credit_eligible.values()
            ):
                raise ValueError(
                    "System-invalid trajectories cannot carry policy credit"
                )

        # Once a prefix is unavailable, later Search prefixes cannot be
        # reconstructed from the trajectory and therefore cannot regain IG.
        prefix_chain_open = True
        for index in indices:
            current = bool(self.ig_reward_eligible[index])
            if current and not prefix_chain_open:
                raise ValueError(
                    "IG eligibility cannot resume after a missing Search prefix"
                )
            prefix_chain_open = prefix_chain_open and current


def trajectory_credit_input_from_record(record: Any) -> TrajectoryCreditInput:
    """Build the advantage input from the explicit trajectory validity fields."""
    search_eligibility = record.ig_reward_eligibility_by_search_index
    policy_eligibility = record.policy_credit_eligibility_by_search_index
    return TrajectoryCreditInput(
        immediate_ig=dict(record.immediate_ig),
        search_turn_indices=tuple(sorted(search_eligibility)),
        ig_reward_eligible=search_eligibility,
        policy_credit_eligible=policy_eligibility,
        outcome=float(record.task_outcome),
        outcome_reward_eligible=bool(record.outcome_reward_eligible),
        format_indicator=int(record.answer_format_indicator),
        answer_policy_credit_eligible=(
            record.terminal_policy_credit_turn_index is not None
        ),
        trajectory_system_valid=bool(record.trajectory_system_valid),
    )


@dataclass(frozen=True)
class TrajectoryAdvantage:
    normalized_ig: dict[int, float]
    future_ig_sum: dict[int, float]
    accumulated_ig_count: dict[int, int]
    future_ig_rescaled: dict[int, float]
    normalized_outcome: float
    centered_format_indicator: float
    search_advantage: dict[int, float]
    answer_advantage: float | None
    search_policy_credit_eligible: dict[int, bool]
    answer_policy_credit_eligible: bool
    search_advantage_old_shadow: dict[int, float] = field(default_factory=dict)
    search_task_advantage: dict[int, float] = field(default_factory=dict)
    stop_continue_by_search_index: dict[int, StopContinueAdvantage] = field(
        default_factory=dict
    )
    search_task_mode: str = NORMALIZED_OUTCOME_MODE
    sufficient_before_search: dict[int, bool] = field(default_factory=dict)
    sufficient_after_search: dict[int, bool] = field(default_factory=dict)
    no_new_observation: dict[int, bool] = field(default_factory=dict)
    search_branch_by_search_index: dict[int, str] = field(default_factory=dict)
    effective_cumulative_ig: dict[int, float] = field(default_factory=dict)
    effective_cumulative_ig_count: dict[int, int] = field(default_factory=dict)
    probe_reward_delta: dict[int, float] = field(default_factory=dict)
    routed_outcome: dict[int, float] = field(default_factory=dict)
    search_main_advantage: dict[int, float] = field(default_factory=dict)
    search_decision_advantage: dict[int, float] = field(default_factory=dict)
    search_query_advantage: dict[int, float] = field(default_factory=dict)
    mica_ig_return: dict[int, float] = field(default_factory=dict)
    mica_peer_count: dict[int, int] = field(default_factory=dict)
    mica_loc_mean: dict[int, float] = field(default_factory=dict)
    mica_loc_std: dict[int, float] = field(default_factory=dict)
    mica_ret_mean: dict[int, float] = field(default_factory=dict)
    mica_ret_std: dict[int, float] = field(default_factory=dict)
    mica_local_advantage: dict[int, float] = field(default_factory=dict)
    mica_return_advantage: dict[int, float] = field(default_factory=dict)
    mica_singleton_fallback: dict[int, bool] = field(default_factory=dict)
    mica_ig_missing_reason: dict[int, str] = field(default_factory=dict)
    mica_singleton_tail_start_depth: int | None = None
    mica_singleton_consecutive_length: int = 0


@dataclass(frozen=True)
class A2TGPOPromptResult:
    trajectories: tuple[TrajectoryAdvantage, ...]
    ig_mean_by_search_index: dict[int, float]
    ig_std_by_search_index: dict[int, float]
    outcome_mean: float
    outcome_std: float
    format_mean: float


def _population_stats(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError("Normalization inputs must be finite")
    return (
        float(np.mean(array, dtype=np.float64)),
        float(np.std(array, ddof=0, dtype=np.float64)),
    )


def compute_prompt_advantages(
    trajectories: Sequence[TrajectoryCreditInput],
    *,
    gamma: float = 1.0,
    lambda_ig: float | None = SEARCH_IG_COEFFICIENT,
    lambda_outcome: float = 1.0,
    lambda_format: float = 1.0,
    normalization_epsilon: float = 1.0e-6,
    zero_variance_tolerance: float = 1.0e-12,
    accumulate_future_ig: bool = True,
) -> A2TGPOPromptResult:
    """Compute same-prompt A-squared-TGPO advantages.

    Malformed diagnostics are not inputs. They therefore cannot create a third
    advantage term or implicitly zero otherwise eligible policy credit.
    """
    if not trajectories:
        raise ValueError("A prompt group cannot be empty")
    if gamma != 1.0:
        raise ValueError("This deployment locks gamma=1.0")
    if normalization_epsilon <= 0 or zero_variance_tolerance < 0:
        raise ValueError("Normalization tolerances are invalid")
    if accumulate_future_ig and lambda_ig is None:
        raise ValueError("Legacy future-IG mode requires lambda_ig")
    for trajectory in trajectories:
        trajectory.validate()

    trajectory_count = len(trajectories)
    normalized_ig: list[dict[int, float]] = [
        {} for _ in range(trajectory_count)
    ]
    ig_means: dict[int, float] = {}
    ig_stds: dict[int, float] = {}
    all_search_indices = sorted(
        {
            index
            for trajectory in trajectories
            for index in trajectory.search_turn_indices
        }
    )
    for search_index in all_search_indices:
        peers = [
            (trajectory_index, float(trajectory.immediate_ig[search_index]))
            for trajectory_index, trajectory in enumerate(trajectories)
            if trajectory.ig_reward_eligible.get(search_index, False)
        ]
        mean, std = _population_stats([value for _, value in peers])
        ig_means[search_index] = mean
        ig_stds[search_index] = std
        supported = len(peers) >= 2
        for trajectory_index, value in peers:
            normalized_ig[trajectory_index][search_index] = (
                0.0
                if not supported or std * std <= zero_variance_tolerance
                else (value - mean) / (std + normalization_epsilon)
            )

    valid_outcomes = [
        float(trajectory.outcome)
        for trajectory in trajectories
        if trajectory.outcome_reward_eligible
    ]
    outcome_mean, outcome_std = _population_stats(valid_outcomes)
    normalized_outcome = [
        (
            0.0
            if (
                not trajectory.outcome_reward_eligible
                or outcome_std * outcome_std <= zero_variance_tolerance
            )
            else (float(trajectory.outcome) - outcome_mean)
            / (outcome_std + normalization_epsilon)
        )
        for trajectory in trajectories
    ]
    format_indicators = [trajectory.format_indicator for trajectory in trajectories]
    format_values = centered_format_advantage(format_indicators)
    format_mean = float(
        np.mean(np.asarray(format_indicators, dtype=np.float64))
    )

    results: list[TrajectoryAdvantage] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        future_sum: dict[int, float] = {}
        future_count: dict[int, int] = {}
        future_rescaled: dict[int, float] = {}
        search_advantage: dict[int, float] = {}
        for search_index in trajectory.search_turn_indices:
            if not trajectory.policy_credit_eligible[search_index]:
                continue
            if not accumulate_future_ig:
                # New production mode uses the current turn's normalized local
                # IG directly.  Missing/unsupported IG peers fail closed to 0.
                search_advantage[search_index] = float(
                    normalized_ig[trajectory_index].get(search_index, 0.0)
                )
                continue
            future_indices = [
                index
                for index in trajectory.search_turn_indices
                if index >= search_index
                and trajectory.ig_reward_eligible[index]
            ]
            total = float(
                math.fsum(
                    normalized_ig[trajectory_index][index]
                    for index in future_indices
                )
            )
            # Every real, eligible IG position counts even when its normalized
            # value is exactly zero due to zero peer variance.
            count = len(future_indices)
            rescaled = total / math.sqrt(count) if count else 0.0
            future_sum[search_index] = total
            future_count[search_index] = count
            future_rescaled[search_index] = rescaled
            search_advantage[search_index] = (
                float(lambda_ig) * rescaled
                + lambda_outcome * normalized_outcome[trajectory_index]
            )

        answer_advantage = (
            lambda_outcome * normalized_outcome[trajectory_index]
            + lambda_format * float(format_values[trajectory_index])
            if trajectory.answer_policy_credit_eligible
            else None
        )
        results.append(
            TrajectoryAdvantage(
                normalized_ig=normalized_ig[trajectory_index],
                future_ig_sum=future_sum,
                accumulated_ig_count=future_count,
                future_ig_rescaled=future_rescaled,
                normalized_outcome=normalized_outcome[trajectory_index],
                centered_format_indicator=float(format_values[trajectory_index]),
                search_advantage=search_advantage,
                answer_advantage=answer_advantage,
                search_policy_credit_eligible={
                    index: bool(trajectory.policy_credit_eligible[index])
                    for index in trajectory.search_turn_indices
                },
                answer_policy_credit_eligible=bool(
                    trajectory.answer_policy_credit_eligible
                ),
                search_advantage_old_shadow=dict(search_advantage),
                search_task_advantage={
                    index: float(normalized_outcome[trajectory_index])
                    for index in search_advantage
                },
                search_task_mode=NORMALIZED_OUTCOME_MODE,
            )
        )

    return A2TGPOPromptResult(
        trajectories=tuple(results),
        ig_mean_by_search_index=ig_means,
        ig_std_by_search_index=ig_stds,
        outcome_mean=outcome_mean,
        outcome_std=outcome_std,
        format_mean=format_mean,
    )


def _rebuild_sufficiency_novelty_local_ig(
    records: Sequence[Any],
    prompt_result: A2TGPOPromptResult,
    *,
    expected_policy_version: int | None,
    expected_scorer_version: str | None,
) -> tuple[A2TGPOPromptResult, Mapping[str, float | int]]:
    """Apply the locked S/N/local-IG Search formula to one Prompt group."""

    rebuilt: list[TrajectoryAdvantage] = []
    local_ig_values: list[float] = []
    local_ig_hat_values: list[float] = []
    a_search_values: list[float] = []
    sufficient_count = 0
    no_new_count = 0
    sufficient_and_no_new_count = 0
    normal_count = 0
    query_repeat_count = 0
    different_query_no_new_count = 0
    state_count = 0
    old_answer_advantages = tuple(
        advantage.answer_advantage for advantage in prompt_result.trajectories
    )

    for record, advantage in zip(
        records,
        prompt_result.trajectories,
        strict=True,
    ):
        search_indices = set(advantage.search_advantage)
        if advantage.future_ig_sum or advantage.accumulated_ig_count or advantage.future_ig_rescaled:
            raise RuntimeError(
                f"{record.trajectory_id}: production local-IG path accumulated future IG"
            )
        probes = record.metadata.get("sufficiency_probes", {})
        if search_indices and not isinstance(probes, Mapping):
            raise ValueError(
                f"{record.trajectory_id}: selected Search has no sufficiency probes"
            )
        turns_by_search = {
            int(turn.search_index): turn
            for turn in record.turns
            if turn.turn_type is TurnType.SEARCH
            and turn.search_index is not None
            and turn.policy_credit_eligible
        }
        if set(turns_by_search) != search_indices:
            raise ValueError(
                f"{record.trajectory_id}: Search turn/advantage coverage mismatch"
            )

        new_search: dict[int, float] = {}
        sufficient_by_search: dict[int, bool] = {}
        no_new_by_search: dict[int, bool] = {}
        branch_by_search: dict[int, str] = {}
        for search_index in sorted(search_indices):
            raw_probe = probes.get(search_index, probes.get(str(search_index)))
            if not isinstance(raw_probe, Mapping):
                raise ValueError(
                    f"{record.trajectory_id}:{search_index}: sufficiency probe is missing"
                )
            sufficient = raw_probe.get("sufficient_before_search")
            if not isinstance(sufficient, bool):
                raise ValueError("S must be a binary bool")
            if int(raw_probe.get("completion_count", -1)) != 1:
                raise ValueError("Sufficiency probe must have exactly one completion")
            if raw_probe.get("do_sample") is not False:
                raise ValueError("Sufficiency probe must use do_sample=false")
            if float(raw_probe.get("temperature", -1.0)) != 0.0:
                raise ValueError("Sufficiency probe temperature must be zero")
            if float(raw_probe.get("top_p", -1.0)) != 1.0:
                raise ValueError("Sufficiency probe top_p must be one")
            if int(raw_probe.get("n", -1)) != 1:
                raise ValueError("Sufficiency probe n must be one")
            if raw_probe.get("detached") is not True:
                raise ValueError("Sufficiency probe must be detached")
            if raw_probe.get("prefix_provenance_valid") is not True:
                raise ValueError("Sufficiency prefix provenance is invalid")
            if raw_probe.get("context_truncated") is not False:
                raise ValueError("Sufficiency probe input context was truncated")
            if str(raw_probe.get("scorer_version")) != SUFFICIENCY_EXACT_SCORER_VERSION:
                raise ValueError("Sufficiency exact scorer version mismatch")
            if expected_scorer_version is not None and str(
                raw_probe.get("task_scorer_version")
            ) != str(expected_scorer_version):
                raise ValueError("Sufficiency shadow task scorer version mismatch")
            if expected_policy_version is not None:
                versions = {
                    int(raw_probe[name])
                    for name in (
                        "candidate_rollout_policy_version",
                        "exact_ig_policy_version",
                        "sufficiency_probe_policy_version",
                        "old_logprob_policy_version",
                    )
                }
                if versions != {int(expected_policy_version)}:
                    raise ValueError("Sufficiency/on-policy versions disagree")

            turn = turns_by_search[search_index]
            no_new = turn.no_new_observation
            if not isinstance(no_new, bool):
                raise ValueError("N must be a binary bool")
            local_ig_hat = float(advantage.normalized_ig.get(search_index, 0.0))
            if sufficient:
                actual = -1.0
                branch = "sufficient_before_search"
            elif no_new:
                actual = -1.0
                branch = "no_new_observation"
            else:
                actual = local_ig_hat
                branch = "normalized_local_ig"
                normal_count += 1
            expected = (
                -1.0
                if sufficient
                else -1.0
                if no_new
                else local_ig_hat
            )
            if not math.isfinite(actual) or not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError("Search advantage formula assertion failed")
            new_search[search_index] = actual
            sufficient_by_search[search_index] = sufficient
            no_new_by_search[search_index] = no_new
            branch_by_search[search_index] = branch
            state_count += 1
            sufficient_count += int(sufficient)
            no_new_count += int(no_new)
            sufficient_and_no_new_count += int(sufficient and no_new)
            query_repeat_count += int(turn.exact_query_repeat)
            different_query_no_new_count += int(
                turn.different_query_no_new_passage
            )
            if search_index in record.immediate_ig:
                local_ig_values.append(float(record.immediate_ig[search_index]))
            local_ig_hat_values.append(local_ig_hat)
            a_search_values.append(actual)

        rebuilt.append(
            replace(
                advantage,
                search_advantage=new_search,
                search_advantage_old_shadow={},
                search_task_advantage={},
                stop_continue_by_search_index={},
                search_task_mode=SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
                sufficient_before_search=sufficient_by_search,
                no_new_observation=no_new_by_search,
                search_branch_by_search_index=branch_by_search,
            )
        )

    result = replace(prompt_result, trajectories=tuple(rebuilt))
    if tuple(
        advantage.answer_advantage for advantage in result.trajectories
    ) != old_answer_advantages:
        raise RuntimeError("S/N/local-IG reconstruction changed A_answer")

    def stats(prefix: str, values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
            f"{prefix}_std": float(array.std(ddof=0)) if array.size else 0.0,
        }

    metrics: dict[str, float | int] = {
        "search/state_count": state_count,
        "search/sufficient_before_search_count": sufficient_count,
        "search/no_new_observation_count": no_new_count,
        "search/sufficient_and_no_new_count": sufficient_and_no_new_count,
        "search/normal_local_ig_branch_count": normal_count,
        "search/exact_query_repeat_count": query_repeat_count,
        "search/different_query_no_new_passage_count": (
            different_query_no_new_count
        ),
        "search/z_o_actor_entry_count": 0,
        "search/a_sc_actor_entry_count": 0,
        "search/future_ig_contribution_count": 0,
        "search/sqrt_n_rescale_call_count": 0,
        "search/external_ig_multiplier_call_count": 0,
    }
    metrics["search/sufficient_before_search_rate"] = (
        sufficient_count / state_count if state_count else 0.0
    )
    metrics["search/no_new_observation_rate"] = (
        no_new_count / state_count if state_count else 0.0
    )
    metrics["search/normal_local_ig_branch_rate"] = (
        normal_count / state_count if state_count else 0.0
    )
    metrics["search/exact_query_repeat_rate"] = (
        query_repeat_count / state_count if state_count else 0.0
    )
    metrics["search/different_query_no_new_passage_rate"] = (
        different_query_no_new_count / state_count if state_count else 0.0
    )
    metrics.update(stats("search/local_ig", local_ig_values))
    metrics.update(stats("search/local_ig_hat", local_ig_hat_values))
    metrics.update(stats("search/A_search", a_search_values))
    return result, metrics


def _validated_probe_state(
    raw_probe: Mapping[str, Any],
    *,
    stage: str,
    expected_policy_version: int | None,
    expected_scorer_version: str | None,
) -> tuple[bool, float]:
    """Recompute one Probe sufficiency bit from persisted raw fields."""

    if stage not in {"pre", "post"}:
        raise ValueError(f"Unsupported Answer Probe stage: {stage}")
    bool_fields = (
        "parser_success",
        "no_answer",
        "output_truncated",
        "alias_aware_exact",
        "prefix_provenance_valid",
        "detached",
    )
    for field_name in bool_fields:
        if not isinstance(raw_probe.get(field_name), bool):
            raise ValueError(f"{stage} Probe {field_name} must be a bool")
    if raw_probe["prefix_provenance_valid"] is not True:
        raise ValueError(f"{stage} Probe prefix provenance is invalid")
    if raw_probe["detached"] is not True:
        raise ValueError(f"{stage} Probe must be detached")
    if not isinstance(raw_probe.get("raw_answer_text"), str):
        raise ValueError(f"{stage} Probe raw_answer_text must be a string")
    reward = float(raw_probe.get("raw_task_reward", float("nan")))
    if not math.isfinite(reward):
        raise ValueError(f"{stage} Probe raw task reward is non-finite")
    if int(raw_probe.get("completion_count", -1)) != 1:
        raise ValueError(f"{stage} Probe must have exactly one completion")
    locked_generation = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "n": 1,
        "max_tokens": 500,
        "stop": ["</answer>"],
    }
    for field_name, expected in locked_generation.items():
        actual = raw_probe.get(field_name)
        if field_name == "stop":
            actual = list(actual) if isinstance(actual, (list, tuple)) else actual
        if actual != expected:
            raise ValueError(
                f"{stage} Probe {field_name}={actual!r}, expected {expected!r}"
            )
    if str(raw_probe.get("scorer_version")) != SUFFICIENCY_EXACT_SCORER_VERSION:
        raise ValueError(f"{stage} Probe exact scorer version mismatch")
    if expected_scorer_version is not None and str(
        raw_probe.get("task_scorer_version")
    ) != str(expected_scorer_version):
        raise ValueError(f"{stage} Probe production task scorer version mismatch")
    if expected_policy_version is not None:
        version_fields = (
            "candidate_rollout_policy_version",
            "exact_ig_policy_version",
            "probe_policy_version",
            "old_logprob_policy_version",
        )
        try:
            versions = {int(raw_probe[name]) for name in version_fields}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{stage} Probe policy version metadata is invalid") from error
        if versions != {int(expected_policy_version)}:
            raise ValueError(f"{stage} Probe/on-policy versions disagree")

    sufficient = bool(
        raw_probe["alias_aware_exact"]
        and raw_probe["parser_success"]
        and not raw_probe["no_answer"]
        and not raw_probe["output_truncated"]
    )
    precomputed_name = (
        "sufficient_before_search" if stage == "pre" else "sufficient_after_search"
    )
    if raw_probe.get(precomputed_name) is not sufficient:
        raise ValueError(
            f"{stage} Probe precomputed sufficiency differs from raw fields"
        )
    return sufficient, reward


def _rebuild_sufficiency_novelty_cumulative_ig_probe_routed_outcome(
    records: Sequence[Any],
    prompt_result: A2TGPOPromptResult,
    *,
    expected_policy_version: int | None,
    expected_scorer_version: str | None,
    probe_epsilon: float,
) -> tuple[A2TGPOPromptResult, Mapping[str, float | int]]:
    """Apply S/N-gated cumulative IG with Probe-routed Outcome credit."""

    if not math.isclose(float(probe_epsilon), 1.0e-6, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Probe-routed Outcome locks probe_epsilon=1e-6")
    rebuilt: list[TrajectoryAdvantage] = []
    old_answers = tuple(
        advantage.answer_advantage for advantage in prompt_result.trajectories
    )
    state_count = 0
    s_before_count = 0
    s_after_count = 0
    post_probe_count = 0
    no_new_count = 0
    normal_count = 0
    s_and_n_count = 0
    truncated_accumulation_count = 0
    n_masked_future_continued_count = 0
    route_positive_count = 0
    route_zero_count = 0
    route_negative_count = 0
    delta_positive_count = 0
    delta_zero_count = 0
    delta_negative_count = 0
    z_o_normal_entry_count = 0
    budget_exhausted_count = 0
    budget_exhausted_post_probe_count = 0
    budget_exhausted_o_route_nonzero_count = 0
    budget_exhausted_ig_entry_count = 0
    budget_exhausted_a_search_not_minus_one_count = 0
    normal_search_missing_post_prefix_count = 0
    local_ig_values: list[float] = []
    local_ig_hat_values: list[float] = []
    cumulative_values: list[float] = []
    cumulative_counts: list[int] = []
    delta_values: list[float] = []
    route_values: list[float] = []
    search_values: list[float] = []
    cumulative_by_index: dict[int, list[float]] = {}

    for record, advantage in zip(records, prompt_result.trajectories, strict=True):
        if (
            advantage.future_ig_sum
            or advantage.accumulated_ig_count
            or advantage.future_ig_rescaled
        ):
            raise RuntimeError(
                f"{record.trajectory_id}: new mode reused legacy future-IG state"
            )
        if advantage.stop_continue_by_search_index:
            raise RuntimeError(
                f"{record.trajectory_id}: A_SC entered Probe-routed Search credit"
            )
        all_turns = {
            int(turn.search_index): turn
            for turn in record.turns
            if turn.turn_type is TurnType.SEARCH and turn.search_index is not None
        }
        if tuple(sorted(all_turns)) != tuple(range(len(all_turns))):
            raise ValueError(f"{record.trajectory_id}: Search indices are not contiguous")
        optimized_indices = set(advantage.search_advantage)
        if not optimized_indices.issubset(all_turns):
            raise ValueError(f"{record.trajectory_id}: optimized Search is not real")
        raw_by_search = record.metadata.get("routed_answer_probes", {})
        if all_turns and not isinstance(raw_by_search, Mapping):
            raise ValueError(f"{record.trajectory_id}: routed Answer Probes are missing")

        s_before: dict[int, bool] = {}
        s_after: dict[int, bool] = {}
        pre_rewards: dict[int, float] = {}
        post_rewards: dict[int, float] = {}
        no_new_by_search: dict[int, bool] = {}
        budget_exhausted_by_search: dict[int, bool] = {}
        for search_index, turn in sorted(all_turns.items()):
            raw_stages = raw_by_search.get(
                search_index,
                raw_by_search.get(str(search_index)),
            )
            if not isinstance(raw_stages, Mapping):
                raise ValueError(
                    f"{record.trajectory_id}:{search_index}: Probe stages are missing"
                )
            raw_pre = raw_stages.get("pre")
            if not isinstance(raw_pre, Mapping):
                raise ValueError(
                    f"{record.trajectory_id}:{search_index}: pre Probe is missing"
                )
            before, pre_reward = _validated_probe_state(
                raw_pre,
                stage="pre",
                expected_policy_version=expected_policy_version,
                expected_scorer_version=expected_scorer_version,
            )
            s_before[search_index] = before
            pre_rewards[search_index] = pre_reward
            if not isinstance(turn.no_new_observation, bool):
                raise ValueError(
                    f"{record.trajectory_id}:{search_index}: N must be a bool"
                )
            no_new_by_search[search_index] = bool(turn.no_new_observation)
            budget_exhausted = is_budget_exhausted_terminal_search(
                record,
                search_index,
            )
            budget_exhausted_by_search[search_index] = budget_exhausted
            budget_exhausted_count += int(budget_exhausted)
            raw_post = raw_stages.get("post")
            if budget_exhausted:
                if raw_post is not None:
                    budget_exhausted_post_probe_count += 1
                    raise ValueError(
                        f"{record.trajectory_id}:{search_index}: budget-exhausted "
                        "Search cannot carry a post Probe"
                    )
                if not no_new_by_search[search_index]:
                    raise ValueError(
                        f"{record.trajectory_id}:{search_index}: budget-exhausted "
                        "Search must have N=True"
                    )
            elif before:
                if raw_post is not None:
                    raise ValueError(
                        f"{record.trajectory_id}:{search_index}: post Probe must not run "
                        "after pre-search sufficiency"
                    )
            else:
                if not isinstance(raw_post, Mapping):
                    normal_search_missing_post_prefix_count += 1
                    raise ValueError(
                        f"{record.trajectory_id}:{search_index}: post Probe is missing"
                    )
                after, post_reward = _validated_probe_state(
                    raw_post,
                    stage="post",
                    expected_policy_version=expected_policy_version,
                    expected_scorer_version=expected_scorer_version,
                )
                s_after[search_index] = after
                post_rewards[search_index] = post_reward
                post_probe_count += 1
                s_after_count += int(after)

        new_search: dict[int, float] = {}
        branch_by_search: dict[int, str] = {}
        d_by_search: dict[int, float] = {}
        d_count_by_search: dict[int, int] = {}
        delta_by_search: dict[int, float] = {}
        route_by_search: dict[int, float] = {}
        for search_index in sorted(optimized_indices):
            sufficient = s_before[search_index]
            no_new = no_new_by_search[search_index]
            state_count += 1
            s_before_count += int(sufficient)
            no_new_count += int(no_new)
            s_and_n_count += int(sufficient and no_new)
            if search_index in record.immediate_ig:
                local_ig_values.append(float(record.immediate_ig[search_index]))
            local_ig_hat = float(advantage.normalized_ig.get(search_index, 0.0))
            local_ig_hat_values.append(local_ig_hat)
            budget_exhausted = budget_exhausted_by_search[search_index]
            if budget_exhausted and (
                search_index in record.immediate_ig
                or search_index in advantage.normalized_ig
            ):
                budget_exhausted_ig_entry_count += 1
                raise RuntimeError(
                    f"{record.trajectory_id}:{search_index}: budget-exhausted "
                    "Search entered Exact-IG credit"
                )
            if sufficient:
                actual = -1.0
                branch = "sufficient_before_search"
            elif no_new:
                actual = -1.0
                branch = "no_new_observation"
            else:
                current_turn = all_turns[search_index]
                if not (
                    current_turn.policy_credit_eligible
                    and current_turn.ig_reward_eligible
                    and search_index in advantage.normalized_ig
                ):
                    raise ValueError(
                        f"{record.trajectory_id}:{search_index}: current Normal Search "
                        "is not IG-credit eligible"
                    )
                values: list[float] = []
                encountered_masked_n = False
                continued_after_masked_n = False
                for future_index in sorted(
                    index for index in all_turns if index >= search_index
                ):
                    if s_before[future_index]:
                        break
                    future_turn = all_turns[future_index]
                    valid_ig = bool(
                        future_turn.policy_credit_eligible
                        and future_turn.ig_reward_eligible
                        and not no_new_by_search[future_index]
                    )
                    if valid_ig:
                        if future_index not in advantage.normalized_ig:
                            raise ValueError(
                                f"{record.trajectory_id}:{future_index}: normalized "
                                "local IG is missing"
                            )
                        values.append(float(advantage.normalized_ig[future_index]))
                        continued_after_masked_n = (
                            continued_after_masked_n or encountered_masked_n
                        )
                    elif no_new_by_search[future_index]:
                        encountered_masked_n = True
                    if s_after.get(future_index, False):
                        truncated_accumulation_count += 1
                        break
                if not values:
                    raise RuntimeError(
                        f"{record.trajectory_id}:{search_index}: effective IG "
                        "accumulation is empty"
                    )
                n_masked_future_continued_count += int(continued_after_masked_n)
                d_ig_eff = float(math.fsum(values) / math.sqrt(len(values)))
                delta_probe = float(
                    post_rewards[search_index] - pre_rewards[search_index]
                )
                z_outcome = float(advantage.normalized_outcome)
                if delta_probe > probe_epsilon:
                    route = max(z_outcome, 0.0)
                    delta_positive_count += 1
                elif delta_probe < -probe_epsilon:
                    route = min(z_outcome, 0.0)
                    delta_negative_count += 1
                else:
                    route = 0.0
                    delta_zero_count += 1
                actual = float(d_ig_eff + route)
                branch = "cumulative_ig_probe_routed_outcome"
                d_by_search[search_index] = d_ig_eff
                d_count_by_search[search_index] = len(values)
                delta_by_search[search_index] = delta_probe
                route_by_search[search_index] = route
                normal_count += 1
                cumulative_values.append(d_ig_eff)
                cumulative_counts.append(len(values))
                delta_values.append(delta_probe)
                route_values.append(route)
                cumulative_by_index.setdefault(search_index, []).append(d_ig_eff)
                route_positive_count += int(route > 0.0)
                route_zero_count += int(route == 0.0)
                route_negative_count += int(route < 0.0)
                z_o_normal_entry_count += int(route != 0.0)
            expected = (
                -1.0
                if sufficient
                else -1.0
                if no_new
                else d_by_search[search_index] + route_by_search[search_index]
            )
            if not math.isfinite(actual) or not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError("Probe-routed Search advantage assertion failed")
            new_search[search_index] = actual
            branch_by_search[search_index] = branch
            search_values.append(actual)
            if budget_exhausted:
                budget_exhausted_o_route_nonzero_count += int(
                    float(route_by_search.get(search_index, 0.0)) != 0.0
                )
                budget_exhausted_a_search_not_minus_one_count += int(
                    actual != -1.0
                )
                if (
                    search_index in d_by_search
                    or search_index in delta_by_search
                    or search_index in route_by_search
                ):
                    raise RuntimeError(
                        f"{record.trajectory_id}:{search_index}: budget-exhausted "
                        "Search entered Normal credit metadata"
                    )

        rebuilt.append(
            replace(
                advantage,
                search_advantage=new_search,
                search_advantage_old_shadow={},
                search_task_advantage={},
                stop_continue_by_search_index={},
                search_task_mode=(
                    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
                ),
                sufficient_before_search=s_before,
                sufficient_after_search=s_after,
                no_new_observation=no_new_by_search,
                search_branch_by_search_index=branch_by_search,
                effective_cumulative_ig=d_by_search,
                effective_cumulative_ig_count=d_count_by_search,
                probe_reward_delta=delta_by_search,
                routed_outcome=route_by_search,
            )
        )

    if any(
        (
            budget_exhausted_post_probe_count,
            budget_exhausted_o_route_nonzero_count,
            budget_exhausted_ig_entry_count,
            budget_exhausted_a_search_not_minus_one_count,
            normal_search_missing_post_prefix_count,
        )
    ):
        raise RuntimeError("Budget-exhausted Search safety counters are non-zero")

    result = replace(prompt_result, trajectories=tuple(rebuilt))
    if tuple(
        advantage.answer_advantage for advantage in result.trajectories
    ) != old_answers:
        raise RuntimeError("Probe-routed Search reconstruction changed A_answer")

    def stats(prefix: str, values: Sequence[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_count": int(array.size),
            f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
            f"{prefix}_std": float(array.std(ddof=0)) if array.size else 0.0,
            f"{prefix}_max": float(array.max()) if array.size else 0.0,
        }

    metrics: dict[str, float | int] = {
        "search/state_count": state_count,
        "search/s_before_count": s_before_count,
        "search/s_after_count": s_after_count,
        "search/post_probe_count": post_probe_count,
        "search/no_new_observation_count": no_new_count,
        "search/normal_count": normal_count,
        "search/s_before_and_n_count": s_and_n_count,
        "search/s_after_truncated_accumulation_count": (
            truncated_accumulation_count
        ),
        "search/n_masked_future_continued_count": (
            n_masked_future_continued_count
        ),
        "search/delta_probe_positive_count": delta_positive_count,
        "search/delta_probe_zero_count": delta_zero_count,
        "search/delta_probe_negative_count": delta_negative_count,
        "search/o_route_positive_count": route_positive_count,
        "search/o_route_zero_count": route_zero_count,
        "search/o_route_negative_count": route_negative_count,
        "search/z_o_normal_entry_count": z_o_normal_entry_count,
        "search/z_o_s_or_n_entry_count": 0,
        "search/a_sc_actor_entry_count": 0,
        "search/future_ig_cross_s_after_boundary_count": 0,
        "search/external_ig_multiplier_call_count": 0,
        "search/budget_exhausted_count": budget_exhausted_count,
        "search/budget_exhausted_post_probe_count": (
            budget_exhausted_post_probe_count
        ),
        "search/budget_exhausted_o_route_nonzero_count": (
            budget_exhausted_o_route_nonzero_count
        ),
        "search/budget_exhausted_ig_entry_count": budget_exhausted_ig_entry_count,
        "search/budget_exhausted_A_search_not_minus_one_count": (
            budget_exhausted_a_search_not_minus_one_count
        ),
        "search/normal_search_missing_post_prefix_count": (
            normal_search_missing_post_prefix_count
        ),
    }
    metrics["search/s_before_rate"] = (
        s_before_count / state_count if state_count else 0.0
    )
    metrics["search/s_after_rate"] = (
        s_after_count / post_probe_count if post_probe_count else 0.0
    )
    metrics["search/no_new_observation_rate"] = (
        no_new_count / state_count if state_count else 0.0
    )
    metrics["search/normal_rate"] = normal_count / state_count if state_count else 0.0
    metrics.update(stats("search/local_ig", local_ig_values))
    metrics.update(stats("search/local_ig_hat", local_ig_hat_values))
    metrics.update(stats("search/cumulative_ig_count", cumulative_counts))
    metrics.update(stats("search/D_ig_eff", cumulative_values))
    metrics.update(stats("search/delta_probe", delta_values))
    metrics.update(stats("search/O_route", route_values))
    metrics.update(stats("search/A_search", search_values))
    for search_index, values in sorted(cumulative_by_index.items()):
        metrics.update(stats(f"search/D_ig_eff_t{search_index}", values))
    return result, metrics


def _rebuild_role_localized_gate(
    records: Sequence[Any],
    prompt_result: A2TGPOPromptResult,
    *,
    expected_policy_version: int | None,
    expected_scorer_version: str | None,
    probe_epsilon: float,
) -> tuple[A2TGPOPromptResult, Mapping[str, float | int]]:
    old_answers = tuple(
        advantage.answer_advantage for advantage in prompt_result.trajectories
    )
    rebuilt: list[TrajectoryAdvantage] = []
    branch_counts: dict[str, int] = {}
    main_values: list[float] = []
    decision_values: list[float] = []
    query_values: list[float] = []
    allowed_overlap_count = 0
    empty_query_count = 0
    for record, advantage in zip(records, prompt_result.trajectories, strict=True):
        if (
            advantage.future_ig_sum
            or advantage.accumulated_ig_count
            or advantage.future_ig_rescaled
        ):
            raise RuntimeError(
                f"{record.trajectory_id}: role-localized mode reused legacy future IG"
            )
        if advantage.stop_continue_by_search_index:
            raise RuntimeError(
                f"{record.trajectory_id}: A_SC entered role-localized credit"
            )
        credits = build_role_localized_trajectory_credits(
            record,
            normalized_ig=advantage.normalized_ig,
            normalized_outcome=advantage.normalized_outcome,
            optimized_search_indices=tuple(advantage.search_advantage),
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
            probe_epsilon=probe_epsilon,
        )
        for branch in credits.branch.values():
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
        main_values.extend(credits.main.values())
        decision_values.extend(credits.decision.values())
        query_values.extend(credits.query.values())
        allowed_overlap_count += (
            credits.allowed_soft_duplicate_main_query_overlap_count
        )
        empty_query_count += credits.empty_query_without_query_span_count
        rebuilt.append(
            replace(
                advantage,
                search_advantage=dict(credits.main),
                search_main_advantage=dict(credits.main),
                search_decision_advantage=dict(credits.decision),
                search_query_advantage=dict(credits.query),
                search_advantage_old_shadow={},
                search_task_advantage={},
                stop_continue_by_search_index={},
                search_task_mode=(
                    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
                ),
                sufficient_before_search=dict(credits.sufficient_before),
                sufficient_after_search=dict(credits.sufficient_after),
                no_new_observation=dict(credits.no_new_observation),
                search_branch_by_search_index=dict(credits.branch),
                effective_cumulative_ig=dict(credits.effective_cumulative_ig),
                effective_cumulative_ig_count=dict(
                    credits.effective_cumulative_ig_count
                ),
                probe_reward_delta=dict(credits.probe_reward_delta),
                routed_outcome=dict(credits.routed_outcome),
            )
        )
    result = replace(prompt_result, trajectories=tuple(rebuilt))
    if tuple(
        advantage.answer_advantage for advantage in result.trajectories
    ) != old_answers:
        raise RuntimeError("Role-localized Search reconstruction changed A_answer")

    def stats(prefix: str, values: Sequence[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_count": int(array.size),
            f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
            f"{prefix}_std": float(array.std(ddof=0)) if array.size else 0.0,
            f"{prefix}_nonzero_rate": (
                float(np.count_nonzero(array)) / float(array.size)
                if array.size
                else 0.0
            ),
        }

    metrics: dict[str, float | int] = {
        **{
            f"role_gate/branch_{branch}_count": count
            for branch, count in sorted(branch_counts.items())
        },
        "role_gate/allowed_soft_duplicate_main_query_overlap_count": (
            allowed_overlap_count
        ),
        "role_gate/empty_query_without_query_span_count": empty_query_count,
        "role_gate/nonzero_decision_and_query_same_token_count": 0,
        "role_gate/unexpected_nonzero_main_gate_overlap_count": 0,
        "role_gate/observation_policy_mask_violation_count": 0,
    }
    metrics.update(stats("role_gate/A_main", main_values))
    metrics.update(stats("role_gate/A_decision", decision_values))
    metrics.update(stats("role_gate/A_query", query_values))
    return result, metrics


def _rebuild_answer_only_ragen2_mica_ig_v1(
    records: Sequence[Any],
    prompt_result: A2TGPOPromptResult,
    *,
    gamma: float,
    alpha: float,
    normalization_epsilon: float,
    zero_variance_tolerance: float,
) -> tuple[A2TGPOPromptResult, Mapping[str, float | int]]:
    """Replace only Search credit with raw-IG MICA for one selected prompt."""

    old_answers = tuple(
        advantage.answer_advantage for advantage in prompt_result.trajectories
    )
    missing_reasons: list[dict[int, str]] = []
    for record in records:
        reasons: dict[int, str] = {}
        for turn in record.turns:
            if turn.turn_type is not TurnType.SEARCH or turn.search_index is None:
                continue
            search_index = int(turn.search_index)
            if turn.ig_reward_eligible:
                continue
            if is_budget_exhausted_terminal_search(record, search_index):
                reason = "budget_exhausted"
            elif bool(getattr(turn, "model_search_invalid", False)):
                reason = "protocol_invalid"
            elif not bool(getattr(turn, "retriever_executed", False)):
                reason = "retriever_not_executed"
            elif not bool(turn.search_prefix_valid):
                reason = "post_observation_unavailable"
            else:
                reason = "exact_ig_undefined"
            reasons[search_index] = reason
        missing_reasons.append(reasons)

    mica = compute_mica_search_advantage(
        trajectory_ids=[str(record.trajectory_id) for record in records],
        search_indices_by_trajectory=[
            tuple(
                int(turn.search_index)
                for turn in record.turns
                if turn.turn_type is TurnType.SEARCH
                and turn.search_index is not None
            )
            for record in records
        ],
        raw_ig_by_trajectory=[dict(record.immediate_ig) for record in records],
        ig_reward_eligible_by_trajectory=[
            record.ig_reward_eligibility_by_search_index for record in records
        ],
        policy_credit_eligible_by_trajectory=[
            record.policy_credit_eligibility_by_search_index for record in records
        ],
        normalized_terminal_outcomes=[
            float(advantage.normalized_outcome)
            for advantage in prompt_result.trajectories
        ],
        ig_missing_reason_by_trajectory=missing_reasons,
        gamma=float(gamma),
        alpha=float(alpha),
        normalization_epsilon=float(normalization_epsilon),
        zero_variance_tolerance=float(zero_variance_tolerance),
    )

    rebuilt: list[TrajectoryAdvantage] = []
    peer_counts: list[float] = []
    raw_values: list[float] = []
    return_values: list[float] = []
    local_values: list[float] = []
    ret_adv_values: list[float] = []
    search_values: list[float] = []
    singleton_values: list[float] = []
    singleton_count = 0
    missing_count = 0
    loc_zero_variance_count = 0
    ret_zero_variance_count = 0
    for record, advantage, trajectory in zip(
        records,
        prompt_result.trajectories,
        mica.trajectories,
        strict=True,
    ):
        if str(record.trajectory_id) != trajectory.trajectory_id:
            raise RuntimeError("MICA trajectory ordering changed")
        search_advantage = {
            index: float(credit.search_advantage)
            for index, credit in trajectory.by_search_index.items()
        }
        local_advantage = {
            index: float(credit.local_advantage)
            for index, credit in trajectory.by_search_index.items()
            if credit.local_advantage is not None
        }
        return_advantage = {
            index: float(credit.return_advantage)
            for index, credit in trajectory.by_search_index.items()
            if credit.return_advantage is not None
        }
        # The existing adaptive clip consumes normalized immediate IG. MICA's
        # A_loc is the same prompt/depth normalization for supported groups;
        # singleton and missing-IG turns intentionally use neutral zero.
        normalized_ig = {
            index: float(local_advantage.get(index, 0.0))
            for index in search_advantage
        }
        for index, value in local_advantage.items():
            previous = advantage.normalized_ig.get(index)
            if previous is None or not math.isclose(
                float(previous),
                value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError("MICA local normalization drifted from production")
        branches = {
            index: (
                "mica_singleton_outcome"
                if credit.singleton_fallback
                else "mica_prompt_depth"
                if credit.ig_reward_eligible
                else "mica_missing_ig_zero_credit"
            )
            for index, credit in trajectory.by_search_index.items()
        }
        ig_return = {
            index: float(credit.ig_return)
            for index, credit in trajectory.by_search_index.items()
            if credit.ig_return is not None
        }
        peer_count = {
            index: int(credit.peer_count)
            for index, credit in trajectory.by_search_index.items()
        }
        loc_mean = {
            index: float(credit.loc_mean)
            for index, credit in trajectory.by_search_index.items()
            if credit.loc_mean is not None
        }
        loc_std = {
            index: float(credit.loc_std)
            for index, credit in trajectory.by_search_index.items()
            if credit.loc_std is not None
        }
        ret_mean = {
            index: float(credit.ret_mean)
            for index, credit in trajectory.by_search_index.items()
            if credit.ret_mean is not None
        }
        ret_std = {
            index: float(credit.ret_std)
            for index, credit in trajectory.by_search_index.items()
            if credit.ret_std is not None
        }
        singleton = {
            index: bool(credit.singleton_fallback)
            for index, credit in trajectory.by_search_index.items()
        }
        ig_missing = {
            index: str(credit.ig_missing_reason)
            for index, credit in trajectory.by_search_index.items()
            if credit.ig_missing_reason is not None
        }
        for credit in trajectory.by_search_index.values():
            peer_counts.append(float(credit.peer_count))
            search_values.append(float(credit.search_advantage))
            singleton_count += int(credit.singleton_fallback)
            missing_count += int(not credit.ig_reward_eligible)
            if credit.raw_ig is not None:
                raw_values.append(float(credit.raw_ig))
            if credit.ig_return is not None:
                return_values.append(float(credit.ig_return))
            if credit.local_advantage is not None:
                local_values.append(float(credit.local_advantage))
                loc_zero_variance_count += int(
                    float(credit.loc_std) ** 2 <= zero_variance_tolerance
                )
            if credit.return_advantage is not None:
                ret_adv_values.append(float(credit.return_advantage))
                ret_zero_variance_count += int(
                    float(credit.ret_std) ** 2 <= zero_variance_tolerance
                )
            if credit.singleton_fallback:
                singleton_values.append(float(credit.normalized_terminal_outcome))

        rebuilt.append(
            replace(
                advantage,
                normalized_ig=normalized_ig,
                future_ig_sum={},
                accumulated_ig_count={},
                future_ig_rescaled={},
                search_advantage=search_advantage,
                search_advantage_old_shadow={},
                search_task_advantage={},
                stop_continue_by_search_index={},
                search_task_mode=ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
                sufficient_before_search={},
                sufficient_after_search={},
                no_new_observation={},
                search_branch_by_search_index=branches,
                effective_cumulative_ig={},
                effective_cumulative_ig_count={},
                probe_reward_delta={},
                routed_outcome={},
                search_main_advantage={},
                search_decision_advantage={},
                search_query_advantage={},
                mica_ig_return=ig_return,
                mica_peer_count=peer_count,
                mica_loc_mean=loc_mean,
                mica_loc_std=loc_std,
                mica_ret_mean=ret_mean,
                mica_ret_std=ret_std,
                mica_local_advantage=local_advantage,
                mica_return_advantage=return_advantage,
                mica_singleton_fallback=singleton,
                mica_ig_missing_reason=ig_missing,
                mica_singleton_tail_start_depth=(
                    trajectory.singleton_tail_start_depth
                ),
                mica_singleton_consecutive_length=(
                    trajectory.singleton_consecutive_length
                ),
            )
        )

    result = replace(prompt_result, trajectories=tuple(rebuilt))
    if tuple(
        advantage.answer_advantage for advantage in result.trajectories
    ) != old_answers:
        raise RuntimeError("MICA Search reconstruction changed A_answer")

    def summary(prefix: str, values: Sequence[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_count": int(array.size),
            f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
            f"{prefix}_std": float(array.std(ddof=0)) if array.size else 0.0,
        }

    state_count = len(search_values)
    metrics: dict[str, float | int] = {
        "mica/singleton_fallback_count": singleton_count,
        "mica/singleton_fallback_rate": (
            singleton_count / state_count if state_count else 0.0
        ),
        "mica/ig_missing_zero_credit_count": missing_count,
        "mica/loc_zero_variance_count": loc_zero_variance_count,
        "mica/ret_zero_variance_count": ret_zero_variance_count,
        "mica/gamma": float(gamma),
        "mica/alpha": float(alpha),
        "mica/role_gate_actor_loss_count": 0,
        "mica/routed_outcome_entry_count": 0,
        "mica/normal_terminal_outcome_entry_count": 0,
    }
    metrics.update(summary("mica/peer_count", peer_counts))
    metrics.update(summary("mica/raw_ig", raw_values))
    metrics.update(summary("mica/ig_return", return_values))
    metrics.update(summary("mica/A_loc", local_values))
    metrics.update(summary("mica/A_ret", ret_adv_values))
    metrics.update(summary("mica/A_search", search_values))
    metrics.update(summary("mica/singleton_Z_O", singleton_values))
    return result, metrics


def rebuild_search_advantages(
    records: Sequence[Any],
    prompt_result: A2TGPOPromptResult,
    *,
    search_task_mode: str,
    group_size: int,
    lambda_ig: float | None = SEARCH_IG_COEFFICIENT,
    lambda_task: float = 1.0,
    reward_epsilon: float = 1.0e-6,
    scale_epsilon: float = 1.0e-8,
    pooled_scale_ddof: int = 0,
    probe_epsilon: float = 1.0e-6,
    mica_gamma: float = 1.0,
    mica_alpha: float = 0.5,
    normalization_epsilon: float = 1.0e-6,
    zero_variance_tolerance: float = 1.0e-12,
    expected_policy_version: int | None = None,
    expected_scorer_version: str | None = None,
) -> tuple[A2TGPOPromptResult, Mapping[str, float | int]]:
    """Rebuild only Search advantages after final Prompt selection.

    Answer advantages are copied byte-for-byte from ``prompt_result``.
    """

    if len(records) != len(prompt_result.trajectories):
        raise ValueError("Record and advantage cardinalities differ")
    if search_task_mode not in {
        NORMALIZED_OUTCOME_MODE,
        STOP_CONTINUE_CONSENSUS_MODE,
        SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
        SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
        SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
        ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
    }:
        raise ValueError(f"Unsupported Search task mode: {search_task_mode}")
    if search_task_mode == SUFFICIENCY_NOVELTY_LOCAL_IG_MODE:
        return _rebuild_sufficiency_novelty_local_ig(
            records,
            prompt_result,
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
        )
    if search_task_mode == ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE:
        return _rebuild_answer_only_ragen2_mica_ig_v1(
            records,
            prompt_result,
            gamma=float(mica_gamma),
            alpha=float(mica_alpha),
            normalization_epsilon=float(normalization_epsilon),
            zero_variance_tolerance=float(zero_variance_tolerance),
        )
    if (
        search_task_mode
        == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
    ):
        return _rebuild_sufficiency_novelty_cumulative_ig_probe_routed_outcome(
            records,
            prompt_result,
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
            probe_epsilon=float(probe_epsilon),
        )
    if (
        search_task_mode
        == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
    ):
        return _rebuild_role_localized_gate(
            records,
            prompt_result,
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
            probe_epsilon=float(probe_epsilon),
        )
    if not math.isclose(
        float(lambda_ig) if lambda_ig is not None else float("nan"),
        SEARCH_IG_COEFFICIENT,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Production Search advantage locks lambda_ig="
            f"{SEARCH_IG_COEFFICIENT}"
        )
    if not math.isclose(float(lambda_task), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Stop/Continue V1 locks lambda_task=1.0")

    old_answer_advantages = tuple(
        advantage.answer_advantage for advantage in prompt_result.trajectories
    )
    if search_task_mode == NORMALIZED_OUTCOME_MODE:
        rebuilt = []
        for advantage in prompt_result.trajectories:
            old_shadow = {
                index: float(
                    lambda_ig * advantage.future_ig_rescaled[index]
                    + advantage.normalized_outcome
                )
                for index in advantage.search_advantage
            }
            rebuilt.append(
                replace(
                    advantage,
                    search_advantage=old_shadow,
                    search_advantage_old_shadow=dict(old_shadow),
                    search_task_advantage={
                        index: float(advantage.normalized_outcome)
                        for index in old_shadow
                    },
                    stop_continue_by_search_index={},
                    search_task_mode=NORMALIZED_OUTCOME_MODE,
                )
            )
        result = replace(prompt_result, trajectories=tuple(rebuilt))
        if tuple(
            advantage.answer_advantage for advantage in result.trajectories
        ) != old_answer_advantages:
            raise RuntimeError("A_answer changed in normalized-Outcome mode")
        return result, {
            "sc/state_count": 0,
            "sc/clear_count": 0,
            "sc/clear_rate": 0.0,
            "sc/fallback_count": 0,
            "sc/fallback_rate": 0.0,
        }

    expected_state_keys: set[tuple[str, str, int]] = set()
    rewards = []
    normalized_outcome_by_trajectory: dict[tuple[str, str], float] = {}
    for record, advantage in zip(
        records,
        prompt_result.trajectories,
        strict=True,
    ):
        prompt_id = str(record.prompt_global_id)
        trajectory_id = str(record.trajectory_id)
        normalized_outcome_by_trajectory[(prompt_id, trajectory_id)] = float(
            advantage.normalized_outcome
        )
        expected_search_indices = set(advantage.search_advantage)
        # A direct-Answer trajectory has no Search action and therefore no
        # counterfactual Stop state. Its unchanged A_answer is the only policy
        # credit path; probe coverage remains strict for every real Search.
        if not expected_search_indices:
            continue
        probes = record.metadata.get("stop_continue_probes")
        if not isinstance(probes, Mapping):
            raise ValueError(
                f"{trajectory_id}: selected trajectory has no Stop probes"
            )
        for search_index in expected_search_indices:
            key = (prompt_id, trajectory_id, int(search_index))
            expected_state_keys.add(key)
            raw_probe = probes.get(search_index, probes.get(str(search_index)))
            if not isinstance(raw_probe, Mapping):
                raise ValueError(f"{key}: Stop probe is missing")
            rewards.append(stop_continue_reward_from_mapping(raw_probe))

    if not expected_state_keys:
        rebuilt = tuple(
            replace(
                advantage,
                search_advantage={},
                search_advantage_old_shadow={},
                search_task_advantage={},
                stop_continue_by_search_index={},
                search_task_mode=STOP_CONTINUE_CONSENSUS_MODE,
            )
            for advantage in prompt_result.trajectories
        )
        result = replace(prompt_result, trajectories=rebuilt)
        if tuple(
            advantage.answer_advantage for advantage in result.trajectories
        ) != old_answer_advantages:
            raise RuntimeError("Empty Search group changed A_answer")
        return result, {
            "sc/state_count": 0,
            "sc/clear_count": 0,
            "sc/clear_rate": 0.0,
            "sc/fallback_count": 0,
            "sc/fallback_rate": 0.0,
            "sc/fallback_z_o_to_search_count": 0,
        }

    computation = compute_stop_continue_advantages(
        rewards,
        normalized_outcome_by_trajectory=normalized_outcome_by_trajectory,
        expected_state_keys=expected_state_keys,
        group_size=int(group_size),
        reward_epsilon=float(reward_epsilon),
        scale_epsilon=float(scale_epsilon),
        pooled_scale_ddof=int(pooled_scale_ddof),
        expected_policy_version=expected_policy_version,
        expected_scorer_version=expected_scorer_version,
    )

    rebuilt = []
    for record, advantage in zip(
        records,
        prompt_result.trajectories,
        strict=True,
    ):
        old_shadow: dict[int, float] = {}
        task_by_search: dict[int, float] = {}
        new_search: dict[int, float] = {}
        sc_by_search: dict[int, StopContinueAdvantage] = {}
        for search_index, a_ig in advantage.future_ig_rescaled.items():
            if search_index not in advantage.search_advantage:
                continue
            state_key = (
                str(record.prompt_global_id),
                str(record.trajectory_id),
                int(search_index),
            )
            if state_key not in computation.by_state:
                raise ValueError(f"{state_key}: computed Stop advantage is missing")
            sc = computation.by_state[state_key]
            old_value = float(
                lambda_ig * a_ig + advantage.normalized_outcome
            )
            expected_task = float(sc.advantage_sc if sc.sc_clear else 0.0)
            if not math.isclose(
                float(sc.task_advantage),
                expected_task,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"{state_key}: A_task must be A_SC when clear and zero otherwise"
                )
            new_value = float(lambda_ig * a_ig + lambda_task * expected_task)
            if not math.isfinite(old_value) or not math.isfinite(new_value):
                raise ValueError(f"{state_key}: Search advantage is non-finite")
            old_shadow[int(search_index)] = old_value
            task_by_search[int(search_index)] = float(sc.task_advantage)
            new_search[int(search_index)] = new_value
            sc_by_search[int(search_index)] = sc
        if set(new_search) != set(advantage.search_advantage):
            raise RuntimeError(
                f"{record.trajectory_id}: Search advantage coverage changed"
            )
        rebuilt.append(
            replace(
                advantage,
                search_advantage=new_search,
                search_advantage_old_shadow=old_shadow,
                search_task_advantage=task_by_search,
                stop_continue_by_search_index=sc_by_search,
                search_task_mode=STOP_CONTINUE_CONSENSUS_MODE,
            )
        )

    result = replace(prompt_result, trajectories=tuple(rebuilt))
    if tuple(
        advantage.answer_advantage for advantage in result.trajectories
    ) != old_answer_advantages:
        raise RuntimeError("Stop/Continue reconstruction changed A_answer")
    return result, computation.metrics


def turn_advantages_from_record(
    record: Any,
    advantage: TrajectoryAdvantage,
) -> dict[int, float]:
    """Map channel advantages to real model turns without terminal reassignment.

    Search turns receive only their indexed Search advantage. A real
    model-generated Answer/fallback turn receives the terminal advantage only
    when it is the trajectory's explicit terminal policy-credit turn. If no
    such model span exists, no Answer advantage is emitted.
    """

    terminal_turn = record.terminal_policy_credit_turn_index
    if terminal_turn is None:
        if advantage.answer_policy_credit_eligible:
            raise ValueError(
                "Advantage claims terminal credit without a real terminal model span"
            )
        if advantage.answer_advantage is not None:
            raise ValueError(
                "No real terminal model span may carry an Answer advantage"
            )
    elif (
        not advantage.answer_policy_credit_eligible
        or advantage.answer_advantage is None
    ):
        raise ValueError(
            "A real terminal policy-credit span is missing its Answer advantage"
        )

    values: dict[int, float] = {}
    for turn in record.turns:
        if not turn.policy_credit_eligible:
            continue
        if turn.turn_type is TurnType.SEARCH:
            if turn.search_index is None:
                raise ValueError("Search turn is missing search_index")
            search_index = int(turn.search_index)
            if search_index not in advantage.search_advantage:
                raise ValueError(
                    f"Eligible Search turn {search_index} has no Search advantage"
                )
            values[int(turn.turn_index)] = float(
                advantage.search_advantage[search_index]
            )
            continue
        if terminal_turn is not None and turn.turn_index == terminal_turn:
            values[int(turn.turn_index)] = float(advantage.answer_advantage)
    return values
