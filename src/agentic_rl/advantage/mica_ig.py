from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE = (
    "answer_only_ragen2_mica_ig_v1_singleton_outcome"
)


@dataclass(frozen=True)
class PromptDepthStats:
    peer_count: int
    mean: float
    std: float


@dataclass(frozen=True)
class MicaSearchCredit:
    raw_ig: float | None
    ig_return: float | None
    peer_count: int
    loc_mean: float | None
    loc_std: float | None
    ret_mean: float | None
    ret_std: float | None
    local_advantage: float | None
    return_advantage: float | None
    singleton_fallback: bool
    normalized_terminal_outcome: float
    search_advantage: float
    ig_reward_eligible: bool
    policy_credit_eligible: bool
    ig_missing_reason: str | None


@dataclass(frozen=True)
class MicaTrajectoryResult:
    trajectory_id: str
    by_search_index: dict[int, MicaSearchCredit]
    singleton_tail_start_depth: int | None
    singleton_consecutive_length: int


@dataclass(frozen=True)
class MicaPromptResult:
    trajectories: tuple[MicaTrajectoryResult, ...]
    local_stats_by_search_index: dict[int, PromptDepthStats]
    return_stats_by_search_index: dict[int, PromptDepthStats]


def _population_stats(values: Sequence[float]) -> PromptDepthStats:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return PromptDepthStats(peer_count=0, mean=0.0, std=0.0)
    if not np.all(np.isfinite(array)):
        raise ValueError("MICA normalization values must be finite")
    return PromptDepthStats(
        peer_count=int(array.size),
        mean=float(np.mean(array, dtype=np.float64)),
        std=float(np.std(array, ddof=0, dtype=np.float64)),
    )


def compute_normalized_terminal_outcomes(
    outcomes: Sequence[float],
    outcome_reward_eligible: Sequence[bool] | None = None,
    *,
    normalization_epsilon: float = 1.0e-6,
    zero_variance_tolerance: float = 1.0e-12,
) -> tuple[float, ...]:
    """Reuse the production within-prompt population normalization contract."""

    if outcome_reward_eligible is None:
        outcome_reward_eligible = [True] * len(outcomes)
    if len(outcomes) != len(outcome_reward_eligible):
        raise ValueError("Outcome eligibility length mismatch")
    if normalization_epsilon <= 0.0 or zero_variance_tolerance < 0.0:
        raise ValueError("Invalid MICA normalization tolerances")
    values = [
        float(value)
        for value, eligible in zip(
            outcomes,
            outcome_reward_eligible,
            strict=True,
        )
        if bool(eligible)
    ]
    stats = _population_stats(values)
    result = []
    for value, eligible in zip(outcomes, outcome_reward_eligible, strict=True):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Terminal task outcomes must be finite")
        result.append(
            0.0
            if (
                not bool(eligible)
                or stats.std * stats.std <= zero_variance_tolerance
            )
            else (numeric - stats.mean) / (stats.std + normalization_epsilon)
        )
    return tuple(float(value) for value in result)


def compute_raw_ig_returns(
    raw_ig_by_trajectory: Sequence[Mapping[int, float]],
    *,
    gamma: float = 1.0,
) -> tuple[dict[int, float], ...]:
    """Compute literal suffix returns over each trajectory's valid IG indices."""

    if not math.isfinite(float(gamma)) or float(gamma) < 0.0:
        raise ValueError("mica_gamma must be finite and non-negative")
    results: list[dict[int, float]] = []
    for raw_values in raw_ig_by_trajectory:
        normalized = {int(index): float(value) for index, value in raw_values.items()}
        if any(index < 0 for index in normalized):
            raise ValueError("Search indices must be non-negative")
        if any(not math.isfinite(value) for value in normalized.values()):
            raise ValueError("Raw Exact-IG rewards must be finite")
        ordered = sorted(normalized)
        returns: dict[int, float] = {}
        for index in ordered:
            returns[index] = float(
                math.fsum(
                    (float(gamma) ** (future_index - index))
                    * normalized[future_index]
                    for future_index in ordered
                    if future_index >= index
                )
            )
        results.append(returns)
    return tuple(results)


def compute_prompt_depth_group_stats(
    values_by_trajectory: Sequence[Mapping[int, float]],
) -> dict[int, PromptDepthStats]:
    """Compute independent population statistics for each Search depth."""

    peers: dict[int, list[float]] = {}
    for values in values_by_trajectory:
        for search_index, value in values.items():
            index = int(search_index)
            numeric = float(value)
            if index < 0 or not math.isfinite(numeric):
                raise ValueError("Invalid prompt/depth normalization value")
            peers.setdefault(index, []).append(numeric)
    return {
        index: _population_stats(values)
        for index, values in sorted(peers.items())
    }


def _group_relative_advantage(
    value: float,
    stats: PromptDepthStats,
    *,
    normalization_epsilon: float,
    zero_variance_tolerance: float,
) -> float:
    if stats.peer_count < 2:
        raise ValueError("Singleton groups do not define a relative advantage")
    if stats.std * stats.std <= zero_variance_tolerance:
        return 0.0
    return float((float(value) - stats.mean) / (stats.std + normalization_epsilon))


def compute_mica_local_advantage(
    raw_ig: float,
    stats: PromptDepthStats,
    *,
    normalization_epsilon: float = 1.0e-6,
    zero_variance_tolerance: float = 1.0e-12,
) -> float:
    return _group_relative_advantage(
        raw_ig,
        stats,
        normalization_epsilon=normalization_epsilon,
        zero_variance_tolerance=zero_variance_tolerance,
    )


def compute_mica_return_advantage(
    ig_return: float,
    stats: PromptDepthStats,
    *,
    normalization_epsilon: float = 1.0e-6,
    zero_variance_tolerance: float = 1.0e-12,
) -> float:
    return _group_relative_advantage(
        ig_return,
        stats,
        normalization_epsilon=normalization_epsilon,
        zero_variance_tolerance=zero_variance_tolerance,
    )


def compute_singleton_outcome_fallback(normalized_terminal_outcome: float) -> float:
    numeric = float(normalized_terminal_outcome)
    if not math.isfinite(numeric):
        raise ValueError("Singleton terminal Outcome fallback must be finite")
    return numeric


def compute_mica_search_advantage(
    *,
    trajectory_ids: Sequence[str],
    search_indices_by_trajectory: Sequence[Sequence[int]],
    raw_ig_by_trajectory: Sequence[Mapping[int, float]],
    ig_reward_eligible_by_trajectory: Sequence[Mapping[int, bool]],
    policy_credit_eligible_by_trajectory: Sequence[Mapping[int, bool]],
    normalized_terminal_outcomes: Sequence[float],
    ig_missing_reason_by_trajectory: Sequence[Mapping[int, str]] | None = None,
    gamma: float = 1.0,
    alpha: float = 0.5,
    normalization_epsilon: float = 1.0e-6,
    zero_variance_tolerance: float = 1.0e-12,
) -> MicaPromptResult:
    """Construct MICA V1 Search credit for one selected prompt group."""

    count = len(trajectory_ids)
    sequences = (
        search_indices_by_trajectory,
        raw_ig_by_trajectory,
        ig_reward_eligible_by_trajectory,
        policy_credit_eligible_by_trajectory,
        normalized_terminal_outcomes,
    )
    if any(len(values) != count for values in sequences):
        raise ValueError("MICA trajectory inputs have different cardinalities")
    if len(set(map(str, trajectory_ids))) != count:
        raise ValueError("MICA trajectory IDs must be unique")
    if not math.isclose(float(gamma), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("MICA-IG V1 locks gamma=1.0")
    if not math.isclose(float(alpha), 0.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("MICA-IG V1 locks alpha=0.5")
    if normalization_epsilon <= 0.0 or zero_variance_tolerance < 0.0:
        raise ValueError("Invalid MICA normalization tolerances")
    missing_reasons = (
        tuple({} for _ in range(count))
        if ig_missing_reason_by_trajectory is None
        else tuple(ig_missing_reason_by_trajectory)
    )
    if len(missing_reasons) != count:
        raise ValueError("MICA missing-reason cardinality mismatch")

    normalized_raw: list[dict[int, float]] = []
    normalized_search_indices: list[tuple[int, ...]] = []
    for trajectory_index in range(count):
        indices = tuple(int(value) for value in search_indices_by_trajectory[trajectory_index])
        if indices != tuple(sorted(set(indices))) or any(value < 0 for value in indices):
            raise ValueError("Real Search indices must be sorted, unique, and non-negative")
        ig_eligibility = {
            int(index): bool(value)
            for index, value in ig_reward_eligible_by_trajectory[trajectory_index].items()
        }
        policy_eligibility = {
            int(index): bool(value)
            for index, value in policy_credit_eligible_by_trajectory[trajectory_index].items()
        }
        if set(ig_eligibility) != set(indices) or set(policy_eligibility) != set(indices):
            raise ValueError("MICA eligibility must cover every real Search turn")
        raw = {
            int(index): float(value)
            for index, value in raw_ig_by_trajectory[trajectory_index].items()
        }
        expected_raw = {index for index in indices if ig_eligibility[index]}
        if set(raw) != expected_raw:
            raise ValueError("Raw IG must exist exactly for IG-eligible Search turns")
        if any(not math.isfinite(value) for value in raw.values()):
            raise ValueError("Raw Exact-IG rewards must be finite")
        normalized_raw.append(raw)
        normalized_search_indices.append(indices)

    returns = compute_raw_ig_returns(normalized_raw, gamma=gamma)
    local_stats = compute_prompt_depth_group_stats(normalized_raw)
    return_stats = compute_prompt_depth_group_stats(returns)
    trajectories: list[MicaTrajectoryResult] = []
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        z_outcome = float(normalized_terminal_outcomes[trajectory_index])
        if not math.isfinite(z_outcome):
            raise ValueError("Normalized terminal outcomes must be finite")
        ig_eligibility = ig_reward_eligible_by_trajectory[trajectory_index]
        policy_eligibility = policy_credit_eligible_by_trajectory[trajectory_index]
        credits: dict[int, MicaSearchCredit] = {}
        singleton_indices: list[int] = []
        for search_index in normalized_search_indices[trajectory_index]:
            if not bool(policy_eligibility[search_index]):
                continue
            if not bool(ig_eligibility[search_index]):
                reason = str(
                    missing_reasons[trajectory_index].get(
                        search_index,
                        "exact_ig_undefined",
                    )
                )
                credits[search_index] = MicaSearchCredit(
                    raw_ig=None,
                    ig_return=None,
                    peer_count=0,
                    loc_mean=None,
                    loc_std=None,
                    ret_mean=None,
                    ret_std=None,
                    local_advantage=None,
                    return_advantage=None,
                    singleton_fallback=False,
                    normalized_terminal_outcome=z_outcome,
                    search_advantage=0.0,
                    ig_reward_eligible=False,
                    policy_credit_eligible=True,
                    ig_missing_reason=reason,
                )
                continue
            loc_stats = local_stats[search_index]
            ret_stats = return_stats[search_index]
            if loc_stats.peer_count != ret_stats.peer_count:
                raise RuntimeError("Local and return MICA peer counts diverged")
            if loc_stats.peer_count == 1:
                singleton_indices.append(search_index)
                fallback = compute_singleton_outcome_fallback(z_outcome)
                credits[search_index] = MicaSearchCredit(
                    raw_ig=normalized_raw[trajectory_index][search_index],
                    ig_return=returns[trajectory_index][search_index],
                    peer_count=1,
                    loc_mean=loc_stats.mean,
                    loc_std=loc_stats.std,
                    ret_mean=ret_stats.mean,
                    ret_std=ret_stats.std,
                    local_advantage=None,
                    return_advantage=None,
                    singleton_fallback=True,
                    normalized_terminal_outcome=z_outcome,
                    search_advantage=fallback,
                    ig_reward_eligible=True,
                    policy_credit_eligible=True,
                    ig_missing_reason=None,
                )
                continue
            if loc_stats.peer_count < 2:
                raise RuntimeError("An IG-eligible Search has no prompt/depth peer group")
            local_advantage = compute_mica_local_advantage(
                normalized_raw[trajectory_index][search_index],
                loc_stats,
                normalization_epsilon=normalization_epsilon,
                zero_variance_tolerance=zero_variance_tolerance,
            )
            return_advantage = compute_mica_return_advantage(
                returns[trajectory_index][search_index],
                ret_stats,
                normalization_epsilon=normalization_epsilon,
                zero_variance_tolerance=zero_variance_tolerance,
            )
            search_advantage = float(
                float(alpha) * return_advantage
                + (1.0 - float(alpha)) * local_advantage
            )
            credits[search_index] = MicaSearchCredit(
                raw_ig=normalized_raw[trajectory_index][search_index],
                ig_return=returns[trajectory_index][search_index],
                peer_count=loc_stats.peer_count,
                loc_mean=loc_stats.mean,
                loc_std=loc_stats.std,
                ret_mean=ret_stats.mean,
                ret_std=ret_stats.std,
                local_advantage=local_advantage,
                return_advantage=return_advantage,
                singleton_fallback=False,
                normalized_terminal_outcome=z_outcome,
                search_advantage=search_advantage,
                ig_reward_eligible=True,
                policy_credit_eligible=True,
                ig_missing_reason=None,
            )

        singleton_start = min(singleton_indices) if singleton_indices else None
        singleton_length = (
            sum(index >= singleton_start for index in singleton_indices)
            if singleton_start is not None
            else 0
        )
        trajectories.append(
            MicaTrajectoryResult(
                trajectory_id=str(trajectory_id),
                by_search_index=credits,
                singleton_tail_start_depth=singleton_start,
                singleton_consecutive_length=int(singleton_length),
            )
        )

    return MicaPromptResult(
        trajectories=tuple(trajectories),
        local_stats_by_search_index=local_stats,
        return_stats_by_search_index=return_stats,
    )
