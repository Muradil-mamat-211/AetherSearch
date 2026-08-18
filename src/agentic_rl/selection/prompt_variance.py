from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IGPromptVariance:
    aggregate: float
    by_search_index: dict[int, float]
    peer_count_by_search_index: dict[int, int]
    natural_weight_by_search_index: dict[int, float]


def sample_variance(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError("Variance inputs must be finite")
    centered = array - np.mean(array, dtype=np.float64)
    return float(np.sum(centered * centered, dtype=np.float64) / (array.size - 1))


def ig_prompt_variance(
    immediate_ig_by_trajectory: Sequence[Mapping[int, float]],
    ig_reward_eligible_by_trajectory: Sequence[Mapping[int, bool]] | None = None,
) -> IGPromptVariance:
    if ig_reward_eligible_by_trajectory is None:
        ig_reward_eligible_by_trajectory = [
            {int(index): True for index in values}
            for values in immediate_ig_by_trajectory
        ]
    if len(ig_reward_eligible_by_trajectory) != len(
        immediate_ig_by_trajectory
    ):
        raise ValueError("IG eligibility length mismatch")

    peers: dict[int, list[float]] = {}
    for turn_values, eligibility in zip(
        immediate_ig_by_trajectory,
        ig_reward_eligible_by_trajectory,
    ):
        eligible_indices = {
            int(index) for index, eligible in eligibility.items() if eligible
        }
        if set(map(int, turn_values)) != eligible_indices:
            raise ValueError(
                "Immediate IG must exist exactly for eligible Search turns"
            )
        for search_index, value in turn_values.items():
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("IG values must be finite")
            peers.setdefault(int(search_index), []).append(numeric)

    peer_counts = {index: len(values) for index, values in peers.items()}
    variances = {
        index: sample_variance(values) if len(values) >= 2 else 0.0
        for index, values in peers.items()
    }
    supported_indices = tuple(
        index for index, count in peer_counts.items() if count >= 2
    )
    supported_peer_total = sum(peer_counts[index] for index in supported_indices)
    if supported_peer_total == 0:
        return IGPromptVariance(
            aggregate=0.0,
            by_search_index=variances,
            peer_count_by_search_index=peer_counts,
            natural_weight_by_search_index={index: 0.0 for index in variances},
        )

    weights = {
        index: (
            peer_counts[index] / supported_peer_total
            if index in supported_indices
            else 0.0
        )
        for index in variances
    }
    aggregate = float(
        np.sum(
            np.asarray(
                [weights[index] * variances[index] for index in sorted(variances)],
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
    )
    return IGPromptVariance(
        aggregate=aggregate,
        by_search_index=variances,
        peer_count_by_search_index=peer_counts,
        natural_weight_by_search_index=weights,
    )


def outcome_prompt_variance(
    outcomes: Sequence[float],
    outcome_reward_eligible: Sequence[bool] | None = None,
) -> float:
    if outcome_reward_eligible is None:
        outcome_reward_eligible = [True] * len(outcomes)
    if len(outcome_reward_eligible) != len(outcomes):
        raise ValueError("outcome eligibility length mismatch")
    values = [
        float(value)
        for value, eligible in zip(outcomes, outcome_reward_eligible)
        if eligible
    ]
    return sample_variance(values)


def compute_answer_outcome_variance(
    terminal_task_outcomes: Sequence[float],
    outcome_reward_eligible: Sequence[bool] | None = None,
) -> float:
    """Named Answer-only RAGEN entry point using the production sample variance."""

    return outcome_prompt_variance(
        terminal_task_outcomes,
        outcome_reward_eligible,
    )
