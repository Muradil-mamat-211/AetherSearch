from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .channel_scale import ChannelPoolStats, ChannelScaleState
from .health_gate import GateDecision
from .paper_ragen2 import (
    compute_ragen2_paper_sample_variance,
    select_ragen2_raw_variance_mass_top_p,
)
from .prompt_variance import (
    ig_prompt_variance,
    outcome_prompt_variance,
)
from .top_p import TopPResult, stable_mass_top_p


DUAL_CHANNEL_SELECTION_SIGNAL = "dual_channel_ig_outcome"
ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL = "answer_outcome_only"
DUAL_CHANNEL_SCALED_TOP_P_MODE = "dual_channel_scaled_top_p"
ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE = "answer_outcome_only_scaled_top_p"
ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE = (
    "answer_outcome_only_ragen2_paper_variance_top_p"
)


@dataclass(frozen=True)
class PromptGroup:
    prompt_global_id: str
    trajectories: tuple[Any, ...]
    ig_variance: float
    outcome_variance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionDecision:
    candidate_count: int
    selected_ids: tuple[str, ...]
    selected_count: int
    requires_refill: bool
    skip_update: bool
    capacity_truncation_count: int
    top_p: TopPResult
    ig_stats: ChannelPoolStats
    outcome_stats: ChannelPoolStats
    score_by_prompt: dict[str, float]
    signal_mode: str = DUAL_CHANNEL_SELECTION_SIGNAL
    selection_mode: str = DUAL_CHANNEL_SCALED_TOP_P_MODE
    raw_top_p: TopPResult | None = None
    health_gate_selection_call_count: int = 0
    scale_selection_call_count: int = 0
    normalized_signal_selection_call_count: int = 0


def _paper_bypass_stats(
    variances: dict[str, float],
) -> ChannelPoolStats:
    raw = {str(prompt_id): float(value) for prompt_id, value in variances.items()}
    for prompt_id, value in raw.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid paper RAGEN variance for {prompt_id}: {value}")
    positive_count = sum(value > 0.0 for value in raw.values())
    values = np.asarray(tuple(raw.values()), dtype=np.float64)
    mean_raw = float(np.mean(values, dtype=np.float64)) if values.size else 0.0
    std_raw = float(np.std(values, ddof=0, dtype=np.float64)) if values.size else 0.0
    return ChannelPoolStats(
        raw_variance=raw,
        excess_variance=dict(raw),
        normalized_signal={prompt_id: 0.0 for prompt_id in raw},
        positive_median=None,
        mean_excess=mean_raw,
        heterogeneity=(std_raw / mean_raw if mean_raw > 0.0 else 0.0),
        positive_prompt_count=int(positive_count),
        scale_used=None,
        gate=GateDecision(
            active=False,
            mode="paper_raw_sample_variance",
            health_ratio=None,
            reason="selection_bypasses_health_and_scale",
        ),
        scale_observation_valid=False,
        scale_update_allowed_after_success=False,
    )


class CandidatePool:
    def __init__(self, *, group_size: int, maximum_prompts: int) -> None:
        self.group_size = int(group_size)
        self.maximum_prompts = int(maximum_prompts)
        self._groups: dict[str, PromptGroup] = {}

    def __len__(self) -> int:
        return len(self._groups)

    def add(self, groups: Iterable[PromptGroup]) -> None:
        incoming = tuple(groups)
        incoming_ids = [group.prompt_global_id for group in incoming]
        if len(set(incoming_ids)) != len(incoming_ids):
            raise ValueError("Incoming candidate groups contain duplicate IDs")
        duplicate = set(incoming_ids).intersection(self._groups)
        if duplicate:
            raise ValueError(f"Duplicate prompt group: {sorted(duplicate)[0]}")
        if len(self._groups) + len(incoming) > self.maximum_prompts:
            raise ValueError("Candidate pool capacity exceeded")
        for group in incoming:
            if len(group.trajectories) != self.group_size:
                raise ValueError(
                    f"{group.prompt_global_id} has {len(group.trajectories)} "
                    f"trajectories, expected {self.group_size}"
                )
        for group in incoming:
            self._groups[group.prompt_global_id] = group

    def get(self, prompt_global_id: str) -> PromptGroup:
        return self._groups[prompt_global_id]

    def groups(self) -> tuple[PromptGroup, ...]:
        return tuple(self._groups[key] for key in sorted(self._groups))

    def select(
        self,
        *,
        ig_state: ChannelScaleState,
        outcome_state: ChannelScaleState,
        top_p_mass: float,
        alpha_ig: float,
        alpha_outcome: float,
        noise_floor_ig: float,
        noise_floor_outcome: float,
        minimum_positive_prompts: int,
        health_threshold_ratio: float,
        minimum_selected_prompts: int,
        maximum_selected_prompts: int,
        allow_provisional_scale: bool,
        epsilon: float = 1.0e-12,
        signal_mode: str = DUAL_CHANNEL_SELECTION_SIGNAL,
        selection_mode: str | None = None,
    ) -> SelectionDecision:
        if signal_mode not in {
            DUAL_CHANNEL_SELECTION_SIGNAL,
            ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
        }:
            raise ValueError(f"Unsupported RAGEN selection signal: {signal_mode}")
        if selection_mode is None:
            selection_mode = (
                ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE
                if signal_mode == ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL
                else DUAL_CHANNEL_SCALED_TOP_P_MODE
            )
        allowed_modes = {
            DUAL_CHANNEL_SCALED_TOP_P_MODE,
            ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE,
            ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE,
        }
        if selection_mode not in allowed_modes:
            raise ValueError(f"Unsupported RAGEN selection mode: {selection_mode}")
        if (
            selection_mode
            == ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE
            and signal_mode != ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL
        ):
            raise ValueError("Paper RAGEN-2 mode requires Answer-only signal")
        if (
            selection_mode == ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE
            and signal_mode != ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL
        ):
            raise ValueError("Answer-only scaled mode requires Answer-only signal")
        if (
            selection_mode == DUAL_CHANNEL_SCALED_TOP_P_MODE
            and signal_mode != DUAL_CHANNEL_SELECTION_SIGNAL
        ):
            raise ValueError("Dual-channel scaled mode requires dual-channel signal")
        ig_variances = {
            prompt_id: group.ig_variance for prompt_id, group in self._groups.items()
        }
        outcome_variances = {
            prompt_id: group.outcome_variance
            for prompt_id, group in self._groups.items()
        }
        if selection_mode == ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE:
            ig_stats = _paper_bypass_stats(ig_variances)
            outcome_stats = _paper_bypass_stats(outcome_variances)
            scores = dict(outcome_variances)
            raw_top_p = select_ragen2_raw_variance_mass_top_p(
                scores,
                rho=top_p_mass,
            )
            health_gate_call_count = 0
            scale_call_count = 0
            normalized_signal_call_count = 0
        else:
            ig_stats = ig_state.inspect_pool(
                ig_variances,
                noise_floor=noise_floor_ig,
                minimum_positive_prompts=minimum_positive_prompts,
                health_threshold_ratio=health_threshold_ratio,
                allow_provisional_scale=allow_provisional_scale,
                epsilon=epsilon,
            )
            outcome_stats = outcome_state.inspect_pool(
                outcome_variances,
                noise_floor=noise_floor_outcome,
                minimum_positive_prompts=minimum_positive_prompts,
                health_threshold_ratio=health_threshold_ratio,
                allow_provisional_scale=allow_provisional_scale,
                epsilon=epsilon,
            )

            if signal_mode == ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL:
                denominator = (
                    alpha_outcome if outcome_stats.gate.active else 0.0
                ) + epsilon
                scores = {
                    prompt_id: (
                        alpha_outcome * outcome_stats.normalized_signal[prompt_id]
                        if outcome_stats.gate.active
                        else 0.0
                    )
                    / denominator
                    for prompt_id in self._groups
                }
            else:
                denominator = (
                    (alpha_ig if ig_stats.gate.active else 0.0)
                    + (alpha_outcome if outcome_stats.gate.active else 0.0)
                    + epsilon
                )
                scores = {
                    prompt_id: (
                        (
                            alpha_ig * ig_stats.normalized_signal[prompt_id]
                            if ig_stats.gate.active
                            else 0.0
                        )
                        + (
                            alpha_outcome
                            * outcome_stats.normalized_signal[prompt_id]
                            if outcome_stats.gate.active
                            else 0.0
                        )
                    )
                    / denominator
                    for prompt_id in self._groups
                }
            raw_top_p = stable_mass_top_p(
                scores,
                rho=top_p_mass,
                include_zero=False,
                zero_tolerance=0.0,
            )
            health_gate_call_count = 2
            scale_call_count = 2
            normalized_signal_call_count = 2
        selected = list(raw_top_p.selected_ids)
        truncation = max(0, len(selected) - maximum_selected_prompts)
        if truncation:
            selected = selected[:maximum_selected_prompts]
        selected_mass = float(
            np.sum(
                np.asarray([scores[prompt_id] for prompt_id in selected], dtype=np.float64),
                dtype=np.float64,
            )
        )
        capped_top_p = TopPResult(
            ordered_positive_ids=raw_top_p.ordered_positive_ids,
            selected_ids=tuple(selected),
            total_mass=raw_top_p.total_mass,
            selected_mass=selected_mass,
            selected_mass_ratio=(
                selected_mass / raw_top_p.total_mass
                if raw_top_p.total_mass > 0
                else 0.0
            ),
        )
        underflow = len(selected) < minimum_selected_prompts
        requires_refill = underflow and len(self) < self.maximum_prompts
        skip_update = underflow and len(self) >= self.maximum_prompts
        return SelectionDecision(
            candidate_count=len(self),
            selected_ids=tuple(selected),
            selected_count=len(selected),
            requires_refill=requires_refill,
            skip_update=skip_update,
            capacity_truncation_count=truncation,
            top_p=capped_top_p,
            ig_stats=ig_stats,
            outcome_stats=outcome_stats,
            score_by_prompt=scores,
            signal_mode=signal_mode,
            selection_mode=selection_mode,
            raw_top_p=raw_top_p,
            health_gate_selection_call_count=health_gate_call_count,
            scale_selection_call_count=scale_call_count,
            normalized_signal_selection_call_count=normalized_signal_call_count,
        )

    def selected_groups(self, decision: SelectionDecision) -> tuple[PromptGroup, ...]:
        groups = tuple(self._groups[prompt_id] for prompt_id in decision.selected_ids)
        for group in groups:
            if len(group.trajectories) != self.group_size:
                raise RuntimeError("Selected group lost trajectories")
        return groups


def prompt_group_from_trajectories(
    group: Any,
    *,
    expected_group_size: int = 16,
) -> PromptGroup:
    group.validate(expected_group_size=expected_group_size)
    if not all(
        trajectory.trajectory_system_valid for trajectory in group.trajectories
    ):
        raise ValueError(
            "System-invalid trajectories cannot enter a candidate prompt group; "
            "the runtime must retry or replace the group"
        )
    if not all(trajectory.optimization_ready for trajectory in group.trajectories):
        raise ValueError(
            "Every candidate trajectory must contain at least one real model "
            "token eligible for policy credit"
        )
    ig_eligibility = [
        trajectory.ig_reward_eligibility_by_search_index
        for trajectory in group.trajectories
    ]
    outcome_eligibility = [
        bool(trajectory.outcome_reward_eligible)
        for trajectory in group.trajectories
    ]
    ig = ig_prompt_variance(
        [trajectory.immediate_ig for trajectory in group.trajectories],
        ig_eligibility,
    )
    outcome = outcome_prompt_variance(
        [trajectory.task_outcome for trajectory in group.trajectories],
        outcome_eligibility,
    )
    return PromptGroup(
        prompt_global_id=group.prompt_global_id,
        trajectories=tuple(group.trajectories),
        ig_variance=ig.aggregate,
        outcome_variance=outcome,
        metadata={
            "ig_variance_by_search_index": ig.by_search_index,
            "ig_peer_count_by_search_index": ig.peer_count_by_search_index,
            "ig_natural_weight_by_search_index": ig.natural_weight_by_search_index,
            "ig_eligible_trajectory_count": sum(
                any(values.values()) for values in ig_eligibility
            ),
            "outcome_eligible_trajectory_count": sum(outcome_eligibility),
            "system_valid_trajectory_count": sum(
                bool(trajectory.trajectory_system_valid)
                for trajectory in group.trajectories
            ),
        },
    )


def prompt_group_from_outcomes(
    group: Any,
    *,
    expected_group_size: int = 16,
) -> PromptGroup:
    """Build a Candidate group without requiring preselection Exact-IG."""

    group.validate(expected_group_size=expected_group_size)
    if not all(
        trajectory.trajectory_system_valid for trajectory in group.trajectories
    ):
        raise ValueError("System-invalid trajectories cannot enter a candidate group")
    if not all(trajectory.optimization_ready for trajectory in group.trajectories):
        raise ValueError("Every candidate trajectory must be optimization-ready")
    outcome_eligibility = [
        bool(trajectory.outcome_reward_eligible)
        for trajectory in group.trajectories
    ]
    outcome = compute_ragen2_paper_sample_variance(
        [trajectory.task_outcome for trajectory in group.trajectories],
        outcome_eligibility,
    )
    return PromptGroup(
        prompt_global_id=group.prompt_global_id,
        trajectories=tuple(group.trajectories),
        ig_variance=0.0,
        outcome_variance=outcome,
        metadata={
            "exact_ig_deferred": True,
            "ig_variance_by_search_index": {},
            "ig_peer_count_by_search_index": {},
            "ig_natural_weight_by_search_index": {},
            "ig_eligible_trajectory_count": 0,
            "outcome_eligible_trajectory_count": sum(outcome_eligibility),
            "system_valid_trajectory_count": sum(
                bool(trajectory.trajectory_system_valid)
                for trajectory in group.trajectories
            ),
        },
    )
