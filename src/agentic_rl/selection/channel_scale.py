from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .health_gate import GateDecision, decide_channel_gate


@dataclass(frozen=True)
class ChannelPoolStats:
    raw_variance: dict[str, float]
    excess_variance: dict[str, float]
    normalized_signal: dict[str, float]
    positive_median: float | None
    mean_excess: float
    heterogeneity: float
    positive_prompt_count: int
    scale_used: float | None
    gate: GateDecision
    scale_observation_valid: bool
    scale_update_allowed_after_success: bool


@dataclass(frozen=True)
class ChannelScaleState:
    committed_scale: float | None = None
    health_observations: tuple[float, ...] = field(default_factory=tuple)
    health_reference: float | None = None
    valid_success_count: int = 0

    @staticmethod
    def ema_eta(half_life: float) -> float:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        return float(1.0 - 2.0 ** (-1.0 / half_life))

    def inspect_pool(
        self,
        variances: Mapping[str, float],
        *,
        noise_floor: float,
        minimum_positive_prompts: int,
        health_threshold_ratio: float,
        allow_provisional_scale: bool,
        epsilon: float = 1.0e-12,
    ) -> ChannelPoolStats:
        if not math.isfinite(noise_floor) or noise_floor < 0:
            raise ValueError("noise_floor must be finite and non-negative")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and positive")
        raw: dict[str, float] = {}
        excess: dict[str, float] = {}
        for prompt_id, value in variances.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"Invalid channel variance for {prompt_id}: {value}")
            raw[str(prompt_id)] = numeric
            excess[str(prompt_id)] = max(numeric - float(noise_floor), 0.0)

        positive = np.asarray([value for value in excess.values() if value > 0], dtype=np.float64)
        positive_median = float(np.median(positive)) if positive.size else None
        all_excess = np.asarray(list(excess.values()), dtype=np.float64)
        mean_excess = float(np.mean(all_excess, dtype=np.float64)) if all_excess.size else 0.0
        std_excess = float(np.std(all_excess, ddof=0, dtype=np.float64)) if all_excess.size else 0.0
        heterogeneity = std_excess / (mean_excess + epsilon)
        gate = decide_channel_gate(
            mean_excess=mean_excess,
            positive_prompt_count=int(positive.size),
            minimum_positive_prompts=minimum_positive_prompts,
            health_reference=self.health_reference,
            health_threshold_ratio=health_threshold_ratio,
            epsilon=epsilon,
        )

        scale_used = self.committed_scale
        if scale_used is None and allow_provisional_scale:
            scale_used = positive_median
        if gate.active and (scale_used is None or scale_used <= 0):
            gate = GateDecision(
                active=False,
                mode=gate.mode,
                health_ratio=gate.health_ratio,
                reason="scale_unavailable_after_update_1",
            )
        scale_observation_valid = bool(
            positive_median is not None
            and positive_median > 0
            and math.isfinite(positive_median)
            and mean_excess > 0
            and math.isfinite(mean_excess)
        )
        # Bootstrap activation controls only current selection. Before the
        # absolute-health gate is ready, it never freezes scale evolution.
        scale_update_allowed = bool(
            scale_observation_valid
            and (self.health_reference is None or gate.active)
        )
        normalized = {
            prompt_id: (
                value / (scale_used + epsilon)
                if gate.active and scale_used is not None and scale_used > 0
                else 0.0
            )
            for prompt_id, value in excess.items()
        }
        return ChannelPoolStats(
            raw_variance=raw,
            excess_variance=excess,
            normalized_signal=normalized,
            positive_median=positive_median,
            mean_excess=mean_excess,
            heterogeneity=heterogeneity,
            positive_prompt_count=int(positive.size),
            scale_used=scale_used,
            gate=gate,
            scale_observation_valid=scale_observation_valid,
            scale_update_allowed_after_success=scale_update_allowed,
        )

    def committed_after_success(
        self,
        stats: ChannelPoolStats,
        *,
        ema_half_life: float,
        health_reference_valid_updates: int,
        allow_initialization: bool,
    ) -> "ChannelScaleState":
        if not stats.scale_observation_valid or stats.positive_median is None:
            return self
        median = float(stats.positive_median)
        if median <= 0 or not math.isfinite(median):
            return self

        next_scale = self.committed_scale
        if not stats.scale_update_allowed_after_success:
            next_scale = self.committed_scale
        elif self.committed_scale is None:
            # Update 1 is the only legal initialization boundary. There is no
            # late channel-initialization state machine.
            next_scale = median if allow_initialization else None
        else:
            eta = self.ema_eta(ema_half_life)
            next_scale = math.exp(
                (1.0 - eta) * math.log(self.committed_scale)
                + eta * math.log(median)
            )

        observations = self.health_observations + (float(stats.mean_excess),)
        reference = self.health_reference
        if reference is None and len(observations) >= health_reference_valid_updates:
            initial = np.asarray(
                observations[:health_reference_valid_updates],
                dtype=np.float64,
            )
            reference = float(np.median(initial))
        return replace(
            self,
            committed_scale=next_scale,
            health_observations=observations,
            health_reference=reference,
            valid_success_count=self.valid_success_count + 1,
        )
