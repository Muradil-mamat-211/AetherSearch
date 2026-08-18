from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    active: bool
    mode: str
    health_ratio: float | None
    reason: str


def decide_channel_gate(
    *,
    mean_excess: float,
    positive_prompt_count: int,
    minimum_positive_prompts: int,
    health_reference: float | None,
    health_threshold_ratio: float,
    epsilon: float,
) -> GateDecision:
    if not math.isfinite(mean_excess) or mean_excess <= 0:
        return GateDecision(False, "health" if health_reference is not None else "bootstrap", None, "non_positive_signal")
    if positive_prompt_count < minimum_positive_prompts:
        return GateDecision(False, "health" if health_reference is not None else "bootstrap", None, "insufficient_positive_prompts")
    if health_reference is None:
        return GateDecision(True, "bootstrap", None, "bootstrap_signal_present")
    ratio = mean_excess / (health_reference + epsilon)
    if ratio < health_threshold_ratio:
        return GateDecision(False, "health", ratio, "below_health_threshold")
    return GateDecision(True, "health", ratio, "healthy")

