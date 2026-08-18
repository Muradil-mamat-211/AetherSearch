"""Strict one-step policy objective and reference KL."""

from .strict_onpolicy_loss import a2tgpo_adaptive_turn_objective
from .turn_ratio import compute_turn_ratios

__all__ = ["a2tgpo_adaptive_turn_objective", "compute_turn_ratios"]
