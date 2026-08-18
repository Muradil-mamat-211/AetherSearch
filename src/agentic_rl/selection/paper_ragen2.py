from __future__ import annotations

from typing import Mapping, Sequence

from .prompt_variance import sample_variance
from .top_p import TopPResult, stable_mass_top_p


def compute_ragen2_paper_sample_variance(
    terminal_outcomes: Sequence[float],
    outcome_reward_eligible: Sequence[bool] | None = None,
) -> float:
    """Return the paper RAGEN-2 within-prompt sample variance (ddof=1)."""

    if outcome_reward_eligible is None:
        outcome_reward_eligible = [True] * len(terminal_outcomes)
    if len(outcome_reward_eligible) != len(terminal_outcomes):
        raise ValueError("outcome eligibility length mismatch")
    values = [
        float(value)
        for value, eligible in zip(
            terminal_outcomes,
            outcome_reward_eligible,
            strict=True,
        )
        if bool(eligible)
    ]
    return sample_variance(values)


def select_ragen2_raw_variance_mass_top_p(
    sample_variance_by_prompt: Mapping[str, float],
    *,
    rho: float,
) -> TopPResult:
    """Select the minimal prompt prefix carrying rho of raw variance mass."""

    return stable_mass_top_p(
        sample_variance_by_prompt,
        rho=rho,
        include_zero=False,
        zero_tolerance=0.0,
    )

