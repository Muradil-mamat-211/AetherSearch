from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReasoningCollapseMetrics:
    answer_rate: float
    mean_search_turns: float
    multi_search_rate: float
    zero_ig_variance_prompt_rate: float
    zero_outcome_variance_prompt_rate: float
    zero_advantage_rate: float
    within_prompt_entropy: float | None = None
    cross_prompt_mi_proxy: float | None = None
    query_diversity: float | None = None
    template_similarity: float | None = None
    repeat_query_rate: float | None = None
    ig_health_ratio: float | None = None
    outcome_health_ratio: float | None = None
    selected_domain_concentration: float | None = None


def compute_reasoning_collapse_metrics(
    *,
    has_answer: Sequence[bool],
    search_turn_counts: Sequence[int],
    ig_variances: Sequence[float],
    outcome_variances: Sequence[float],
    advantages: Sequence[float],
    zero_tolerance: float = 1.0e-12,
) -> ReasoningCollapseMetrics:
    answer = np.asarray(has_answer, dtype=np.float64)
    searches = np.asarray(search_turn_counts, dtype=np.float64)
    ig = np.asarray(ig_variances, dtype=np.float64)
    outcome = np.asarray(outcome_variances, dtype=np.float64)
    advantage = np.asarray(advantages, dtype=np.float64)
    return ReasoningCollapseMetrics(
        answer_rate=float(answer.mean()) if answer.size else 0.0,
        mean_search_turns=float(searches.mean()) if searches.size else 0.0,
        multi_search_rate=float((searches >= 2).mean()) if searches.size else 0.0,
        zero_ig_variance_prompt_rate=float((ig <= zero_tolerance).mean())
        if ig.size
        else 0.0,
        zero_outcome_variance_prompt_rate=float(
            (outcome <= zero_tolerance).mean()
        )
        if outcome.size
        else 0.0,
        zero_advantage_rate=float((np.abs(advantage) <= zero_tolerance).mean())
        if advantage.size
        else 0.0,
    )
