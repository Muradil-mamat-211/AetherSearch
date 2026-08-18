from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TopPResult:
    ordered_positive_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    total_mass: float
    selected_mass: float
    selected_mass_ratio: float


def stable_mass_top_p(
    scores: Mapping[str, float],
    *,
    rho: float,
    include_zero: bool = False,
    zero_tolerance: float = 0.0,
) -> TopPResult:
    if not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1]")
    normalized: list[tuple[str, float]] = []
    for prompt_id, score in scores.items():
        numeric = float(score)
        if not np.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Invalid Top-p score for {prompt_id}: {score}")
        if include_zero or numeric > zero_tolerance:
            normalized.append((str(prompt_id), numeric))
    normalized.sort(key=lambda item: (-item[1], item[0]))
    total = float(
        np.sum(np.asarray([score for _, score in normalized], dtype=np.float64), dtype=np.float64)
    )
    if not normalized or total <= zero_tolerance:
        return TopPResult(tuple(), tuple(), total, 0.0, 0.0)

    target = rho * total
    cumulative = np.float64(0.0)
    selected: list[str] = []
    for prompt_id, score in normalized:
        selected.append(prompt_id)
        cumulative = np.float64(cumulative + np.float64(score))
        if cumulative >= target:
            break
    selected_mass = float(cumulative)
    return TopPResult(
        ordered_positive_ids=tuple(prompt_id for prompt_id, _ in normalized),
        selected_ids=tuple(selected),
        total_mass=total,
        selected_mass=selected_mass,
        selected_mass_ratio=selected_mass / total,
    )

