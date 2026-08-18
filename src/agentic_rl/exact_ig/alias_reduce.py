from __future__ import annotations

import math
from typing import Sequence


def immediate_ig_from_prefix_scores(
    prefix_scores: Sequence[float],
) -> tuple[float, ...]:
    if len(prefix_scores) < 2:
        return tuple()
    return tuple(
        float(prefix_scores[index] - prefix_scores[index - 1])
        for index in range(1, len(prefix_scores))
    )


def telescoping_error(
    prefix_scores: Sequence[float],
    immediate_ig: Sequence[float],
) -> float:
    if len(prefix_scores) != len(immediate_ig) + 1:
        raise ValueError("Prefix scores and immediate IG lengths do not align")
    if not prefix_scores:
        return 0.0
    return abs(
        float(math.fsum(float(value) for value in immediate_ig))
        - float(prefix_scores[-1] - prefix_scores[0])
    )
