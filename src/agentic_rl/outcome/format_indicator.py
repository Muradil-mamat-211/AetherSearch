from __future__ import annotations

from typing import Sequence

import numpy as np


def centered_format_advantage(indicators: Sequence[int]) -> np.ndarray:
    values = np.asarray(indicators, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Format indicators must be a non-empty 1D sequence")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("Format indicators must be binary")
    return values - np.mean(values, dtype=np.float64)
