from __future__ import annotations

from typing import Sequence

import numpy as np


def build_logical_position_ids(
    original_position_ids: Sequence[int],
    prefix_end_positions: Sequence[int],
    target_lengths: Sequence[int],
    *,
    maximum_position_id_exclusive: int,
) -> np.ndarray:
    original = np.asarray(original_position_ids, dtype=np.int64)
    if original.ndim != 1 or original.size == 0:
        raise ValueError("original_position_ids must be a non-empty 1D sequence")
    if len(prefix_end_positions) != len(target_lengths):
        raise ValueError("prefix_end_positions and target_lengths must align")
    if maximum_position_id_exclusive <= 0:
        raise ValueError("maximum_position_id_exclusive must be positive")
    if np.any(original < 0):
        raise ValueError("Position IDs must be non-negative")

    segments: list[np.ndarray] = [original]
    for prefix_end, target_length in zip(prefix_end_positions, target_lengths):
        if prefix_end <= 0 or prefix_end > original.size:
            raise ValueError("Invalid prefix endpoint")
        if target_length <= 0:
            raise ValueError("Target length must be positive")
        first_target_position = int(original[prefix_end - 1]) + 1
        if first_target_position + int(target_length) > maximum_position_id_exclusive:
            raise ValueError(
                "Exact-IG logical position would exceed the model context limit"
            )
        segments.append(
            np.arange(
                first_target_position,
                first_target_position + int(target_length),
                dtype=np.int64,
            )
        )
    result = np.concatenate(segments)
    if np.any(result >= maximum_position_id_exclusive):
        raise ValueError("Position ID exceeds the model context limit")
    return result
