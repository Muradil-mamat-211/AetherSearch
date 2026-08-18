from __future__ import annotations

from typing import Sequence

import numpy as np


def build_structural_attention_mask(
    original_token_count: int,
    prefix_end_positions: Sequence[int],
    segment_lengths: Sequence[int],
    *,
    original_attention_mask: Sequence[int],
) -> np.ndarray:
    if len(prefix_end_positions) != len(segment_lengths):
        raise ValueError("prefix_end_positions and segment_lengths must align")
    if original_token_count <= 0:
        raise ValueError("Original trajectory must contain at least one token")
    if any(end <= 0 or end > original_token_count for end in prefix_end_positions):
        raise ValueError("Prefix endpoints must be in [1, original_token_count]")
    if any(length <= 0 for length in segment_lengths):
        raise ValueError("Each GT copy must contain target tokens")
    original_keys = np.asarray(original_attention_mask, dtype=np.int64)
    if original_keys.shape != (original_token_count,):
        raise ValueError("original_attention_mask shape mismatch")
    if not np.all((original_keys == 0) | (original_keys == 1)):
        raise ValueError("original_attention_mask must be binary")

    total_length = original_token_count + sum(int(length) for length in segment_lengths)
    mask = np.zeros((total_length, total_length), dtype=np.bool_)
    mask[:original_token_count, :original_token_count] = np.tril(
        np.ones((original_token_count, original_token_count), dtype=np.bool_)
    ) & original_keys.astype(np.bool_)[None, :]

    segment_start = original_token_count
    for prefix_end, segment_length in zip(prefix_end_positions, segment_lengths):
        segment_end = segment_start + int(segment_length)
        for query_position in range(segment_start, segment_end):
            mask[query_position, : int(prefix_end)] = original_keys[
                : int(prefix_end)
            ].astype(np.bool_)
            mask[query_position, segment_start : query_position + 1] = True
        segment_start = segment_end
    return mask


def to_additive_attention_mask(
    boolean_mask: np.ndarray,
    *,
    masked_value: float = float("-inf"),
) -> np.ndarray:
    if boolean_mask.ndim != 2:
        raise ValueError("Structural attention mask must be rank 2")
    return np.where(boolean_mask, 0.0, masked_value).astype(np.float32)
