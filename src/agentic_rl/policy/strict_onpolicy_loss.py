from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Collection, Mapping


ADAPTIVE_CLIP_BETA = 0.3
ADAPTIVE_CLIP_EPSILON_LOW = 0.003
ADAPTIVE_CLIP_EPSILON_HIGH = 0.004
ANSWER_CLIP_SCALE = 1.0
CLIPPING_MODE = "a2tgpo_adaptive_turn_level"


@dataclass(frozen=True)
class TurnObjectiveResult:
    objective_by_turn: dict[int, Any]
    ratio_by_turn: dict[int, Any]
    clip_scale_by_turn: dict[int, float]
    lower_bound_by_turn: dict[int, float]
    upper_bound_by_turn: dict[int, float]
    clipped_by_turn: dict[int, bool]


def adaptive_clip_scale(
    normalized_ig: float,
    *,
    beta_c: float = ADAPTIVE_CLIP_BETA,
) -> float:
    """Return the stop-gradient A-squared-TGPO Search-turn clip scale."""
    value = float(normalized_ig)
    beta = float(beta_c)
    if not math.isfinite(value):
        raise ValueError("normalized_ig must be finite")
    if beta != ADAPTIVE_CLIP_BETA:
        raise ValueError(f"beta_c is frozen at {ADAPTIVE_CLIP_BETA}")
    sigmoid = (
        1.0 / (1.0 + math.exp(-value))
        if value >= 0
        else math.exp(value) / (1.0 + math.exp(value))
    )
    scale = 1.0 + beta * (2.0 * sigmoid - 1.0)
    # Finite-precision sigmoid can round to exactly zero or one for large
    # magnitudes. Preserve the mathematical open interval without changing the
    # interior formula.
    lower_open = 1.0 - beta
    upper_open = 1.0 + beta
    if scale <= lower_open:
        scale = math.nextafter(lower_open, 1.0)
    elif scale >= upper_open:
        scale = math.nextafter(upper_open, 1.0)
    return float(scale)


def a2tgpo_adaptive_turn_objective(
    ratio_by_turn: Mapping[int, Any],
    advantage_by_turn: Mapping[int, float],
    normalized_ig_by_search_turn: Mapping[int, float],
    *,
    answer_turn_ids: Collection[int],
    beta_c: float = ADAPTIVE_CLIP_BETA,
    epsilon_low: float = ADAPTIVE_CLIP_EPSILON_LOW,
    epsilon_high: float = ADAPTIVE_CLIP_EPSILON_HIGH,
) -> TurnObjectiveResult:
    """Apply the unique adaptive turn-level clipped surrogate.

    Search turns derive their detached scale from normalized immediate IG.
    Answer/fallback turns use the official neutral scale of one and never
    receive a fabricated IG clipping signal.
    """
    import torch

    if float(beta_c) != ADAPTIVE_CLIP_BETA:
        raise ValueError(f"beta_c is frozen at {ADAPTIVE_CLIP_BETA}")
    if float(epsilon_low) != ADAPTIVE_CLIP_EPSILON_LOW:
        raise ValueError(
            f"epsilon_low is frozen at {ADAPTIVE_CLIP_EPSILON_LOW}"
        )
    if float(epsilon_high) != ADAPTIVE_CLIP_EPSILON_HIGH:
        raise ValueError(
            f"epsilon_high is frozen at {ADAPTIVE_CLIP_EPSILON_HIGH}"
        )
    turn_ids = set(map(int, ratio_by_turn))
    if turn_ids != set(map(int, advantage_by_turn)):
        raise ValueError("Ratio and advantage turn IDs must match")
    search_ids = set(map(int, normalized_ig_by_search_turn))
    answer_ids = set(map(int, answer_turn_ids))
    if search_ids & answer_ids:
        raise ValueError("Search and answer turn IDs must be disjoint")
    if search_ids | answer_ids != turn_ids:
        raise ValueError(
            "Every optimized turn must be exactly one Search or answer/fallback turn"
        )

    objectives: dict[int, Any] = {}
    scales: dict[int, float] = {}
    lowers: dict[int, float] = {}
    uppers: dict[int, float] = {}
    clipped_flags: dict[int, bool] = {}
    for turn_id in sorted(turn_ids):
        ratio = ratio_by_turn[turn_id]
        if not bool(torch.isfinite(ratio.detach()).item()):
            raise ValueError(f"Turn {turn_id} ratio must be finite")
        advantage = torch.as_tensor(
            float(advantage_by_turn[turn_id]),
            dtype=ratio.dtype,
            device=ratio.device,
        ).detach()
        scale = (
            adaptive_clip_scale(
                normalized_ig_by_search_turn[turn_id],
                beta_c=beta_c,
            )
            if turn_id in search_ids
            else ANSWER_CLIP_SCALE
        )
        lower = 1.0 - scale * float(epsilon_low)
        upper = 1.0 + scale * float(epsilon_high)
        clipped_ratio = torch.clamp(ratio, min=lower, max=upper)
        objectives[turn_id] = torch.minimum(
            ratio * advantage,
            clipped_ratio * advantage,
        )
        scales[turn_id] = scale
        lowers[turn_id] = lower
        uppers[turn_id] = upper
        clipped_flags[turn_id] = bool(
            (ratio.detach() < lower).item() or (ratio.detach() > upper).item()
        )
    return TurnObjectiveResult(
        objective_by_turn=objectives,
        ratio_by_turn=dict(ratio_by_turn),
        clip_scale_by_turn=scales,
        lower_bound_by_turn=lowers,
        upper_bound_by_turn=uppers,
        clipped_by_turn=clipped_flags,
    )


def fixed_gate_turn_objective(
    ratio_by_turn: Mapping[int, Any],
    advantage_by_turn: Mapping[int, float],
    *,
    epsilon_low: float = ADAPTIVE_CLIP_EPSILON_LOW,
    epsilon_high: float = ADAPTIVE_CLIP_EPSILON_HIGH,
) -> TurnObjectiveResult:
    """Apply fixed DAPO clipping to Decision or Query segment ratios."""

    import torch

    if float(epsilon_low) != ADAPTIVE_CLIP_EPSILON_LOW:
        raise ValueError(
            f"gate epsilon_low is frozen at {ADAPTIVE_CLIP_EPSILON_LOW}"
        )
    if float(epsilon_high) != ADAPTIVE_CLIP_EPSILON_HIGH:
        raise ValueError(
            f"gate epsilon_high is frozen at {ADAPTIVE_CLIP_EPSILON_HIGH}"
        )
    turn_ids = set(map(int, ratio_by_turn))
    if turn_ids != set(map(int, advantage_by_turn)):
        raise ValueError("Gate ratio and advantage turn IDs must match")
    lower = 1.0 - float(epsilon_low)
    upper = 1.0 + float(epsilon_high)
    objectives: dict[int, Any] = {}
    clipped: dict[int, bool] = {}
    for turn_id in sorted(turn_ids):
        ratio = ratio_by_turn[turn_id]
        if not bool(torch.isfinite(ratio.detach()).item()):
            raise ValueError(f"Gate turn {turn_id} ratio must be finite")
        advantage = torch.as_tensor(
            float(advantage_by_turn[turn_id]),
            dtype=ratio.dtype,
            device=ratio.device,
        ).detach()
        clipped_ratio = torch.clamp(ratio, min=lower, max=upper)
        objectives[turn_id] = torch.minimum(
            ratio * advantage,
            clipped_ratio * advantage,
        )
        clipped[turn_id] = bool(
            (ratio.detach() < lower).item() or (ratio.detach() > upper).item()
        )
    return TurnObjectiveResult(
        objective_by_turn=objectives,
        ratio_by_turn=dict(ratio_by_turn),
        clip_scale_by_turn={turn_id: 1.0 for turn_id in turn_ids},
        lower_bound_by_turn={turn_id: lower for turn_id in turn_ids},
        upper_bound_by_turn={turn_id: upper for turn_id in turn_ids},
        clipped_by_turn=clipped,
    )


def expand_turn_values_to_tokens(
    values_by_turn: Mapping[int, Any],
    turn_ids: Any,
    action_mask: Any,
) -> Any:
    import torch

    result = torch.zeros_like(
        action_mask,
        dtype=next(iter(values_by_turn.values())).dtype,
    )
    for turn_id, value in values_by_turn.items():
        mask = action_mask.bool() & (turn_ids == int(turn_id))
        if not bool(mask.any().item()):
            raise ValueError(f"Turn {turn_id} has no action tokens")
        result = torch.where(mask, value.expand_as(result), result)
    return result


def combine_task_and_kl(
    local_task_prompt_sum: Any,
    local_kl_prompt_sum: Any,
    *,
    global_prompt_count: int,
    world_size: int,
    kl_coefficient: float,
) -> tuple[Any, Any, Any]:
    from .reduction import distributed_local_mean_loss

    local_task = distributed_local_mean_loss(
        local_task_prompt_sum,
        global_prompt_count=global_prompt_count,
        world_size=world_size,
    )
    local_kl = distributed_local_mean_loss(
        local_kl_prompt_sum,
        global_prompt_count=global_prompt_count,
        world_size=world_size,
    )
    total = -local_task + float(kl_coefficient) * local_kl
    return total, local_task, local_kl
