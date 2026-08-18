from __future__ import annotations

from typing import Any, Sequence


def compute_turn_ratios(
    current_logprobs: Any,
    old_logprobs: Any,
    action_mask: Any,
    turn_ids: Any,
    *,
    expected_turn_ids: Sequence[int] | None = None,
) -> dict[int, Any]:
    import torch

    if not (
        current_logprobs.shape
        == old_logprobs.shape
        == action_mask.shape
        == turn_ids.shape
    ):
        raise ValueError("Logprobs, action mask, and turn IDs must share a shape")
    if old_logprobs.requires_grad:
        raise ValueError("Old-policy logprobs must be detached")
    if not current_logprobs.requires_grad:
        raise ValueError("Current-policy logprobs must retain gradients")
    if not torch.isfinite(current_logprobs).all() or not torch.isfinite(
        old_logprobs
    ).all():
        raise ValueError("Policy logprobs must be finite")

    present = torch.unique(turn_ids[action_mask.bool()]).detach().cpu().tolist()
    ordered = (
        [int(turn_id) for turn_id in expected_turn_ids]
        if expected_turn_ids is not None
        else sorted(int(turn_id) for turn_id in present)
    )
    ratios: dict[int, Any] = {}
    for turn_id in ordered:
        mask = action_mask.bool() & (turn_ids == int(turn_id))
        token_count = int(mask.sum().detach().cpu().item())
        if token_count == 0:
            raise ValueError(f"Turn {turn_id} has no action tokens")
        mean_log_ratio = (current_logprobs[mask] - old_logprobs[mask]).mean()
        ratio = torch.exp(mean_log_ratio)
        if not bool(torch.isfinite(ratio.detach()).item()):
            raise ValueError(f"Turn {turn_id} produced a non-finite ratio")
        ratios[int(turn_id)] = ratio
    return ratios
