from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TrajectoryTokenValues:
    prompt_global_id: str
    trajectory_id: str
    values: Any
    policy_mask: Any

    @property
    def action_mask(self) -> Any:
        """Compatibility alias; reduction always consumes the policy mask."""
        return self.policy_mask


@dataclass(frozen=True)
class NestedReductionResult:
    local_prompt_sum: Any
    local_prompt_count: int
    local_trajectory_count: int
    local_action_token_count: int
    prompt_means: dict[str, Any]


def prompt_trajectory_action_token_reduce(
    records: Sequence[TrajectoryTokenValues],
    *,
    expected_group_size: int,
) -> NestedReductionResult:
    import torch

    if not records:
        raise ValueError("At least one local trajectory is required")
    grouped: dict[str, list[Any]] = {}
    action_count = 0
    seen_trajectories: set[tuple[str, str]] = set()
    for record in records:
        identity = (record.prompt_global_id, record.trajectory_id)
        if identity in seen_trajectories:
            raise ValueError(
                f"Duplicate trajectory in reduction: {record.prompt_global_id}/"
                f"{record.trajectory_id}"
            )
        seen_trajectories.add(identity)
        if record.values.shape != record.policy_mask.shape:
            raise ValueError("values and policy_mask must align")
        mask = record.policy_mask.bool()
        token_count = int(mask.sum().detach().cpu().item())
        if token_count < 1:
            raise ValueError(
                f"Trajectory {record.trajectory_id} has no action tokens"
            )
        action_count += token_count
        trajectory_mean = record.values[mask].mean()
        grouped.setdefault(record.prompt_global_id, []).append(trajectory_mean)

    prompt_means: dict[str, Any] = {}
    for prompt_id, trajectory_means in grouped.items():
        if len(trajectory_means) != expected_group_size:
            raise ValueError(
                f"Prompt {prompt_id} has {len(trajectory_means)} trajectories; "
                f"expected {expected_group_size}"
            )
        prompt_means[prompt_id] = torch.stack(trajectory_means).mean()
    local_prompt_sum = torch.stack(
        [prompt_means[prompt_id] for prompt_id in sorted(prompt_means)]
    ).sum()
    return NestedReductionResult(
        local_prompt_sum=local_prompt_sum,
        local_prompt_count=len(prompt_means),
        local_trajectory_count=len(records),
        local_action_token_count=action_count,
        prompt_means=prompt_means,
    )


def distributed_local_mean_loss(
    local_prompt_sum: Any,
    *,
    global_prompt_count: int,
    world_size: int,
) -> Any:
    if global_prompt_count <= 0:
        raise ValueError("global_prompt_count must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    # DDP/FSDP averages rank gradients. Multiplication by world_size makes
    # the averaged local gradient equal the exact global prompt mean.
    return local_prompt_sum * (float(world_size) / float(global_prompt_count))
