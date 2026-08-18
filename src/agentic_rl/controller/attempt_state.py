from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentic_rl.selection.channel_scale import ChannelScaleState


class AttemptPhase(str, Enum):
    CREATED = "created"
    SNAPSHOT_FROZEN = "snapshot_frozen"
    ROLLOUT_COMPLETE = "rollout_complete"
    SCORED = "scored"
    SELECTED = "selected"
    BACKWARD_COMPLETE = "backward_complete"
    OPTIMIZER_STEP_IN_PROGRESS = "optimizer_step_in_progress"
    OPTIMIZER_STEPPED = "optimizer_stepped"
    SCHEDULER_STEP_IN_PROGRESS = "scheduler_step_in_progress"
    SCHEDULER_STEPPED = "scheduler_stepped"
    COMMIT_IN_PROGRESS = "commit_in_progress"
    COMMITTED = "committed"
    SKIPPED = "skipped"
    ABORTED = "aborted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class SnapshotVersions:
    actor: int
    rollout: int
    old_policy: int
    reward_policy: int

    def assert_rollout_boundary_parity(self) -> None:
        versions = {
            self.actor,
            self.rollout,
            self.old_policy,
            self.reward_policy,
        }
        if len(versions) != 1:
            raise RuntimeError(
                "Rollout-start snapshot mismatch: "
                f"actor={self.actor}, rollout={self.rollout}, "
                f"old={self.old_policy}, reward={self.reward_policy}"
            )


@dataclass(frozen=True)
class TrainingState:
    attempt_id: int = 0
    successful_update_step: int = 0
    data_cursor: int = 0
    ig_channel: ChannelScaleState = field(default_factory=ChannelScaleState)
    outcome_channel: ChannelScaleState = field(default_factory=ChannelScaleState)
    rng_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptRecord:
    attempt_id: int
    starting_successful_update_step: int
    phase: AttemptPhase = AttemptPhase.CREATED
    snapshots: SnapshotVersions | None = None
    candidate_prompt_count: int = 0
    selected_prompt_count: int = 0
    refill_count: int = 0
    skip_reason: str | None = None
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    zero_grad_calls: int = 0
    backward_microbatches: int = 0
    pre_step_parameter_checksums: list[str] = field(default_factory=list)
