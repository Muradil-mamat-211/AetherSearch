from __future__ import annotations

from agentic_rl.controller.attempt_state import SnapshotVersions, TrainingState
from agentic_rl.controller.transaction import StrictUpdateTransaction
from agentic_rl.selection.candidate_pool import CandidatePool, PromptGroup
from agentic_rl.selection.channel_scale import ChannelScaleState


def exercise_forced_skip_transaction() -> dict[str, object]:
    """Exercise the 128-prompt underflow path without a runtime or optimizer."""

    pool = CandidatePool(group_size=16, maximum_prompts=128)
    pool.add(
        PromptGroup(
            prompt_global_id=f"forced-skip-{index:03d}",
            trajectories=tuple(range(16)),
            ig_variance=1.0 if index < 4 else 0.0,
            outcome_variance=0.0,
        )
        for index in range(128)
    )
    state = TrainingState(
        data_cursor=321,
        ig_channel=ChannelScaleState(committed_scale=1.0),
        outcome_channel=ChannelScaleState(committed_scale=1.0),
    )
    decision = pool.select(
        ig_state=state.ig_channel,
        outcome_state=state.outcome_channel,
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=1.0e-12,
        noise_floor_outcome=1.0e-12,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=False,
    )
    if not decision.skip_update or decision.requires_refill:
        raise RuntimeError("Synthetic 128-prompt pool did not enter skip")
    transaction = StrictUpdateTransaction(state)
    transaction.freeze_snapshot(SnapshotVersions(0, 0, 0, 0))
    transaction.complete_rollout(candidate_prompt_count=128, refill_count=2)
    transaction.complete_scoring()
    next_state = transaction.skip(
        "selected_below_minimum_after_refill",
        data_cursor=999,
    )
    checks = {
        "optimizer_steps": transaction.record.optimizer_steps,
        "scheduler_steps": transaction.record.scheduler_steps,
        "successful_update_before": state.successful_update_step,
        "successful_update_after": next_state.successful_update_step,
        "scale_unchanged": (
            next_state.ig_channel == state.ig_channel
            and next_state.outcome_channel == state.outcome_channel
        ),
        "health_unchanged": (
            next_state.ig_channel.health_observations
            == state.ig_channel.health_observations
            and next_state.outcome_channel.health_observations
            == state.outcome_channel.health_observations
        ),
        "selected_count": decision.selected_count,
        "skip_update": decision.skip_update,
        "ephemeral_cursor_after": next_state.data_cursor,
        "formal_cursor_persisted": False,
    }
    if checks != {
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "successful_update_before": 0,
        "successful_update_after": 0,
        "scale_unchanged": True,
        "health_unchanged": True,
        "selected_count": 4,
        "skip_update": True,
        "ephemeral_cursor_after": 999,
        "formal_cursor_persisted": False,
    }:
        raise RuntimeError(f"Forced-skip transaction contract failed: {checks}")
    return {"status": "PASS", **checks}
