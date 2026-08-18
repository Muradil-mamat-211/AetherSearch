import pytest

from agentic_rl.controller.attempt_state import SnapshotVersions, TrainingState
from agentic_rl.controller.transaction import (
    StrictUpdateTransaction,
    TransactionError,
    UntrustedPostStepState,
)
from agentic_rl.selection.channel_scale import ChannelScaleState


def _stats():
    state = ChannelScaleState()
    return state.inspect_pool(
        {f"p{index}": float(index + 1) for index in range(8)},
        noise_floor=0.0,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        allow_provisional_scale=True,
    )


def _selected_transaction() -> StrictUpdateTransaction:
    transaction = StrictUpdateTransaction(TrainingState())
    transaction.freeze_snapshot(SnapshotVersions(0, 0, 0, 0))
    transaction.complete_rollout(candidate_prompt_count=64, refill_count=0)
    transaction.complete_scoring()
    transaction.select(32)
    return transaction


def test_successful_update_exactly_one_global_step() -> None:
    transaction = _selected_transaction()
    transaction.record_zero_grad()
    transaction.record_backward_microbatch("same")
    transaction.record_backward_microbatch("same")
    transaction.complete_backward("same")
    transaction.begin_optimizer_step()
    transaction.record_optimizer_step()
    transaction.begin_scheduler_step()
    transaction.record_scheduler_step()
    state = transaction.propose_commit(
        ig_stats=_stats(),
        outcome_stats=_stats(),
        ema_half_life=10,
        health_reference_valid_updates=10,
        data_cursor=64,
        rng_state={"python": "opaque"},
    )
    transaction.record_commit_success()
    assert state.successful_update_step == 1
    assert state.attempt_id == 1
    assert transaction.record.optimizer_steps == 1
    assert transaction.record.scheduler_steps == 1


def test_microbatch_parameter_mutation_is_rejected() -> None:
    transaction = _selected_transaction()
    transaction.record_zero_grad()
    transaction.record_backward_microbatch("before")
    with pytest.raises(TransactionError, match="changed between micro-batches"):
        transaction.record_backward_microbatch("after")


def test_last_backward_parameter_mutation_is_rejected() -> None:
    transaction = _selected_transaction()
    transaction.record_zero_grad()
    transaction.record_backward_microbatch("before")
    with pytest.raises(TransactionError, match="during backward accumulation"):
        transaction.complete_backward("after")


def test_skip_after_refill_advances_attempt_and_cursor_only() -> None:
    transaction = StrictUpdateTransaction(TrainingState(data_cursor=10))
    transaction.freeze_snapshot(SnapshotVersions(0, 0, 0, 0))
    transaction.complete_rollout(candidate_prompt_count=128, refill_count=2)
    transaction.complete_scoring()
    state = transaction.skip("selected_below_32", data_cursor=138)
    assert state.attempt_id == 1
    assert state.successful_update_step == 0
    assert state.data_cursor == 138


def test_post_step_precommit_failure_requires_restore() -> None:
    transaction = _selected_transaction()
    transaction.record_zero_grad()
    transaction.record_backward_microbatch("same")
    transaction.complete_backward("same")
    transaction.begin_optimizer_step()
    transaction.record_optimizer_step()
    with pytest.raises(UntrustedPostStepState):
        transaction.abort("metadata writer failed")


def test_pre_step_failure_is_locally_rollback_eligible() -> None:
    transaction = _selected_transaction()
    transaction.record_zero_grad()
    transaction.record_backward_microbatch("same")
    transaction.complete_backward("same")
    assert transaction.can_rollback_pre_step
    state = transaction.abort("PRE_STEP_FAILURE")
    assert state.successful_update_step == 0


def test_optimizer_call_failure_is_post_step_untrusted_even_before_return() -> None:
    transaction = _selected_transaction()
    transaction.record_zero_grad()
    transaction.record_backward_microbatch("same")
    transaction.complete_backward("same")
    transaction.begin_optimizer_step()
    assert not transaction.can_rollback_pre_step
    with pytest.raises(UntrustedPostStepState):
        transaction.abort("POST_STEP_PRE_COMMIT_FAILURE")


@pytest.mark.parametrize(
    "failure_name",
    ["CHECKPOINT_WRITE_FAILURE", "LATEST_POINTER_FAILURE"],
)
def test_durable_commit_failure_requires_resume_from_last_success(
    failure_name,
) -> None:
    transaction = _selected_transaction()
    transaction.record_zero_grad()
    transaction.record_backward_microbatch("same")
    transaction.complete_backward("same")
    transaction.begin_optimizer_step()
    transaction.record_optimizer_step()
    transaction.begin_scheduler_step()
    transaction.record_scheduler_step()
    transaction.propose_commit(
        ig_stats=_stats(),
        outcome_stats=_stats(),
        ema_half_life=10,
        health_reference_valid_updates=10,
        data_cursor=64,
        rng_state={},
    )
    with pytest.raises(UntrustedPostStepState, match="restore"):
        transaction.abort(failure_name)


def test_resume_from_last_success_state_does_not_include_failed_proposal() -> None:
    last_success = TrainingState(
        attempt_id=4,
        successful_update_step=3,
        data_cursor=192,
    )
    transaction = StrictUpdateTransaction(last_success)
    transaction.freeze_snapshot(SnapshotVersions(3, 3, 3, 3))
    transaction.complete_rollout(candidate_prompt_count=64, refill_count=0)
    transaction.complete_scoring()
    state = transaction.abort("RESUME_FROM_LAST_SUCCESS")
    assert state.successful_update_step == 3
    assert state.data_cursor == 192
    assert state.attempt_id == 5
