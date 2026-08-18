from __future__ import annotations

from dataclasses import replace

from agentic_rl.selection.channel_scale import ChannelPoolStats

from .attempt_state import AttemptPhase, AttemptRecord, SnapshotVersions, TrainingState


class TransactionError(RuntimeError):
    pass


class UntrustedPostStepState(TransactionError):
    pass


class StrictUpdateTransaction:
    def __init__(
        self,
        state: TrainingState,
        *,
        allowed_candidate_counts: tuple[int, ...] = (64, 96, 128),
        minimum_selected_prompts: int = 32,
        maximum_selected_prompts: int = 36,
    ) -> None:
        if not allowed_candidate_counts or any(
            count <= 0 for count in allowed_candidate_counts
        ):
            raise ValueError("allowed_candidate_counts must be positive")
        if (
            minimum_selected_prompts <= 0
            or maximum_selected_prompts < minimum_selected_prompts
        ):
            raise ValueError("Invalid selected-prompt transaction bounds")
        self.allowed_candidate_counts = tuple(
            int(count) for count in allowed_candidate_counts
        )
        self.minimum_selected_prompts = int(minimum_selected_prompts)
        self.maximum_selected_prompts = int(maximum_selected_prompts)
        self.starting_state = state
        self.record = AttemptRecord(
            attempt_id=state.attempt_id + 1,
            starting_successful_update_step=state.successful_update_step,
        )

    def _require(self, *allowed: AttemptPhase) -> None:
        if self.record.phase not in allowed:
            raise TransactionError(
                f"Invalid transition from {self.record.phase.value}; "
                f"expected one of {[phase.value for phase in allowed]}"
            )

    @property
    def can_rollback_pre_step(self) -> bool:
        return self.record.phase in {
            AttemptPhase.CREATED,
            AttemptPhase.SNAPSHOT_FROZEN,
            AttemptPhase.ROLLOUT_COMPLETE,
            AttemptPhase.SCORED,
            AttemptPhase.SELECTED,
            AttemptPhase.BACKWARD_COMPLETE,
        }

    def freeze_snapshot(self, versions: SnapshotVersions) -> None:
        self._require(AttemptPhase.CREATED)
        versions.assert_rollout_boundary_parity()
        self.record.snapshots = versions
        self.record.phase = AttemptPhase.SNAPSHOT_FROZEN

    def complete_rollout(self, *, candidate_prompt_count: int, refill_count: int) -> None:
        self._require(AttemptPhase.SNAPSHOT_FROZEN)
        if candidate_prompt_count not in self.allowed_candidate_counts:
            raise TransactionError(
                "Candidate pool count is outside the configured transaction "
                f"contract: {candidate_prompt_count} not in "
                f"{self.allowed_candidate_counts}"
            )
        self.record.candidate_prompt_count = int(candidate_prompt_count)
        self.record.refill_count = int(refill_count)
        self.record.phase = AttemptPhase.ROLLOUT_COMPLETE

    def complete_scoring(self) -> None:
        self._require(AttemptPhase.ROLLOUT_COMPLETE)
        self.record.phase = AttemptPhase.SCORED

    def select(self, selected_prompt_count: int) -> None:
        self._require(AttemptPhase.SCORED)
        if not (
            self.minimum_selected_prompts
            <= selected_prompt_count
            <= self.maximum_selected_prompts
        ):
            raise TransactionError(
                "A successful selection is outside the configured transaction "
                f"bounds [{self.minimum_selected_prompts}, "
                f"{self.maximum_selected_prompts}]"
            )
        self.record.selected_prompt_count = int(selected_prompt_count)
        self.record.phase = AttemptPhase.SELECTED

    def skip(self, reason: str, *, data_cursor: int) -> TrainingState:
        self._require(AttemptPhase.SCORED)
        if self.record.candidate_prompt_count != max(self.allowed_candidate_counts):
            raise TransactionError(
                "Underflow may be skipped only after the maximum candidate pool"
            )
        self.record.skip_reason = str(reason)
        self.record.phase = AttemptPhase.SKIPPED
        return replace(
            self.starting_state,
            attempt_id=self.record.attempt_id,
            data_cursor=int(data_cursor),
        )

    def record_zero_grad(self) -> None:
        self._require(AttemptPhase.SELECTED)
        self.record.zero_grad_calls += 1
        if self.record.zero_grad_calls > 1:
            raise TransactionError("zero_grad may be called only once")

    def record_backward_microbatch(self, parameter_checksum: str) -> None:
        self._require(AttemptPhase.SELECTED)
        if self.record.zero_grad_calls != 1:
            raise TransactionError("zero_grad must occur before backward")
        if (
            self.record.pre_step_parameter_checksums
            and parameter_checksum != self.record.pre_step_parameter_checksums[0]
        ):
            raise TransactionError("Actor parameters changed between micro-batches")
        self.record.pre_step_parameter_checksums.append(str(parameter_checksum))
        self.record.backward_microbatches += 1

    def complete_backward(self, final_parameter_checksum: str) -> None:
        self._require(AttemptPhase.SELECTED)
        if self.record.backward_microbatches < 1:
            raise TransactionError("At least one backward micro-batch is required")
        if (
            not self.record.pre_step_parameter_checksums
            or str(final_parameter_checksum)
            != self.record.pre_step_parameter_checksums[0]
        ):
            raise TransactionError(
                "Actor parameters changed during backward accumulation"
            )
        self.record.phase = AttemptPhase.BACKWARD_COMPLETE

    def begin_optimizer_step(self) -> None:
        self._require(AttemptPhase.BACKWARD_COMPLETE)
        if self.record.optimizer_steps != 0:
            raise TransactionError("optimizer.step may begin only once")
        self.record.phase = AttemptPhase.OPTIMIZER_STEP_IN_PROGRESS

    def record_optimizer_step(self) -> None:
        self._require(AttemptPhase.OPTIMIZER_STEP_IN_PROGRESS)
        self.record.optimizer_steps += 1
        if self.record.optimizer_steps != 1:
            raise TransactionError("Exactly one optimizer.step is permitted")
        self.record.phase = AttemptPhase.OPTIMIZER_STEPPED

    def begin_scheduler_step(self) -> None:
        self._require(AttemptPhase.OPTIMIZER_STEPPED)
        if self.record.scheduler_steps != 0:
            raise TransactionError("scheduler.step may begin only once")
        self.record.phase = AttemptPhase.SCHEDULER_STEP_IN_PROGRESS

    def record_scheduler_step(self) -> None:
        self._require(AttemptPhase.SCHEDULER_STEP_IN_PROGRESS)
        self.record.scheduler_steps += 1
        if self.record.scheduler_steps != 1:
            raise TransactionError("Exactly one scheduler.step is permitted")
        self.record.phase = AttemptPhase.SCHEDULER_STEPPED

    def propose_commit(
        self,
        *,
        ig_stats: ChannelPoolStats,
        outcome_stats: ChannelPoolStats,
        ema_half_life: float,
        health_reference_valid_updates: int,
        data_cursor: int,
        rng_state: dict,
    ) -> TrainingState:
        self._require(AttemptPhase.SCHEDULER_STEPPED)
        if self.record.optimizer_steps != 1 or self.record.scheduler_steps != 1:
            raise TransactionError("Optimizer and scheduler must each step exactly once")
        next_state = TrainingState(
            attempt_id=self.record.attempt_id,
            successful_update_step=self.starting_state.successful_update_step + 1,
            data_cursor=int(data_cursor),
            ig_channel=self.starting_state.ig_channel.committed_after_success(
                ig_stats,
                ema_half_life=ema_half_life,
                health_reference_valid_updates=health_reference_valid_updates,
                allow_initialization=self.starting_state.successful_update_step == 0,
            ),
            outcome_channel=self.starting_state.outcome_channel.committed_after_success(
                outcome_stats,
                ema_half_life=ema_half_life,
                health_reference_valid_updates=health_reference_valid_updates,
                allow_initialization=self.starting_state.successful_update_step == 0,
            ),
            rng_state=dict(rng_state),
        )
        self.record.phase = AttemptPhase.COMMIT_IN_PROGRESS
        return next_state

    def record_commit_success(self) -> None:
        self._require(AttemptPhase.COMMIT_IN_PROGRESS)
        self.record.phase = AttemptPhase.COMMITTED

    def abort(self, reason: str, *, data_cursor: int | None = None) -> TrainingState:
        if self.record.phase in {
            AttemptPhase.OPTIMIZER_STEP_IN_PROGRESS,
            AttemptPhase.OPTIMIZER_STEPPED,
            AttemptPhase.SCHEDULER_STEP_IN_PROGRESS,
            AttemptPhase.SCHEDULER_STEPPED,
            AttemptPhase.COMMIT_IN_PROGRESS,
        }:
            self.record.phase = AttemptPhase.UNTRUSTED
            raise UntrustedPostStepState(
                f"Failure after optimizer.step and before commit: {reason}. "
                "Terminate and restore the last successful checkpoint."
            )
        if self.record.phase in {
            AttemptPhase.COMMITTED,
            AttemptPhase.SKIPPED,
            AttemptPhase.UNTRUSTED,
        }:
            raise TransactionError(f"Cannot abort finalized phase {self.record.phase.value}")
        self.record.skip_reason = str(reason)
        self.record.phase = AttemptPhase.ABORTED
        return replace(
            self.starting_state,
            attempt_id=self.record.attempt_id,
            data_cursor=(
                self.starting_state.data_cursor
                if data_cursor is None
                else int(data_cursor)
            ),
        )
