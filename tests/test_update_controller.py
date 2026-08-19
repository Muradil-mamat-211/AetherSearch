from agentic_rl.config import load_config

from config_support import TEST_CONFIG
from agentic_rl.controller.attempt_state import SnapshotVersions, TrainingState
import pytest

from agentic_rl.controller.transaction import UntrustedPostStepState
from agentic_rl.controller.update_controller import StrictAttemptController
from agentic_rl.selection.candidate_pool import PromptGroup


class FakeRuntime:
    def __init__(
        self,
        *,
        sparse_signal: bool = False,
        optimizer_raises: bool = False,
        commit_raises: bool = False,
    ) -> None:
        self.sparse_signal = sparse_signal
        self.next_prompt = 0
        self.cursor_value = 0
        self.zero_grad_count = 0
        self.backward_count = 0
        self.optimizer_count = 0
        self.scheduler_count = 0
        self.optimizer_raises = optimizer_raises
        self.commit_raises = commit_raises
        self.committed_state = None
        self.rollback_count = 0
        self.events = []
        self.collection_sizes = []
        self.selection_rounds = None

    def freeze_rollout_boundary(self, successful_update_step):
        return SnapshotVersions(
            successful_update_step,
            successful_update_step,
            successful_update_step,
            successful_update_step,
        )

    def collect_scored_prompt_groups(self, prompt_count, *, snapshot_step):
        self.collection_sizes.append(int(prompt_count))
        groups = []
        for _ in range(prompt_count):
            index = self.next_prompt
            self.next_prompt += 1
            signal = 1.0
            if self.sparse_signal and index >= 4:
                signal = 0.0
            groups.append(
                PromptGroup(
                    prompt_global_id=f"p{index:03d}",
                    trajectories=tuple(range(16)),
                    ig_variance=signal,
                    outcome_variance=0.0,
                )
            )
        self.cursor_value += prompt_count
        return groups

    def selected_microbatches(self, groups):
        self.events.append("advantages_and_old_logprobs")
        return [groups[: len(groups) // 2], groups[len(groups) // 2 :]]

    def record_selection_metrics(
        self,
        state,
        groups,
        decision,
        *,
        refill_count,
        selection_seconds,
        selection_rounds,
    ):
        del state, groups, decision, selection_seconds
        self.selection_rounds = list(selection_rounds)
        assert refill_count == len(self.selection_rounds) - 1

    def prepare_selected_stop_branches(self, groups):
        assert groups
        self.events.append("stop_branches")

    def zero_grad(self):
        self.events.append("zero_grad")
        self.zero_grad_count += 1

    def actor_parameter_checksum(self):
        return "unchanged"

    def backward_microbatch(self, microbatch):
        assert microbatch
        self.backward_count += 1

    def clip_gradients(self, max_grad_norm):
        return 0.5

    def optimizer_step(self):
        self.optimizer_count += 1
        if self.optimizer_raises:
            raise RuntimeError("optimizer failed after entering step")

    def scheduler_step(self):
        self.scheduler_count += 1

    def rollback_pre_step_attempt(self):
        self.rollback_count += 1

    def commit_successful_update(self, state):
        if self.commit_raises:
            raise RuntimeError("durable checkpoint commit failed")
        self.committed_state = state

    def data_cursor(self):
        return self.cursor_value

    def rng_state(self):
        return {"test": 1}


def test_controller_caps_to_36_and_performs_one_step() -> None:
    controller = StrictAttemptController(load_config(TEST_CONFIG))
    runtime = FakeRuntime()
    result = controller.run_attempt(TrainingState(), runtime)
    assert result.optimizer_committed
    assert result.selection.selected_count == 36
    assert result.state.successful_update_step == 1
    assert runtime.zero_grad_count == 1
    assert runtime.backward_count == 2
    assert runtime.optimizer_count == 1
    assert runtime.scheduler_count == 1
    assert runtime.committed_state == result.state
    assert runtime.events[:3] == [
        "stop_branches",
        "advantages_and_old_logprobs",
        "zero_grad",
    ]
    assert runtime.selection_rounds == [
        {
            "pool_size": 64,
            "selected_count": 36,
            "requires_refill": False,
            "skip_update": False,
        }
    ]


def test_controller_refills_full_pool_then_skips_without_step() -> None:
    controller = StrictAttemptController(load_config(TEST_CONFIG))
    runtime = FakeRuntime(sparse_signal=True)
    result = controller.run_attempt(TrainingState(), runtime)
    assert not result.optimizer_committed
    assert result.selection.candidate_count == 128
    assert result.selection.selected_count == 4
    assert result.state.attempt_id == 1
    assert result.state.successful_update_step == 0
    assert result.state.data_cursor == 128
    assert result.state.ig_channel.valid_success_count == 0
    assert result.state.outcome_channel.valid_success_count == 0
    assert result.state.ig_channel.health_reference is None
    assert result.state.outcome_channel.health_reference is None
    assert runtime.optimizer_count == 0
    assert runtime.collection_sizes == [32, 32, 32, 32]
    assert [row["pool_size"] for row in runtime.selection_rounds] == [64, 96, 128]
    assert [row["requires_refill"] for row in runtime.selection_rounds] == [
        True,
        True,
        False,
    ]
    assert runtime.selection_rounds[-1]["skip_update"] is True


def test_controller_second_refill_recomputes_full_128_pool_and_can_succeed() -> None:
    class SecondRefillRuntime(FakeRuntime):
        def collect_scored_prompt_groups(self, prompt_count, *, snapshot_step):
            groups = super().collect_scored_prompt_groups(
                prompt_count,
                snapshot_step=snapshot_step,
            )
            rebuilt = []
            for group in groups:
                index = int(group.prompt_global_id[1:])
                signal = 1.0 if index < 34 else (1.1 if index >= 96 else 0.0)
                rebuilt.append(
                    PromptGroup(
                        prompt_global_id=group.prompt_global_id,
                        trajectories=group.trajectories,
                        ig_variance=signal,
                        outcome_variance=0.0,
                    )
                )
            return rebuilt

    controller = StrictAttemptController(load_config(TEST_CONFIG))
    runtime = SecondRefillRuntime()
    result = controller.run_attempt(TrainingState(), runtime)

    assert result.optimizer_committed
    assert result.selection.candidate_count == 128
    assert result.selection.selected_count == 36
    assert result.state.data_cursor == 128
    assert runtime.collection_sizes == [32, 32, 32, 32]
    assert [row["pool_size"] for row in runtime.selection_rounds] == [64, 96, 128]
    assert runtime.selection_rounds[1]["requires_refill"] is True
    assert runtime.selection_rounds[2]["requires_refill"] is False
    assert any(value.startswith("p1") for value in result.selection.selected_ids)


def test_optimizer_call_failure_never_uses_pre_step_rollback() -> None:
    controller = StrictAttemptController(load_config(TEST_CONFIG))
    runtime = FakeRuntime(optimizer_raises=True)
    state = TrainingState()
    with pytest.raises(UntrustedPostStepState):
        controller.run_attempt(state, runtime)
    assert runtime.optimizer_count == 1
    assert runtime.rollback_count == 0
    assert state.ig_channel.valid_success_count == 0
    assert state.outcome_channel.valid_success_count == 0
    assert state.ig_channel.health_reference is None
    assert state.outcome_channel.health_reference is None


def test_durable_commit_failure_marks_process_state_untrusted() -> None:
    controller = StrictAttemptController(load_config(TEST_CONFIG))
    runtime = FakeRuntime(commit_raises=True)
    with pytest.raises(UntrustedPostStepState):
        controller.run_attempt(TrainingState(), runtime)
    assert runtime.optimizer_count == 1
    assert runtime.scheduler_count == 1
    assert runtime.rollback_count == 0
