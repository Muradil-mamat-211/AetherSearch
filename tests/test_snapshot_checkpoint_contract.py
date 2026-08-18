import pytest
import torch

from agentic_rl.checkpoint.state_schema import (
    ChannelCheckpointState,
    CheckpointMetadata,
)
from agentic_rl.controller.snapshot import (
    SnapshotCoordinator,
    assert_snapshot_logprob_parity,
)
from agentic_rl.rollout.vllm_manager import VersionedVLLMManager


class FakeReplica:
    def sleep(self):
        pass

    def wake_up(self):
        pass


def test_snapshot_detects_actor_mutation_before_optimizer_boundary() -> None:
    actor = torch.nn.Linear(2, 2)
    synchronized = []
    def synchronize(step):
        synchronized.append(step)
        return step

    boundary = SnapshotCoordinator().freeze(
        actor=actor,
        successful_update_step=7,
        synchronize_vllm=synchronize,
        materialize_old_policy=lambda step: step,
        materialize_reward_policy=lambda step: step,
        reference_model_fingerprint="reference",
    )
    boundary.validate_after_pre_step_work(actor)
    assert synchronized == [7]
    with torch.no_grad():
        actor.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="changed"):
        boundary.validate_after_pre_step_work(actor)


def test_snapshot_versions_are_sourced_from_each_materialization_callback() -> None:
    actor = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="snapshot mismatch"):
        SnapshotCoordinator().freeze(
            actor=actor,
            successful_update_step=3,
            synchronize_vllm=lambda step: step,
            materialize_old_policy=lambda step: step - 1,
            materialize_reward_policy=lambda step: step,
            reference_model_fingerprint="reference",
        )


def test_snapshot_teacher_forced_logprob_parity_without_optimizer_step() -> None:
    actor = torch.tensor([-1.0, -2.0, -3.0])
    old = actor.clone()
    reward = actor.clone() + torch.tensor([0.0, 1.0e-7, 0.0])
    maximum = assert_snapshot_logprob_parity(
        actor,
        old,
        reward,
        absolute_tolerance=1.0e-6,
    )
    assert maximum <= 1.0e-6


def test_vllm_manager_requires_four_equal_versioned_replicas() -> None:
    replicas = [FakeReplica() for _ in range(4)]
    manager = VersionedVLLMManager(
        replicas,
        data_parallel_size=4,
        tensor_parallel_size=1,
    )
    versions = manager.synchronize_weights(
        snapshot_step=3,
        sync_one_replica=lambda index, replica: "same",
    )
    assert len(versions) == 4
    manager.assert_snapshot(3)
    manager.wake_for_rollout()
    manager.sleep_after_rollout()


def test_checkpoint_metadata_contains_resumable_state_without_writing() -> None:
    channel = ChannelCheckpointState(1.0, (0.5,), None, 1)
    metadata = CheckpointMetadata(
        schema_version=1,
        successful_update_step=1,
        attempt_id=2,
        data_cursor=64,
        dataset_mixture_state={"epoch": 0},
        rng_state_files={"rank0": "rng/rank0.pt"},
        ig_channel=channel,
        outcome_channel=channel,
        algorithm_config={"group_size": 16},
        model_fingerprint="actor",
        reference_model_fingerprint="reference",
        train_data_sha256="train",
        validation_data_sha256="validation",
        retriever_index_sha256="index",
        retriever_config_sha256="retriever",
        tokenizer_hash="tokenizer",
        chat_template_hash="template",
        source_commit="source",
        framework_versions={"torch": "2.8.0"},
        fsdp_world_size=4,
        vllm_data_parallel_size=4,
        vllm_tensor_parallel_size=1,
        optimizer_state_present=True,
        scheduler_state_present=True,
        actor_state_present=True,
    )
    payload = metadata.as_dict()
    assert payload["successful_update_step"] == 1
    assert payload["ig_channel"]["committed_scale"] == 1.0
