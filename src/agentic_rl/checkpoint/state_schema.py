from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ChannelCheckpointState:
    committed_scale: float | None
    health_observations: tuple[float, ...]
    health_reference: float | None
    valid_success_count: int


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    successful_update_step: int
    attempt_id: int
    data_cursor: int
    dataset_mixture_state: Mapping[str, Any]
    rng_state_files: Mapping[str, str]
    ig_channel: ChannelCheckpointState
    outcome_channel: ChannelCheckpointState
    algorithm_config: Mapping[str, Any]
    model_fingerprint: str
    reference_model_fingerprint: str
    train_data_sha256: str
    validation_data_sha256: str
    retriever_index_sha256: str
    retriever_config_sha256: str
    tokenizer_hash: str
    chat_template_hash: str
    source_commit: str
    framework_versions: Mapping[str, str | None]
    fsdp_world_size: int
    vllm_data_parallel_size: int
    vllm_tensor_parallel_size: int
    optimizer_state_present: bool
    scheduler_state_present: bool
    actor_state_present: bool

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported checkpoint metadata schema")
        if self.successful_update_step < 0 or self.attempt_id < 0:
            raise ValueError("Checkpoint counters cannot be negative")
        if self.attempt_id < self.successful_update_step:
            raise ValueError("attempt_id cannot trail successful_update_step")
        if self.data_cursor < 0:
            raise ValueError("data_cursor cannot be negative")
        if self.fsdp_world_size < 1:
            raise ValueError("Checkpoint FSDP2 world_size must be positive")
        if self.vllm_data_parallel_size < 1 or self.vllm_tensor_parallel_size != 1:
            raise ValueError("Checkpoint vLLM topology must have positive DP and TP=1")
        if not (
            self.actor_state_present
            and self.optimizer_state_present
            and self.scheduler_state_present
        ):
            raise ValueError("A resumable checkpoint must contain actor/optimizer/scheduler")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
