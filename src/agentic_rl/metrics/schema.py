from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class MetricScope(str, Enum):
    ATTEMPT = "attempt"
    UPDATE = "update"
    CHANNEL = "channel"
    PROMPT = "prompt"
    TRAJECTORY = "trajectory"
    TURN = "turn"
    BEHAVIOR = "behavior"
    SYSTEM = "system"
    CHECKPOINT = "checkpoint"
    EVAL = "eval"
    FIXED_EVAL = "fixed_eval"


@dataclass(frozen=True)
class AttemptMetrics:
    attempt_id: int
    successful_update_step: int
    status: str
    skip_reason: str | None
    candidate_prompt_count: int
    candidate_trajectory_count: int
    selected_prompt_count: int
    selected_trajectory_count: int
    refill_count: int
    refill_used: bool
    parser_success_rate: float
    format_success_rate: float
    malformed_search_rate: float
    trajectory_valid_rate: float
    environment_failure_rate: float
    ig_variance_mean: float
    outcome_variance_mean: float
    ig_positive_prompt_count: int
    outcome_positive_prompt_count: int
    ig_channel_active: bool
    outcome_channel_active: bool
    ig_scale_used: float | None
    outcome_scale_used: float | None
    ig_scale_update_allowed_after_success: bool
    outcome_scale_update_allowed_after_success: bool
    ig_health_ratio: float | None
    outcome_health_ratio: float | None
    ig_health_reference: float | None
    outcome_health_reference: float | None
    ig_mean_excess: float
    outcome_mean_excess: float
    ig_heterogeneity: float
    outcome_heterogeneity: float
    top_p_selected_count: int
    top_p_actual_mass: float
    task_objective: float | None
    full_vocab_reference_kl: float | None
    total_loss: float | None
    gradient_norm: float | None
    ratio_mean: float | None
    ratio_max: float | None
    clip_fraction: float | None
    optimizer_steps_in_attempt: int
    scheduler_steps_in_attempt: int
    actor_snapshot_step: int
    rollout_snapshot_step: int
    old_policy_snapshot_step: int
    reward_policy_snapshot_step: int

    def validate(self) -> None:
        if self.optimizer_steps_in_attempt not in {0, 1}:
            raise ValueError("optimizer_steps_in_attempt must be 0 or 1")
        if self.scheduler_steps_in_attempt not in {0, 1}:
            raise ValueError("scheduler_steps_in_attempt must be 0 or 1")
        if self.status == "committed":
            if self.optimizer_steps_in_attempt != 1 or self.scheduler_steps_in_attempt != 1:
                raise ValueError("Committed attempts require exactly one optimizer/scheduler step")
        versions = {
            self.actor_snapshot_step,
            self.rollout_snapshot_step,
            self.old_policy_snapshot_step,
            self.reward_policy_snapshot_step,
        }
        if len(versions) != 1:
            raise ValueError("Rollout-start snapshot versions disagree")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PromptMetrics:
    attempt_id: int
    prompt_global_id: str
    dataset_source: str
    domain: str
    trajectory_count: int
    valid_trajectory_count: int
    ig_variance: float
    outcome_variance: float
    ig_excess_variance: float
    outcome_excess_variance: float
    selection_score: float
    selected: bool
    search_peer_counts: dict[int, int]
    ig_variance_by_search_index: dict[int, float]
    ig_natural_weights: dict[int, float]
    ig_channel_score: float
    outcome_channel_score: float
    selection_rank: int | None


@dataclass(frozen=True)
class TrajectoryMetrics:
    attempt_id: int
    prompt_global_id: str
    trajectory_id: str
    parser_status: str
    parser_error_type: str | None
    fallback_status: str | None
    trajectory_validity: bool
    terminal_answer_valid: bool
    trajectory_system_valid: bool
    outcome_reward_eligible: bool
    environment_failure_code: str | None
    search_turn_count: int
    task_outcome: float
    format_indicator: int
    action_token_count: int
    information_token_count: int
    phi_values: tuple[float, ...]
    immediate_ig_values: tuple[float, ...]
    search_queries: tuple[str, ...]
    search_action_span_valid: tuple[bool, ...]
    search_prefix_valid: tuple[bool, ...]
    ig_reward_eligible: tuple[bool, ...]
    policy_credit_eligible: tuple[bool, ...]


@dataclass(frozen=True)
class TurnMetrics:
    attempt_id: int
    prompt_global_id: str
    trajectory_id: str
    turn_index: int
    turn_type: str
    search_action_span_valid: bool
    search_prefix_valid: bool
    ig_reward_eligible: bool
    policy_credit_eligible: bool
    raw_ig: float | None
    normalized_ig: float | None
    future_ig_sum: float | None
    accumulated_ig_count: int
    future_ig_rescaled: float | None
    normalized_outcome: float
    centered_format_indicator: float | None
    final_advantage: float
    ratio: float | None
    clip_scale: float | None
    clip_lower: float | None
    clip_upper: float | None
    clipped: bool | None
    action_span_start: int | None
    action_span_end: int | None


@dataclass(frozen=True)
class SystemMetrics:
    attempt_id: int
    successful_update_step: int
    component: str
    wall_seconds: float
    gpu_memory_allocated_bytes: int | None
    gpu_memory_reserved_bytes: int | None
    cpu_utilization: float | None
    ray_object_store_bytes: int | None
    vllm_snapshot_step: int | None
    fsdp_world_size: int | None
    retriever_request_count: int | None
