from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from agentic_rl.advantage.a2tgpo import (
    SEARCH_IG_COEFFICIENT,
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
    SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
    compute_prompt_advantages,
    rebuild_search_advantages,
    trajectory_credit_input_from_record,
    turn_advantages_from_record,
)
from agentic_rl.advantage.mica_ig import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
)
from agentic_rl.rollout.trajectory_schema import TurnType
from agentic_rl.selection.candidate_pool import PromptGroup


@dataclass(frozen=True)
class PreparedTrajectory:
    record: Any
    advantage_by_turn: dict[int, float]
    normalized_ig_by_turn: dict[int, float]
    answer_turn_ids: tuple[int, ...]
    expected_turn_ids: tuple[int, ...]
    advantage: Any | None = None
    search_task_mode: str = "normalized_outcome"
    decision_advantage_by_turn: dict[int, float] = field(default_factory=dict)
    query_advantage_by_turn: dict[int, float] = field(default_factory=dict)
    decision_token_mask: tuple[int, ...] = ()
    query_token_mask: tuple[int, ...] = ()
    decision_turn_ids: tuple[int, ...] = ()
    query_turn_ids: tuple[int, ...] = ()
    search_turn_count: int = 0


@dataclass(frozen=True)
class WeightedPreparedTrajectory:
    prepared: PreparedTrajectory
    weight: float


def prepare_selected_trajectories(
    groups: Sequence[PromptGroup],
    *,
    expected_group_size: int,
    advantage_config: dict[str, Any] | None = None,
    expected_policy_version: int | None = None,
    expected_scorer_version: str | None = None,
    stop_continue_metrics: dict[str, float | int] | None = None,
) -> tuple[tuple[PreparedTrajectory, ...], ...]:
    if not groups:
        raise ValueError("Selected prompt groups cannot be empty")
    config = dict(advantage_config or {})
    search_task_mode = str(
        config.get("search_task_mode", "normalized_outcome")
    )
    local_normalization_modes = {
        SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
        SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
        SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
        ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
    }
    legacy_lambda_ig = (
        None
        if search_task_mode in local_normalization_modes
        else float(config.get("lambda_ig", SEARCH_IG_COEFFICIENT))
    )
    sc_config = dict(config.get("sc", {}))
    stop_continue_values: list[Any] = []
    prepared_groups: list[tuple[PreparedTrajectory, ...]] = []
    for group in groups:
        if len(group.trajectories) != expected_group_size:
            raise ValueError("Selected prompt lost one or more trajectories")
        credit_inputs = [
            trajectory_credit_input_from_record(record)
            for record in group.trajectories
        ]
        advantages = compute_prompt_advantages(
            credit_inputs,
            gamma=float(config.get("gamma", 1.0)),
            lambda_ig=legacy_lambda_ig,
            lambda_outcome=float(config.get("lambda_outcome", 1.0)),
            lambda_format=float(config.get("lambda_format", 1.0)),
            normalization_epsilon=float(
                config.get("normalization_epsilon", 1.0e-6)
            ),
            zero_variance_tolerance=float(
                config.get("zero_variance_tolerance", 1.0e-12)
            ),
            accumulate_future_ig=(
                search_task_mode not in local_normalization_modes
            ),
        )
        advantages, group_sc_metrics = rebuild_search_advantages(
            group.trajectories,
            advantages,
            search_task_mode=search_task_mode,
            group_size=int(expected_group_size),
            lambda_ig=legacy_lambda_ig,
            lambda_task=float(config.get("lambda_task", 1.0)),
            reward_epsilon=float(sc_config.get("reward_epsilon", 1.0e-6)),
            scale_epsilon=float(sc_config.get("scale_epsilon", 1.0e-8)),
            pooled_scale_ddof=int(sc_config.get("pooled_scale_ddof", 0)),
            probe_epsilon=float(config.get("probe_epsilon", 1.0e-6)),
            mica_gamma=float(dict(config.get("mica", {})).get("gamma", 1.0)),
            mica_alpha=float(dict(config.get("mica", {})).get("alpha", 0.5)),
            normalization_epsilon=float(
                config.get("normalization_epsilon", 1.0e-6)
            ),
            zero_variance_tolerance=float(
                config.get("zero_variance_tolerance", 1.0e-12)
            ),
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
        )
        del group_sc_metrics
        prepared: list[PreparedTrajectory] = []
        for record, advantage in zip(
            group.trajectories,
            advantages.trajectories,
            strict=True,
        ):
            values = turn_advantages_from_record(record, advantage)
            normalized_by_turn: dict[int, float] = {}
            for turn in record.turns:
                if (
                    turn.turn_type is TurnType.SEARCH
                    and turn.policy_credit_eligible
                ):
                    if turn.search_index is None:
                        raise ValueError("Search turn is missing search_index")
                    normalized_by_turn[int(turn.turn_index)] = float(
                        advantage.normalized_ig.get(int(turn.search_index), 0.0)
                    )
            terminal = record.terminal_policy_credit_turn_index
            answer_ids = () if terminal is None else (int(terminal),)
            expected = tuple(sorted(values))
            if set(normalized_by_turn) | set(answer_ids) != set(expected):
                raise RuntimeError(
                    "Every optimized turn must be exactly one Search or terminal turn"
                )
            decision_by_turn: dict[int, float] = {}
            query_by_turn: dict[int, float] = {}
            decision_mask: list[int] = []
            query_mask: list[int] = []
            if (
                search_task_mode
                == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
            ):
                decision_mask = [0] * len(record.input_ids)
                query_mask = [0] * len(record.input_ids)
                if advantage.search_main_advantage != advantage.search_advantage:
                    raise RuntimeError("Role-localized Main credit changed in transport")
                turns_by_search = {
                    int(turn.search_index): turn
                    for turn in record.turns
                    if turn.turn_type is TurnType.SEARCH
                    and turn.search_index is not None
                    and turn.policy_credit_eligible
                }
                if not (
                    set(advantage.search_main_advantage)
                    == set(advantage.search_decision_advantage)
                    == set(advantage.search_query_advantage)
                    == set(turns_by_search)
                ):
                    raise RuntimeError("Role-localized credit coverage mismatch")
                for search_index, turn in sorted(turns_by_search.items()):
                    turn_id = int(turn.turn_index)
                    decision_credit = float(
                        advantage.search_decision_advantage[search_index]
                    )
                    query_credit = float(
                        advantage.search_query_advantage[search_index]
                    )
                    if decision_credit != 0.0:
                        start, end = map(int, turn.decision_token_span)
                        if end <= start:
                            raise RuntimeError("Decision gate has an empty D span")
                        decision_by_turn[turn_id] = decision_credit
                        for position in range(start, end):
                            if not record.policy_mask[position]:
                                raise RuntimeError("Decision span escaped policy mask")
                            decision_mask[position] = 1
                    if query_credit != 0.0:
                        start, end = map(int, turn.query_token_span)
                        if end <= start:
                            raise RuntimeError("Query gate has an empty Q span")
                        query_by_turn[turn_id] = query_credit
                        for position in range(start, end):
                            if not record.policy_mask[position]:
                                raise RuntimeError("Query span escaped policy mask")
                            query_mask[position] = 1
                if any(
                    left and right
                    for left, right in zip(
                        decision_mask,
                        query_mask,
                        strict=True,
                    )
                ):
                    raise RuntimeError("Decision and Query token masks overlap")
            prepared.append(
                PreparedTrajectory(
                    record=record,
                    advantage_by_turn=values,
                    normalized_ig_by_turn=normalized_by_turn,
                    answer_turn_ids=answer_ids,
                    expected_turn_ids=expected,
                    advantage=advantage,
                    search_task_mode=search_task_mode,
                    decision_advantage_by_turn=decision_by_turn,
                    query_advantage_by_turn=query_by_turn,
                    decision_token_mask=tuple(decision_mask),
                    query_token_mask=tuple(query_mask),
                    decision_turn_ids=tuple(sorted(decision_by_turn)),
                    query_turn_ids=tuple(sorted(query_by_turn)),
                    search_turn_count=sum(
                        turn.turn_type is TurnType.SEARCH
                        and turn.policy_credit_eligible
                        for turn in record.turns
                    ),
                )
            )
            stop_continue_values.extend(
                advantage.stop_continue_by_search_index.values()
            )
        prepared_groups.append(tuple(prepared))
    if stop_continue_metrics is not None:
        stop_continue_metrics.clear()
        state_count = len(stop_continue_values)
        clear_count = sum(value.sc_clear for value in stop_continue_values)
        fallback_count = state_count - clear_count
        clear_positive_count = sum(
            value.clear_positive for value in stop_continue_values
        )
        clear_negative_count = sum(
            value.clear_negative for value in stop_continue_values
        )
        stop_continue_metrics.update(
            {
                "sc/state_count": state_count,
                "sc/clear_count": clear_count,
                "sc/fallback_count": fallback_count,
                "sc/fallback_z_o_to_search_count": 0,
                "sc/clear_positive_count": clear_positive_count,
                "sc/clear_negative_count": clear_negative_count,
            }
        )
        stop_continue_metrics["sc/clear_rate"] = (
            clear_count / state_count if state_count else 0.0
        )
        stop_continue_metrics["sc/fallback_rate"] = (
            fallback_count / state_count if state_count else 0.0
        )
        scalar_fields = {
            "sc/delta": "delta_sc",
            "sc/scale": "pooled_scale",
            "sc/raw_advantage": "raw_advantage_sc",
            "sc/advantage": "advantage_sc",
            "sc/stop_reward_1": "stop_reward_1",
            "sc/stop_reward_2": "stop_reward_2",
        }
        arrays: dict[str, np.ndarray] = {}
        for metric_prefix, attribute in scalar_fields.items():
            array = np.asarray(
                [
                    float(getattr(value, attribute))
                    for value in stop_continue_values
                ],
                dtype=np.float64,
            )
            arrays[metric_prefix] = array
            stop_continue_metrics[f"{metric_prefix}_mean"] = (
                float(np.mean(array, dtype=np.float64))
                if array.size
                else 0.0
            )
        for metric_prefix in (
            "sc/delta",
            "sc/advantage",
        ):
            array = arrays[metric_prefix]
            stop_continue_metrics[f"{metric_prefix}_std"] = (
                float(np.std(array, ddof=0, dtype=np.float64))
                if array.size
                else 0.0
            )
        stop_continue_metrics["sc/clip_fraction"] = (
            sum(value.clipped for value in stop_continue_values) / state_count
            if state_count
            else 0.0
        )
        reward_epsilon = float(sc_config.get("reward_epsilon", 1.0e-6))
        stop_continue_metrics["sc/stop_reward_agreement_rate"] = (
            sum(
                abs(value.stop_reward_1 - value.stop_reward_2)
                <= reward_epsilon
                for value in stop_continue_values
            )
            / state_count
            if state_count
            else 0.0
        )
    return tuple(prepared_groups)


def pack_prompt_groups_by_action_tokens(
    groups: Sequence[tuple[PreparedTrajectory, ...]],
    *,
    world_size: int,
) -> tuple[tuple[PreparedTrajectory, ...], ...]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    ranked = sorted(
        groups,
        key=lambda group: (
            -sum(item.record.action_token_count for item in group),
            group[0].record.prompt_global_id,
        ),
    )
    assignments: list[list[PreparedTrajectory]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    prompt_counts = [0] * world_size
    for group in ranked:
        rank = min(
            range(world_size),
            key=lambda index: (loads[index], prompt_counts[index], index),
        )
        assignments[rank].extend(group)
        loads[rank] += sum(item.record.action_token_count for item in group)
        prompt_counts[rank] += 1
    return tuple(tuple(values) for values in assignments)


def build_synchronized_microbatch_rounds(
    assignments: Sequence[Sequence[PreparedTrajectory]],
    *,
    micro_batch_size_per_rank: int,
    pad_token_id: int,
    snapshot_step: int,
    global_prompt_count: int,
    group_size: int,
    action_state_chunk_size: int,
    vocabulary_chunk_size: int,
    kl_coefficient: float,
    lambda_decision: float = 0.0,
    lambda_query: float = 0.0,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Build equal-shape rank payloads for every FSDP collective round.

    Short ranks are padded with a real local trajectory carrying weight zero.
    The dummy still executes the same collectives, but contributes no task or
    KL gradient.
    """

    if micro_batch_size_per_rank <= 0:
        raise ValueError("micro_batch_size_per_rank must be positive")
    if not assignments:
        raise ValueError("At least one FSDP rank assignment is required")
    global_filler = next(
        (trajectory for values in assignments for trajectory in values),
        None,
    )
    if global_filler is None:
        raise ValueError("At least one selected trajectory is required")
    chunks = [
        [
            tuple(values[start : start + micro_batch_size_per_rank])
            for start in range(0, len(values), micro_batch_size_per_rank)
        ]
        for values in assignments
    ]
    round_count = max(len(values) for values in chunks)
    rounds: list[tuple[dict[str, Any], ...]] = []
    for round_index in range(round_count):
        weighted_by_rank: list[list[WeightedPreparedTrajectory]] = []
        for rank, rank_chunks in enumerate(chunks):
            real = (
                list(rank_chunks[round_index])
                if round_index < len(rank_chunks)
                else []
            )
            weighted = [
                WeightedPreparedTrajectory(item, 1.0) for item in real
            ]
            filler = (
                assignments[rank][0]
                if assignments[rank]
                else global_filler
            )
            while len(weighted) < micro_batch_size_per_rank:
                weighted.append(WeightedPreparedTrajectory(filler, 0.0))
            weighted_by_rank.append(weighted)

        maximum_sequence = max(
            len(item.prepared.record.input_ids)
            for rank_values in weighted_by_rank
            for item in rank_values
        )
        rank_payloads = tuple(
            _collate_rank_payload(
                values,
                sequence_length=maximum_sequence,
                pad_token_id=pad_token_id,
                snapshot_step=snapshot_step,
                global_prompt_count=global_prompt_count,
                group_size=group_size,
                action_state_chunk_size=action_state_chunk_size,
                vocabulary_chunk_size=vocabulary_chunk_size,
                kl_coefficient=kl_coefficient,
                lambda_decision=lambda_decision,
                lambda_query=lambda_query,
            )
            for values in weighted_by_rank
        )
        rounds.append(rank_payloads)
    return tuple(rounds)


def _collate_rank_payload(
    values: Sequence[WeightedPreparedTrajectory],
    *,
    sequence_length: int,
    pad_token_id: int,
    snapshot_step: int,
    global_prompt_count: int,
    group_size: int,
    action_state_chunk_size: int,
    vocabulary_chunk_size: int,
    kl_coefficient: float,
    lambda_decision: float,
    lambda_query: float,
) -> dict[str, Any]:
    batch_size = len(values)
    input_ids = torch.full(
        (batch_size, sequence_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.long,
    )
    position_ids = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.long,
    )
    policy_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )
    turn_ids = torch.full(
        (batch_size, sequence_length),
        -1,
        dtype=torch.long,
    )
    labels = torch.full(
        (batch_size, sequence_length),
        -100,
        dtype=torch.long,
    )
    decision_token_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )
    query_token_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )
    for batch_index, weighted in enumerate(values):
        record = weighted.prepared.record
        length = len(record.input_ids)
        input_ids[batch_index, :length] = torch.tensor(
            record.input_ids,
            dtype=torch.long,
        )
        attention_mask[batch_index, :length] = 1
        position_ids[batch_index, :length] = torch.arange(
            length,
            dtype=torch.long,
        )
        policy_mask[batch_index, :length] = torch.tensor(
            record.policy_mask,
            dtype=torch.bool,
        )
        turn_ids[batch_index, :length] = torch.tensor(
            record.turn_ids,
            dtype=torch.long,
        )
        labels[batch_index, :length] = torch.where(
            policy_mask[batch_index, :length],
            input_ids[batch_index, :length],
            torch.full((length,), -100, dtype=torch.long),
        )
        if weighted.prepared.decision_token_mask:
            decision_token_mask[batch_index, :length] = torch.tensor(
                weighted.prepared.decision_token_mask,
                dtype=torch.bool,
            )
        if weighted.prepared.query_token_mask:
            query_token_mask[batch_index, :length] = torch.tensor(
                weighted.prepared.query_token_mask,
                dtype=torch.bool,
            )
    modes = {weighted.prepared.search_task_mode for weighted in values}
    if len(modes) != 1:
        raise RuntimeError("A rank payload cannot mix Search task modes")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "policy_mask": policy_mask,
        "turn_ids": turn_ids,
        "labels": labels,
        "decision_token_mask": decision_token_mask,
        "query_token_mask": query_token_mask,
        "search_task_mode": modes.pop(),
        "old_logprobs": torch.zeros(
            (batch_size, sequence_length),
            dtype=torch.float32,
        ),
        "advantage_by_turn": [
            weighted.prepared.advantage_by_turn for weighted in values
        ],
        "normalized_ig_by_turn": [
            weighted.prepared.normalized_ig_by_turn for weighted in values
        ],
        "answer_turn_ids": [
            weighted.prepared.answer_turn_ids for weighted in values
        ],
        "expected_turn_ids": [
            weighted.prepared.expected_turn_ids for weighted in values
        ],
        "decision_advantage_by_turn": [
            weighted.prepared.decision_advantage_by_turn or {}
            for weighted in values
        ],
        "query_advantage_by_turn": [
            weighted.prepared.query_advantage_by_turn or {}
            for weighted in values
        ],
        "decision_turn_ids": [
            weighted.prepared.decision_turn_ids for weighted in values
        ],
        "query_turn_ids": [
            weighted.prepared.query_turn_ids for weighted in values
        ],
        "search_turn_counts": [
            int(weighted.prepared.search_turn_count) for weighted in values
        ],
        "trajectory_weights": [weighted.weight for weighted in values],
        "prompt_global_ids": [
            weighted.prepared.record.prompt_global_id for weighted in values
        ],
        "trajectory_ids": [
            weighted.prepared.record.trajectory_id for weighted in values
        ],
        "snapshot_step": int(snapshot_step),
        "global_prompt_count": int(global_prompt_count),
        "group_size": int(group_size),
        "action_state_chunk_size": int(action_state_chunk_size),
        "vocabulary_chunk_size": int(vocabulary_chunk_size),
        "kl_coefficient": float(kl_coefficient),
        "lambda_decision": float(lambda_decision),
        "lambda_query": float(lambda_query),
    }
