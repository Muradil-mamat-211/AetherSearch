from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

from agentic_rl.rollout.trajectory_schema import TokenSource, TurnType
from agentic_rl.selection.candidate_pool import (
    ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE,
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    position = (len(finite) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _normalized_query(value: str) -> str:
    return " ".join(str(value).lower().split())


def _action_tokens_by_turn(record: Any) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for source, turn_id, eligible in zip(
        record.token_sources,
        record.turn_ids,
        record.policy_mask,
        strict=True,
    ):
        if (
            source is TokenSource.MODEL
            and bool(eligible)
            and int(turn_id) >= 0
        ):
            counts[int(turn_id)] += 1
    return dict(counts)


def _prepared_map(
    prepared_groups: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    return {
        str(item.record.trajectory_id): item
        for group in prepared_groups
        for item in group
    }


def build_channel_records(
    *,
    attempt_id: int,
    successful_update_before: int,
    successful_update_after: int,
    decision: Any,
    state_before: Any,
    state_after: Any,
    committed: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for channel, stats, before, after in (
        (
            "IG",
            decision.ig_stats,
            state_before.ig_channel,
            state_after.ig_channel,
        ),
        (
            "Outcome",
            decision.outcome_stats,
            state_before.outcome_channel,
            state_after.outcome_channel,
        ),
    ):
        scale_changed = before.committed_scale != after.committed_scale
        observations_changed = (
            len(after.health_observations) > len(before.health_observations)
        )
        records.append(
            {
                "attempt_id": int(attempt_id),
                "successful_update_before": int(successful_update_before),
                "successful_update_after": int(successful_update_after),
                "channel": channel,
                "stage_mode": str(stats.gate.mode),
                "m": stats.positive_median,
                "M": float(stats.mean_excess),
                "N_positive": int(stats.positive_prompt_count),
                "b_use": stats.scale_used,
                "b_before": before.committed_scale,
                "b_after": after.committed_scale,
                "B_ref": after.health_reference,
                "health_ratio": stats.gate.health_ratio,
                "activation": bool(stats.gate.active),
                "activation_reason": str(stats.gate.reason),
                "EMA_update_allowed": bool(
                    stats.scale_update_allowed_after_success
                ),
                "EMA_updated": bool(committed and scale_changed),
                "EMA_frozen": bool(
                    committed
                    and stats.scale_observation_valid
                    and not scale_changed
                    and before.committed_scale is not None
                ),
                "health_observation_committed": bool(
                    committed and observations_changed
                ),
                "valid_health_observation_count": int(
                    after.valid_success_count
                ),
                "heterogeneity": float(stats.heterogeneity),
            }
        )
    return records


def build_prompt_records(
    *,
    attempt_id: int,
    groups: Sequence[Any],
    decision: Any,
) -> list[dict[str, Any]]:
    ordered = list(decision.top_p.ordered_positive_ids)
    rank = {prompt_id: index + 1 for index, prompt_id in enumerate(ordered)}
    total_mass = float(decision.top_p.total_mass)
    cumulative = 0.0
    cumulative_by_id: dict[str, float] = {}
    for prompt_id in ordered:
        cumulative += float(decision.score_by_prompt[prompt_id])
        cumulative_by_id[prompt_id] = (
            cumulative / total_mass if total_mass > 0 else 0.0
        )
    selected = set(decision.selected_ids)
    boundary_score = (
        float(decision.score_by_prompt[decision.selected_ids[-1]])
        if decision.selected_ids
        else 0.0
    )
    records = []
    for group in groups:
        prompt_id = str(group.prompt_global_id)
        metadata = dict(group.metadata)
        peer_counts = {
            int(key): int(value)
            for key, value in metadata.get(
                "ig_peer_count_by_search_index",
                {},
            ).items()
        }
        natural_weights = {
            int(key): float(value)
            for key, value in metadata.get(
                "ig_natural_weight_by_search_index",
                {},
            ).items()
        }
        first = group.trajectories[0]
        records.append(
            {
                "attempt_id": int(attempt_id),
                "prompt_global_id": prompt_id,
                "dataset_row_id": first.metadata.get("dataset_row_id"),
                "domain": first.metadata.get("data_source", ""),
                "V_IG": float(group.ig_variance),
                "V_Outcome": float(group.outcome_variance),
                "ragen_signal_mode": str(decision.signal_mode),
                "ragen_selection_mode": str(decision.selection_mode),
                "paper_raw_sample_outcome_variance": (
                    float(decision.score_by_prompt[prompt_id])
                    if str(decision.selection_mode)
                    == ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE
                    else None
                ),
                "paper_health_gate_selection_call_count": (
                    int(decision.health_gate_selection_call_count)
                    if str(decision.selection_mode)
                    == ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE
                    else None
                ),
                "paper_scale_selection_call_count": (
                    int(decision.scale_selection_call_count)
                    if str(decision.selection_mode)
                    == ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE
                    else None
                ),
                "e_IG": float(decision.ig_stats.excess_variance[prompt_id]),
                "e_Outcome": float(
                    decision.outcome_stats.excess_variance[prompt_id]
                ),
                "U_IG": float(
                    decision.ig_stats.normalized_signal[prompt_id]
                ),
                "U_Outcome": float(
                    decision.outcome_stats.normalized_signal[prompt_id]
                ),
                "S": float(decision.score_by_prompt[prompt_id]),
                "rank": rank.get(prompt_id),
                "selected": prompt_id in selected,
                "T_plus": sorted(
                    index for index, count in peer_counts.items() if count >= 2
                ),
                "peer_counts": peer_counts,
                "natural_weights": natural_weights,
                "top_p_cumulative_mass": cumulative_by_id.get(prompt_id),
                "selection_mass": float(decision.top_p.selected_mass),
                "selection_mass_ratio": float(
                    decision.top_p.selected_mass_ratio
                ),
                "selection_boundary_distance": float(
                    decision.score_by_prompt[prompt_id] - boundary_score
                ),
                "trajectory_count": len(group.trajectories),
            }
        )
    return records


def build_trajectory_and_turn_records(
    *,
    attempt_id: int,
    groups: Sequence[Any],
    prepared_groups: Sequence[Sequence[Any]],
    turn_runtime: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = _prepared_map(prepared_groups)
    trajectory_records: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []
    for group in groups:
        for record in group.trajectories:
            item = prepared.get(str(record.trajectory_id))
            advantage = None if item is None else item.advantage
            action_by_turn = _action_tokens_by_turn(record)
            score_by_prefix = tuple(
                float(value)
                for value in record.metadata.get(
                    "exact_ig_score_by_prefix",
                    (),
                )
            )
            trajectory_records.append(
                {
                    "attempt_id": int(attempt_id),
                    "prompt_global_id": str(record.prompt_global_id),
                    "trajectory_id": str(record.trajectory_id),
                    "search_task_mode": (
                        None if advantage is None else str(advantage.search_task_mode)
                    ),
                    "R_task": float(record.task_outcome),
                    "F_ans": int(record.answer_format_indicator),
                    "search_count": int(record.search_turn_count),
                    "system_valid": bool(record.trajectory_system_valid),
                    "terminal_answer_valid": bool(
                        record.terminal_answer_valid
                    ),
                    "answer_policy_credit_eligible": bool(
                        record.terminal_policy_credit_turn_index is not None
                    ),
                    "action_token_count": int(record.action_token_count),
                    "outcome_z": (
                        None
                        if advantage is None
                        else float(advantage.normalized_outcome)
                    ),
                    "parser_status": str(record.parser_status),
                    "parser_error_type": record.parser_error_type,
                    "fallback_status": record.fallback_status,
                    "environment_failure_code": record.environment_failure_code,
                    "Phi": list(score_by_prefix),
                    "IG": [
                        float(value)
                        for _, value in sorted(record.immediate_ig.items())
                    ],
                    "queries": [
                        str(turn.query)
                        for turn in record.turns
                        if turn.query is not None
                    ],
                    "no_new_observation_count": sum(
                        turn.no_new_observation is True
                        for turn in record.turns
                        if turn.turn_type is TurnType.SEARCH
                    ),
                    "exact_query_repeat_count": sum(
                        bool(turn.exact_query_repeat)
                        for turn in record.turns
                        if turn.turn_type is TurnType.SEARCH
                    ),
                    "mica_singleton_tail_start_depth": (
                        None
                        if advantage is None
                        else advantage.mica_singleton_tail_start_depth
                    ),
                    "mica_singleton_consecutive_length": (
                        0
                        if advantage is None
                        else int(advantage.mica_singleton_consecutive_length)
                    ),
                }
            )
            eligible_phi_index = 0
            for turn in record.turns:
                search_index = (
                    None
                    if turn.search_index is None
                    else int(turn.search_index)
                )
                raw_ig = (
                    None
                    if search_index is None
                    else record.immediate_ig.get(search_index)
                )
                phi_before = None
                phi_after = None
                if raw_ig is not None and score_by_prefix:
                    if eligible_phi_index + 1 >= len(score_by_prefix):
                        raise RuntimeError(
                            "Exact-IG prefix scores do not cover every eligible turn"
                        )
                    phi_before = score_by_prefix[eligible_phi_index]
                    phi_after = score_by_prefix[eligible_phi_index + 1]
                    eligible_phi_index += 1
                normalized_ig = (
                    None
                    if (
                        advantage is None
                        or search_index is None
                        or search_index not in advantage.normalized_ig
                    )
                    else float(advantage.normalized_ig[search_index])
                )
                runtime = turn_runtime.get(
                    (str(record.trajectory_id), int(turn.turn_index)),
                    {},
                )
                is_search = turn.turn_type is TurnType.SEARCH
                is_terminal = (
                    record.terminal_policy_credit_turn_index
                    == int(turn.turn_index)
                )
                stop_continue = (
                    None
                    if (
                        advantage is None
                        or search_index is None
                        or search_index
                        not in advantage.stop_continue_by_search_index
                    )
                    else advantage.stop_continue_by_search_index[search_index]
                )
                routed_stages = None
                if search_index is not None:
                    routed = record.metadata.get("routed_answer_probes", {})
                    if isinstance(routed, Mapping):
                        routed_stages = routed.get(
                            search_index,
                            routed.get(str(search_index)),
                        )
                routed_pre = (
                    routed_stages.get("pre")
                    if isinstance(routed_stages, Mapping)
                    and isinstance(routed_stages.get("pre"), Mapping)
                    else None
                )
                routed_post = (
                    routed_stages.get("post")
                    if isinstance(routed_stages, Mapping)
                    and isinstance(routed_stages.get("post"), Mapping)
                    else None
                )
                turn_records.append(
                    {
                        "attempt_id": int(attempt_id),
                        "prompt_global_id": str(record.prompt_global_id),
                        "trajectory_id": str(record.trajectory_id),
                        "turn_id": int(turn.turn_index),
                        "search_index": search_index,
                        "turn_type": turn.turn_type.value,
                        "Phi_before": phi_before,
                        "Phi_after": phi_after,
                        "raw_IG": None if raw_ig is None else float(raw_ig),
                        "normalized_IG": normalized_ig,
                        "D": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.future_ig_sum
                            )
                            else float(
                                advantage.future_ig_sum[search_index]
                            )
                        ),
                        "n_acc": (
                            0
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.accumulated_ig_count
                            )
                            else int(
                                advantage.accumulated_ig_count[search_index]
                            )
                        ),
                        "D_bar": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.future_ig_rescaled
                            )
                            else float(
                                advantage.future_ig_rescaled[search_index]
                            )
                        ),
                        "A_IG": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.normalized_ig
                            )
                            else float(
                                advantage.normalized_ig[search_index]
                            )
                        ),
                        "local_ig_hat": normalized_ig,
                        "mica_ig_return": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.mica_ig_return
                            )
                            else float(advantage.mica_ig_return[search_index])
                        ),
                        "mica_peer_count": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.mica_peer_count
                            )
                            else int(advantage.mica_peer_count[search_index])
                        ),
                        "mica_loc_mean": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.mica_loc_mean
                            )
                            else float(advantage.mica_loc_mean[search_index])
                        ),
                        "mica_loc_std": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.mica_loc_std
                            )
                            else float(advantage.mica_loc_std[search_index])
                        ),
                        "mica_ret_mean": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.mica_ret_mean
                            )
                            else float(advantage.mica_ret_mean[search_index])
                        ),
                        "mica_ret_std": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.mica_ret_std
                            )
                            else float(advantage.mica_ret_std[search_index])
                        ),
                        "mica_A_loc": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.mica_local_advantage
                            )
                            else float(
                                advantage.mica_local_advantage[search_index]
                            )
                        ),
                        "mica_A_ret": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.mica_return_advantage
                            )
                            else float(
                                advantage.mica_return_advantage[search_index]
                            )
                        ),
                        "mica_singleton_fallback": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.mica_singleton_fallback
                            )
                            else bool(
                                advantage.mica_singleton_fallback[search_index]
                            )
                        ),
                        "mica_Z_O": (
                            None
                            if advantage is None or search_index is None
                            else float(advantage.normalized_outcome)
                        ),
                        "mica_ig_missing_reason": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.mica_ig_missing_reason
                            )
                            else str(
                                advantage.mica_ig_missing_reason[search_index]
                            )
                        ),
                        "sufficient_before_search": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.sufficient_before_search
                            )
                            else bool(
                                advantage.sufficient_before_search[search_index]
                            )
                        ),
                        "sufficient_after_search": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.sufficient_after_search
                            )
                            else bool(
                                advantage.sufficient_after_search[search_index]
                            )
                        ),
                        "D_ig_eff": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.effective_cumulative_ig
                            )
                            else float(
                                advantage.effective_cumulative_ig[search_index]
                            )
                        ),
                        "D_ig_eff_count": (
                            0
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.effective_cumulative_ig_count
                            )
                            else int(
                                advantage.effective_cumulative_ig_count[
                                    search_index
                                ]
                            )
                        ),
                        "delta_probe": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.probe_reward_delta
                            )
                            else float(
                                advantage.probe_reward_delta[search_index]
                            )
                        ),
                        "O_route": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index not in advantage.routed_outcome
                            )
                            else float(advantage.routed_outcome[search_index])
                        ),
                        "pre_probe_raw_task_reward": (
                            None
                            if routed_pre is None
                            else float(routed_pre["raw_task_reward"])
                        ),
                        "post_probe_raw_task_reward": (
                            None
                            if routed_post is None
                            else float(routed_post["raw_task_reward"])
                        ),
                        "no_new_observation": (
                            None
                            if search_index is None
                            else turn.no_new_observation
                        ),
                        "search_advantage_branch": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.search_branch_by_search_index
                            )
                            else str(
                                advantage.search_branch_by_search_index[
                                    search_index
                                ]
                            )
                        ),
                        "exact_query_repeat": (
                            bool(turn.exact_query_repeat) if is_search else None
                        ),
                        "different_query_no_new_passage": (
                            bool(turn.different_query_no_new_passage)
                            if is_search
                            else None
                        ),
                        "current_passage_keys": (
                            list(turn.current_passage_keys) if is_search else None
                        ),
                        "new_passage_keys": (
                            list(turn.new_passage_keys) if is_search else None
                        ),
                        "z_outcome": (
                            None
                            if advantage is None
                            else float(advantage.normalized_outcome)
                        ),
                        "A_search": (
                            float(advantage.search_advantage[search_index])
                            if (
                                advantage is not None
                                and is_search
                                and search_index
                                in advantage.search_advantage
                            )
                            else None
                        ),
                        "R_C": (
                            None
                            if stop_continue is None
                            else float(stop_continue.continue_reward)
                        ),
                        "R_S1": (
                            None
                            if stop_continue is None
                            else float(stop_continue.stop_reward_1)
                        ),
                        "R_S2": (
                            None
                            if stop_continue is None
                            else float(stop_continue.stop_reward_2)
                        ),
                        "Delta_SC": (
                            None
                            if stop_continue is None
                            else float(stop_continue.delta_sc)
                        ),
                        "s_SC": (
                            None
                            if stop_continue is None
                            else float(stop_continue.pooled_scale)
                        ),
                        "raw_A_SC": (
                            None
                            if stop_continue is None
                            else float(stop_continue.raw_advantage_sc)
                        ),
                        "A_SC": (
                            None
                            if stop_continue is None
                            else float(stop_continue.advantage_sc)
                        ),
                        "sc_clear": (
                            None
                            if stop_continue is None
                            else bool(stop_continue.sc_clear)
                        ),
                        "A_task": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.search_task_advantage
                            )
                            else float(
                                advantage.search_task_advantage[search_index]
                            )
                        ),
                        "A_search_old_shadow": (
                            None
                            if (
                                advantage is None
                                or search_index is None
                                or search_index
                                not in advantage.search_advantage_old_shadow
                            )
                            else float(
                                advantage.search_advantage_old_shadow[
                                    search_index
                                ]
                            )
                        ),
                        "A_search_new": (
                            float(advantage.search_advantage[search_index])
                            if (
                                advantage is not None
                                and is_search
                                and search_index
                                in advantage.search_advantage
                            )
                            else None
                        ),
                        "A_answer": (
                            float(advantage.answer_advantage)
                            if (
                                advantage is not None
                                and is_terminal
                                and advantage.answer_advantage is not None
                            )
                            else None
                        ),
                        "A_format": (
                            float(advantage.centered_format_indicator)
                            if advantage is not None and is_terminal
                            else None
                        ),
                        "turn_ratio": runtime.get("ratio"),
                        "clip_scale": runtime.get("clip_scale"),
                        "clip_lower": runtime.get("clip_lower"),
                        "clip_upper": runtime.get("clip_upper"),
                        "clipped_low": runtime.get("clipped_low"),
                        "clipped_high": runtime.get("clipped_high"),
                        "action_token_count": int(
                            action_by_turn.get(int(turn.turn_index), 0)
                        ),
                        "policy_credit_eligible": bool(
                            turn.policy_credit_eligible
                        ),
                        "search_action_span_valid": bool(
                            turn.search_action_span_valid
                        ),
                        "search_prefix_valid": bool(
                            turn.search_prefix_valid
                        ),
                        "ig_reward_eligible": bool(
                            turn.ig_reward_eligible
                        ),
                        "retriever_executed": (
                            bool(turn.retriever_executed) if is_search else None
                        ),
                        "protocol_invalid": (
                            bool(turn.model_search_invalid) if is_search else None
                        ),
                        "budget_exhausted": (
                            bool(turn.retrieval_budget_exhausted)
                            if is_search
                            else None
                        ),
                        "post_observation_available": (
                            bool(turn.search_prefix_valid) if is_search else None
                        ),
                    }
                )
    return trajectory_records, turn_records


def build_behavior_record(
    *,
    attempt_id: int,
    successful_update_step: int,
    groups: Sequence[Any],
) -> dict[str, Any]:
    trajectories = [
        trajectory for group in groups for trajectory in group.trajectories
    ]
    outcomes = [float(item.task_outcome) for item in trajectories]
    search_counts = [int(item.search_turn_count) for item in trajectories]
    queries = [
        _normalized_query(str(turn.query))
        for item in trajectories
        for turn in item.turns
        if turn.query
    ]
    repeated = 0
    gold_seen_then_search = 0
    for item in trajectories:
        item_queries = [
            _normalized_query(str(turn.query))
            for turn in item.turns
            if turn.query
        ]
        repeated += int(len(item_queries) != len(set(item_queries)))
        aliases = [
            str(value).lower()
            for value in item.metadata.get("gold_aliases", ())
            if str(value).strip()
        ]
        search_turns = [
            turn for turn in item.turns if turn.turn_type is TurnType.SEARCH
        ]
        seen = False
        continued = False
        for index, turn in enumerate(search_turns):
            information = str(turn.information_text or "").lower()
            if any(alias in information for alias in aliases):
                seen = True
                continued = index + 1 < len(search_turns)
                break
        gold_seen_then_search += int(seen and continued)
    count = max(1, len(trajectories))
    unique_queries = len(set(queries))
    query_count = len(queries)
    return {
        "attempt_id": int(attempt_id),
        "successful_update_step": int(successful_update_step),
        "answer_rate": sum(
            bool(item.terminal_answer_valid) for item in trajectories
        )
        / count,
        "format_rate": sum(
            int(item.answer_format_indicator) for item in trajectories
        )
        / count,
        "no_answer_rate": sum(
            item.terminal_policy_credit_turn_index is None
            for item in trajectories
        )
        / count,
        "task_f1_mean": statistics.fmean(outcomes) if outcomes else 0.0,
        "task_f1_p50": _percentile(outcomes, 0.50),
        "task_f1_p95": _percentile(outcomes, 0.95),
        "avg_search_count": statistics.fmean(search_counts)
        if search_counts
        else 0.0,
        "multi_search_rate": sum(value >= 2 for value in search_counts) / count,
        "repeat_query_rate": repeated / count,
        "max_turn_rate": sum(value >= 5 for value in search_counts) / count,
        "gold_seen_then_search_rate": gold_seen_then_search / count,
        "query_diversity": (
            unique_queries / query_count if query_count else 0.0
        ),
        "template_similarity": (
            1.0 - unique_queries / query_count if query_count else 0.0
        ),
        "malformed_rate": sum(
            str(item.parser_status) != "valid" for item in trajectories
        )
        / count,
        "system_invalid_rate": sum(
            not bool(item.trajectory_system_valid) for item in trajectories
        )
        / count,
    }
