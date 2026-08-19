from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from agentic_rl.outcome.workers import score_trajectory_outcome
from agentic_rl.outcome.workers import PRODUCTION_TASK_SCORER_VERSION
from agentic_rl.rollout.trajectory_schema import (
    PromptTrajectoryGroup,
    TokenSource,
    TrajectoryRecord,
    TurnRecord,
    TurnType,
)
from agentic_rl.selection.candidate_pool import (
    PromptGroup,
    prompt_group_from_outcomes,
    prompt_group_from_trajectories,
)


def trajectory_record_from_extra(
    extra: Mapping[str, Any],
    *,
    outcome_override: Mapping[str, Any] | None = None,
) -> TrajectoryRecord:
    turns = [
        TurnRecord(
            turn_index=int(item["turn_index"]),
            turn_type=TurnType(str(item["turn_type"])),
            model_text=str(item["model_text"]),
            search_index=(
                None
                if item.get("search_index") is None
                else int(item["search_index"])
            ),
            query=None if item.get("query") is None else str(item["query"]),
            information_text=(
                None
                if item.get("information_text") is None
                else str(item["information_text"])
            ),
            parser_status=str(item.get("parser_status", "valid")),
            parser_error_type=(
                None
                if item.get("parser_error_type") is None
                else str(item["parser_error_type"])
            ),
            search_action_span_valid=bool(
                item.get("search_action_span_valid", False)
            ),
            search_prefix_valid=bool(item.get("search_prefix_valid", False)),
            ig_reward_eligible=bool(item.get("ig_reward_eligible", False)),
            policy_credit_eligible=bool(
                item.get("policy_credit_eligible", False)
            ),
            no_new_observation=(
                None
                if item.get("no_new_observation") is None
                else bool(item["no_new_observation"])
            ),
            exact_query_repeat=bool(item.get("exact_query_repeat", False)),
            different_query_no_new_passage=bool(
                item.get("different_query_no_new_passage", False)
            ),
            current_passage_keys=tuple(
                str(value) for value in item.get("current_passage_keys", ())
            ),
            new_passage_keys=tuple(
                str(value) for value in item.get("new_passage_keys", ())
            ),
            role_localized_gate_enabled=bool(
                item.get("role_localized_gate_enabled", False)
            ),
            retriever_executed=bool(item.get("retriever_executed", False)),
            retrieval_budget_exhausted=bool(
                item.get("retrieval_budget_exhausted", False)
            ),
            model_search_invalid=bool(item.get("model_search_invalid", False)),
            main_credit_eligible=bool(item.get("main_credit_eligible", False)),
            branch_type=(
                None if item.get("branch_type") is None else str(item["branch_type"])
            ),
            no_new_reason=(
                None
                if item.get("no_new_reason") is None
                else str(item["no_new_reason"])
            ),
            raw_query=(
                None if item.get("raw_query") is None else str(item["raw_query"])
            ),
            canonical_query=(
                None
                if item.get("canonical_query") is None
                else str(item["canonical_query"])
            ),
            new_passage_count=int(item.get("new_passage_count", 0)),
            stable_passage_keys_before=tuple(
                str(value)
                for value in item.get("stable_passage_keys_before", ())
            ),
            stable_passage_keys_after=tuple(
                str(value)
                for value in item.get("stable_passage_keys_after", ())
            ),
            action_token_span=(
                None
                if item.get("action_token_span") is None
                else tuple(map(int, item["action_token_span"]))
            ),
            think_token_span=(
                None
                if item.get("think_token_span") is None
                else tuple(map(int, item["think_token_span"]))
            ),
            decision_token_span=(
                None
                if item.get("decision_token_span") is None
                else tuple(map(int, item["decision_token_span"]))
            ),
            query_token_span=(
                None
                if item.get("query_token_span") is None
                else tuple(map(int, item["query_token_span"]))
            ),
            observation_token_span=(
                None
                if item.get("observation_token_span") is None
                else tuple(map(int, item["observation_token_span"]))
            ),
        )
        for item in extra["turn_records"]
    ]
    aliases = tuple(str(value) for value in extra["gold_aliases"])
    outcome = None
    if outcome_override is None:
        outcome = score_trajectory_outcome(
            [str(value) for value in extra["model_actions"]],
            aliases,
            data_source=str(extra.get("data_source", "")),
            trajectory_system_valid=bool(extra["trajectory_system_valid"]),
        )
        task_outcome = float(outcome.task_outcome)
        format_indicator = int(outcome.format_indicator)
        terminal_answer_valid = bool(outcome.terminal_answer_valid)
        trajectory_protocol_valid = bool(outcome.parse.trajectory_valid)
        trajectory_system_valid = bool(outcome.trajectory_system_valid)
        parser_status = str(outcome.parse.parser_status)
        parser_error_type = outcome.parse.parser_error_type
        fallback_status = outcome.parse.fallback_status
    else:
        if str(outcome_override["trajectory_id"]) != str(extra["trajectory_id"]):
            raise ValueError("Outcome result does not match trajectory_id")
        task_outcome = float(outcome_override["task_outcome"])
        format_indicator = int(outcome_override["format_indicator"])
        terminal_answer_valid = bool(
            outcome_override["terminal_answer_valid"]
        )
        trajectory_protocol_valid = bool(
            terminal_answer_valid
            and str(outcome_override["parser_status"]) == "valid"
        )
        trajectory_system_valid = bool(
            outcome_override["trajectory_system_valid"]
        )
        parser_status = str(outcome_override["parser_status"])
        parser_error_type = (
            None
            if outcome_override.get("parser_error_type") is None
            else str(outcome_override["parser_error_type"])
        )
        fallback_status = (
            None if trajectory_protocol_valid else "malformed_fallback"
        )
    record = TrajectoryRecord(
        prompt_global_id=str(extra["prompt_global_id"]),
        trajectory_id=str(extra["trajectory_id"]),
        input_ids=[int(value) for value in extra["full_input_ids_unpadded"]],
        token_sources=[
            TokenSource(str(value)) for value in extra["token_sources_unpadded"]
        ],
        turn_ids=[int(value) for value in extra["turn_ids_unpadded"]],
        turns=turns,
        search_prefix_end_positions=[
            int(value) for value in extra["prefix_end_positions"]
        ],
        search_prefix_before_search_end_positions={
            int(index): int(endpoint)
            for index, endpoint in dict(
                extra["prefix_before_search_end_positions"]
            ).items()
        },
        sampled_action_logprobs=[
            float(value)
            for value in extra.get(
                "sampled_action_logprobs_unpadded",
                [],
            )
        ],
        task_outcome=task_outcome,
        answer_format_indicator=format_indicator,
        terminal_answer_valid=terminal_answer_valid,
        trajectory_protocol_valid=trajectory_protocol_valid,
        trajectory_system_valid=trajectory_system_valid,
        parser_status=parser_status,
        parser_error_type=parser_error_type,
        fallback_status=fallback_status,
        environment_failure_code=(
            None
            if extra.get("environment_failure_code") is None
            else str(extra["environment_failure_code"])
        ),
        metadata={
            "snapshot_step": int(extra["snapshot_step"]),
            "data_source": str(extra.get("data_source", "")),
            "dataset_row_id": str(extra.get("dataset_row_id", "")),
            "dataset_source_index": int(
                extra.get("dataset_source_index", -1)
            ),
            "gold_aliases": aliases,
            "canonical_answer": str(extra["canonical_answer"]),
            "termination_reason": str(extra["termination_reason"]),
            "retrieval_records": list(extra.get("retrieval_records", [])),
            "turn_records_raw": list(extra.get("turn_records", [])),
            "prefix_before_search_end_positions": {
                int(index): int(endpoint)
                for index, endpoint in dict(
                    extra["prefix_before_search_end_positions"]
                ).items()
            },
            "task_scorer_version": (
                PRODUCTION_TASK_SCORER_VERSION
                if outcome_override is None
                else str(outcome_override["scorer_version"])
            ),
        },
    )
    record.validate()
    return record


def attach_exact_ig(
    record: TrajectoryRecord,
    immediate_ig: Sequence[float],
) -> TrajectoryRecord:
    eligible_indices = [
        index
        for index, eligible in sorted(
            record.ig_reward_eligibility_by_search_index.items()
        )
        if eligible
    ]
    values = [float(value) for value in immediate_ig]
    if len(values) != len(eligible_indices):
        raise ValueError(
            f"{record.trajectory_id}: Exact-IG count {len(values)} does not "
            f"match eligible Search count {len(eligible_indices)}"
        )
    record.immediate_ig = dict(zip(eligible_indices, values, strict=True))
    record.validate()
    return record


def prompt_groups_from_records(
    records: Sequence[TrajectoryRecord],
    *,
    aliases_by_prompt: Mapping[str, Sequence[str]],
    expected_group_size: int,
    outcome_only_selection: bool = False,
) -> tuple[PromptGroup, ...]:
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.prompt_global_id].append(record)
    results: list[PromptGroup] = []
    for prompt_id in sorted(grouped):
        trajectories = tuple(
            sorted(grouped[prompt_id], key=lambda item: item.trajectory_id)
        )
        aliases = tuple(str(value) for value in aliases_by_prompt[prompt_id])
        group = PromptTrajectoryGroup(
            prompt_global_id=prompt_id,
            trajectories=trajectories,
            aliases=aliases,
        )
        builder = (
            prompt_group_from_outcomes
            if outcome_only_selection
            else prompt_group_from_trajectories
        )
        results.append(builder(group, expected_group_size=expected_group_size))
    return tuple(results)
