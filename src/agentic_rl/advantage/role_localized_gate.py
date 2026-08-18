from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agentic_rl.outcome.workers import SUFFICIENCY_EXACT_SCORER_VERSION
from agentic_rl.rollout.search_role_provenance import (
    ROLE_LOCALIZED_BRANCH_N_BUDGET,
    ROLE_LOCALIZED_BRANCH_N_INVALID,
    ROLE_LOCALIZED_BRANCH_N_SOFT,
    ROLE_LOCALIZED_BRANCH_NORMAL,
    ROLE_LOCALIZED_BRANCH_S_BEFORE,
    classify_role_localized_search_branch,
)
from agentic_rl.rollout.trajectory_schema import TurnType


@dataclass(frozen=True)
class RoleLocalizedTrajectoryCredits:
    main: dict[int, float]
    decision: dict[int, float]
    query: dict[int, float]
    branch: dict[int, str]
    sufficient_before: dict[int, bool]
    sufficient_after: dict[int, bool]
    no_new_observation: dict[int, bool]
    effective_cumulative_ig: dict[int, float]
    effective_cumulative_ig_count: dict[int, int]
    probe_reward_delta: dict[int, float]
    routed_outcome: dict[int, float]
    allowed_soft_duplicate_main_query_overlap_count: int
    empty_query_without_query_span_count: int


def _validated_probe(
    raw_probe: Mapping[str, Any],
    *,
    stage: str,
    expected_policy_version: int | None,
    expected_scorer_version: str | None,
) -> tuple[bool, float]:
    if stage not in {"pre", "post"}:
        raise ValueError(f"Unsupported Answer Probe stage: {stage}")
    for field_name in (
        "parser_success",
        "no_answer",
        "output_truncated",
        "alias_aware_exact",
        "prefix_provenance_valid",
        "detached",
    ):
        if not isinstance(raw_probe.get(field_name), bool):
            raise ValueError(f"{stage} Probe {field_name} must be a bool")
    if raw_probe["prefix_provenance_valid"] is not True:
        raise ValueError(f"{stage} Probe prefix provenance is invalid")
    if raw_probe["detached"] is not True:
        raise ValueError(f"{stage} Probe must be detached")
    reward = float(raw_probe.get("raw_task_reward", float("nan")))
    if not math.isfinite(reward):
        raise ValueError(f"{stage} Probe reward is non-finite")
    if int(raw_probe.get("completion_count", -1)) != 1:
        raise ValueError(f"{stage} Probe must contain one completion")
    locked = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "n": 1,
        "max_tokens": 500,
        "stop": ["</answer>"],
    }
    for field_name, expected in locked.items():
        actual = raw_probe.get(field_name)
        if field_name == "stop" and isinstance(actual, (tuple, list)):
            actual = list(actual)
        if actual != expected:
            raise ValueError(
                f"{stage} Probe {field_name}={actual!r}, expected {expected!r}"
            )
    if str(raw_probe.get("scorer_version")) != SUFFICIENCY_EXACT_SCORER_VERSION:
        raise ValueError(f"{stage} Probe exact scorer version mismatch")
    if expected_scorer_version is not None and str(
        raw_probe.get("task_scorer_version")
    ) != str(expected_scorer_version):
        raise ValueError(f"{stage} Probe task scorer version mismatch")
    if expected_policy_version is not None:
        version_fields = (
            "candidate_rollout_policy_version",
            "exact_ig_policy_version",
            "probe_policy_version",
            "old_logprob_policy_version",
        )
        try:
            versions = {int(raw_probe[name]) for name in version_fields}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{stage} Probe policy metadata is invalid") from error
        if versions != {int(expected_policy_version)}:
            raise ValueError(f"{stage} Probe/on-policy versions disagree")

    sufficient = bool(
        raw_probe["alias_aware_exact"]
        and raw_probe["parser_success"]
        and not raw_probe["no_answer"]
        and not raw_probe["output_truncated"]
    )
    field_name = (
        "sufficient_before_search" if stage == "pre" else "sufficient_after_search"
    )
    if raw_probe.get(field_name) is not sufficient:
        raise ValueError(f"{stage} Probe precomputed sufficiency differs from raw fields")
    return sufficient, reward


def _query_token_count(turn: Any) -> int:
    span = getattr(turn, "query_token_span", None)
    if span is None:
        raise ValueError("Role-localized Search is missing query_token_span")
    if len(span) != 2:
        raise ValueError("query_token_span must be a half-open pair")
    count = int(span[1]) - int(span[0])
    if count < 0:
        raise ValueError("query_token_span is inverted")
    return count


def build_role_localized_trajectory_credits(
    record: Any,
    *,
    normalized_ig: Mapping[int, float],
    normalized_outcome: float,
    optimized_search_indices: Sequence[int],
    expected_policy_version: int | None,
    expected_scorer_version: str | None,
    probe_epsilon: float,
) -> RoleLocalizedTrajectoryCredits:
    """Recompute B and construct independent Main/Decision/Query credits."""

    if not math.isclose(float(probe_epsilon), 1.0e-6, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Role-localized mode locks probe_epsilon=1e-6")
    all_turns = {
        int(turn.search_index): turn
        for turn in record.turns
        if turn.turn_type is TurnType.SEARCH and turn.search_index is not None
    }
    if tuple(sorted(all_turns)) != tuple(range(len(all_turns))):
        raise ValueError(f"{record.trajectory_id}: Search indices are not contiguous")
    optimized = set(map(int, optimized_search_indices))
    expected_optimized = {
        index for index, turn in all_turns.items() if turn.policy_credit_eligible
    }
    if optimized != expected_optimized:
        raise ValueError(f"{record.trajectory_id}: optimized Search coverage is invalid")

    raw_by_search = record.metadata.get("routed_answer_probes", {})
    if all_turns and not isinstance(raw_by_search, Mapping):
        raise ValueError(f"{record.trajectory_id}: routed Answer Probes are missing")
    sufficient_before: dict[int, bool] = {}
    sufficient_after: dict[int, bool] = {}
    pre_rewards: dict[int, float] = {}
    post_rewards: dict[int, float] = {}
    branches: dict[int, str] = {}
    no_new: dict[int, bool] = {}

    for search_index, turn in sorted(all_turns.items()):
        if not getattr(turn, "role_localized_gate_enabled", False):
            raise ValueError(
                f"{record.trajectory_id}:{search_index}: role provenance is disabled"
            )
        raw_stages = raw_by_search.get(
            search_index,
            raw_by_search.get(str(search_index)),
        )
        if not isinstance(raw_stages, Mapping):
            raise ValueError(
                f"{record.trajectory_id}:{search_index}: Probe stages are missing"
            )
        raw_pre = raw_stages.get("pre")
        if not isinstance(raw_pre, Mapping):
            raise ValueError(
                f"{record.trajectory_id}:{search_index}: pre Probe is missing"
            )
        before, pre_reward = _validated_probe(
            raw_pre,
            stage="pre",
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
        )
        branch = classify_role_localized_search_branch(
            retrieval_budget_exhausted=bool(turn.retrieval_budget_exhausted),
            model_search_invalid=bool(turn.model_search_invalid),
            sufficient_before_search=before,
            retriever_executed=bool(turn.retriever_executed),
            no_new_observation=turn.no_new_observation,
        )
        if str(turn.branch_type) != branch:
            raise ValueError(
                f"{record.trajectory_id}:{search_index}: persisted branch mismatch"
            )
        expected_main = branch in {
            ROLE_LOCALIZED_BRANCH_N_SOFT,
            ROLE_LOCALIZED_BRANCH_NORMAL,
        }
        if bool(turn.main_credit_eligible) is not expected_main:
            raise ValueError(
                f"{record.trajectory_id}:{search_index}: Main eligibility mismatch"
            )
        sufficient_before[search_index] = before
        pre_rewards[search_index] = pre_reward
        branches[search_index] = branch
        no_new[search_index] = bool(turn.no_new_observation)

        raw_post = raw_stages.get("post")
        if branch in {
            ROLE_LOCALIZED_BRANCH_N_BUDGET,
            ROLE_LOCALIZED_BRANCH_N_INVALID,
            ROLE_LOCALIZED_BRANCH_S_BEFORE,
        }:
            if raw_post is not None:
                raise ValueError(
                    f"{record.trajectory_id}:{search_index}: {branch} has a post Probe"
                )
            continue
        if not isinstance(raw_post, Mapping):
            raise ValueError(
                f"{record.trajectory_id}:{search_index}: post Probe is missing"
            )
        after, post_reward = _validated_probe(
            raw_post,
            stage="post",
            expected_policy_version=expected_policy_version,
            expected_scorer_version=expected_scorer_version,
        )
        sufficient_after[search_index] = after
        post_rewards[search_index] = post_reward

    main: dict[int, float] = {}
    decision: dict[int, float] = {}
    query: dict[int, float] = {}
    cumulative: dict[int, float] = {}
    cumulative_count: dict[int, int] = {}
    probe_delta: dict[int, float] = {}
    routed_outcome: dict[int, float] = {}
    allowed_soft_overlap_count = 0
    empty_invalid_query_count = 0

    for search_index in sorted(optimized):
        turn = all_turns[search_index]
        branch = branches[search_index]
        decision_credit = (
            -1.0
            if branch == ROLE_LOCALIZED_BRANCH_N_BUDGET
            else -0.5
            if branch in {
                ROLE_LOCALIZED_BRANCH_N_INVALID,
                ROLE_LOCALIZED_BRANCH_S_BEFORE,
            }
            else 0.0
        )
        query_count = _query_token_count(turn)
        if branch == ROLE_LOCALIZED_BRANCH_N_INVALID:
            query_credit = -0.5 if query_count > 0 else 0.0
            empty_invalid_query_count += int(query_count == 0)
        else:
            raw_ig = record.immediate_ig.get(search_index)
            duplicate_gate = bool(
                branch == ROLE_LOCALIZED_BRANCH_N_SOFT
                and turn.exact_query_repeat
                and int(turn.new_passage_count) == 0
                and raw_ig is not None
                and float(raw_ig) <= 0.0
            )
            query_credit = -0.25 if duplicate_gate else 0.0
            allowed_soft_overlap_count += int(duplicate_gate)

        if branch in {
            ROLE_LOCALIZED_BRANCH_N_SOFT,
            ROLE_LOCALIZED_BRANCH_NORMAL,
        }:
            values: list[float] = []
            for future_index in sorted(
                index for index in all_turns if index >= search_index
            ):
                future_branch = branches[future_index]
                if future_branch == ROLE_LOCALIZED_BRANCH_S_BEFORE:
                    break
                future_turn = all_turns[future_index]
                if future_branch in {
                    ROLE_LOCALIZED_BRANCH_N_SOFT,
                    ROLE_LOCALIZED_BRANCH_NORMAL,
                }:
                    if not (
                        future_turn.policy_credit_eligible
                        and future_turn.ig_reward_eligible
                        and future_turn.main_credit_eligible
                        and future_index in normalized_ig
                    ):
                        raise ValueError(
                            f"{record.trajectory_id}:{future_index}: B requires "
                            "eligible normalized local IG"
                        )
                    values.append(float(normalized_ig[future_index]))
                if sufficient_after.get(future_index, False):
                    break
            if not values:
                raise RuntimeError(
                    f"{record.trajectory_id}:{search_index}: B accumulation is empty"
                )
            d_ig_eff = float(math.fsum(values) / math.sqrt(len(values)))
            delta = float(post_rewards[search_index] - pre_rewards[search_index])
            z_outcome = float(normalized_outcome)
            route = (
                max(z_outcome, 0.0)
                if delta > probe_epsilon
                else min(z_outcome, 0.0)
                if delta < -probe_epsilon
                else 0.0
            )
            main_credit = float(d_ig_eff + route)
            cumulative[search_index] = d_ig_eff
            cumulative_count[search_index] = len(values)
            probe_delta[search_index] = delta
            routed_outcome[search_index] = route
        else:
            main_credit = 0.0

        allowed_overlap = bool(
            branch == ROLE_LOCALIZED_BRANCH_N_SOFT
            and decision_credit == 0.0
            and query_credit == -0.25
        )
        if main_credit != 0.0 and (
            decision_credit != 0.0 or query_credit != 0.0
        ) and not allowed_overlap:
            raise RuntimeError("Unexpected Main/Gate overlap")
        if query_credit != 0.0 and query_count == 0:
            raise RuntimeError("Nonzero Query credit has no Query span")
        if not all(
            math.isfinite(value)
            for value in (main_credit, decision_credit, query_credit)
        ):
            raise RuntimeError("Role-localized Search credit is non-finite")
        main[search_index] = main_credit
        decision[search_index] = decision_credit
        query[search_index] = query_credit

    return RoleLocalizedTrajectoryCredits(
        main=main,
        decision=decision,
        query=query,
        branch=branches,
        sufficient_before=sufficient_before,
        sufficient_after=sufficient_after,
        no_new_observation=no_new,
        effective_cumulative_ig=cumulative,
        effective_cumulative_ig_count=cumulative_count,
        probe_reward_delta=probe_delta,
        routed_outcome=routed_outcome,
        allowed_soft_duplicate_main_query_overlap_count=allowed_soft_overlap_count,
        empty_query_without_query_span_count=empty_invalid_query_count,
    )
