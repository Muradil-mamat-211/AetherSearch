from __future__ import annotations

import math

import numpy as np

from agentic_rl.advantage.a2tgpo import (
    TrajectoryCreditInput,
    compute_prompt_advantages,
    rebuild_search_advantages,
    turn_advantages_from_record,
)
from agentic_rl.advantage.mica_ig import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
    compute_mica_search_advantage,
    compute_normalized_terminal_outcomes,
)
from agentic_rl.rollout.trajectory_schema import (
    TokenSource,
    TrajectoryRecord,
    TurnRecord,
    TurnType,
)


def _mica(
    raw,
    *,
    search_indices=None,
    eligible=None,
    policy=None,
    z_outcomes=None,
):
    count = len(raw)
    indices = (
        [tuple(sorted(values)) for values in raw]
        if search_indices is None
        else search_indices
    )
    eligibility = (
        [{index: True for index in values} for values in indices]
        if eligible is None
        else eligible
    )
    policy_eligibility = (
        [{index: True for index in values} for values in indices]
        if policy is None
        else policy
    )
    return compute_mica_search_advantage(
        trajectory_ids=[f"t{index}" for index in range(count)],
        search_indices_by_trajectory=indices,
        raw_ig_by_trajectory=raw,
        ig_reward_eligible_by_trajectory=eligibility,
        policy_credit_eligible_by_trajectory=policy_eligibility,
        normalized_terminal_outcomes=(
            [0.0] * count if z_outcomes is None else z_outcomes
        ),
        gamma=1.0,
        alpha=0.5,
    )


def _population_z(values):
    array = np.asarray(values, dtype=np.float64)
    return (array - array.mean()) / (array.std(ddof=0) + 1.0e-6)


def test_case_1_standard_group_local_and_return_are_hand_computed() -> None:
    raw = [
        {0: 1.0, 1: 0.0},
        {0: 2.0, 1: 0.0},
        {0: 3.0, 1: 0.0},
        {0: 4.0, 1: 4.0},
    ]
    result = _mica(raw)
    expected_local = _population_z([1.0, 2.0, 3.0, 4.0])
    expected_return = _population_z([1.0, 2.0, 3.0, 8.0])
    for index, trajectory in enumerate(result.trajectories):
        credit = trajectory.by_search_index[0]
        assert math.isclose(credit.local_advantage, expected_local[index])
        assert math.isclose(credit.return_advantage, expected_return[index])
        assert math.isclose(
            credit.search_advantage,
            0.5 * expected_local[index] + 0.5 * expected_return[index],
        )


def test_case_2_no_cross_depth_contamination() -> None:
    first = _mica([{0: 1.0, 1: 0.1}, {0: 2.0, 1: 0.3}])
    changed = _mica([{0: -1.0e9, 1: 0.1}, {0: 1.0e9, 1: 0.3}])
    for left, right in zip(first.trajectories, changed.trajectories, strict=True):
        assert left.by_search_index[1].local_advantage == (
            right.by_search_index[1].local_advantage
        )
        assert left.by_search_index[1].return_advantage == (
            right.by_search_index[1].return_advantage
        )


def test_case_3_no_cross_prompt_contamination() -> None:
    prompt_a_before = _mica([{0: 1.0}, {0: 3.0}])
    _mica([{0: -1.0e12}, {0: 1.0e12}, {0: 7.0}])
    prompt_a_after = _mica([{0: 1.0}, {0: 3.0}])
    assert prompt_a_before == prompt_a_after


def test_case_4_singleton_uses_normalized_outcome_without_relative_channels() -> None:
    result = _mica([{0: 2.0}], z_outcomes=[0.75])
    credit = result.trajectories[0].by_search_index[0]
    assert credit.peer_count == 1
    assert credit.singleton_fallback
    assert credit.local_advantage is None
    assert credit.return_advantage is None
    assert credit.search_advantage == 0.75


def test_case_5_singleton_tail_is_not_assumed_terminal() -> None:
    result = _mica(
        [
            {0: 0.0, 1: 0.0, 2: 1.0, 3: 2.0},
            {0: 1.0, 1: 1.0},
            {0: 2.0, 1: 2.0},
        ],
        z_outcomes=[1.25, -0.5, -0.5],
    )
    trajectory = result.trajectories[0]
    assert trajectory.by_search_index[2].search_advantage == 1.25
    assert trajectory.by_search_index[3].search_advantage == 1.25
    assert trajectory.singleton_tail_start_depth == 2
    assert trajectory.singleton_consecutive_length == 2


def test_case_6_local_zero_variance_keeps_fixed_half_return_weight() -> None:
    result = _mica([{0: 1.0, 1: 0.0}, {0: 1.0, 1: 2.0}])
    for trajectory in result.trajectories:
        credit = trajectory.by_search_index[0]
        assert credit.local_advantage == 0.0
        assert credit.return_advantage != 0.0
        assert credit.search_advantage == 0.5 * credit.return_advantage
        assert not credit.singleton_fallback


def test_case_7_return_zero_variance_keeps_fixed_half_local_weight() -> None:
    result = _mica([{0: 0.0, 1: 2.0}, {0: 2.0, 1: 0.0}])
    for trajectory in result.trajectories:
        credit = trajectory.by_search_index[0]
        assert credit.return_advantage == 0.0
        assert credit.local_advantage != 0.0
        assert credit.search_advantage == 0.5 * credit.local_advantage
        assert not credit.singleton_fallback


def test_case_8_both_zero_variance_is_zero_not_outcome_fallback() -> None:
    result = _mica(
        [{0: 1.0}, {0: 1.0}],
        z_outcomes=[1.0, -1.0],
    )
    for trajectory in result.trajectories:
        credit = trajectory.by_search_index[0]
        assert credit.local_advantage == 0.0
        assert credit.return_advantage == 0.0
        assert credit.search_advantage == 0.0
        assert not credit.singleton_fallback


def test_case_9_singleton_with_zero_outcome_variance_is_zero() -> None:
    z_outcomes = compute_normalized_terminal_outcomes([1.0])
    assert z_outcomes == (0.0,)
    credit = _mica([{0: 1.0}], z_outcomes=z_outcomes).trajectories[
        0
    ].by_search_index[0]
    assert credit.search_advantage == 0.0


def test_case_10_gamma_one_is_literal_suffix_sum_not_telescoping_assertion() -> None:
    result = _mica([{0: 1.0, 1: -3.0, 2: 5.0}])
    credit = result.trajectories[0].by_search_index
    assert credit[0].ig_return == 3.0
    assert credit[1].ig_return == 2.0
    assert credit[2].ig_return == 5.0


def test_case_11_invalid_future_search_is_absent_not_fake_zero() -> None:
    result = _mica(
        [{0: 2.0}, {0: -2.0}],
        search_indices=[(0, 1), (0,)],
        eligible=[{0: True, 1: False}, {0: True}],
        policy=[{0: True, 1: True}, {0: True}],
    )
    first = result.trajectories[0].by_search_index
    assert first[0].ig_return == 2.0
    assert first[0].peer_count == 2
    assert first[1].raw_ig is None
    assert first[1].ig_return is None
    assert first[1].peer_count == 0
    assert first[1].search_advantage == 0.0


def _record(trajectory_id: str, raw_ig: float, outcome: float, fmt: int):
    record = TrajectoryRecord(
        prompt_global_id="p",
        trajectory_id=trajectory_id,
        input_ids=[10, 20, 21, 30, 40],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.MODEL,
        ],
        turn_ids=[-1, 0, 0, -1, 1],
        turns=[
            TurnRecord(
                turn_index=0,
                turn_type=TurnType.SEARCH,
                search_index=0,
                model_text="<think>x</think><search>q</search>",
                search_action_span_valid=True,
                search_prefix_valid=True,
                ig_reward_eligible=True,
                policy_credit_eligible=True,
            ),
            TurnRecord(
                turn_index=1,
                turn_type=TurnType.ANSWER,
                model_text="<answer>a</answer>",
                policy_credit_eligible=True,
            ),
        ],
        search_prefix_end_positions=[1, 4],
        search_prefix_before_search_end_positions={0: 1},
        immediate_ig={0: raw_ig},
        task_outcome=outcome,
        answer_format_indicator=fmt,
        terminal_answer_valid=True,
        trajectory_protocol_valid=True,
        trajectory_system_valid=True,
    )
    record.validate()
    return record


def _credit_from_record(record):
    return TrajectoryCreditInput(
        immediate_ig=dict(record.immediate_ig),
        search_turn_indices=(0,),
        ig_reward_eligible={0: True},
        policy_credit_eligible={0: True},
        outcome=record.task_outcome,
        outcome_reward_eligible=True,
        format_indicator=record.answer_format_indicator,
        answer_policy_credit_eligible=True,
    )


def test_case_12_answer_advantage_is_bitwise_unchanged() -> None:
    records = [_record("t0", -1.0, 0.0, 0), _record("t1", 1.0, 1.0, 1)]
    old = compute_prompt_advantages(
        [_credit_from_record(record) for record in records],
        accumulate_future_ig=False,
        lambda_ig=None,
    )
    expected = tuple(value.answer_advantage for value in old.trajectories)
    rebuilt, _ = rebuild_search_advantages(
        records,
        old,
        search_task_mode=ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
        group_size=2,
    )
    assert tuple(value.answer_advantage for value in rebuilt.trajectories) == expected


def test_case_13_policy_mask_excludes_observation_and_broadcasts_whole_search() -> None:
    records = [_record("t0", -1.0, 0.0, 0), _record("t1", 1.0, 1.0, 1)]
    base = compute_prompt_advantages(
        [_credit_from_record(record) for record in records],
        accumulate_future_ig=False,
        lambda_ig=None,
    )
    rebuilt, _ = rebuild_search_advantages(
        records,
        base,
        search_task_mode=ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
        group_size=2,
    )
    record = records[0]
    mapped = turn_advantages_from_record(record, rebuilt.trajectories[0])
    assert record.policy_mask == [0, 1, 1, 0, 1]
    assert mapped[0] == rebuilt.trajectories[0].search_advantage[0]
    assert record.turn_ids[1] == record.turn_ids[2] == 0


def test_case_14_old_mode_regression_fixture_is_unchanged() -> None:
    records = [_record("t0", -1.0, 0.0, 0), _record("t1", 1.0, 1.0, 1)]
    original = compute_prompt_advantages(
        [_credit_from_record(record) for record in records]
    )
    rebuilt, _ = rebuild_search_advantages(
        records,
        original,
        search_task_mode="normalized_outcome",
        group_size=2,
        lambda_ig=0.3,
    )
    for before, after in zip(
        original.trajectories,
        rebuilt.trajectories,
        strict=True,
    ):
        assert after.answer_advantage == before.answer_advantage
        assert after.search_advantage == {
            0: 0.3 * before.future_ig_rescaled[0] + before.normalized_outcome
        }
