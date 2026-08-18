import inspect
import math

import pytest

from agentic_rl.advantage.a2tgpo import (
    TrajectoryCreditInput,
    compute_prompt_advantages,
    turn_advantages_from_record,
)
from agentic_rl.rollout.trajectory_schema import (
    TokenSource,
    TrajectoryRecord,
    TurnRecord,
    TurnType,
)


def _credit(
    ig,
    *,
    search_indices=None,
    ig_eligible=None,
    policy_eligible=None,
    outcome=0.0,
    outcome_eligible=True,
    format_indicator=1,
    answer_credit=True,
    system_valid=True,
):
    indices = tuple(ig if search_indices is None else search_indices)
    if ig_eligible is None:
        ig_eligible = {index: index in ig for index in indices}
    if policy_eligible is None:
        policy_eligible = {index: True for index in indices}
    return TrajectoryCreditInput(
        immediate_ig=ig,
        search_turn_indices=indices,
        ig_reward_eligible=ig_eligible,
        policy_credit_eligible=policy_eligible,
        outcome=outcome,
        outcome_reward_eligible=outcome_eligible,
        format_indicator=format_indicator,
        answer_policy_credit_eligible=answer_credit,
        trajectory_system_valid=system_valid,
    )


def test_search_advantage_has_only_future_ig_and_outcome_terms() -> None:
    result = compute_prompt_advantages(
        [
            _credit({0: 1.0, 1: 3.0}, outcome=1.0, format_indicator=1),
            _credit({0: -1.0, 1: 1.0}, outcome=0.0, format_indicator=0),
        ]
    )
    first = result.trajectories[0]
    expected_search = 0.3 * (
        first.normalized_ig[0] + first.normalized_ig[1]
    ) / math.sqrt(2.0) + first.normalized_outcome
    expected_answer = first.normalized_outcome + 0.5
    assert math.isclose(first.search_advantage[0], expected_search)
    assert math.isclose(first.answer_advantage, expected_answer)
    source = inspect.getsource(compute_prompt_advantages).lower()
    assert ("lambda_" + "mal") not in source
    assert "parser_status" not in source


def test_rescale_n_counts_real_zero_normalized_position() -> None:
    result = compute_prompt_advantages(
        [
            _credit({0: 1.0, 1: 0.0, 2: 1.0}),
            _credit({0: -1.0, 1: 0.0, 2: -1.0}),
        ]
    )
    first = result.trajectories[0]
    assert math.isclose(first.normalized_ig[0], 1.0, abs_tol=1.0e-6)
    assert first.normalized_ig[1] == 0.0
    assert math.isclose(first.normalized_ig[2], 1.0, abs_tol=1.0e-6)
    expected_sum = first.normalized_ig[0] + first.normalized_ig[2]
    assert math.isclose(first.future_ig_sum[0], expected_sum)
    assert first.accumulated_ig_count[0] == 3
    assert math.isclose(
        first.future_ig_rescaled[0], expected_sum / math.sqrt(3.0)
    )


def test_two_real_search_turns_have_n_two() -> None:
    result = compute_prompt_advantages(
        [_credit({0: 1.0, 1: 1.0}), _credit({0: -1.0, 1: -1.0})]
    )
    assert result.trajectories[0].accumulated_ig_count[0] == 2


def test_malformed_final_answer_does_not_remove_valid_search_ig() -> None:
    result = compute_prompt_advantages(
        [
            _credit(
                {0: 1.0},
                outcome=0.0,
                outcome_eligible=False,
                format_indicator=0,
            ),
            _credit({0: -1.0}, outcome=1.0, format_indicator=1),
        ]
    )
    first = result.trajectories[0]
    assert math.isclose(first.normalized_ig[0], 1.0, abs_tol=1.0e-6)
    assert first.normalized_outcome == 0.0
    assert math.isclose(first.search_advantage[0], 0.3, abs_tol=1.0e-6)
    assert first.answer_advantage == -0.5


def test_invalid_second_search_has_no_fake_ig_but_keeps_normal_outcome_anchor() -> None:
    first = _credit(
        {0: 1.0},
        search_indices=(0, 1),
        ig_eligible={0: True, 1: False},
        policy_eligible={0: True, 1: True},
        outcome=1.0,
        outcome_eligible=True,
    )
    second = _credit(
        {0: -1.0},
        search_indices=(0,),
        outcome=0.0,
        outcome_eligible=True,
    )
    result = compute_prompt_advantages([first, second]).trajectories[0]
    assert set(result.normalized_ig) == {0}
    assert result.accumulated_ig_count[0] == 1
    assert result.accumulated_ig_count[1] == 0
    assert result.search_advantage[1] == result.normalized_outcome


def test_system_failure_is_not_wrong_outcome_or_variance_input() -> None:
    system_failure = _credit(
        {},
        search_indices=(0, 1),
        ig_eligible={0: False, 1: False},
        policy_eligible={0: False, 1: False},
        outcome=0.0,
        outcome_eligible=False,
        format_indicator=0,
        answer_credit=False,
        system_valid=False,
    )
    valid = _credit({0: 0.0}, outcome=1.0)
    result = compute_prompt_advantages([system_failure, valid])
    failed = result.trajectories[0]
    assert failed.normalized_ig == {}
    assert failed.normalized_outcome == 0.0
    assert failed.search_advantage == {}
    assert failed.answer_advantage is None
    assert not failed.answer_policy_credit_eligible


def test_ig_eligibility_cannot_resume_after_missing_prefix() -> None:
    invalid = _credit(
        {0: 1.0, 2: 1.0},
        search_indices=(0, 1, 2),
        ig_eligible={0: True, 1: False, 2: True},
    )
    with pytest.raises(ValueError, match="cannot resume"):
        compute_prompt_advantages([invalid])


def _record_with_fallback(
    *,
    fallback_source: TokenSource,
    fallback_policy_credit: bool,
) -> TrajectoryRecord:
    fallback_turn_id = 1 if fallback_source is TokenSource.MODEL else -1
    record = TrajectoryRecord(
        prompt_global_id="p",
        trajectory_id=f"t-{fallback_source.value}",
        input_ids=[10, 11, 12],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.MODEL,
            fallback_source,
        ],
        turn_ids=[-1, 0, fallback_turn_id],
        turns=[
            TurnRecord(
                turn_index=0,
                turn_type=TurnType.SEARCH,
                search_index=0,
                model_text="<search>x</search>",
                search_action_span_valid=True,
                search_prefix_valid=True,
                ig_reward_eligible=True,
                policy_credit_eligible=True,
            ),
            TurnRecord(
                turn_index=1,
                turn_type=TurnType.FALLBACK,
                model_text="<answer>broken"
                if fallback_source is TokenSource.MODEL
                else "",
                parser_status="invalid",
                parser_error_type="unbalanced_tags",
                policy_credit_eligible=fallback_policy_credit,
            ),
        ],
        search_prefix_end_positions=[2],
        search_prefix_before_search_end_positions={0: 1},
        immediate_ig={0: 1.0},
        terminal_answer_valid=False,
        trajectory_protocol_valid=False,
        trajectory_system_valid=True,
    )
    record.validate()
    return record


def test_no_model_terminal_span_emits_no_answer_advantage_or_reassignment() -> None:
    result = compute_prompt_advantages(
        [
            _credit(
                {0: 1.0},
                outcome=1.0,
                format_indicator=0,
                answer_credit=False,
            ),
            _credit(
                {0: -1.0},
                outcome=0.0,
                format_indicator=1,
                answer_credit=True,
            ),
        ]
    ).trajectories[0]
    record = _record_with_fallback(
        fallback_source=TokenSource.CODE_INSERTED,
        fallback_policy_credit=False,
    )
    mapped = turn_advantages_from_record(record, result)
    assert result.answer_advantage is None
    assert set(mapped) == {0}
    assert mapped[0] == result.search_advantage[0]
    assert math.isclose(
        mapped[0],
        0.3 * result.future_ig_rescaled[0] + result.normalized_outcome,
    )
    assert mapped[0] != (
        result.search_advantage[0] + result.centered_format_indicator
    )


def test_real_model_generated_malformed_fallback_receives_answer_advantage() -> None:
    result = compute_prompt_advantages(
        [
            _credit(
                {0: 1.0},
                outcome=0.0,
                outcome_eligible=False,
                format_indicator=0,
                answer_credit=True,
            ),
            _credit({0: -1.0}, outcome=1.0, format_indicator=1),
        ]
    ).trajectories[0]
    record = _record_with_fallback(
        fallback_source=TokenSource.MODEL,
        fallback_policy_credit=True,
    )
    mapped = turn_advantages_from_record(record, result)
    assert result.answer_advantage is not None
    assert set(mapped) == {0, 1}
    assert mapped[0] == result.search_advantage[0]
    assert mapped[1] == result.answer_advantage


def test_environment_fallback_never_receives_answer_advantage() -> None:
    result = compute_prompt_advantages(
        [
            _credit(
                {0: 1.0},
                outcome=0.0,
                format_indicator=0,
                answer_credit=False,
            ),
            _credit({0: -1.0}, outcome=1.0, format_indicator=1),
        ]
    ).trajectories[0]
    record = _record_with_fallback(
        fallback_source=TokenSource.ENVIRONMENT,
        fallback_policy_credit=False,
    )
    mapped = turn_advantages_from_record(record, result)
    assert result.answer_advantage is None
    assert set(mapped) == {0}
