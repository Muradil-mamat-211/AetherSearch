import pytest

from agentic_rl.rollout.token_provenance import (
    assert_environment_information_masked,
    assign_model_turns_with_fallback,
    build_policy_credit_mask,
)
from agentic_rl.rollout.trajectory_schema import (
    TokenSource,
    TrajectoryRecord,
    TurnRecord,
    TurnType,
)


def test_model_tokens_use_fallback_and_environment_is_masked() -> None:
    sources = [
        TokenSource.PROMPT,
        TokenSource.MODEL,
        TokenSource.MODEL,
        TokenSource.ENVIRONMENT,
        TokenSource.CODE_INSERTED,
        TokenSource.PADDING,
    ]
    assignment = assign_model_turns_with_fallback(
        sources,
        [None, 0, None, None, None, None],
        next_fallback_turn_index=1,
    )
    assert assignment.turn_ids == (-1, 0, 1, -1, -1, -1)
    assert assignment.action_mask == (0, 1, 1, 0, 0, 0)
    assert assignment.unmatched_model_token_count == 1
    assert assignment.fallback_turn_index == 1
    assert_environment_information_masked(sources, assignment.action_mask)


def test_malformed_diagnostic_does_not_implicitly_remove_model_credit() -> None:
    sources = [TokenSource.PROMPT, TokenSource.MODEL, TokenSource.MODEL]
    mask = build_policy_credit_mask(
        sources,
        [-1, 0, 0],
        {0: True},
        trajectory_system_valid=True,
    )
    assert mask == (0, 1, 1)


def test_environment_and_code_tokens_never_receive_policy_credit() -> None:
    mask = build_policy_credit_mask(
        [
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.CODE_INSERTED,
        ],
        [0, -1, -1],
        {0: True},
        trajectory_system_valid=True,
    )
    assert mask == (1, 0, 0)


def test_system_invalid_trajectory_masks_all_policy_credit() -> None:
    mask = build_policy_credit_mask(
        [TokenSource.MODEL, TokenSource.MODEL],
        [0, 1],
        {0: False, 1: False},
        trajectory_system_valid=False,
    )
    assert mask == (0, 0)


def test_trajectory_validity_is_channel_specific() -> None:
    trajectory = TrajectoryRecord(
        prompt_global_id="p",
        trajectory_id="t",
        input_ids=[1, 2, 3, 4],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.MODEL,
        ],
        turn_ids=[-1, 0, -1, 1],
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
                model_text="<answer>broken",
                parser_status="invalid",
                parser_error_type="unbalanced_tags",
                policy_credit_eligible=True,
            ),
        ],
        search_prefix_end_positions=[3],
        search_prefix_before_search_end_positions={0: 1},
        immediate_ig={0: 0.25},
        terminal_answer_valid=False,
        trajectory_protocol_valid=False,
        trajectory_system_valid=True,
    )
    trajectory.validate()
    assert not trajectory.trajectory_valid
    assert not trajectory.outcome_reward_eligible
    assert trajectory.ig_reward_eligibility_by_search_index == {0: True}
    assert trajectory.policy_credit_mask == [0, 1, 0, 1]
    assert trajectory.terminal_policy_credit_turn_index == 1
    assert trajectory.optimization_ready
    assert trajectory.policy_mask == [0, 1, 0, 1]
    assert trajectory.kl_mask == trajectory.policy_mask


def test_code_inserted_fallback_receives_no_terminal_policy_credit() -> None:
    trajectory = TrajectoryRecord(
        prompt_global_id="p",
        trajectory_id="t",
        input_ids=[1, 2, 3],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.CODE_INSERTED,
        ],
        turn_ids=[-1, 0, -1],
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
                model_text="",
                policy_credit_eligible=False,
            ),
        ],
        search_prefix_end_positions=[2],
        search_prefix_before_search_end_positions={0: 1},
        immediate_ig={0: 0.1},
        terminal_answer_valid=False,
        trajectory_protocol_valid=False,
        trajectory_system_valid=True,
    )
    trajectory.validate()
    assert trajectory.terminal_policy_credit_turn_index is None
    assert trajectory.policy_credit_mask == [0, 1, 0]
    assert trajectory.policy_mask == [0, 1, 0]
    assert trajectory.kl_mask == [0, 1, 0]
    assert trajectory.optimization_ready


def test_only_last_real_terminal_model_turn_can_receive_terminal_credit() -> None:
    trajectory = TrajectoryRecord(
        prompt_global_id="p",
        trajectory_id="t",
        input_ids=[1, 2, 3],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.MODEL,
        ],
        turn_ids=[-1, 0, 1],
        turns=[
            TurnRecord(
                turn_index=0,
                turn_type=TurnType.FALLBACK,
                model_text="<answer>first",
                policy_credit_eligible=True,
            ),
            TurnRecord(
                turn_index=1,
                turn_type=TurnType.FALLBACK,
                model_text="<answer>last",
                policy_credit_eligible=True,
            ),
        ],
        search_prefix_end_positions=[1],
        terminal_answer_valid=False,
        trajectory_protocol_valid=False,
        trajectory_system_valid=True,
    )
    trajectory.validate()
    assert trajectory.terminal_policy_credit_turn_index == 1
    assert trajectory.policy_mask == [0, 0, 1]
    assert trajectory.kl_mask == [0, 0, 1]


def test_system_invalid_turn_cannot_claim_ig_or_policy_credit() -> None:
    turn = TurnRecord(
        turn_index=0,
        turn_type=TurnType.SEARCH,
        search_index=0,
        model_text="<search>x</search>",
        search_action_span_valid=True,
        search_prefix_valid=True,
        ig_reward_eligible=True,
        policy_credit_eligible=True,
    )
    with pytest.raises(ValueError, match="ig_reward_eligible"):
        turn.validate(trajectory_system_valid=False)
