from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_rl.advantage.a2tgpo import (
    TrajectoryCreditInput,
    compute_prompt_advantages,
    rebuild_search_advantages,
)
from agentic_rl.advantage.stop_continue import (
    NORMALIZED_OUTCOME_MODE,
    STOP_CONTINUE_CONSENSUS_MODE,
    StopContinueRewardTriple,
    compute_stop_continue_advantages,
)
from agentic_rl.exact_ig.target_schema import select_canonical_answer
from agentic_rl.outcome.token_f1 import max_alias_token_f1
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    score_stop_answer_completion,
    score_trajectory_outcome,
)


POLICY_VERSION = 7


def _reward(
    trajectory_id: str,
    continue_reward: float,
    stop_reward_1: float,
    stop_reward_2: float,
    *,
    search_index: int = 0,
) -> StopContinueRewardTriple:
    return StopContinueRewardTriple(
        prompt_global_id="prompt",
        trajectory_id=trajectory_id,
        search_index=search_index,
        continue_reward=continue_reward,
        stop_reward_1=stop_reward_1,
        stop_reward_2=stop_reward_2,
        continue_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
        stop_scorer_version_1=PRODUCTION_TASK_SCORER_VERSION,
        stop_scorer_version_2=PRODUCTION_TASK_SCORER_VERSION,
        candidate_rollout_policy_version=POLICY_VERSION,
        exact_ig_policy_version=POLICY_VERSION,
        stop_branch_policy_version=POLICY_VERSION,
        old_logprob_policy_version=POLICY_VERSION,
        prefix_provenance_valid=True,
        context_truncated=False,
        completion_count=2,
        detached=True,
    )


def _compute(
    rewards: list[StopContinueRewardTriple],
    *,
    z_outcome: float = -0.25,
):
    return compute_stop_continue_advantages(
        rewards,
        normalized_outcome_by_trajectory={
            (value.prompt_global_id, value.trajectory_id): z_outcome
            for value in rewards
        },
        expected_state_keys=[value.state_key for value in rewards],
        group_size=16,
        reward_epsilon=1.0e-6,
        scale_epsilon=1.0e-8,
        pooled_scale_ddof=0,
        expected_policy_version=POLICY_VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )


def test_clear_continue_uses_positive_stop_continue_advantage() -> None:
    result = _compute([_reward("t0", 1.0, 0.0, 0.6)])
    value = result.by_state[("prompt", "t0", 0)]
    assert value.sc_clear
    assert value.clear_positive
    assert not value.clear_negative
    assert value.delta_sc == pytest.approx(0.7)
    assert value.advantage_sc > 0.0
    assert value.task_advantage == value.advantage_sc


def test_clear_stop_uses_negative_stop_continue_advantage() -> None:
    result = _compute([_reward("t0", 0.0, 0.6, 1.0)])
    value = result.by_state[("prompt", "t0", 0)]
    assert value.sc_clear
    assert value.clear_negative
    assert value.delta_sc == pytest.approx(-0.8)
    assert value.advantage_sc < 0.0
    assert value.task_advantage == value.advantage_sc


@pytest.mark.parametrize(
    ("continue_reward", "stop_reward_1", "stop_reward_2"),
    (
        (0.6, 0.0, 1.0),
        (0.6, 0.0, 0.6),
        (0.6, 0.6, 0.6),
    ),
)
def test_non_consensus_cases_add_no_search_task_advantage(
    continue_reward: float,
    stop_reward_1: float,
    stop_reward_2: float,
) -> None:
    result = _compute(
        [_reward("t0", continue_reward, stop_reward_1, stop_reward_2)],
        z_outcome=-0.375,
    )
    value = result.by_state[("prompt", "t0", 0)]
    assert not value.sc_clear
    assert value.task_advantage == 0.0
    assert math.isfinite(value.advantage_sc)
    if continue_reward == stop_reward_1 == stop_reward_2:
        assert value.delta_sc == 0.0
        assert value.pooled_scale == 0.0
        assert value.advantage_sc == 0.0


def test_reward_epsilon_boundary_is_strict() -> None:
    epsilon = 1.0e-6
    exact = _compute([_reward("t0", 0.6 + epsilon, 0.0, 0.6)])
    above = _compute(
        [_reward("t0", 0.6 + epsilon + 1.0e-12, 0.0, 0.6)]
    )
    assert not exact.by_state[("prompt", "t0", 0)].sc_clear
    assert above.by_state[("prompt", "t0", 0)].sc_clear

    exact_negative = _compute(
        [_reward("t0", 0.6 - epsilon, 0.6, 1.0)]
    )
    below_negative = _compute(
        [_reward("t0", 0.6 - epsilon - 1.0e-12, 0.6, 1.0)]
    )
    assert not exact_negative.by_state[("prompt", "t0", 0)].sc_clear
    assert below_negative.by_state[("prompt", "t0", 0)].sc_clear


def test_pooled_scale_flattens_three_rewards_per_peer_with_ddof_zero() -> None:
    rewards = [
        _reward("t0", 1.0, 0.0, 0.6),
        _reward("t1", 0.2, 0.4, 0.8),
    ]
    result = _compute(rewards)
    expected = np.std([1.0, 0.0, 0.6, 0.2, 0.4, 0.8], ddof=0)
    assert result.pooled_scale_by_prompt_search[("prompt", 0)] == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="ddof=0"):
        compute_stop_continue_advantages(
            rewards,
            normalized_outcome_by_trajectory={
                ("prompt", "t0"): 0.0,
                ("prompt", "t1"): 0.0,
            },
            expected_state_keys=[value.state_key for value in rewards],
            group_size=16,
            pooled_scale_ddof=1,
        )


def test_advantage_clips_at_sqrt_group_size_minus_one() -> None:
    positive = [_reward(f"t{index}", 0.0, 0.0, 0.0) for index in range(15)]
    positive.append(_reward("t15", 1.0, 0.0, 0.0))
    positive_value = _compute(positive).by_state[("prompt", "t15", 0)]
    assert positive_value.raw_advantage_sc > math.sqrt(15)
    assert positive_value.advantage_sc == pytest.approx(math.sqrt(15))

    negative = [_reward(f"t{index}", 0.0, 0.0, 0.0) for index in range(15)]
    negative.append(_reward("t15", 0.0, 1.0, 1.0))
    negative_value = _compute(negative).by_state[("prompt", "t15", 0)]
    assert negative_value.raw_advantage_sc < -math.sqrt(15)
    assert negative_value.advantage_sc == pytest.approx(-math.sqrt(15))


def _credit(immediate_ig: float, outcome: float, format_indicator: int):
    return TrajectoryCreditInput(
        immediate_ig={0: immediate_ig},
        search_turn_indices=(0,),
        ig_reward_eligible={0: True},
        policy_credit_eligible={0: True},
        outcome=outcome,
        outcome_reward_eligible=True,
        format_indicator=format_indicator,
        answer_policy_credit_eligible=True,
    )


def _direct_answer_credit(outcome: float, format_indicator: int):
    return TrajectoryCreditInput(
        immediate_ig={},
        search_turn_indices=(),
        ig_reward_eligible={},
        policy_credit_eligible={},
        outcome=outcome,
        outcome_reward_eligible=True,
        format_indicator=format_indicator,
        answer_policy_credit_eligible=True,
    )


def _probe_mapping(reward: StopContinueRewardTriple) -> dict[str, object]:
    return {
        field: getattr(reward, field)
        for field in reward.__dataclass_fields__
    }


def test_feature_flag_and_answer_advantage_regression() -> None:
    base = compute_prompt_advantages(
        [_credit(1.0, 1.0, 1), _credit(-1.0, 0.0, 0)]
    )
    records = [
        SimpleNamespace(
            prompt_global_id="prompt",
            trajectory_id=f"t{index}",
            metadata={
                "stop_continue_probes": {
                    0: _probe_mapping(
                        _reward(
                            f"t{index}",
                            1.0 if index == 0 else 0.0,
                            0.0 if index == 0 else 0.6,
                            0.6 if index == 0 else 1.0,
                        )
                    )
                }
            },
        )
        for index in range(2)
    ]
    old_answers = tuple(value.answer_advantage for value in base.trajectories)
    legacy, _ = rebuild_search_advantages(
        records,
        base,
        search_task_mode=NORMALIZED_OUTCOME_MODE,
        group_size=16,
    )
    for value in legacy.trajectories:
        assert value.search_advantage[0] == pytest.approx(
            0.3 * value.future_ig_rescaled[0] + value.normalized_outcome
        )

    rebuilt, _ = rebuild_search_advantages(
        records,
        base,
        search_task_mode=STOP_CONTINUE_CONSENSUS_MODE,
        group_size=16,
        expected_policy_version=POLICY_VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )
    assert tuple(
        value.answer_advantage for value in rebuilt.trajectories
    ) == old_answers
    for value in rebuilt.trajectories:
        sc = value.stop_continue_by_search_index[0]
        assert value.search_advantage[0] == pytest.approx(
            0.3 * value.future_ig_rescaled[0]
            + (sc.advantage_sc if sc.sc_clear else 0.0)
        )
        assert value.search_advantage_old_shadow[0] == pytest.approx(
            0.3 * value.future_ig_rescaled[0] + value.normalized_outcome
        )


def test_mixed_group_direct_answer_needs_no_stop_probe_and_keeps_answer_credit() -> None:
    base = compute_prompt_advantages(
        [_credit(1.0, 0.0, 1), _direct_answer_credit(1.0, 1)]
    )
    records = [
        SimpleNamespace(
            prompt_global_id="prompt",
            trajectory_id="searched",
            metadata={
                "stop_continue_probes": {
                    0: _probe_mapping(_reward("searched", 0.0, 0.6, 1.0))
                }
            },
        ),
        SimpleNamespace(
            prompt_global_id="prompt",
            trajectory_id="direct-answer",
            metadata={},
        ),
    ]
    old_answers = tuple(value.answer_advantage for value in base.trajectories)

    rebuilt, metrics = rebuild_search_advantages(
        records,
        base,
        search_task_mode=STOP_CONTINUE_CONSENSUS_MODE,
        group_size=16,
        expected_policy_version=POLICY_VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )

    direct = rebuilt.trajectories[1]
    assert direct.search_advantage == {}
    assert direct.search_task_advantage == {}
    assert direct.stop_continue_by_search_index == {}
    assert direct.answer_advantage == old_answers[1]
    assert tuple(value.answer_advantage for value in rebuilt.trajectories) == old_answers
    assert metrics["sc/state_count"] == 1


def test_all_direct_answer_group_uses_only_unchanged_answer_advantage() -> None:
    base = compute_prompt_advantages(
        [_direct_answer_credit(1.0, 1), _direct_answer_credit(0.0, 0)]
    )
    records = [
        SimpleNamespace(
            prompt_global_id="prompt",
            trajectory_id=f"direct-{index}",
            metadata={},
        )
        for index in range(2)
    ]
    old_answers = tuple(value.answer_advantage for value in base.trajectories)

    rebuilt, metrics = rebuild_search_advantages(
        records,
        base,
        search_task_mode=STOP_CONTINUE_CONSENSUS_MODE,
        group_size=16,
        expected_policy_version=POLICY_VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )

    assert tuple(value.answer_advantage for value in rebuilt.trajectories) == old_answers
    assert all(value.search_advantage == {} for value in rebuilt.trajectories)
    assert metrics == {
        "sc/state_count": 0,
        "sc/clear_count": 0,
        "sc/clear_rate": 0.0,
        "sc/fallback_count": 0,
        "sc/fallback_rate": 0.0,
        "sc/fallback_z_o_to_search_count": 0,
    }


def test_real_search_still_fails_closed_when_stop_probes_are_absent() -> None:
    base = compute_prompt_advantages([_credit(1.0, 1.0, 1)])
    records = [
        SimpleNamespace(
            prompt_global_id="prompt",
            trajectory_id="searched",
            metadata={},
        )
    ]
    with pytest.raises(ValueError, match="selected trajectory has no Stop probes"):
        rebuild_search_advantages(
            records,
            base,
            search_task_mode=STOP_CONTINUE_CONSENSUS_MODE,
            group_size=16,
            expected_policy_version=POLICY_VERSION,
            expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
        )


def test_exact_ig_canonical_first_but_task_reward_keeps_alias_max() -> None:
    aliases = ["New York City", "NYC"]
    assert select_canonical_answer(aliases) == "New York City"
    expected = max(
        max_alias_token_f1("NYC", [aliases[0]]),
        max_alias_token_f1("NYC", [aliases[1]]),
    )
    outcome = score_trajectory_outcome(
        ["<answer>NYC</answer>"],
        aliases,
    )
    stop = score_stop_answer_completion("NYC</answer>", aliases)
    assert outcome.task_outcome == pytest.approx(expected)
    assert stop.task_outcome == pytest.approx(expected)
    assert outcome.task_outcome == 1.0


def test_missing_or_mismatched_stop_contract_fails_closed() -> None:
    invalid = _reward("t0", 1.0, 0.0, 0.6)
    invalid = StopContinueRewardTriple(
        **{
            **_probe_mapping(invalid),
            "stop_branch_policy_version": POLICY_VERSION + 1,
        }
    )
    with pytest.raises(ValueError, match="versions"):
        _compute([invalid])
