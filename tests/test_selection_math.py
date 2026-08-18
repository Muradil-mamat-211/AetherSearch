import math

import pytest

from agentic_rl.selection.candidate_pool import CandidatePool, PromptGroup
from agentic_rl.selection.channel_scale import ChannelScaleState
from agentic_rl.selection.prompt_variance import (
    ig_prompt_variance,
    outcome_prompt_variance,
    sample_variance,
)
from agentic_rl.selection.top_p import stable_mass_top_p


def test_sample_variances_are_ddof_one() -> None:
    assert sample_variance([1.0, 2.0, 3.0]) == 1.0
    result = ig_prompt_variance(
        [
            {0: 1.0, 1: 10.0},
            {0: 3.0, 1: 14.0},
            {0: 5.0},
        ]
    )
    assert result.by_search_index[0] == 4.0
    assert result.by_search_index[1] == 8.0
    assert result.peer_count_by_search_index == {0: 3, 1: 2}
    assert math.isclose(result.aggregate, (3.0 / 5.0) * 4.0 + (2.0 / 5.0) * 8.0)
    assert outcome_prompt_variance([0.0, 0.5, 1.0]) == 0.25


def test_invalid_trajectories_do_not_enter_channel_variance() -> None:
    result = ig_prompt_variance(
        [{0: 1.0}, {0: 3.0}, {}],
        [{0: True}, {0: True}, {0: False}],
    )
    assert result.by_search_index[0] == 2.0
    assert outcome_prompt_variance(
        [0.0, 1.0, 1000.0],
        [True, True, False],
    ) == 0.5


def test_final_answer_invalidity_does_not_remove_prior_ig_peer() -> None:
    result = ig_prompt_variance(
        [{0: 1.0}, {0: 3.0}],
        [{0: True}, {0: True}],
    )
    assert result.peer_count_by_search_index == {0: 2}
    assert result.by_search_index[0] == 2.0
    assert outcome_prompt_variance([0.0, 1.0], [False, True]) == 0.0


def test_singleton_search_position_does_not_dilute_supported_position() -> None:
    result = ig_prompt_variance(
        [{0: 1.0, 1: 5.0}, {0: 3.0}],
    )
    assert result.peer_count_by_search_index == {0: 2, 1: 1}
    assert result.by_search_index[1] == 0.0
    assert result.natural_weight_by_search_index == {
        0: 1.0,
        1: 0.0,
    }
    assert math.isclose(result.aggregate, 2.0)


def test_multiple_singletons_do_not_dilute_supported_position() -> None:
    result = ig_prompt_variance(
        [
            {0: 0.0, 1: 9.0},
            {0: 2.0, 2: 8.0},
            {0: 1.0, 3: 7.0},
        ]
        + [{0: float(index % 2)} for index in range(13)]
    )
    assert result.peer_count_by_search_index == {0: 16, 1: 1, 2: 1, 3: 1}
    assert result.natural_weight_by_search_index == {
        0: 1.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }
    assert result.aggregate == result.by_search_index[0]


def test_singleton_positions_do_not_dilute_exact_unit_sample_variance() -> None:
    # Sixteen peers with these symmetric values have sample variance exactly 1.
    magnitude = math.sqrt(15.0 / 16.0)
    trajectories = [
        {0: (-magnitude if index < 8 else magnitude)}
        for index in range(16)
    ]
    trajectories[0].update({1: 5.0, 2: 7.0, 3: 9.0})
    result = ig_prompt_variance(trajectories)
    assert result.peer_count_by_search_index == {0: 16, 1: 1, 2: 1, 3: 1}
    assert math.isclose(result.by_search_index[0], 1.0)
    assert result.natural_weight_by_search_index == {
        0: 1.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }
    assert math.isclose(result.aggregate, 1.0)


def test_supported_positions_weight_only_by_supported_peer_counts() -> None:
    result = ig_prompt_variance(
        [
            {0: 0.0, 1: 0.0, 2: 4.0},
            {0: 2.0, 1: 4.0},
            {0: 1.0},
            {0: 3.0},
        ]
    )
    assert result.peer_count_by_search_index == {0: 4, 1: 2, 2: 1}
    assert result.natural_weight_by_search_index == {
        0: 4.0 / 6.0,
        1: 2.0 / 6.0,
        2: 0.0,
    }
    expected = (
        result.by_search_index[0] * 4.0 / 6.0
        + result.by_search_index[1] * 2.0 / 6.0
    )
    assert math.isclose(result.aggregate, expected)


def test_stable_top_p_minimal_prefix_and_tie_order() -> None:
    result = stable_mass_top_p(
        {"p3": 1.0, "p1": 4.0, "p2": 1.0},
        rho=0.8,
    )
    assert result.ordered_positive_ids == ("p1", "p2", "p3")
    assert result.selected_ids == ("p1", "p2")
    assert math.isclose(result.selected_mass_ratio, 5.0 / 6.0)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"p0": 0.0, "p1": 0.0}, tuple()),
        ({"p0": 2.0, "p1": 0.0}, ("p0",)),
        ({"p2": 1.0, "p1": 1.0, "p0": 1.0}, ("p0", "p1", "p2")),
    ],
)
def test_top_p_zero_single_positive_and_stable_ties(scores, expected) -> None:
    assert stable_mass_top_p(scores, rho=0.9).selected_ids == expected


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.0])
def test_top_p_fails_closed_on_invalid_scores(score) -> None:
    with pytest.raises(ValueError, match="Invalid Top-p score"):
        stable_mass_top_p({"p": score}, rho=0.9)


def test_candidate_pool_retains_whole_g16_groups() -> None:
    pool = CandidatePool(group_size=16, maximum_prompts=96)
    groups = [
        PromptGroup(
            prompt_global_id=f"p{index:02d}",
            trajectories=tuple(range(16)),
            ig_variance=float(64 - index),
            outcome_variance=float(index + 1),
        )
        for index in range(64)
    ]
    pool.add(groups)
    decision = pool.select(
        ig_state=ChannelScaleState(),
        outcome_state=ChannelScaleState(),
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=1.0e-12,
        noise_floor_outcome=1.0e-12,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=True,
    )
    assert decision.candidate_count == 64
    assert decision.selected_count <= 36
    assert all(len(group.trajectories) == 16 for group in pool.selected_groups(decision))


def test_candidate_pool_add_is_atomic_on_capacity_failure() -> None:
    pool = CandidatePool(group_size=1, maximum_prompts=2)
    pool.add([PromptGroup("p0", (0,), 1.0, 0.0)])
    with pytest.raises(ValueError, match="capacity"):
        pool.add(
            [
                PromptGroup("p1", (0,), 1.0, 0.0),
                PromptGroup("p2", (0,), 1.0, 0.0),
            ]
        )
    assert tuple(group.prompt_global_id for group in pool.groups()) == ("p0",)


def test_candidate_pool_add_is_atomic_on_group_size_failure() -> None:
    pool = CandidatePool(group_size=16, maximum_prompts=96)
    pool.add([PromptGroup("p0", tuple(range(16)), 1.0, 0.0)])
    with pytest.raises(ValueError, match="expected 16"):
        pool.add(
            [
                PromptGroup("p1", tuple(range(16)), 1.0, 0.0),
                PromptGroup("p2", tuple(range(15)), 1.0, 0.0),
            ]
        )
    assert tuple(group.prompt_global_id for group in pool.groups()) == ("p0",)
