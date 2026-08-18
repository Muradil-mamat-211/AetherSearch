import math

from agentic_rl.selection.candidate_pool import CandidatePool, PromptGroup
from agentic_rl.selection.channel_scale import ChannelScaleState
from agentic_rl.selection.top_p import stable_mass_top_p


def _equal_scores(count):
    return {f"p{index:03d}": 1.0 for index in range(count)}


def test_nucleus_count_boundaries_31_32_36_37() -> None:
    assert len(stable_mass_top_p(_equal_scores(34), rho=0.9).selected_ids) == 31
    assert len(stable_mass_top_p(_equal_scores(35), rho=0.9).selected_ids) == 32
    assert len(stable_mass_top_p(_equal_scores(40), rho=0.9).selected_ids) == 36
    assert len(stable_mass_top_p(_equal_scores(41), rho=0.9).selected_ids) == 37


def test_top36_capacity_records_actual_mass_below_nucleus_when_needed() -> None:
    pool = CandidatePool(group_size=16, maximum_prompts=96)
    pool.add(
        PromptGroup(f"p{index:03d}", tuple(range(16)), 1.0, 0.0)
        for index in range(41)
    )
    decision = pool.select(
        ig_state=ChannelScaleState(),
        outcome_state=ChannelScaleState(),
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=0.0,
        noise_floor_outcome=0.0,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=True,
    )
    assert decision.selected_count == 36
    assert decision.capacity_truncation_count == 1
    assert math.isclose(decision.top_p.selected_mass_ratio, 36.0 / 41.0)
    assert decision.top_p.selected_mass_ratio < 0.9


def test_two_refills_recompute_full_128_pool_instead_of_union_of_rounds() -> None:
    pool = CandidatePool(group_size=1, maximum_prompts=128)
    pool.add(
        PromptGroup(
            f"old-{index:03d}",
            (index,),
            1.0 if index < 34 else 0.0,
            0.0,
        )
        for index in range(64)
    )
    state = ChannelScaleState()
    first = pool.select(
        ig_state=state,
        outcome_state=state,
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=0.0,
        noise_floor_outcome=0.0,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=True,
    )
    assert first.selected_count == 31
    assert first.requires_refill
    pool.add(
        PromptGroup(f"refill-one-{index:03d}", (index,), 0.0, 0.0)
        for index in range(32)
    )
    second = pool.select(
        ig_state=state,
        outcome_state=state,
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=0.0,
        noise_floor_outcome=0.0,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=True,
    )
    assert second.candidate_count == 96
    assert second.selected_count == 31
    assert second.requires_refill
    pool.add(
        PromptGroup(f"refill-two-{index:03d}", (index,), 1.1, 0.0)
        for index in range(32)
    )
    third = pool.select(
        ig_state=state,
        outcome_state=state,
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=0.0,
        noise_floor_outcome=0.0,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=True,
    )
    assert third.candidate_count == 128
    assert third.selected_count == 36
    assert not third.requires_refill
    assert not third.skip_update
    assert any(prompt_id.startswith("refill-two-") for prompt_id in third.selected_ids)
    assert not set(first.selected_ids).issubset(third.selected_ids)
