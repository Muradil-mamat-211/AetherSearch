import math
from pathlib import Path

from agentic_rl.config import load_config

from config_support import MICA_CONFIG, PAPER_MICA_CONFIG
from agentic_rl.selection.candidate_pool import (
    ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE,
    ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE,
    ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
    CandidatePool,
    PromptGroup,
)
from agentic_rl.selection.channel_scale import ChannelScaleState
from agentic_rl.selection.paper_ragen2 import (
    compute_ragen2_paper_sample_variance,
    select_ragen2_raw_variance_mass_top_p,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _select(
    pool: CandidatePool,
    *,
    ig_state: ChannelScaleState | None = None,
    outcome_state: ChannelScaleState | None = None,
    selection_mode: str = ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE,
):
    return pool.select(
        ig_state=ig_state or ChannelScaleState(),
        outcome_state=outcome_state or ChannelScaleState(),
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
        signal_mode=ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
        selection_mode=selection_mode,
    )


def _pool(variances, *, ig_scale=1.0, maximum=128):
    pool = CandidatePool(group_size=1, maximum_prompts=maximum)
    pool.add(
        PromptGroup(str(prompt_id), (index,), ig_scale * (index + 1), variance)
        for index, (prompt_id, variance) in enumerate(variances.items())
    )
    return pool


def test_paper_sample_variance_is_ddof_one() -> None:
    result = compute_ragen2_paper_sample_variance([0.0, 1.0, 0.0, 1.0])
    assert result == 1.0 / 3.0
    assert result != 0.25


def test_paper_formal_config_activates_only_the_new_selector() -> None:
    old_config = load_config(MICA_CONFIG)
    paper_config = load_config(PAPER_MICA_CONFIG)
    assert paper_config["selection"]["mode"] == (
        ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE
    )
    assert paper_config["selection"]["health_gate_active_for_selection"] is False
    assert paper_config["selection"]["scale_active_for_selection"] is False
    for section in (
        "mica",
        "advantage",
        "policy",
        "optimizer",
        "scheduler",
        "rollout",
        "learner",
        "checkpoint",
        "evaluation",
    ):
        assert paper_config[section] == old_config[section]


def test_paper_selector_accumulates_variance_not_standard_deviation() -> None:
    result = select_ragen2_raw_variance_mass_top_p(
        {"variance-nine": 9.0, "variance-one": 1.0},
        rho=0.9,
    )
    assert result.selected_ids == ("variance-nine",)
    assert result.selected_mass_ratio == 0.9


def test_paper_selector_is_independent_of_health_scale_and_median_state() -> None:
    variances = {f"p{index:02d}": float(64 - index) for index in range(64)}
    pool = _pool(variances)
    baseline = _select(pool)
    extreme = _select(
        pool,
        ig_state=ChannelScaleState(
            committed_scale=1.0e100,
            health_observations=(1.0e-100,) * 10,
            health_reference=1.0e100,
            valid_success_count=999,
        ),
        outcome_state=ChannelScaleState(
            committed_scale=1.0e-100,
            health_observations=(1.0e100,) * 10,
            health_reference=1.0e-100,
            valid_success_count=999,
        ),
    )
    assert baseline.selected_ids == extreme.selected_ids
    assert baseline.score_by_prompt == extreme.score_by_prompt == variances
    assert baseline.health_gate_selection_call_count == 0
    assert baseline.scale_selection_call_count == 0
    assert baseline.normalized_signal_selection_call_count == 0
    assert not baseline.outcome_stats.scale_observation_valid
    assert not baseline.outcome_stats.scale_update_allowed_after_success
    assert (
        extreme.outcome_stats.gate.reason
        == "selection_bypasses_health_and_scale"
    )
    assert (
        ChannelScaleState(committed_scale=7.0).committed_after_success(
            baseline.outcome_stats,
            ema_half_life=10.0,
            health_reference_valid_updates=10,
            allow_initialization=True,
        )
        == ChannelScaleState(committed_scale=7.0)
    )


def test_paper_selector_never_calls_channel_scale_inspection(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("ChannelScaleState.inspect_pool entered paper mode")

    monkeypatch.setattr(ChannelScaleState, "inspect_pool", forbidden)
    variances = {f"p{index:02d}": float(64 - index) for index in range(64)}
    decision = _select(_pool(variances))
    assert decision.selected_ids


def test_paper_selector_has_no_ig_or_mica_credit_leakage() -> None:
    variances = {f"p{index:02d}": float((index % 11) + 1) for index in range(64)}
    low_ig = _select(_pool(variances, ig_scale=1.0))
    high_ig = _select(_pool(variances, ig_scale=1.0e9))
    assert low_ig.selected_ids == high_ig.selected_ids
    assert low_ig.score_by_prompt == high_ig.score_by_prompt


def test_paper_selector_ties_use_prompt_id_order() -> None:
    result = select_ragen2_raw_variance_mass_top_p(
        {"p2": 1.0, "p0": 1.0, "p1": 1.0},
        rho=0.9,
    )
    assert result.ordered_positive_ids == ("p0", "p1", "p2")
    assert result.selected_ids == ("p0", "p1", "p2")


def test_paper_zero_signal_uses_existing_refill_skip_contract() -> None:
    pool = _pool({f"p{index:02d}": 0.0 for index in range(64)}, maximum=64)
    decision = _select(pool)
    assert decision.raw_top_p is not None
    assert decision.raw_top_p.selected_ids == ()
    assert decision.selected_ids == ()
    assert not decision.requires_refill
    assert decision.skip_update


def test_paper_mode_preserves_two_refill_orchestration() -> None:
    pool = _pool(
        {
            f"old-{index:03d}": 1.0 if index < 34 else 0.0
            for index in range(64)
        }
    )
    first = _select(pool)
    assert first.selected_count == 31
    assert first.requires_refill

    pool.add(
        PromptGroup(f"refill-one-{index:03d}", (index,), 0.0, 0.0)
        for index in range(32)
    )
    second = _select(pool)
    assert second.candidate_count == 96
    assert second.selected_count == 31
    assert second.requires_refill

    pool.add(
        PromptGroup(f"refill-two-{index:03d}", (index,), 0.0, 1.1)
        for index in range(32)
    )
    third = _select(pool)
    assert third.candidate_count == 128
    assert third.selected_count == 36
    assert not third.requires_refill
    assert not third.skip_update
    assert any(value.startswith("refill-two-") for value in third.selected_ids)


def test_old_answer_scaled_mode_is_unchanged_by_explicit_dispatch() -> None:
    variances = {f"p{index:02d}": float((index % 7) + 1) for index in range(64)}
    pool = _pool(variances)
    state = ChannelScaleState(committed_scale=2.5, health_reference=0.1)
    implicit = pool.select(
        ig_state=state,
        outcome_state=state,
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=1.0e-12,
        noise_floor_outcome=1.0e-12,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=False,
        signal_mode=ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
    )
    explicit = pool.select(
        ig_state=state,
        outcome_state=state,
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=1.0e-12,
        noise_floor_outcome=1.0e-12,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=32,
        maximum_selected_prompts=36,
        allow_provisional_scale=False,
        signal_mode=ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
        selection_mode=ANSWER_OUTCOME_ONLY_SCALED_TOP_P_MODE,
    )
    assert implicit.selected_ids == explicit.selected_ids
    assert implicit.score_by_prompt == explicit.score_by_prompt
    assert math.isclose(
        implicit.top_p.selected_mass_ratio,
        explicit.top_p.selected_mass_ratio,
    )
