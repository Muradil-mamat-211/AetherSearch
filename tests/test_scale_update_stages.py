import math

from agentic_rl.selection.channel_scale import ChannelScaleState


def _stats(
    state: ChannelScaleState,
    scale: float,
    *,
    positive_count: int = 8,
    allow_provisional_scale: bool,
):
    return state.inspect_pool(
        {
            f"p{index}": (
                scale * float(index + 1) if index < positive_count else 0.0
            )
            for index in range(8)
        },
        noise_floor=0.0,
        minimum_positive_prompts=4,
        health_threshold_ratio=0.1,
        allow_provisional_scale=allow_provisional_scale,
    )


def _commit(
    state: ChannelScaleState,
    stats,
    *,
    allow_initialization: bool,
) -> ChannelScaleState:
    return state.committed_after_success(
        stats,
        ema_half_life=10,
        health_reference_valid_updates=10,
        allow_initialization=allow_initialization,
    )


def test_update1_initializes_scale_only_on_success_commit() -> None:
    state = ChannelScaleState()
    stats = _stats(state, 1.0, allow_provisional_scale=True)
    assert state.committed_scale is None
    assert stats.scale_used == stats.positive_median
    committed = _commit(state, stats, allow_initialization=True)
    assert committed.committed_scale == stats.positive_median
    assert committed.valid_success_count == 1


def test_update1_inactive_bootstrap_channel_still_commits_existing_median() -> None:
    state = ChannelScaleState()
    stats = _stats(
        state,
        2.0,
        positive_count=1,
        allow_provisional_scale=True,
    )
    assert not stats.gate.active
    assert stats.positive_median == 2.0
    assert stats.scale_update_allowed_after_success
    committed = _commit(state, stats, allow_initialization=True)
    assert committed.committed_scale == 2.0
    assert committed.health_observations == (stats.mean_excess,)


def test_update2_selection_uses_lagged_scale_then_log_ema_commits() -> None:
    state0 = ChannelScaleState()
    stats1 = _stats(state0, 1.0, allow_provisional_scale=True)
    state1 = _commit(state0, stats1, allow_initialization=True)
    stats2 = _stats(state1, 100.0, allow_provisional_scale=False)
    assert stats2.scale_used == state1.committed_scale
    assert stats2.scale_used != stats2.positive_median
    state2 = _commit(state1, stats2, allow_initialization=False)
    eta = ChannelScaleState.ema_eta(10)
    expected = math.exp(
        (1.0 - eta) * math.log(state1.committed_scale)
        + eta * math.log(stats2.positive_median)
    )
    assert math.isclose(state2.committed_scale, expected)


def test_update2_bootstrap_inactive_channel_still_updates_ema() -> None:
    state0 = ChannelScaleState()
    state1 = _commit(
        state0,
        _stats(state0, 1.0, allow_provisional_scale=True),
        allow_initialization=True,
    )
    stats2 = _stats(
        state1,
        100.0,
        positive_count=1,
        allow_provisional_scale=False,
    )
    assert stats2.gate.mode == "bootstrap"
    assert not stats2.gate.active
    assert stats2.scale_update_allowed_after_success
    state2 = _commit(state1, stats2, allow_initialization=False)
    assert state2.committed_scale != state1.committed_scale


def test_update11_uses_first_ten_valid_success_observations_for_health() -> None:
    state = ChannelScaleState()
    means = []
    for update_index in range(10):
        stats = _stats(
            state,
            float(update_index + 1),
            allow_provisional_scale=update_index == 0,
        )
        means.append(stats.mean_excess)
        state = _commit(
            state,
            stats,
            allow_initialization=update_index == 0,
        )
    assert state.valid_success_count == 10
    assert state.health_reference is not None
    expected = (means[4] + means[5]) / 2.0
    assert math.isclose(state.health_reference, expected)
    update11 = _stats(state, 1.0, allow_provisional_scale=False)
    assert update11.gate.mode == "health"
    assert update11.gate.health_ratio is not None


def test_updates_1_through_10_select_with_bootstrap_then_update11_uses_health() -> None:
    state = ChannelScaleState()
    for update_number in range(1, 10):
        stats = _stats(
            state,
            float(update_number),
            allow_provisional_scale=update_number == 1,
        )
        assert stats.gate.mode == "bootstrap"
        assert state.health_reference is None
        state = _commit(
            state,
            stats,
            allow_initialization=update_number == 1,
        )
    update10 = _stats(state, 10.0, allow_provisional_scale=False)
    assert state.valid_success_count == 9
    assert state.health_reference is None
    assert update10.gate.mode == "bootstrap"
    state = _commit(state, update10, allow_initialization=False)
    assert state.valid_success_count == 10
    assert state.health_reference is not None
    update11 = _stats(state, 11.0, allow_provisional_scale=False)
    assert update11.gate.mode == "health"


def test_channels_switch_to_health_gate_independently() -> None:
    ig_state = ChannelScaleState()
    outcome_state = ChannelScaleState()
    for update_number in range(1, 11):
        ig_stats = _stats(
            ig_state,
            float(update_number),
            allow_provisional_scale=update_number == 1,
        )
        ig_state = _commit(
            ig_state,
            ig_stats,
            allow_initialization=update_number == 1,
        )
        if update_number <= 8:
            outcome_stats = _stats(
                outcome_state,
                float(update_number),
                allow_provisional_scale=update_number == 1,
            )
            outcome_state = _commit(
                outcome_state,
                outcome_stats,
                allow_initialization=update_number == 1,
            )
    assert ig_state.valid_success_count == 10
    assert outcome_state.valid_success_count == 8
    assert _stats(
        ig_state, 11.0, allow_provisional_scale=False
    ).gate.mode == "health"
    assert _stats(
        outcome_state, 9.0, allow_provisional_scale=False
    ).gate.mode == "bootstrap"


def test_update11_low_health_channel_freezes_ema_but_records_observation() -> None:
    state = ChannelScaleState()
    for update_index in range(10):
        stats = _stats(
            state,
            float(update_index + 1),
            allow_provisional_scale=update_index == 0,
        )
        state = _commit(
            state,
            stats,
            allow_initialization=update_index == 0,
        )
    previous_scale = state.committed_scale
    low_health = _stats(state, 0.001, allow_provisional_scale=False)
    assert low_health.gate.mode == "health"
    assert not low_health.gate.active
    assert not low_health.scale_update_allowed_after_success
    committed = _commit(state, low_health, allow_initialization=False)
    assert committed.committed_scale == previous_scale
    assert committed.valid_success_count == 11
    assert committed.health_observations[-1] == low_health.mean_excess


def test_scale_cannot_late_initialize_after_update1() -> None:
    state = ChannelScaleState()
    stats = _stats(
        state,
        1.0,
        allow_provisional_scale=False,
    )
    assert not stats.gate.active
    assert stats.gate.reason == "scale_unavailable_after_update_1"
    committed = _commit(state, stats, allow_initialization=False)
    assert committed.committed_scale is None
