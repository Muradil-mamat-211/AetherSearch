from __future__ import annotations

from types import SimpleNamespace

from agentic_rl.runtime.learner_batch import (
    PreparedTrajectory,
    build_synchronized_microbatch_rounds,
)


def _prepared(trajectory_id: str) -> PreparedTrajectory:
    record = SimpleNamespace(
        input_ids=(10, 11, 12),
        policy_mask=(False, True, True),
        turn_ids=(-1, 0, 0),
        prompt_global_id="prompt-0",
        trajectory_id=trajectory_id,
    )
    return PreparedTrajectory(
        record=record,
        advantage_by_turn={0: 1.0},
        normalized_ig_by_turn={0: 0.0},
        answer_turn_ids=(),
        expected_turn_ids=(0,),
    )


def test_empty_fsdp_ranks_receive_zero_weight_collective_fillers() -> None:
    real = _prepared("trajectory-0")
    rounds = build_synchronized_microbatch_rounds(
        ((real,), (), (), ()),
        micro_batch_size_per_rank=1,
        pad_token_id=0,
        snapshot_step=0,
        global_prompt_count=1,
        group_size=1,
        action_state_chunk_size=1,
        vocabulary_chunk_size=8,
        kl_coefficient=0.01,
    )
    assert len(rounds) == 1
    assert len(rounds[0]) == 4
    assert rounds[0][0]["trajectory_weights"] == [1.0]
    for payload in rounds[0][1:]:
        assert payload["trajectory_weights"] == [0.0]
        assert payload["trajectory_ids"] == ["trajectory-0"]
        assert payload["policy_mask"].sum().item() == 2


def test_collective_filler_requires_at_least_one_selected_trajectory() -> None:
    try:
        build_synchronized_microbatch_rounds(
            ((), (), (), ()),
            micro_batch_size_per_rank=1,
            pad_token_id=0,
            snapshot_step=0,
            global_prompt_count=1,
            group_size=1,
            action_state_chunk_size=1,
            vocabulary_chunk_size=8,
            kl_coefficient=0.01,
        )
    except ValueError as error:
        assert "selected trajectory" in str(error)
    else:
        raise AssertionError("An all-empty distributed batch must fail closed")
