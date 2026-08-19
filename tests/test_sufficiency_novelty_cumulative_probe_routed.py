from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentic_rl.advantage.a2tgpo import (
    A2TGPOPromptResult,
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
    TrajectoryAdvantage,
    rebuild_search_advantages,
)
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    SUFFICIENCY_EXACT_SCORER_VERSION,
)
from agentic_rl.rollout.trajectory_schema import (
    TokenSource,
    TrajectoryRecord,
    TurnRecord,
    TurnType,
)
from agentic_rl.runtime.learner_batch import prepare_selected_trajectories
from agentic_rl.runtime.verl_runtime_adapter import VerlAttemptRuntimeAdapter

from config_support import PILOT_CONFIG
from agentic_rl.config import load_config
from pathlib import Path


VERSION = 11
ROOT = Path(__file__).resolve().parents[1]


def _probe(
    *,
    stage: str,
    sufficient: bool,
    reward: float,
) -> dict[str, object]:
    raw = {
        "raw_answer_text": "answer</answer>",
        "parser_success": True,
        "no_answer": False,
        "output_truncated": False,
        "alias_aware_exact": sufficient,
        "raw_task_reward": reward,
        "completion_count": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "n": 1,
        "max_tokens": 500,
        "stop": ["</answer>"],
        "detached": True,
        "prefix_provenance_valid": True,
        "scorer_version": SUFFICIENCY_EXACT_SCORER_VERSION,
        "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
        "candidate_rollout_policy_version": VERSION,
        "exact_ig_policy_version": VERSION,
        "probe_policy_version": VERSION,
        "old_logprob_policy_version": VERSION,
    }
    raw[
        "sufficient_before_search"
        if stage == "pre"
        else "sufficient_after_search"
    ] = sufficient
    return raw


def _fixture(
    values: list[float],
    *,
    s_before: list[bool] | None = None,
    s_after: list[bool] | None = None,
    no_new: list[bool] | None = None,
    pre_rewards: list[float] | None = None,
    post_rewards: list[float] | None = None,
    z_outcome: float = 0.0,
) -> tuple[SimpleNamespace, A2TGPOPromptResult]:
    count = len(values)
    s_before = s_before or [False] * count
    s_after = s_after or [False] * count
    no_new = no_new or [False] * count
    pre_rewards = pre_rewards or [0.0] * count
    post_rewards = post_rewards or [0.0] * count
    turns = [
        TurnRecord(
            turn_index=index,
            turn_type=TurnType.SEARCH,
            model_text=f"<search>q{index}</search>",
            search_index=index,
            search_action_span_valid=True,
            search_prefix_valid=True,
            ig_reward_eligible=True,
            policy_credit_eligible=True,
            no_new_observation=no_new[index],
            current_passage_keys=(f"p:{index}",),
            new_passage_keys=(() if no_new[index] else (f"p:{index}",)),
        )
        for index in range(count)
    ]
    probes: dict[int, dict[str, object]] = {}
    for index in range(count):
        stages: dict[str, object] = {
            "pre": _probe(
                stage="pre",
                sufficient=s_before[index],
                reward=pre_rewards[index],
            )
        }
        if not s_before[index]:
            stages["post"] = _probe(
                stage="post",
                sufficient=s_after[index],
                reward=post_rewards[index],
            )
        probes[index] = stages
    record = SimpleNamespace(
        prompt_global_id="p0",
        trajectory_id="t0",
        turns=turns,
        immediate_ig={index: value for index, value in enumerate(values)},
        metadata={"routed_answer_probes": probes},
    )
    advantage = TrajectoryAdvantage(
        normalized_ig={index: value for index, value in enumerate(values)},
        future_ig_sum={},
        accumulated_ig_count={},
        future_ig_rescaled={},
        normalized_outcome=z_outcome,
        centered_format_indicator=0.25,
        search_advantage={index: value for index, value in enumerate(values)},
        answer_advantage=z_outcome + 0.25,
        search_policy_credit_eligible={index: True for index in range(count)},
        answer_policy_credit_eligible=True,
    )
    result = A2TGPOPromptResult(
        trajectories=(advantage,),
        ig_mean_by_search_index={},
        ig_std_by_search_index={},
        outcome_mean=0.0,
        outcome_std=1.0,
        format_mean=0.0,
    )
    return record, result


def _rebuild(record: object, result: A2TGPOPromptResult) -> TrajectoryAdvantage:
    rebuilt, _ = rebuild_search_advantages(
        [record],
        result,
        search_task_mode=(
            SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
        ),
        group_size=16,
        lambda_ig=None,
        expected_policy_version=VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
        probe_epsilon=1.0e-6,
    )
    return rebuilt.trajectories[0]


def test_all_normal_accumulates_current_and_future_effective_ig() -> None:
    record, result = _fixture([1.0, 2.0, 3.0])
    actual = _rebuild(record, result)
    assert actual.search_advantage[0] == pytest.approx(6.0 / math.sqrt(3.0))
    assert actual.search_advantage[1] == pytest.approx(5.0 / math.sqrt(2.0))
    assert actual.search_advantage[2] == 3.0
    assert actual.effective_cumulative_ig_count == {0: 3, 1: 2, 2: 1}


def test_first_s_after_includes_that_search_then_stops_future_credit() -> None:
    record, result = _fixture([1.0, 2.0, 100.0], s_after=[False, True, False])
    actual = _rebuild(record, result)
    assert actual.search_advantage[0] == pytest.approx(3.0 / math.sqrt(2.0))
    assert actual.effective_cumulative_ig_count[0] == 2


def test_middle_n_masks_own_ig_without_stopping_later_propagation() -> None:
    record, result = _fixture([1.0, 100.0, 3.0], no_new=[False, True, False])
    actual = _rebuild(record, result)
    assert actual.search_advantage[0] == pytest.approx(4.0 / math.sqrt(2.0))
    assert actual.search_advantage[1] == -1.0
    assert actual.search_advantage[2] == 3.0


def test_s_before_and_s_plus_n_take_one_negative_branch_only() -> None:
    record, result = _fixture(
        [5.0, 7.0],
        s_before=[True, True],
        no_new=[False, True],
    )
    actual = _rebuild(record, result)
    assert actual.search_advantage == {0: -1.0, 1: -1.0}
    assert not actual.effective_cumulative_ig
    assert not actual.routed_outcome


@pytest.mark.parametrize(
    ("delta", "z_outcome", "expected_route"),
    [
        (0.5, 2.0, 2.0),
        (0.5, -2.0, 0.0),
        (-0.5, -2.0, -2.0),
        (-0.5, 2.0, 0.0),
        (1.0e-6, 2.0, 0.0),
        (-1.0e-6, -2.0, 0.0),
    ],
)
def test_probe_delta_routes_but_is_not_added_directly(
    delta: float,
    z_outcome: float,
    expected_route: float,
) -> None:
    pre_reward = 0.0 if abs(delta) == 1.0e-6 else 0.2
    record, result = _fixture(
        [1.25],
        pre_rewards=[pre_reward],
        post_rewards=[pre_reward + delta],
        z_outcome=z_outcome,
    )
    actual = _rebuild(record, result)
    assert actual.routed_outcome[0] == expected_route
    assert actual.search_advantage[0] == pytest.approx(1.25 + expected_route)
    assert not math.isclose(
        actual.search_advantage[0],
        1.25 + expected_route + delta,
        rel_tol=0.0,
        abs_tol=0.0,
    )


def test_answer_is_byte_for_byte_unchanged_and_a_sc_is_absent() -> None:
    record, result = _fixture([1.0], z_outcome=0.75)
    before = result.trajectories[0].answer_advantage
    actual = _rebuild(record, result)
    assert actual.answer_advantage == before
    assert actual.stop_continue_by_search_index == {}
    assert actual.search_task_advantage == {}


def test_non_normal_search_never_receives_outcome() -> None:
    record, result = _fixture(
        [1.0, 2.0],
        s_before=[True, False],
        no_new=[False, True],
        z_outcome=9.0,
    )
    actual = _rebuild(record, result)
    assert actual.search_advantage == {0: -1.0, 1: -1.0}
    assert actual.routed_outcome == {}


def _budget_exhausted_fixture() -> tuple[TrajectoryRecord, A2TGPOPromptResult]:
    record = TrajectoryRecord(
        prompt_global_id="budget-prompt",
        trajectory_id="budget-trajectory",
        input_ids=[10, 11, 20, 21, 30, 31, 22, 23],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.ENVIRONMENT,
            TokenSource.MODEL,
            TokenSource.MODEL,
        ],
        turn_ids=[-1, -1, 0, 0, -1, -1, 1, 1],
        turns=[
            TurnRecord(
                turn_index=0,
                turn_type=TurnType.SEARCH,
                model_text="<search>valid</search>",
                search_index=0,
                information_text="<information>new</information>",
                search_action_span_valid=True,
                search_prefix_valid=True,
                ig_reward_eligible=True,
                policy_credit_eligible=True,
                no_new_observation=False,
                current_passage_keys=("id:new",),
                new_passage_keys=("id:new",),
            ),
            TurnRecord(
                turn_index=1,
                turn_type=TurnType.SEARCH,
                model_text="<search>over-budget</search>",
                search_index=1,
                information_text=None,
                search_action_span_valid=True,
                search_prefix_valid=False,
                ig_reward_eligible=False,
                policy_credit_eligible=True,
                no_new_observation=True,
            ),
        ],
        search_prefix_end_positions=[2, 6],
        search_prefix_before_search_end_positions={0: 2, 1: 6},
        immediate_ig={0: 1.0},
        metadata={
            "termination_reason": "maximum_search_turns_reached",
            "routed_answer_probes": {
                0: {
                    "pre": _probe(stage="pre", sufficient=False, reward=0.0),
                    "post": _probe(stage="post", sufficient=False, reward=0.0),
                },
                1: {
                    "pre": _probe(stage="pre", sufficient=False, reward=0.0),
                },
            },
        },
    )
    record.validate()
    advantage = TrajectoryAdvantage(
        normalized_ig={0: 1.0},
        normalized_outcome=4.0,
        centered_format_indicator=0.0,
        future_ig_sum={},
        accumulated_ig_count={},
        future_ig_rescaled={},
        search_advantage={0: 1.0, 1: 0.0},
        answer_advantage=None,
        search_policy_credit_eligible={0: True, 1: True},
        answer_policy_credit_eligible=False,
    )
    return record, A2TGPOPromptResult(
        trajectories=(advantage,),
        ig_mean_by_search_index={},
        ig_std_by_search_index={},
        outcome_mean=0.0,
        outcome_std=1.0,
        format_mean=0.0,
    )


def test_budget_exhausted_terminal_search_is_n_only_without_post_credit() -> None:
    record, result = _budget_exhausted_fixture()
    rebuilt, metrics = rebuild_search_advantages(
        [record],
        result,
        search_task_mode=(
            SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
        ),
        group_size=16,
        lambda_ig=None,
        expected_policy_version=VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
        probe_epsilon=1.0e-6,
    )
    actual = rebuilt.trajectories[0]
    assert actual.search_advantage == {0: 1.0, 1: -1.0}
    assert 1 not in actual.sufficient_after_search
    assert 1 not in actual.probe_reward_delta
    assert 1 not in actual.routed_outcome
    assert 1 not in actual.effective_cumulative_ig
    assert metrics["search/budget_exhausted_count"] == 1
    assert metrics["search/budget_exhausted_post_probe_count"] == 0
    assert metrics["search/budget_exhausted_o_route_nonzero_count"] == 0
    assert metrics["search/budget_exhausted_ig_entry_count"] == 0
    assert metrics["search/budget_exhausted_A_search_not_minus_one_count"] == 0
    assert metrics["search/normal_search_missing_post_prefix_count"] == 0


def test_budget_exhausted_terminal_search_rejects_post_probe() -> None:
    record, result = _budget_exhausted_fixture()
    record.metadata["routed_answer_probes"][1]["post"] = _probe(
        stage="post",
        sufficient=False,
        reward=1.0,
    )
    with pytest.raises(ValueError, match="cannot carry a post Probe"):
        rebuild_search_advantages(
            [record],
            result,
            search_task_mode=(
                SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
            ),
            group_size=16,
            lambda_ig=None,
            expected_policy_version=VERSION,
            expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
        )


def test_advantage_recomputes_probe_bool_and_rejects_tampering() -> None:
    record, result = _fixture([1.0])
    record.metadata["routed_answer_probes"][0]["pre"][
        "sufficient_before_search"
    ] = True
    with pytest.raises(ValueError, match="precomputed sufficiency"):
        _rebuild(record, result)


def test_legacy_mode_object_remains_independent() -> None:
    record, result = _fixture([1.0])
    legacy = replace(
        result.trajectories[0],
        search_task_mode="sufficiency_novelty_local_ig",
    )
    assert legacy.search_task_mode == "sufficiency_novelty_local_ig"
    assert (
        SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
        != legacy.search_task_mode
    )


def test_new_formula_reaches_turn_payload_and_runtime_revalidation() -> None:
    records = []
    for trajectory_index, (ig_value, outcome) in enumerate(
        ((-1.0, 0.0), (1.0, 1.0))
    ):
        search = TurnRecord(
            turn_index=0,
            turn_type=TurnType.SEARCH,
            model_text="<search>q</search>",
            search_index=0,
            search_action_span_valid=True,
            search_prefix_valid=True,
            ig_reward_eligible=True,
            policy_credit_eligible=True,
            no_new_observation=False,
            current_passage_keys=(f"p:{trajectory_index}",),
            new_passage_keys=(f"p:{trajectory_index}",),
        )
        answer = TurnRecord(
            turn_index=1,
            turn_type=TurnType.ANSWER,
            model_text="<answer>x</answer>",
            policy_credit_eligible=True,
        )
        records.append(
            SimpleNamespace(
                prompt_global_id="prompt",
                trajectory_id=f"trajectory-{trajectory_index}",
                turns=(search, answer),
                immediate_ig={0: ig_value},
                ig_reward_eligibility_by_search_index={0: True},
                policy_credit_eligibility_by_search_index={0: True},
                task_outcome=outcome,
                outcome_reward_eligible=True,
                answer_format_indicator=1,
                terminal_policy_credit_turn_index=1,
                trajectory_system_valid=True,
                metadata={
                    "routed_answer_probes": {
                        0: {
                            "pre": _probe(
                                stage="pre", sufficient=False, reward=0.0
                            ),
                            "post": _probe(
                                stage="post", sufficient=False, reward=1.0
                            ),
                        }
                    }
                },
            )
        )
    group = SimpleNamespace(trajectories=tuple(records))
    config = load_config(PILOT_CONFIG)
    prepared = prepare_selected_trajectories(
        [group],
        expected_group_size=2,
        advantage_config=dict(config["advantage"]),
        expected_policy_version=VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )
    for item in prepared[0]:
        assert item.advantage_by_turn[0] == item.advantage.search_advantage[0]
        assert item.advantage_by_turn[1] == item.advantage.answer_advantage
    adapter = VerlAttemptRuntimeAdapter(config)
    adapter._last_snapshot_step = VERSION
    adapter._prepared_groups = prepared
    adapter._attempt_context = {}
    adapter._validate_and_record_search_advantage_components()
    metrics = adapter._attempt_context["advantage_component_metrics"]
    assert metrics["search_advantage_formula_assertion_pass"] is True
    assert metrics["answer_advantage_formula_assertion_pass"] is True
    assert metrics["z_O_S_or_N_search_entry_count"] == 0
    assert metrics["A_SC_search_entry_count"] == 0
