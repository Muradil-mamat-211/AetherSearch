from __future__ import annotations

import math
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from agentic_rl.advantage.a2tgpo import (
    TrajectoryCreditInput,
    compute_prompt_advantages,
    rebuild_search_advantages,
    turn_advantages_from_record,
)
from agentic_rl.advantage.stop_continue import (
    STOP_CONTINUE_CONSENSUS_MODE,
    StopContinueRewardTriple,
)
from agentic_rl.exact_ig.target_schema import select_canonical_answer
from agentic_rl.outcome.token_f1 import max_alias_token_f1
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    score_stop_answer_completion,
    score_trajectory_outcome,
)
from agentic_rl.policy.strict_onpolicy_loss import (
    a2tgpo_adaptive_turn_objective,
)
from agentic_rl.rollout.trajectory_schema import TurnType
from agentic_rl.config import load_config

from config_support import TEST_CONFIG
from agentic_rl.controller.attempt_state import TrainingState
from agentic_rl.controller.update_controller import StrictAttemptController
from agentic_rl.selection.candidate_pool import CandidatePool, PromptGroup


POLICY_VERSION = 31


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


def _probe(
    trajectory_id: str,
    continue_reward: float,
    stop_reward_1: float,
    stop_reward_2: float,
) -> dict[str, object]:
    value = StopContinueRewardTriple(
        prompt_global_id="prompt",
        trajectory_id=trajectory_id,
        search_index=0,
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
    return {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
    }


def test_rebuilt_search_advantage_is_the_value_consumed_by_policy_objective() -> None:
    base = compute_prompt_advantages(
        [_credit(1.0, 1.0, 1), _credit(-1.0, 0.0, 0)]
    )
    records = [
        SimpleNamespace(
            prompt_global_id="prompt",
            trajectory_id=f"trajectory-{index}",
            metadata={
                "stop_continue_probes": {
                    0: _probe(
                        f"trajectory-{index}",
                        1.0 if index == 0 else 0.0,
                        0.0 if index == 0 else 0.6,
                        0.6 if index == 0 else 1.0,
                    )
                }
            },
        )
        for index in range(2)
    ]
    old_answer_values = tuple(
        value.answer_advantage for value in base.trajectories
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
    ) == old_answer_values

    advantage = rebuilt.trajectories[0]
    record = SimpleNamespace(
        terminal_policy_credit_turn_index=2,
        turns=(
            SimpleNamespace(
                turn_index=1,
                turn_type=TurnType.SEARCH,
                search_index=0,
                policy_credit_eligible=True,
            ),
            SimpleNamespace(
                turn_index=2,
                turn_type=TurnType.ANSWER,
                search_index=None,
                policy_credit_eligible=True,
            ),
        ),
    )
    by_turn = turn_advantages_from_record(record, advantage)
    assert by_turn[1] == advantage.search_advantage[0]
    assert by_turn[2] == advantage.answer_advantage
    assert not math.isclose(
        by_turn[1],
        advantage.search_advantage_old_shadow[0],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    objective = a2tgpo_adaptive_turn_objective(
        {
            1: torch.tensor(1.0, dtype=torch.float32),
            2: torch.tensor(1.0, dtype=torch.float32),
        },
        by_turn,
        {1: advantage.normalized_ig[0]},
        answer_turn_ids=(2,),
    )
    assert float(objective.objective_by_turn[1]) == float(
        torch.tensor(by_turn[1], dtype=torch.float32)
    )
    assert float(objective.objective_by_turn[2]) == float(
        torch.tensor(by_turn[2], dtype=torch.float32)
    )


def test_stop_probe_policy_versions_are_all_part_of_the_hard_contract() -> None:
    probe = _probe("trajectory", 1.0, 0.0, 0.6)
    value = StopContinueRewardTriple(**probe)
    value.validate(
        expected_policy_version=POLICY_VERSION,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )
    for field in (
        "candidate_rollout_policy_version",
        "exact_ig_policy_version",
        "stop_branch_policy_version",
        "old_logprob_policy_version",
    ):
        invalid = dict(probe)
        invalid[field] = POLICY_VERSION + 1
        try:
            StopContinueRewardTriple(**invalid).validate(
                expected_policy_version=POLICY_VERSION,
                expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"{field} mismatch must fail closed")


def test_fixed_tensor_search_formula_uses_point_three_ig_weight() -> None:
    a_ig = np.asarray([2.0, -3.0, 0.5, -0.25], dtype=np.float64)
    a_sc = np.asarray([0.4, -0.7, 1.2, -1.4], dtype=np.float64)
    z_o = np.asarray([-0.2, 0.3, -0.6, 0.8], dtype=np.float64)
    sc_clear = np.asarray([True, True, False, False], dtype=np.bool_)
    expected_task = np.where(sc_clear, a_sc, 0.0)
    expected_search = 0.3 * a_ig + expected_task

    actual_search = np.asarray(
        [
            0.3 * ig + (sc if clear else 0.0)
            for ig, sc, outcome, clear in zip(
                a_ig,
                a_sc,
                z_o,
                sc_clear,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual_search, expected_search)

    a_format = np.asarray([0.5, -0.5, 0.25, -0.25], dtype=np.float64)
    answer_before = z_o + a_format
    answer_after = z_o + a_format
    np.testing.assert_array_equal(answer_after, answer_before)


def test_ragen_selected_set_is_independent_of_search_advantage_lambda() -> None:
    base = load_config(TEST_CONFIG)
    old_shadow_config = copy.deepcopy(base)
    old_shadow_config["advantage"]["lambda_ig"] = 1.0
    production_config = copy.deepcopy(base)
    production_config["advantage"]["lambda_ig"] = 0.3

    pool = CandidatePool(group_size=16, maximum_prompts=96)
    pool.add(
        PromptGroup(
            prompt_global_id=f"p{index:03d}",
            trajectories=tuple(range(16)),
            ig_variance=float(65 - index),
            outcome_variance=float(index + 1),
        )
        for index in range(64)
    )
    state = TrainingState()
    old_decision = StrictAttemptController(old_shadow_config)._select(pool, state)
    new_decision = StrictAttemptController(production_config)._select(pool, state)

    assert old_decision.score_by_prompt == new_decision.score_by_prompt
    assert old_decision.top_p.ordered_positive_ids == (
        new_decision.top_p.ordered_positive_ids
    )
    assert old_decision.selected_ids == new_decision.selected_ids
    assert old_decision.selected_count == new_decision.selected_count
    old_hash = hashlib.sha256(
        "\n".join(old_decision.selected_ids).encode("utf-8")
    ).hexdigest()
    new_hash = hashlib.sha256(
        "\n".join(new_decision.selected_ids).encode("utf-8")
    ).hexdigest()
    assert old_hash == new_hash


def test_task_reward_alias_max_is_independent_of_search_advantage_lambda() -> None:
    aliases = ["New York City", "NYC"]
    prediction = "NYC"
    assert select_canonical_answer(aliases) == aliases[0]
    expected = max(
        max_alias_token_f1(prediction, [aliases[0]]),
        max_alias_token_f1(prediction, [aliases[1]]),
    )

    before = score_trajectory_outcome(
        [f"<answer>{prediction}</answer>"],
        aliases,
    ).task_outcome
    after = score_trajectory_outcome(
        [f"<answer>{prediction}</answer>"],
        aliases,
    ).task_outcome
    stop = score_stop_answer_completion(f"{prediction}</answer>", aliases)

    assert before == after == expected == 1.0
    assert stop.task_outcome == expected
