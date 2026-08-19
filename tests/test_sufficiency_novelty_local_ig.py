from __future__ import annotations

from gpu_test_guard import skip_if_no_gpu

skip_if_no_gpu()

import math
from types import SimpleNamespace

import pytest

from agentic_rl.advantage.a2tgpo import (
    SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
    TrajectoryCreditInput,
    compute_prompt_advantages,
    rebuild_search_advantages,
)
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    SUFFICIENCY_EXACT_SCORER_VERSION,
    score_sufficiency_probe_completion,
)
from agentic_rl.retriever.protocol import RetrievalDocument
from agentic_rl.rollout.trajectory_schema import TurnRecord, TurnType
from agentic_rl.runtime.capped_vllm import (
    _build_sufficiency_probe_sampling_params,
)
from agentic_rl.runtime.search_agent_loop import (
    compute_and_commit_passage_novelty,
    normalize_passage_text,
    stable_passage_key,
)


def _credit(first: float, second: float, outcome: float) -> TrajectoryCreditInput:
    return TrajectoryCreditInput(
        immediate_ig={0: first, 1: second},
        search_turn_indices=(0, 1),
        ig_reward_eligible={0: True, 1: True},
        policy_credit_eligible={0: True, 1: True},
        outcome=outcome,
        outcome_reward_eligible=True,
        format_indicator=1,
        answer_policy_credit_eligible=True,
    )


def _probe(sufficient: bool, *, version: int = 7) -> dict[str, object]:
    return {
        "sufficient_before_search": sufficient,
        "completion_count": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "detached": True,
        "prefix_provenance_valid": True,
        "context_truncated": False,
        "scorer_version": SUFFICIENCY_EXACT_SCORER_VERSION,
        "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
        "candidate_rollout_policy_version": version,
        "exact_ig_policy_version": version,
        "sufficiency_probe_policy_version": version,
        "old_logprob_policy_version": version,
    }


def _record(
    trajectory_id: str,
    *,
    sufficient: tuple[bool, bool],
    no_new: tuple[bool, bool],
    immediate_ig: tuple[float, float],
) -> SimpleNamespace:
    turns = [
        TurnRecord(
            turn_index=index,
            turn_type=TurnType.SEARCH,
            model_text=f"<search>q{index}</search>",
            search_index=index,
            query=f"q{index}",
            search_action_span_valid=True,
            search_prefix_valid=True,
            ig_reward_eligible=True,
            policy_credit_eligible=True,
            no_new_observation=no_new[index],
            current_passage_keys=(f"passage_id:{index}",),
            new_passage_keys=(() if no_new[index] else (f"passage_id:{index}",)),
        )
        for index in range(2)
    ]
    return SimpleNamespace(
        prompt_global_id="prompt-1",
        trajectory_id=trajectory_id,
        turns=turns,
        immediate_ig={0: immediate_ig[0], 1: immediate_ig[1]},
        metadata={
            "sufficiency_probes": {
                0: _probe(sufficient[0]),
                1: _probe(sufficient[1]),
            }
        },
    )


def test_production_formula_uses_s_then_n_then_only_local_ig() -> None:
    credits = (
        _credit(1.0, 100.0, 0.0),
        _credit(2.0, -50.0, 0.5),
        _credit(3.0, 0.0, 1.0),
    )
    baseline = compute_prompt_advantages(
        credits,
        accumulate_future_ig=False,
        lambda_ig=None,
    )
    answer_before = tuple(item.answer_advantage for item in baseline.trajectories)
    records = (
        _record(
            "t0",
            sufficient=(True, True),
            no_new=(True, False),
            immediate_ig=(1.0, 100.0),
        ),
        _record(
            "t1",
            sufficient=(False, False),
            no_new=(True, False),
            immediate_ig=(2.0, -50.0),
        ),
        _record(
            "t2",
            sufficient=(False, False),
            no_new=(False, False),
            immediate_ig=(3.0, 0.0),
        ),
    )
    rebuilt, metrics = rebuild_search_advantages(
        records,
        baseline,
        search_task_mode=SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
        group_size=3,
        lambda_ig=None,
        expected_policy_version=7,
        expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
    )

    first, second, third = rebuilt.trajectories
    assert first.search_advantage[0] == -1.0  # S and N do not stack.
    assert first.search_advantage[1] == -1.0
    assert second.search_advantage[0] == -1.0
    assert second.search_advantage[1] == second.normalized_ig[1]
    assert third.search_advantage[0] == third.normalized_ig[0]
    assert third.search_advantage[1] == third.normalized_ig[1]
    # Turn zero is not contaminated by the deliberately extreme turn-one IG.
    assert third.search_advantage[0] > 0.0
    assert all(not item.future_ig_sum for item in rebuilt.trajectories)
    assert all(not item.future_ig_rescaled for item in rebuilt.trajectories)
    assert all(not item.accumulated_ig_count for item in rebuilt.trajectories)
    assert all(not item.stop_continue_by_search_index for item in rebuilt.trajectories)
    assert all(not item.search_task_advantage for item in rebuilt.trajectories)
    assert tuple(item.answer_advantage for item in rebuilt.trajectories) == answer_before
    assert metrics["search/z_o_actor_entry_count"] == 0
    assert metrics["search/a_sc_actor_entry_count"] == 0
    assert metrics["search/future_ig_contribution_count"] == 0
    assert metrics["search/sqrt_n_rescale_call_count"] == 0
    assert metrics["search/external_ig_multiplier_call_count"] == 0


def test_local_ig_normalization_fails_closed_for_low_variance() -> None:
    credits = (
        _credit(2.0, 2.0, 0.0),
        _credit(2.0, 2.0, 1.0),
    )
    result = compute_prompt_advantages(
        credits,
        accumulate_future_ig=False,
        lambda_ig=None,
    )
    assert all(item.normalized_ig == {0: 0.0, 1: 0.0} for item in result.trajectories)


def test_sufficiency_probe_requires_parser_valid_alias_exact_and_not_truncated() -> None:
    aliases = ["New York City", "NYC"]
    exact = score_sufficiency_probe_completion("NYC</answer>", aliases)
    assert exact["sufficient_before_search"] is True
    assert exact["alias_exact_match"] is True
    assert exact["scorer_version"] == SUFFICIENCY_EXACT_SCORER_VERSION

    partial = score_sufficiency_probe_completion("New York</answer>", aliases)
    assert partial["partial_task_reward_shadow"] > 0.0
    assert partial["sufficient_before_search"] is False
    truncated = score_sufficiency_probe_completion(
        "NYC</answer>", aliases, truncated=True
    )
    assert truncated["sufficient_before_search"] is False
    malformed = score_sufficiency_probe_completion("NYC", aliases)
    assert malformed["sufficient_before_search"] is False


def test_passage_keys_prefer_unique_id_and_fallback_to_normalized_full_text() -> None:
    identified = RetrievalDocument("row-123", "Title\nBody")
    assert stable_passage_key(identified) == "passage_id:row-123"
    first = RetrievalDocument("", "  Title\n\nBody  ")
    second = RetrievalDocument("", "Title Body")
    assert normalize_passage_text(first.contents) == "Title Body"
    assert stable_passage_key(first) == stable_passage_key(second)
    assert stable_passage_key(first).startswith("text_sha256:")


def test_passage_novelty_is_computed_before_current_results_are_committed() -> None:
    seen: set[str] = set()
    a = RetrievalDocument("a", "A")
    b = RetrievalDocument("b", "B")
    current, new, no_new = compute_and_commit_passage_novelty(seen, [a, a])
    assert current == {"passage_id:a"}
    assert new == {"passage_id:a"}
    assert no_new is False
    assert seen == {"passage_id:a"}

    current, new, no_new = compute_and_commit_passage_novelty(seen, [a, b])
    assert current == {"passage_id:a", "passage_id:b"}
    assert new == {"passage_id:b"}
    assert no_new is False
    current, new, no_new = compute_and_commit_passage_novelty(seen, [a])
    assert new == set()
    assert no_new is True
    current, new, no_new = compute_and_commit_passage_novelty(seen, [])
    assert current == set()
    assert new == set()
    assert no_new is True


def test_sufficiency_sampling_is_one_deterministic_completion() -> None:
    params = _build_sufficiency_probe_sampling_params(
        {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "n": 1,
        },
        max_tokens=500,
    )
    assert params.n == 1
    assert params.temperature == 0.0
    assert params.top_p == 1.0
    assert params.logprobs is None
    assert params.prompt_logprobs is None
    assert params.stop == ["</answer>"]


def test_sufficiency_sampling_rejects_stochastic_settings() -> None:
    with pytest.raises(ValueError, match="do_sample=false"):
        _build_sufficiency_probe_sampling_params(
            {
                "do_sample": True,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
            },
            max_tokens=8,
        )
