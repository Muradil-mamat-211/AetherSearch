from __future__ import annotations

import pytest

from vllm.sampling_params import RequestOutputKind

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
from agentic_rl.runtime.stop_branching import (
    attach_routed_answer_probe_results,
    attach_sufficiency_probe_results,
    attach_stop_branch_rewards,
    build_routed_answer_probe_plan,
    build_sufficiency_probe_plan,
    build_stop_branch_plan,
)
from agentic_rl.runtime.capped_vllm import _build_stop_pair_sampling_params
from agentic_rl.selection.candidate_pool import PromptGroup


def _record(prompt_id: str, trajectory_index: int) -> TrajectoryRecord:
    record = TrajectoryRecord(
        prompt_global_id=prompt_id,
        trajectory_id=f"{prompt_id}:t{trajectory_index}",
        input_ids=[10, 11, 20, 21, 30, 31, 22, 23, 32, 33, 40],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.ENVIRONMENT,
            TokenSource.MODEL,
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.ENVIRONMENT,
            TokenSource.MODEL,
        ],
        turn_ids=[-1, -1, 0, 0, -1, -1, 1, 1, -1, -1, 2],
        turns=[
            TurnRecord(
                turn_index=0,
                turn_type=TurnType.SEARCH,
                model_text="<search>first</search>",
                search_index=0,
                query="first",
                information_text="<information>one</information>",
                search_action_span_valid=True,
                search_prefix_valid=True,
                ig_reward_eligible=True,
                policy_credit_eligible=True,
            ),
            TurnRecord(
                turn_index=1,
                turn_type=TurnType.SEARCH,
                model_text="<search>second</search>",
                search_index=1,
                query="second",
                information_text="<information>two</information>",
                search_action_span_valid=True,
                search_prefix_valid=True,
                ig_reward_eligible=True,
                policy_credit_eligible=True,
            ),
            TurnRecord(
                turn_index=2,
                turn_type=TurnType.ANSWER,
                model_text="<answer>NYC</answer>",
                policy_credit_eligible=True,
            ),
        ],
        search_prefix_end_positions=[2, 6, 10],
        search_prefix_before_search_end_positions={0: 2, 1: 6},
        immediate_ig={0: 0.1, 1: -0.2},
        task_outcome=1.0,
        answer_format_indicator=1,
        terminal_answer_valid=True,
        trajectory_protocol_valid=True,
        trajectory_system_valid=True,
        metadata={
            "snapshot_step": 3,
            "reward_snapshot_step": 3,
            "gold_aliases": ("New York City", "NYC"),
            "data_source": "nq",
            "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
        },
    )
    record.validate()
    return record


def _group(prompt_id: str) -> PromptGroup:
    records = tuple(_record(prompt_id, index) for index in range(2))
    return PromptGroup(
        prompt_global_id=prompt_id,
        trajectories=records,
        ig_variance=1.0,
        outcome_variance=1.0,
    )


def _terminal_empty_observation_record(prompt_id: str) -> TrajectoryRecord:
    record = TrajectoryRecord(
        prompt_global_id=prompt_id,
        trajectory_id=f"{prompt_id}:terminal-empty-observation",
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
                model_text="<search>first</search>",
                search_index=0,
                query="first",
                information_text="<information>one</information>",
                search_action_span_valid=True,
                search_prefix_valid=True,
                ig_reward_eligible=True,
                policy_credit_eligible=True,
                no_new_observation=False,
                current_passage_keys=("id:first",),
                new_passage_keys=("id:first",),
            ),
            TurnRecord(
                turn_index=1,
                turn_type=TurnType.SEARCH,
                model_text="<search>over-budget</search>",
                search_index=1,
                query="over-budget",
                information_text=None,
                search_action_span_valid=True,
                search_prefix_valid=False,
                ig_reward_eligible=False,
                policy_credit_eligible=True,
                no_new_observation=True,
                current_passage_keys=(),
                new_passage_keys=(),
            ),
        ],
        search_prefix_end_positions=[2, 6],
        search_prefix_before_search_end_positions={0: 2, 1: 6},
        immediate_ig={0: 0.1},
        task_outcome=0.0,
        answer_format_indicator=0,
        terminal_answer_valid=False,
        trajectory_protocol_valid=False,
        trajectory_system_valid=True,
        metadata={
            "snapshot_step": 3,
            "reward_snapshot_step": 3,
            "gold_aliases": ("answer",),
            "data_source": "nq",
            "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
            "termination_reason": "maximum_search_turns_reached",
        },
    )
    record.validate()
    return record


def _not_sufficient_raw_probe() -> dict[str, object]:
    return {
        "parser_success": True,
        "no_answer": False,
        "output_truncated": False,
        "alias_aware_exact": False,
        "sufficient_before_search": False,
    }


def test_stop_pair_is_one_final_only_parallel_sampling_request() -> None:
    params = _build_stop_pair_sampling_params(
        {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        },
        max_tokens=32,
    )
    assert params.n == 2
    assert params.output_kind is RequestOutputKind.FINAL_ONLY
    assert params.stop == ["</answer>"]
    assert params.include_stop_str_in_output is True
    assert params.logprobs is None
    assert params.prompt_logprobs is None


def test_terminal_budget_exhausted_search_has_no_post_probe() -> None:
    record = _terminal_empty_observation_record("terminal")
    with pytest.raises(ValueError, match="missing post-Search prefix 1"):
        record.prefix_token_ids_after_search_observation(1)
    record.metadata["routed_answer_probes"] = {
        0: {"pre": _not_sufficient_raw_probe()},
        1: {"pre": _not_sufficient_raw_probe()},
    }
    group = PromptGroup(
        prompt_global_id=record.prompt_global_id,
        trajectories=(record,),
        ig_variance=1.0,
        outcome_variance=1.0,
    )
    scaffold = (91, 92, 93)
    plan = build_routed_answer_probe_plan(
        [group],
        probe_stage="post",
        scaffold_token_ids=scaffold,
        rollout_config={
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        stop_answer_max_new_tokens=500,
        maximum_model_length=1024,
        expected_snapshot_step=3,
    )
    jobs = [job for replica_jobs in plan.jobs_by_replica for job in replica_jobs]
    assert plan.state_count == 1
    assert [job["search_index"] for job in jobs] == [0]
    assert tuple(jobs[0]["probe_input_ids"]) == (10, 11, 20, 21, 30, 31, *scaffold)


def test_missing_post_prefix_remains_fail_closed_for_nonterminal_search() -> None:
    record = _terminal_empty_observation_record("nonterminal")
    record.metadata["termination_reason"] = "model_answer"
    record.input_ids.append(40)
    record.token_sources.append(TokenSource.MODEL)
    record.turn_ids.append(2)
    record.turns.append(
        TurnRecord(
            turn_index=2,
            turn_type=TurnType.ANSWER,
            model_text="<answer>answer</answer>",
            policy_credit_eligible=True,
        )
    )
    record.validate()
    with pytest.raises(ValueError, match="missing post-Search prefix 1"):
        record.prefix_token_ids_after_search_observation(1)


def test_stop_plan_uses_direct_prefix_ids_and_prompt_affinity() -> None:
    groups = tuple(_group(f"p{index}") for index in range(4))
    scaffold = (91, 92, 93)
    plan = build_stop_branch_plan(
        groups,
        scaffold_token_ids=scaffold,
        rollout_config={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        stop_answer_max_new_tokens=8,
        maximum_model_length=64,
        expected_snapshot_step=3,
    )
    assert plan.state_count == 16
    assert plan.request_count == 16
    assert plan.expected_completion_count == 32
    observed_prompt_replicas: dict[str, set[int]] = {}
    for replica, jobs in enumerate(plan.jobs_by_replica):
        for job in jobs:
            prompt_id = job["metadata"]["prompt_global_id"]
            observed_prompt_replicas.setdefault(prompt_id, set()).add(replica)
            trajectory_id = job["metadata"]["trajectory_id"]
            record = next(
                record
                for group in groups
                for record in group.trajectories
                if record.trajectory_id == trajectory_id
            )
            prefix = record.prefix_token_ids_before_search(job["search_index"])
            assert tuple(job["stop_input_ids"]) == (*prefix, *scaffold)
            assert job["sampling_params"]["temperature"] == 1.0
            assert job["sampling_params"]["top_p"] == 0.95
            assert job["sampling_params"]["logprobs"] is False
            assert job["sampling_params"]["prompt_logprobs"] is False
    assert all(len(replicas) == 1 for replicas in observed_prompt_replicas.values())
    assert {
        prompt_id: next(iter(replicas))
        for prompt_id, replicas in observed_prompt_replicas.items()
    } == dict(plan.prompt_to_replica)


def test_attached_stop_completions_remain_detached_from_policy_tokens() -> None:
    group = _group("p0")
    plan = build_stop_branch_plan(
        [group],
        scaffold_token_ids=(91, 92),
        rollout_config={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        stop_answer_max_new_tokens=8,
        maximum_model_length=64,
        expected_snapshot_step=3,
    )
    before_masks = {
        record.trajectory_id: tuple(record.policy_mask)
        for record in group.trajectories
    }
    generated = []
    scored = []
    for replica, jobs in enumerate(plan.jobs_by_replica):
        for job in jobs:
            metadata = dict(job["metadata"])
            generated.append(
                {
                    **metadata,
                    "snapshot_step": 3,
                    "source_checksum": "checksum",
                    "automatic_prefix_caching": True,
                    "completion_count": 2,
                    "completions": [
                        {"sample_index": 0, "text": "NYC</answer>"},
                        {"sample_index": 1, "text": "New York City</answer>"},
                    ],
                    "assigned_replica": replica,
                    "decode_tokens": 8,
                    "prompt_tokens": len(job["stop_input_ids"]),
                    "cached_prompt_tokens": 2,
                    "generation_seconds": 0.01,
                    "prefix_token_ids_sha256": metadata[
                        "prefix_token_ids_sha256"
                    ],
                }
            )
            for sample_index in range(2):
                scored.append(
                    {
                        "prompt_global_id": metadata["prompt_global_id"],
                        "trajectory_id": metadata["trajectory_id"],
                        "search_index": metadata["search_index"],
                        "sample_index": sample_index,
                        "task_outcome": 1.0,
                        "scorer_version": PRODUCTION_TASK_SCORER_VERSION,
                    }
                )
    metrics = attach_stop_branch_rewards(
        [group],
        generated,
        scored,
        expected_snapshot_step=3,
        expected_source_checksum="checksum",
    )
    assert metrics["sc/completion_count"] == 2 * metrics["sc/request_count"]
    for record in group.trajectories:
        assert tuple(record.policy_mask) == before_masks[record.trajectory_id]


def test_sufficiency_plan_is_selected_prompt_affine_and_n_one() -> None:
    groups = tuple(_group(f"p{index}") for index in range(4))
    scaffold = (91, 92, 93)
    plan = build_sufficiency_probe_plan(
        groups,
        scaffold_token_ids=scaffold,
        rollout_config={
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        stop_answer_max_new_tokens=8,
        maximum_model_length=64,
        expected_snapshot_step=3,
    )
    assert plan.state_count == 16
    assert plan.request_count == 16
    assert plan.expected_completion_count == 16
    prompt_replicas: dict[str, set[int]] = {}
    for replica, jobs in enumerate(plan.jobs_by_replica):
        for job in jobs:
            prompt_id = job["metadata"]["prompt_global_id"]
            prompt_replicas.setdefault(prompt_id, set()).add(replica)
            assert job["sampling_params"]["do_sample"] is False
            assert job["sampling_params"]["temperature"] == 0.0
            assert job["sampling_params"]["top_p"] == 1.0
            assert job["sampling_params"]["n"] == 1
    assert all(len(values) == 1 for values in prompt_replicas.values())


def test_sufficiency_attachment_keeps_probe_tokens_detached() -> None:
    group = _group("p-s")
    plan = build_sufficiency_probe_plan(
        [group],
        scaffold_token_ids=(91, 92),
        rollout_config={
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        stop_answer_max_new_tokens=8,
        maximum_model_length=64,
        expected_snapshot_step=3,
    )
    before_masks = {
        record.trajectory_id: tuple(record.policy_mask)
        for record in group.trajectories
    }
    generated = []
    scored = []
    for replica, jobs in enumerate(plan.jobs_by_replica):
        for job in jobs:
            metadata = dict(job["metadata"])
            generated.append(
                {
                    **metadata,
                    "snapshot_step": 3,
                    "source_checksum": "checksum",
                    "automatic_prefix_caching": True,
                    "completion_count": 1,
                    "completions": [
                        {
                            "sample_index": 0,
                            "text": "NYC</answer>",
                            "finish_reason": "stop",
                        }
                    ],
                    "assigned_replica": replica,
                    "decode_tokens": 4,
                    "prompt_tokens": len(job["probe_input_ids"]),
                    "cached_prompt_tokens": 2,
                    "generation_seconds": 0.01,
                    "prefix_token_ids_sha256": metadata[
                        "prefix_token_ids_sha256"
                    ],
                }
            )
            scored.append(
                {
                    "prompt_global_id": metadata["prompt_global_id"],
                    "trajectory_id": metadata["trajectory_id"],
                    "search_index": metadata["search_index"],
                    "sufficient_before_search": True,
                    "alias_exact_match": True,
                    "partial_task_reward_shadow": 1.0,
                    "terminal_answer_valid": True,
                    "parser_status": "valid",
                    "parser_error_type": None,
                    "parsed_answer": "NYC",
                    "truncated": False,
                    "scorer_version": SUFFICIENCY_EXACT_SCORER_VERSION,
                    "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
                }
            )
    metrics = attach_sufficiency_probe_results(
        [group],
        generated,
        scored,
        expected_snapshot_step=3,
        expected_source_checksum="checksum",
    )
    assert metrics["s_probe/completion_count"] == metrics["s_probe/state_count"]
    for record in group.trajectories:
        assert tuple(record.policy_mask) == before_masks[record.trajectory_id]
        assert len(record.metadata["sufficiency_probes"]) == 2
        assert all(
            probe["detached"] is True
            for probe in record.metadata["sufficiency_probes"].values()
        )


def test_routed_pre_and_post_probes_use_direct_state_prefixes() -> None:
    group = _group("p-routed")
    scaffold = (91, 92)
    masks_before = {
        record.trajectory_id: tuple(record.policy_mask)
        for record in group.trajectories
    }

    def generated_and_scored(plan, stage: str):
        generated = []
        scored = []
        for replica, jobs in enumerate(plan.jobs_by_replica):
            for job in jobs:
                metadata = dict(job["metadata"])
                generated.append(
                    {
                        **metadata,
                        "snapshot_step": 3,
                        "source_checksum": "checksum",
                        "automatic_prefix_caching": True,
                        "completion_count": 1,
                        "completions": [
                            {
                                "sample_index": 0,
                                "text": "partial</answer>",
                                "finish_reason": "stop",
                            }
                        ],
                        "assigned_replica": replica,
                        "decode_tokens": 3,
                        "prompt_tokens": len(job["probe_input_ids"]),
                        "cached_prompt_tokens": 1,
                        "generation_seconds": 0.01,
                    }
                )
                scored.append(
                    {
                        "prompt_global_id": metadata["prompt_global_id"],
                        "trajectory_id": metadata["trajectory_id"],
                        "search_index": metadata["search_index"],
                        "probe_stage": stage,
                        "parser_success": True,
                        "no_answer": False,
                        "output_truncated": False,
                        "alias_aware_exact": False,
                        "raw_task_reward": 0.25,
                        "parser_status": "valid",
                        "parser_error_type": None,
                        "parsed_answer": "partial",
                        "scorer_version": SUFFICIENCY_EXACT_SCORER_VERSION,
                        "task_scorer_version": PRODUCTION_TASK_SCORER_VERSION,
                    }
                )
        return generated, scored

    pre_plan = build_routed_answer_probe_plan(
        [group],
        probe_stage="pre",
        scaffold_token_ids=scaffold,
        rollout_config={
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        stop_answer_max_new_tokens=500,
        maximum_model_length=1024,
        expected_snapshot_step=3,
    )
    assert pre_plan.state_count == 4
    for jobs in pre_plan.jobs_by_replica:
        for job in jobs:
            record = next(
                record
                for record in group.trajectories
                if record.trajectory_id == job["metadata"]["trajectory_id"]
            )
            expected = record.prefix_token_ids_before_search(job["search_index"])
            assert tuple(job["probe_input_ids"]) == (*expected, *scaffold)
            assert job["metadata"]["max_tokens"] == 500
            assert job["metadata"]["stop"] == ["</answer>"]
    generated, scored = generated_and_scored(pre_plan, "pre")
    attach_routed_answer_probe_results(
        [group],
        generated,
        scored,
        probe_stage="pre",
        expected_snapshot_step=3,
        expected_source_checksum="checksum",
    )

    post_plan = build_routed_answer_probe_plan(
        [group],
        probe_stage="post",
        scaffold_token_ids=scaffold,
        rollout_config={
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        stop_answer_max_new_tokens=500,
        maximum_model_length=1024,
        expected_snapshot_step=3,
    )
    assert post_plan.state_count == 4
    for jobs in post_plan.jobs_by_replica:
        for job in jobs:
            record = next(
                record
                for record in group.trajectories
                if record.trajectory_id == job["metadata"]["trajectory_id"]
            )
            expected = record.prefix_token_ids_after_search_observation(
                job["search_index"]
            )
            assert tuple(job["probe_input_ids"]) == (*expected, *scaffold)
    generated, scored = generated_and_scored(post_plan, "post")
    attach_routed_answer_probe_results(
        [group],
        generated,
        scored,
        probe_stage="post",
        expected_snapshot_step=3,
        expected_source_checksum="checksum",
    )
    for record in group.trajectories:
        assert tuple(record.policy_mask) == masks_before[record.trajectory_id]
        for stages in record.metadata["routed_answer_probes"].values():
            assert set(stages) == {"pre", "post"}
            assert stages["pre"]["detached"] is True
            assert stages["post"]["detached"] is True
