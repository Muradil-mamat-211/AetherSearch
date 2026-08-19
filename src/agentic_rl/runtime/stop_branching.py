from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from agentic_rl.exact_ig.target_schema import ANSWER_SCAFFOLD_TEXT
from agentic_rl.outcome.workers import (
    PRODUCTION_TASK_SCORER_VERSION,
    SUFFICIENCY_EXACT_SCORER_VERSION,
)
from agentic_rl.rollout.trajectory_schema import (
    TurnType,
    is_budget_exhausted_terminal_search,
)
from agentic_rl.rollout.search_role_provenance import (
    ROLE_LOCALIZED_BRANCH_N_SOFT,
    ROLE_LOCALIZED_BRANCH_NORMAL,
    classify_role_localized_search_branch,
)
from agentic_rl.selection.candidate_pool import PromptGroup


@dataclass(frozen=True)
class StopBranchPlan:
    jobs_by_replica: tuple[tuple[dict[str, Any], ...], ...]
    prompt_to_replica: Mapping[str, int]
    estimated_tokens_by_replica: tuple[int, ...]
    state_count: int
    request_count: int
    expected_completion_count: int


@dataclass(frozen=True)
class SufficiencyProbePlan:
    jobs_by_replica: tuple[tuple[dict[str, Any], ...], ...]
    prompt_to_replica: Mapping[str, int]
    estimated_tokens_by_replica: tuple[int, ...]
    state_count: int
    request_count: int
    expected_completion_count: int


def tokenize_stop_scaffold(tokenizer: Any) -> tuple[int, ...]:
    encoded = tokenizer(
        ANSWER_SCAFFOLD_TEXT,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    token_ids = tuple(int(value) for value in encoded["input_ids"])
    if not token_ids:
        raise ValueError("Stop scaffold tokenization is empty")
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    if decoded != ANSWER_SCAFFOLD_TEXT:
        raise ValueError("Stop scaffold tokenizer round-trip changed the locked text")
    return token_ids


def _sampling_params(
    rollout: Mapping[str, Any],
    *,
    stop_answer_max_new_tokens: int,
) -> dict[str, Any]:
    top_k = int(rollout["top_k"])
    return {
        "temperature": float(rollout["temperature"]),
        "top_p": float(rollout.get("sampling_top_p", rollout["top_p"])),
        "top_k": -1 if top_k == 0 else top_k,
        "min_p": float(rollout.get("min_p", 0.0)),
        "repetition_penalty": float(
            rollout.get("repetition_penalty", 1.0)
        ),
        "presence_penalty": float(rollout.get("presence_penalty", 0.0)),
        "frequency_penalty": float(rollout.get("frequency_penalty", 0.0)),
        "project_max_tokens": int(stop_answer_max_new_tokens),
        "logprobs": False,
        "prompt_logprobs": False,
    }


def _sufficiency_sampling_params(
    rollout: Mapping[str, Any],
    *,
    stop_answer_max_new_tokens: int,
) -> dict[str, Any]:
    """Locked deterministic Answer-now probe parameters."""

    return {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": float(
            rollout.get("repetition_penalty", 1.0)
        ),
        "presence_penalty": float(rollout.get("presence_penalty", 0.0)),
        "frequency_penalty": float(rollout.get("frequency_penalty", 0.0)),
        "n": 1,
        "project_max_tokens": int(stop_answer_max_new_tokens),
        "logprobs": False,
        "prompt_logprobs": False,
    }


def build_sufficiency_probe_plan(
    groups: Sequence[PromptGroup],
    *,
    scaffold_token_ids: Sequence[int],
    rollout_config: Mapping[str, Any],
    stop_answer_max_new_tokens: int,
    maximum_model_length: int,
    expected_snapshot_step: int,
    replica_count: int = 4,
) -> SufficiencyProbePlan:
    """Route one deterministic pre-Search probe per selected Search state."""

    if not groups:
        raise ValueError("Sufficiency probing requires selected Prompt groups")
    if int(replica_count) < 1:
        raise ValueError("Production sufficiency probing requires at least one replica")
    scaffold = tuple(int(value) for value in scaffold_token_ids)
    if not scaffold:
        raise ValueError("Sufficiency scaffold token IDs cannot be empty")
    stop_answer_max_new_tokens = int(stop_answer_max_new_tokens)
    maximum_model_length = int(maximum_model_length)
    if stop_answer_max_new_tokens <= 0 or maximum_model_length <= 0:
        raise ValueError("Probe/model length limits must be positive")
    params = _sufficiency_sampling_params(
        rollout_config,
        stop_answer_max_new_tokens=stop_answer_max_new_tokens,
    )

    group_rows: list[tuple[int, str, list[dict[str, Any]]]] = []
    seen_prompts: set[str] = set()
    state_keys: set[tuple[str, str, int]] = set()
    for group in groups:
        prompt_id = str(group.prompt_global_id)
        if prompt_id in seen_prompts:
            raise ValueError(f"Duplicate selected Prompt group: {prompt_id}")
        seen_prompts.add(prompt_id)
        jobs: list[dict[str, Any]] = []
        estimated_group_tokens = 0
        for record in group.trajectories:
            if int(record.metadata.get("snapshot_step", -1)) != int(
                expected_snapshot_step
            ):
                raise ValueError(
                    f"{record.trajectory_id}: candidate policy version mismatch"
                )
            if not record.trajectory_system_valid:
                raise ValueError(
                    "System-invalid trajectory reached sufficiency probing"
                )
            record.validate()
            for turn in record.turns:
                if (
                    turn.turn_type is not TurnType.SEARCH
                    or not turn.policy_credit_eligible
                ):
                    continue
                search_index = int(turn.search_index)
                state_key = (prompt_id, str(record.trajectory_id), search_index)
                if state_key in state_keys:
                    raise ValueError(f"Duplicate sufficiency state: {state_key}")
                state_keys.add(state_key)
                prefix = record.prefix_token_ids_before_search(search_index)
                probe_input_ids = (*prefix, *scaffold)
                if (
                    len(probe_input_ids) + stop_answer_max_new_tokens
                    > maximum_model_length
                ):
                    raise ValueError(
                        f"{state_key}: sufficiency probe would truncate context"
                    )
                prefix_hash = hashlib.sha256(
                    ",".join(str(value) for value in prefix).encode("ascii")
                ).hexdigest()
                jobs.append(
                    {
                        "search_index": search_index,
                        "probe_input_ids": probe_input_ids,
                        "sampling_params": dict(params),
                        "request_id": (
                            f"sprobe:{expected_snapshot_step}:{prompt_id}:"
                            f"{record.trajectory_id}:{search_index}"
                        ),
                        "metadata": {
                            "prompt_global_id": prompt_id,
                            "trajectory_id": str(record.trajectory_id),
                            "search_index": search_index,
                            "prefix_token_count": len(prefix),
                            "prefix_token_ids_sha256": prefix_hash,
                            "probe_input_token_count": len(probe_input_ids),
                            "candidate_rollout_policy_version": int(
                                expected_snapshot_step
                            ),
                            "gold_aliases": tuple(
                                str(value)
                                for value in record.metadata["gold_aliases"]
                            ),
                            "data_source": str(
                                record.metadata.get("data_source", "")
                            ),
                        },
                    }
                )
                estimated_group_tokens += (
                    len(probe_input_ids) + stop_answer_max_new_tokens
                )
        if jobs:
            group_rows.append((estimated_group_tokens, prompt_id, jobs))

    assignments: list[list[dict[str, Any]]] = [
        [] for _ in range(replica_count)
    ]
    loads = [0] * replica_count
    prompt_to_replica: dict[str, int] = {}
    for cost, prompt_id, jobs in sorted(
        group_rows,
        key=lambda row: (-row[0], row[1]),
    ):
        replica = min(range(replica_count), key=lambda rank: (loads[rank], rank))
        prompt_to_replica[prompt_id] = replica
        assignments[replica].extend(
            sorted(
                jobs,
                key=lambda job: (
                    int(job["search_index"]),
                    str(job["metadata"]["trajectory_id"]),
                ),
            )
        )
        loads[replica] += int(cost)

    for replica, jobs in enumerate(assignments):
        for job in jobs:
            prompt_id = str(job["metadata"]["prompt_global_id"])
            if prompt_to_replica[prompt_id] != replica:
                raise RuntimeError("Prompt affinity changed during probe routing")
    state_count = len(state_keys)
    return SufficiencyProbePlan(
        jobs_by_replica=tuple(tuple(values) for values in assignments),
        prompt_to_replica=prompt_to_replica,
        estimated_tokens_by_replica=tuple(loads),
        state_count=state_count,
        request_count=state_count,
        expected_completion_count=state_count,
    )


def _sufficiency_from_raw_probe(raw_probe: Mapping[str, Any]) -> bool:
    required = (
        "parser_success",
        "no_answer",
        "output_truncated",
        "alias_aware_exact",
    )
    for field_name in required:
        if not isinstance(raw_probe.get(field_name), bool):
            raise ValueError(f"Probe {field_name} must be a bool")
    return bool(
        raw_probe["alias_aware_exact"]
        and raw_probe["parser_success"]
        and not raw_probe["no_answer"]
        and not raw_probe["output_truncated"]
    )


def build_routed_answer_probe_plan(
    groups: Sequence[PromptGroup],
    *,
    probe_stage: str,
    scaffold_token_ids: Sequence[int],
    rollout_config: Mapping[str, Any],
    stop_answer_max_new_tokens: int,
    maximum_model_length: int,
    expected_snapshot_step: int,
    replica_count: int = 4,
) -> SufficiencyProbePlan:
    """Build selected-only pre/post Probes for Probe-routed Search credit."""

    stage = str(probe_stage)
    if stage not in {"pre", "post"}:
        raise ValueError(f"Unsupported routed Answer Probe stage: {stage}")
    if not groups:
        raise ValueError("Routed Answer probing requires selected Prompt groups")
    if int(replica_count) < 1:
        raise ValueError("Production routed probing requires at least one replica")
    scaffold = tuple(int(value) for value in scaffold_token_ids)
    if not scaffold:
        raise ValueError("Routed Answer Probe scaffold cannot be empty")
    max_tokens = int(stop_answer_max_new_tokens)
    model_length = int(maximum_model_length)
    if max_tokens != 500:
        raise ValueError("Routed Answer Probe max_tokens must equal 500")
    if model_length <= 0:
        raise ValueError("Routed Answer Probe model length must be positive")
    params = _sufficiency_sampling_params(
        rollout_config,
        stop_answer_max_new_tokens=max_tokens,
    )
    locked_params = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "n": 1,
        "max_tokens": 500,
        "stop": ["</answer>"],
    }

    group_rows: list[tuple[int, str, list[dict[str, Any]]]] = []
    seen_prompts: set[str] = set()
    state_keys: set[tuple[str, str, int]] = set()
    for group in groups:
        prompt_id = str(group.prompt_global_id)
        if prompt_id in seen_prompts:
            raise ValueError(f"Duplicate selected Prompt group: {prompt_id}")
        seen_prompts.add(prompt_id)
        jobs: list[dict[str, Any]] = []
        group_cost = 0
        for record in group.trajectories:
            if int(record.metadata.get("snapshot_step", -1)) != int(
                expected_snapshot_step
            ):
                raise ValueError(
                    f"{record.trajectory_id}: candidate policy version mismatch"
                )
            if not record.trajectory_system_valid:
                raise ValueError("System-invalid trajectory reached routed probing")
            record.validate()
            for turn in record.turns:
                if turn.turn_type is not TurnType.SEARCH:
                    continue
                search_index = int(turn.search_index)
                if stage == "post":
                    routed = record.metadata.get("routed_answer_probes", {})
                    stages = routed.get(search_index, routed.get(str(search_index)))
                    if not isinstance(stages, Mapping) or not isinstance(
                        stages.get("pre"), Mapping
                    ):
                        raise ValueError(
                            f"{record.trajectory_id}:{search_index}: pre Probe "
                            "metadata is missing before post planning"
                        )
                    raw_pre = stages["pre"]
                    sufficient = _sufficiency_from_raw_probe(raw_pre)
                    if raw_pre.get("sufficient_before_search") is not sufficient:
                        raise ValueError(
                            f"{record.trajectory_id}:{search_index}: pre Probe bool "
                            "does not match raw fields"
                        )
                    if is_budget_exhausted_terminal_search(record, search_index):
                        if stages.get("post") is not None:
                            raise ValueError(
                                f"{record.trajectory_id}:{search_index}: budget-"
                                "exhausted Search cannot carry a post Probe"
                            )
                        continue
                    if bool(
                        getattr(turn, "role_localized_gate_enabled", False)
                        and getattr(turn, "model_search_invalid", False)
                    ):
                        if stages.get("post") is not None:
                            raise ValueError(
                                f"{record.trajectory_id}:{search_index}: hard-invalid "
                                "Search cannot carry a post Probe"
                            )
                        continue
                    if sufficient:
                        continue
                state_key = (prompt_id, str(record.trajectory_id), search_index)
                if state_key in state_keys:
                    raise ValueError(f"Duplicate routed Probe state: {state_key}")
                state_keys.add(state_key)
                prefix = (
                    record.prefix_token_ids_before_search(search_index)
                    if stage == "pre"
                    else record.prefix_token_ids_after_search_observation(
                        search_index
                    )
                )
                probe_input_ids = (*prefix, *scaffold)
                if len(probe_input_ids) + max_tokens > model_length:
                    raise ValueError(
                        f"{state_key}: routed {stage} Probe would truncate context"
                    )
                prefix_hash = hashlib.sha256(
                    ",".join(str(value) for value in prefix).encode("ascii")
                ).hexdigest()
                jobs.append(
                    {
                        "search_index": search_index,
                        "probe_input_ids": probe_input_ids,
                        "sampling_params": dict(params),
                        "request_id": (
                            f"rprobe:{stage}:{expected_snapshot_step}:{prompt_id}:"
                            f"{record.trajectory_id}:{search_index}"
                        ),
                        "metadata": {
                            "prompt_global_id": prompt_id,
                            "trajectory_id": str(record.trajectory_id),
                            "search_index": search_index,
                            "probe_stage": stage,
                            "prefix_token_count": len(prefix),
                            "prefix_token_ids_sha256": prefix_hash,
                            "probe_input_token_count": len(probe_input_ids),
                            "candidate_rollout_policy_version": int(
                                expected_snapshot_step
                            ),
                            "gold_aliases": tuple(
                                str(value)
                                for value in record.metadata["gold_aliases"]
                            ),
                            "data_source": str(
                                record.metadata.get("data_source", "")
                            ),
                            **locked_params,
                        },
                    }
                )
                group_cost += len(probe_input_ids) + max_tokens
        if jobs:
            group_rows.append((group_cost, prompt_id, jobs))

    assignments: list[list[dict[str, Any]]] = [
        [] for _ in range(replica_count)
    ]
    loads = [0] * replica_count
    prompt_to_replica: dict[str, int] = {}
    for cost, prompt_id, jobs in sorted(
        group_rows,
        key=lambda row: (-row[0], row[1]),
    ):
        replica = min(range(replica_count), key=lambda rank: (loads[rank], rank))
        prompt_to_replica[prompt_id] = replica
        assignments[replica].extend(
            sorted(
                jobs,
                key=lambda job: (
                    int(job["search_index"]),
                    str(job["metadata"]["trajectory_id"]),
                ),
            )
        )
        loads[replica] += int(cost)
    for replica, jobs in enumerate(assignments):
        for job in jobs:
            prompt_id = str(job["metadata"]["prompt_global_id"])
            if prompt_to_replica[prompt_id] != replica:
                raise RuntimeError("Prompt affinity changed during routed probing")
    state_count = len(state_keys)
    return SufficiencyProbePlan(
        jobs_by_replica=tuple(tuple(values) for values in assignments),
        prompt_to_replica=prompt_to_replica,
        estimated_tokens_by_replica=tuple(loads),
        state_count=state_count,
        request_count=state_count,
        expected_completion_count=state_count,
    )


def build_stop_branch_plan(
    groups: Sequence[PromptGroup],
    *,
    scaffold_token_ids: Sequence[int],
    rollout_config: Mapping[str, Any],
    stop_answer_max_new_tokens: int,
    maximum_model_length: int,
    expected_snapshot_step: int,
    replica_count: int = 4,
) -> StopBranchPlan:
    """Build deterministic Prompt-affine jobs from original token provenance."""

    if not groups:
        raise ValueError("Stop branching requires selected Prompt groups")
    if int(replica_count) < 1:
        raise ValueError("Production Stop branching requires at least one replica")
    scaffold = tuple(int(value) for value in scaffold_token_ids)
    if not scaffold:
        raise ValueError("Stop scaffold token IDs cannot be empty")
    stop_answer_max_new_tokens = int(stop_answer_max_new_tokens)
    maximum_model_length = int(maximum_model_length)
    if stop_answer_max_new_tokens <= 0 or maximum_model_length <= 0:
        raise ValueError("Stop/model length limits must be positive")
    params = _sampling_params(
        rollout_config,
        stop_answer_max_new_tokens=stop_answer_max_new_tokens,
    )

    group_rows: list[tuple[int, str, list[dict[str, Any]]]] = []
    seen_prompts: set[str] = set()
    state_keys: set[tuple[str, str, int]] = set()
    for group in groups:
        prompt_id = str(group.prompt_global_id)
        if prompt_id in seen_prompts:
            raise ValueError(f"Duplicate selected Prompt group: {prompt_id}")
        seen_prompts.add(prompt_id)
        jobs: list[dict[str, Any]] = []
        estimated_group_tokens = 0
        for record in group.trajectories:
            if int(record.metadata.get("snapshot_step", -1)) != int(
                expected_snapshot_step
            ):
                raise ValueError(
                    f"{record.trajectory_id}: candidate policy version mismatch"
                )
            if not record.trajectory_system_valid:
                raise ValueError("System-invalid trajectory reached Stop branching")
            record.validate()
            for turn in record.turns:
                if (
                    turn.turn_type is not TurnType.SEARCH
                    or not turn.policy_credit_eligible
                ):
                    continue
                search_index = int(turn.search_index)
                state_key = (
                    prompt_id,
                    str(record.trajectory_id),
                    search_index,
                )
                if state_key in state_keys:
                    raise ValueError(f"Duplicate Stop state: {state_key}")
                state_keys.add(state_key)
                prefix = record.prefix_token_ids_before_search(search_index)
                stop_input_ids = (*prefix, *scaffold)
                if (
                    len(stop_input_ids) + stop_answer_max_new_tokens
                    > maximum_model_length
                ):
                    raise ValueError(
                        f"{state_key}: Stop branch would truncate context"
                    )
                prefix_hash = hashlib.sha256(
                    ",".join(str(value) for value in prefix).encode("ascii")
                ).hexdigest()
                jobs.append(
                    {
                        "search_index": search_index,
                        "stop_input_ids": stop_input_ids,
                        "sampling_params": dict(params),
                        "request_id": (
                            f"sc:{expected_snapshot_step}:{prompt_id}:"
                            f"{record.trajectory_id}:{search_index}"
                        ),
                        "metadata": {
                            "prompt_global_id": prompt_id,
                            "trajectory_id": str(record.trajectory_id),
                            "search_index": search_index,
                            "prefix_token_count": len(prefix),
                            "prefix_token_ids_sha256": prefix_hash,
                            "stop_input_token_count": len(stop_input_ids),
                            "candidate_rollout_policy_version": int(
                                expected_snapshot_step
                            ),
                            "gold_aliases": tuple(
                                str(value)
                                for value in record.metadata["gold_aliases"]
                            ),
                            "data_source": str(
                                record.metadata.get("data_source", "")
                            ),
                        },
                    }
                )
                estimated_group_tokens += (
                    len(stop_input_ids) + 2 * stop_answer_max_new_tokens
                )
        if jobs:
            group_rows.append((estimated_group_tokens, prompt_id, jobs))

    if not group_rows:
        raise ValueError("Selected Prompt groups contain no trainable Search state")
    assignments: list[list[dict[str, Any]]] = [
        [] for _ in range(replica_count)
    ]
    loads = [0] * replica_count
    prompt_to_replica: dict[str, int] = {}
    for cost, prompt_id, jobs in sorted(
        group_rows,
        key=lambda row: (-row[0], row[1]),
    ):
        replica = min(range(replica_count), key=lambda rank: (loads[rank], rank))
        prompt_to_replica[prompt_id] = replica
        assignments[replica].extend(
            sorted(
                jobs,
                key=lambda job: (
                    int(job["search_index"]),
                    str(job["metadata"]["trajectory_id"]),
                ),
            )
        )
        loads[replica] += int(cost)

    for replica, jobs in enumerate(assignments):
        for job in jobs:
            prompt_id = str(job["metadata"]["prompt_global_id"])
            if prompt_to_replica[prompt_id] != replica:
                raise RuntimeError("Prompt affinity changed during Stop routing")
    state_count = len(state_keys)
    return StopBranchPlan(
        jobs_by_replica=tuple(tuple(values) for values in assignments),
        prompt_to_replica=prompt_to_replica,
        estimated_tokens_by_replica=tuple(loads),
        state_count=state_count,
        request_count=state_count,
        expected_completion_count=2 * state_count,
    )


def attach_stop_branch_rewards(
    groups: Sequence[PromptGroup],
    generated_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    expected_snapshot_step: int,
    expected_source_checksum: str,
) -> dict[str, Any]:
    """Validate generated/scored Stop pairs and attach detached reward triples."""

    replica_count = max(
        1,
        max(
            (int(row["assigned_replica"]) for row in generated_rows),
            default=0,
        )
        + 1,
    )
    records = {
        (str(record.prompt_global_id), str(record.trajectory_id)): record
        for group in groups
        for record in group.trajectories
    }
    generated_by_state: dict[
        tuple[str, str, int],
        Mapping[str, Any],
    ] = {}
    completion_keys: set[tuple[str, str, int, int]] = set()
    completion_count = 0
    decode_tokens = 0
    prefill_tokens = 0
    cached_prompt_tokens = 0
    generation_seconds = 0.0
    per_replica_jobs = [0] * replica_count
    per_replica_tokens = [0] * replica_count
    truncated_completion_count = 0
    for row in generated_rows:
        state_key = (
            str(row["prompt_global_id"]),
            str(row["trajectory_id"]),
            int(row["search_index"]),
        )
        if state_key in generated_by_state:
            raise ValueError(f"{state_key}: duplicate generated Stop state")
        if int(row["completion_count"]) != 2 or len(row["completions"]) != 2:
            raise ValueError(f"{state_key}: Stop completion count is not two")
        if int(row["snapshot_step"]) != int(expected_snapshot_step):
            raise ValueError(f"{state_key}: Stop policy version mismatch")
        if str(row["source_checksum"]) != str(expected_source_checksum):
            raise ValueError(f"{state_key}: Stop policy checksum mismatch")
        if row.get("automatic_prefix_caching") is not True:
            raise ValueError(f"{state_key}: Automatic Prefix Caching is disabled")
        generated_by_state[state_key] = row
        replica = int(row["assigned_replica"])
        if replica not in range(replica_count):
            raise ValueError(f"{state_key}: invalid Stop replica {replica}")
        per_replica_jobs[replica] += 1
        per_replica_tokens[replica] += int(row["decode_tokens"])
        completion_count += int(row["completion_count"])
        decode_tokens += int(row["decode_tokens"])
        prefill_tokens += int(row["prompt_tokens"])
        cached_prompt_tokens += int(row.get("cached_prompt_tokens", 0))
        generation_seconds += float(row["generation_seconds"])
        for completion in row["completions"]:
            truncated_completion_count += int(
                str(completion.get("finish_reason", "")).lower() == "length"
            )
            completion_key = (*state_key, int(completion["sample_index"]))
            if completion_key in completion_keys:
                raise ValueError(f"{completion_key}: duplicate Stop completion")
            completion_keys.add(completion_key)

    scored_by_completion: dict[
        tuple[str, str, int, int],
        Mapping[str, Any],
    ] = {}
    for row in scored_rows:
        key = (
            str(row["prompt_global_id"]),
            str(row["trajectory_id"]),
            int(row["search_index"]),
            int(row["sample_index"]),
        )
        if key in scored_by_completion:
            raise ValueError(f"{key}: duplicate Stop reward")
        reward = float(row["task_outcome"])
        if not math.isfinite(reward):
            raise ValueError(f"{key}: Stop reward is non-finite")
        if str(row["scorer_version"]) != PRODUCTION_TASK_SCORER_VERSION:
            raise ValueError(f"{key}: Stop scorer version mismatch")
        scored_by_completion[key] = row
    if set(scored_by_completion) != completion_keys:
        raise ValueError("Generated Stop completions and rewards do not align")

    stop_text_equal = 0
    parse_invalid_count = 0
    for state_key, generated in generated_by_state.items():
        record_key = (state_key[0], state_key[1])
        if record_key not in records:
            raise ValueError(f"{state_key}: Stop result has no trajectory")
        record = records[record_key]
        search_index = state_key[2]
        if search_index not in record.policy_credit_eligibility_by_search_index:
            raise ValueError(f"{state_key}: Stop result has no real Search")
        expected_prefix = record.prefix_token_ids_before_search(search_index)
        observed_prefix_hash = str(generated["prefix_token_ids_sha256"])
        expected_prefix_hash = hashlib.sha256(
            ",".join(str(value) for value in expected_prefix).encode("ascii")
        ).hexdigest()
        if observed_prefix_hash != expected_prefix_hash:
            raise ValueError(f"{state_key}: pre-Search prefix provenance changed")
        exact_ig_version = int(record.metadata["reward_snapshot_step"])
        candidate_version = int(record.metadata["snapshot_step"])
        if {exact_ig_version, candidate_version} != {
            int(expected_snapshot_step)
        }:
            raise ValueError(f"{state_key}: upstream policy versions mismatch")
        continue_reward = float(record.task_outcome)
        if not math.isfinite(continue_reward):
            raise ValueError(f"{state_key}: Continue reward is non-finite")
        continue_scorer = str(record.metadata["task_scorer_version"])
        if continue_scorer != PRODUCTION_TASK_SCORER_VERSION:
            raise ValueError(f"{state_key}: Continue scorer version mismatch")
        stop_1 = scored_by_completion[(*state_key, 0)]
        stop_2 = scored_by_completion[(*state_key, 1)]
        parse_invalid_count += int(
            not bool(stop_1.get("terminal_answer_valid", False))
        )
        parse_invalid_count += int(
            not bool(stop_2.get("terminal_answer_valid", False))
        )
        texts = [str(value["text"]) for value in generated["completions"]]
        stop_text_equal += int(texts[0] == texts[1])
        record.metadata.setdefault("stop_continue_probes", {})[search_index] = {
            "prompt_global_id": state_key[0],
            "trajectory_id": state_key[1],
            "search_index": search_index,
            "continue_reward": continue_reward,
            "stop_reward_1": float(stop_1["task_outcome"]),
            "stop_reward_2": float(stop_2["task_outcome"]),
            "continue_scorer_version": continue_scorer,
            "stop_scorer_version_1": str(stop_1["scorer_version"]),
            "stop_scorer_version_2": str(stop_2["scorer_version"]),
            "candidate_rollout_policy_version": candidate_version,
            "exact_ig_policy_version": exact_ig_version,
            "stop_branch_policy_version": int(generated["snapshot_step"]),
            "old_logprob_policy_version": int(expected_snapshot_step),
            "prefix_provenance_valid": True,
            "context_truncated": False,
            "completion_count": 2,
            "detached": True,
            "stop_completion_text_1": texts[0],
            "stop_completion_text_2": texts[1],
            "assigned_replica": int(generated["assigned_replica"]),
            "prefix_token_ids_sha256": expected_prefix_hash,
        }

    expected_states = {
        (
            str(record.prompt_global_id),
            str(record.trajectory_id),
            int(turn.search_index),
        )
        for group in groups
        for record in group.trajectories
        for turn in record.turns
        if turn.turn_type is TurnType.SEARCH and turn.policy_credit_eligible
    }
    if set(generated_by_state) != expected_states:
        raise ValueError("Stop results do not cover every selected Search state")
    state_count = len(expected_states)
    if completion_count != 2 * state_count:
        raise ValueError("Stop completion total is not twice the state count")
    return {
        "sc/request_count": state_count,
        "sc/completion_count": completion_count,
        "sc/prefill_tokens": prefill_tokens,
        "sc/cached_prompt_tokens": cached_prompt_tokens,
        "sc/decode_tokens": decode_tokens,
        "sc/generation_seconds": generation_seconds,
        "sc/cache_hit_rate": (
            cached_prompt_tokens / prefill_tokens if prefill_tokens else 0.0
        ),
        "sc/per_replica_jobs": per_replica_jobs,
        "sc/per_replica_tokens": per_replica_tokens,
        "sc/stop_text_exact_match_rate": stop_text_equal / state_count,
        "sc/parse_invalid_rate": (
            parse_invalid_count / completion_count
            if completion_count
            else 0.0
        ),
        "sc/truncated_rate": (
            truncated_completion_count / completion_count
            if completion_count
            else 0.0
        ),
        "sc/policy_version_match": True,
        "sc/parameter_checksum_before": str(expected_source_checksum),
        "sc/parameter_checksum_after": str(expected_source_checksum),
    }


def attach_sufficiency_probe_results(
    groups: Sequence[PromptGroup],
    generated_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    expected_snapshot_step: int,
    expected_source_checksum: str,
) -> dict[str, Any]:
    """Attach one validated deterministic sufficiency bit per Search state."""

    replica_count = max(
        1,
        max(
            (int(row["assigned_replica"]) for row in generated_rows),
            default=0,
        )
        + 1,
    )
    records = {
        (str(record.prompt_global_id), str(record.trajectory_id)): record
        for group in groups
        for record in group.trajectories
    }
    generated_by_state: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    completion_count = 0
    decode_tokens = 0
    prefill_tokens = 0
    cached_prompt_tokens = 0
    generation_seconds = 0.0
    per_replica_jobs = [0] * replica_count
    per_replica_tokens = [0] * replica_count
    truncated_count = 0
    for row in generated_rows:
        state_key = (
            str(row["prompt_global_id"]),
            str(row["trajectory_id"]),
            int(row["search_index"]),
        )
        if state_key in generated_by_state:
            raise ValueError(f"{state_key}: duplicate generated sufficiency state")
        if int(row["completion_count"]) != 1 or len(row["completions"]) != 1:
            raise ValueError(f"{state_key}: sufficiency completion count is not one")
        if int(row["snapshot_step"]) != int(expected_snapshot_step):
            raise ValueError(f"{state_key}: sufficiency policy version mismatch")
        if str(row["source_checksum"]) != str(expected_source_checksum):
            raise ValueError(f"{state_key}: sufficiency policy checksum mismatch")
        if row.get("automatic_prefix_caching") is not True:
            raise ValueError(f"{state_key}: Automatic Prefix Caching is disabled")
        generated_by_state[state_key] = row
        replica = int(row["assigned_replica"])
        if replica not in range(replica_count):
            raise ValueError(f"{state_key}: invalid sufficiency replica {replica}")
        per_replica_jobs[replica] += 1
        per_replica_tokens[replica] += int(row["decode_tokens"])
        completion_count += 1
        decode_tokens += int(row["decode_tokens"])
        prefill_tokens += int(row["prompt_tokens"])
        cached_prompt_tokens += int(row.get("cached_prompt_tokens", 0))
        generation_seconds += float(row["generation_seconds"])
        completion = row["completions"][0]
        truncated_count += int(
            str(completion.get("finish_reason", "")).lower() == "length"
        )

    scored_by_state: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in scored_rows:
        state_key = (
            str(row["prompt_global_id"]),
            str(row["trajectory_id"]),
            int(row["search_index"]),
        )
        if state_key in scored_by_state:
            raise ValueError(f"{state_key}: duplicate sufficiency score")
        if str(row["scorer_version"]) != SUFFICIENCY_EXACT_SCORER_VERSION:
            raise ValueError(f"{state_key}: sufficiency scorer version mismatch")
        if not math.isfinite(float(row["partial_task_reward_shadow"])):
            raise ValueError(f"{state_key}: sufficiency shadow reward is non-finite")
        scored_by_state[state_key] = row
    if set(scored_by_state) != set(generated_by_state):
        raise ValueError("Generated sufficiency completions and scores do not align")

    sufficient_count = 0
    exact_count = 0
    parser_valid_count = 0
    partial_shadow_values: list[float] = []
    for state_key, generated in generated_by_state.items():
        record_key = (state_key[0], state_key[1])
        if record_key not in records:
            raise ValueError(f"{state_key}: sufficiency result has no trajectory")
        record = records[record_key]
        search_index = state_key[2]
        if search_index not in record.policy_credit_eligibility_by_search_index:
            raise ValueError(f"{state_key}: sufficiency result has no real Search")
        expected_prefix = record.prefix_token_ids_before_search(search_index)
        expected_prefix_hash = hashlib.sha256(
            ",".join(str(value) for value in expected_prefix).encode("ascii")
        ).hexdigest()
        if str(generated["prefix_token_ids_sha256"]) != expected_prefix_hash:
            raise ValueError(f"{state_key}: pre-Search prefix provenance changed")
        exact_ig_version = int(record.metadata["reward_snapshot_step"])
        candidate_version = int(record.metadata["snapshot_step"])
        if {exact_ig_version, candidate_version} != {
            int(expected_snapshot_step)
        }:
            raise ValueError(f"{state_key}: upstream policy versions mismatch")
        scored = scored_by_state[state_key]
        completion = generated["completions"][0]
        truncated = str(completion.get("finish_reason", "")).lower() == "length"
        if bool(scored["truncated"]) != truncated:
            raise ValueError(f"{state_key}: truncation metadata mismatch")
        sufficient = bool(scored["sufficient_before_search"])
        sufficient_count += int(sufficient)
        exact_count += int(bool(scored["alias_exact_match"]))
        parser_valid_count += int(bool(scored["terminal_answer_valid"]))
        partial_shadow_values.append(float(scored["partial_task_reward_shadow"]))
        record.metadata.setdefault("sufficiency_probes", {})[search_index] = {
            "prompt_global_id": state_key[0],
            "trajectory_id": state_key[1],
            "search_index": search_index,
            "sufficient_before_search": sufficient,
            "alias_exact_match": bool(scored["alias_exact_match"]),
            "partial_task_reward_shadow": float(
                scored["partial_task_reward_shadow"]
            ),
            "terminal_answer_valid": bool(scored["terminal_answer_valid"]),
            "parser_status": str(scored["parser_status"]),
            "parser_error_type": scored.get("parser_error_type"),
            "parsed_answer": scored.get("parsed_answer"),
            "truncated": truncated,
            "completion_text": str(completion["text"]),
            "completion_count": 1,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "candidate_rollout_policy_version": candidate_version,
            "exact_ig_policy_version": exact_ig_version,
            "sufficiency_probe_policy_version": int(generated["snapshot_step"]),
            "old_logprob_policy_version": int(expected_snapshot_step),
            "prefix_provenance_valid": True,
            "context_truncated": False,
            "detached": True,
            "scorer_version": str(scored["scorer_version"]),
            "task_scorer_version": str(scored["task_scorer_version"]),
            "assigned_replica": int(generated["assigned_replica"]),
            "prefix_token_ids_sha256": expected_prefix_hash,
        }

    expected_states = {
        (
            str(record.prompt_global_id),
            str(record.trajectory_id),
            int(turn.search_index),
        )
        for group in groups
        for record in group.trajectories
        for turn in record.turns
        if turn.turn_type is TurnType.SEARCH and turn.policy_credit_eligible
    }
    if set(generated_by_state) != expected_states:
        raise ValueError(
            "Sufficiency results do not cover every selected Search state"
        )
    state_count = len(expected_states)
    if completion_count != state_count:
        raise ValueError("Sufficiency completion total is not the state count")
    partial_mean = (
        sum(partial_shadow_values) / len(partial_shadow_values)
        if partial_shadow_values
        else 0.0
    )
    return {
        "s_probe/state_count": state_count,
        "s_probe/request_count": state_count,
        "s_probe/completion_count": completion_count,
        "s_probe/sufficient_count": sufficient_count,
        "s_probe/sufficient_rate": (
            sufficient_count / state_count if state_count else 0.0
        ),
        "s_probe/alias_exact_rate": (
            exact_count / state_count if state_count else 0.0
        ),
        "s_probe/parser_valid_rate": (
            parser_valid_count / state_count if state_count else 0.0
        ),
        "s_probe/truncated_rate": (
            truncated_count / state_count if state_count else 0.0
        ),
        "s_probe/partial_reward_shadow_mean": partial_mean,
        "s_probe/prefill_tokens": prefill_tokens,
        "s_probe/cached_prompt_tokens": cached_prompt_tokens,
        "s_probe/decode_tokens": decode_tokens,
        "s_probe/generation_seconds": generation_seconds,
        "s_probe/cache_hit_rate": (
            cached_prompt_tokens / prefill_tokens if prefill_tokens else 0.0
        ),
        "s_probe/per_replica_jobs": per_replica_jobs,
        "s_probe/per_replica_tokens": per_replica_tokens,
        "s_probe/policy_version_match": True,
        "s_probe/parameter_checksum_before": str(expected_source_checksum),
        "s_probe/parameter_checksum_after": str(expected_source_checksum),
    }


def attach_routed_answer_probe_results(
    groups: Sequence[PromptGroup],
    generated_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    probe_stage: str,
    expected_snapshot_step: int,
    expected_source_checksum: str,
) -> dict[str, Any]:
    """Persist validated raw pre/post Probe evidence for advantage routing."""

    stage = str(probe_stage)
    if stage not in {"pre", "post"}:
        raise ValueError(f"Unsupported routed Answer Probe stage: {stage}")
    replica_count = max(
        1,
        max(
            (int(row["assigned_replica"]) for row in generated_rows),
            default=0,
        )
        + 1,
    )
    records = {
        (str(record.prompt_global_id), str(record.trajectory_id)): record
        for group in groups
        for record in group.trajectories
    }
    generated_by_state: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    completion_count = 0
    prefill_tokens = 0
    cached_prompt_tokens = 0
    decode_tokens = 0
    generation_seconds = 0.0
    per_replica_jobs = [0] * replica_count
    per_replica_tokens = [0] * replica_count
    for row in generated_rows:
        state_key = (
            str(row["prompt_global_id"]),
            str(row["trajectory_id"]),
            int(row["search_index"]),
        )
        if str(row.get("probe_stage")) != stage:
            raise ValueError(f"{state_key}: routed Probe stage mismatch")
        if state_key in generated_by_state:
            raise ValueError(f"{state_key}: duplicate routed {stage} Probe")
        if int(row.get("completion_count", -1)) != 1 or len(
            row.get("completions", ())
        ) != 1:
            raise ValueError(f"{state_key}: routed Probe completion count is not one")
        if int(row.get("snapshot_step", -1)) != int(expected_snapshot_step):
            raise ValueError(f"{state_key}: routed Probe policy version mismatch")
        if str(row.get("source_checksum")) != str(expected_source_checksum):
            raise ValueError(f"{state_key}: routed Probe checksum mismatch")
        if row.get("automatic_prefix_caching") is not True:
            raise ValueError(f"{state_key}: routed Probe APC is disabled")
        locked = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "n": 1,
            "max_tokens": 500,
            "stop": ["</answer>"],
        }
        for field_name, expected in locked.items():
            actual = row.get(field_name)
            if field_name == "stop" and isinstance(actual, (tuple, list)):
                actual = list(actual)
            if actual != expected:
                raise ValueError(
                    f"{state_key}: routed Probe {field_name} is not locked"
                )
        generated_by_state[state_key] = row
        replica = int(row["assigned_replica"])
        if replica not in range(replica_count):
            raise ValueError(f"{state_key}: invalid routed Probe replica")
        per_replica_jobs[replica] += 1
        per_replica_tokens[replica] += int(row["decode_tokens"])
        completion_count += 1
        prefill_tokens += int(row["prompt_tokens"])
        cached_prompt_tokens += int(row.get("cached_prompt_tokens", 0))
        decode_tokens += int(row["decode_tokens"])
        generation_seconds += float(row["generation_seconds"])

    scored_by_state: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in scored_rows:
        state_key = (
            str(row["prompt_global_id"]),
            str(row["trajectory_id"]),
            int(row["search_index"]),
        )
        if str(row.get("probe_stage")) != stage:
            raise ValueError(f"{state_key}: routed Probe score stage mismatch")
        if state_key in scored_by_state:
            raise ValueError(f"{state_key}: duplicate routed Probe score")
        if str(row.get("scorer_version")) != SUFFICIENCY_EXACT_SCORER_VERSION:
            raise ValueError(f"{state_key}: routed Probe scorer mismatch")
        if not math.isfinite(float(row.get("raw_task_reward", float("nan")))):
            raise ValueError(f"{state_key}: routed Probe reward is non-finite")
        scored_by_state[state_key] = row
    if set(scored_by_state) != set(generated_by_state):
        raise ValueError("Generated routed Probes and scored rows do not align")

    expected_states: set[tuple[str, str, int]] = set()
    budget_exhausted_states: set[tuple[str, str, int]] = set()
    hard_invalid_states: set[tuple[str, str, int]] = set()
    for group in groups:
        for record in group.trajectories:
            for turn in record.turns:
                if turn.turn_type is not TurnType.SEARCH:
                    continue
                search_index = int(turn.search_index)
                state_key = (
                    str(record.prompt_global_id),
                    str(record.trajectory_id),
                    search_index,
                )
                if stage == "post":
                    routed = record.metadata.get("routed_answer_probes", {})
                    stages = routed.get(search_index, routed.get(str(search_index)))
                    if not isinstance(stages, Mapping) or not isinstance(
                        stages.get("pre"), Mapping
                    ):
                        raise ValueError(f"{state_key}: pre Probe metadata is missing")
                    if is_budget_exhausted_terminal_search(record, search_index):
                        if stages.get("post") is not None:
                            raise ValueError(
                                f"{state_key}: budget-exhausted Search cannot carry "
                                "a post Probe"
                            )
                        budget_exhausted_states.add(state_key)
                        continue
                    if bool(
                        getattr(turn, "role_localized_gate_enabled", False)
                        and getattr(turn, "model_search_invalid", False)
                    ):
                        if stages.get("post") is not None:
                            raise ValueError(
                                f"{state_key}: hard-invalid Search cannot carry a post Probe"
                            )
                        hard_invalid_states.add(state_key)
                        continue
                    if _sufficiency_from_raw_probe(stages["pre"]):
                        continue
                expected_states.add(state_key)
    if set(generated_by_state) != expected_states:
        raise ValueError(
            f"Routed {stage} Probe coverage does not match required Search states"
        )

    sufficient_count = 0
    reward_values: list[float] = []
    truncated_count = 0
    for state_key, generated in generated_by_state.items():
        record = records.get((state_key[0], state_key[1]))
        if record is None:
            raise ValueError(f"{state_key}: routed Probe has no trajectory")
        search_index = state_key[2]
        expected_prefix = (
            record.prefix_token_ids_before_search(search_index)
            if stage == "pre"
            else record.prefix_token_ids_after_search_observation(search_index)
        )
        prefix_hash = hashlib.sha256(
            ",".join(str(value) for value in expected_prefix).encode("ascii")
        ).hexdigest()
        if str(generated["prefix_token_ids_sha256"]) != prefix_hash:
            raise ValueError(f"{state_key}: routed Probe prefix provenance changed")
        candidate_version = int(record.metadata["snapshot_step"])
        exact_ig_version = int(record.metadata["reward_snapshot_step"])
        if {candidate_version, exact_ig_version} != {int(expected_snapshot_step)}:
            raise ValueError(f"{state_key}: routed Probe upstream versions mismatch")
        completion = generated["completions"][0]
        truncated = str(completion.get("finish_reason", "")).lower() == "length"
        scored = scored_by_state[state_key]
        if bool(scored["output_truncated"]) != truncated:
            raise ValueError(f"{state_key}: routed Probe truncation mismatch")
        raw_fields = {
            "raw_answer_text": str(completion["text"]),
            "parser_success": bool(scored["parser_success"]),
            "no_answer": bool(scored["no_answer"]),
            "output_truncated": truncated,
            "alias_aware_exact": bool(scored["alias_aware_exact"]),
            "raw_task_reward": float(scored["raw_task_reward"]),
        }
        sufficient = _sufficiency_from_raw_probe(raw_fields)
        sufficient_name = (
            "sufficient_before_search"
            if stage == "pre"
            else "sufficient_after_search"
        )
        raw_fields[sufficient_name] = sufficient
        sufficient_count += int(sufficient)
        reward_values.append(float(scored["raw_task_reward"]))
        truncated_count += int(truncated)
        record.metadata.setdefault("routed_answer_probes", {}).setdefault(
            search_index,
            {},
        )[stage] = {
            "prompt_global_id": state_key[0],
            "trajectory_id": state_key[1],
            "search_index": search_index,
            "probe_stage": stage,
            **raw_fields,
            "policy_version": int(generated["snapshot_step"]),
            "scorer_version": str(scored["scorer_version"]),
            "task_scorer_version": str(scored["task_scorer_version"]),
            "prefix_provenance_valid": True,
            "detached": True,
            "context_truncated": False,
            "completion_count": 1,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "n": 1,
            "max_tokens": 500,
            "stop": ["</answer>"],
            "candidate_rollout_policy_version": candidate_version,
            "exact_ig_policy_version": exact_ig_version,
            "probe_policy_version": int(generated["snapshot_step"]),
            "old_logprob_policy_version": int(expected_snapshot_step),
            "assigned_replica": int(generated["assigned_replica"]),
            "prefix_token_ids_sha256": prefix_hash,
            "parser_status": str(scored["parser_status"]),
            "parser_error_type": scored.get("parser_error_type"),
            "parsed_answer": scored.get("parsed_answer"),
        }
        if stage == "pre":
            matching = [
                (position, turn)
                for position, turn in enumerate(record.turns)
                if turn.turn_type is TurnType.SEARCH
                and int(turn.search_index) == int(search_index)
            ]
            if len(matching) != 1:
                raise ValueError(f"{state_key}: Search turn coverage is invalid")
            turn_position, turn = matching[0]
            if getattr(turn, "role_localized_gate_enabled", False):
                branch = classify_role_localized_search_branch(
                    retrieval_budget_exhausted=bool(
                        turn.retrieval_budget_exhausted
                    ),
                    model_search_invalid=bool(turn.model_search_invalid),
                    sufficient_before_search=bool(sufficient),
                    retriever_executed=bool(turn.retriever_executed),
                    no_new_observation=turn.no_new_observation,
                )
                main_eligible = branch in {
                    ROLE_LOCALIZED_BRANCH_N_SOFT,
                    ROLE_LOCALIZED_BRANCH_NORMAL,
                }
                record.turns[turn_position] = replace(
                    turn,
                    branch_type=branch,
                    main_credit_eligible=main_eligible,
                )
                record.metadata.setdefault(
                    "role_localized_search_branches", {}
                )[search_index] = branch
                record.validate()

    state_count = len(expected_states)
    if completion_count != state_count:
        raise ValueError("Routed Probe completion total is invalid")
    prefix = f"answer_probe/{stage}"
    reward_array = [float(value) for value in reward_values]
    return {
        f"{prefix}/state_count": state_count,
        f"{prefix}/request_count": state_count,
        f"{prefix}/completion_count": completion_count,
        f"{prefix}/sufficient_count": sufficient_count,
        f"{prefix}/sufficient_rate": (
            sufficient_count / state_count if state_count else 0.0
        ),
        f"{prefix}/truncated_count": truncated_count,
        f"{prefix}/raw_task_reward_mean": (
            sum(reward_array) / len(reward_array) if reward_array else 0.0
        ),
        f"{prefix}/prefill_tokens": prefill_tokens,
        f"{prefix}/cached_prompt_tokens": cached_prompt_tokens,
        f"{prefix}/decode_tokens": decode_tokens,
        f"{prefix}/generation_seconds": generation_seconds,
        f"{prefix}/cache_hit_rate": (
            cached_prompt_tokens / prefill_tokens if prefill_tokens else 0.0
        ),
        f"{prefix}/per_replica_jobs": per_replica_jobs,
        f"{prefix}/per_replica_tokens": per_replica_tokens,
        f"{prefix}/policy_version_match": True,
        f"{prefix}/budget_exhausted_skipped_count": len(
            budget_exhausted_states
        ),
        f"{prefix}/budget_exhausted_post_probe_count": len(
            set(generated_by_state) & budget_exhausted_states
        ),
        f"{prefix}/hard_invalid_skipped_count": len(hard_invalid_states),
        f"{prefix}/hard_invalid_post_probe_count": len(
            set(generated_by_state) & hard_invalid_states
        ),
        f"{prefix}/parameter_checksum_before": str(expected_source_checksum),
        f"{prefix}/parameter_checksum_after": str(expected_source_checksum),
    }
