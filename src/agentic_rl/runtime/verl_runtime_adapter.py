from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from agentic_rl.checkpoint import (
    AtomicCheckpointCommitter,
    CheckpointMetadata,
    release_file_cache,
)
from agentic_rl.exact_ig.target_schema import (
    EXACT_IG_VERSION,
    PRODUCTION_PRECISION_MODE,
    assert_exact_ig_checkpoint_compatible,
)
from agentic_rl.exact_ig.task_builder import (
    assert_same_prompt_target_consistency,
)
from agentic_rl.checkpoint.state_schema import ChannelCheckpointState
from agentic_rl.controller.attempt_state import SnapshotVersions, TrainingState
from agentic_rl.controller.update_controller import StrictAttemptController
from agentic_rl.metrics.runtime_records import (
    build_behavior_record,
    build_channel_records,
    build_prompt_records,
    build_trajectory_and_turn_records,
)
from agentic_rl.outcome.workers import PRODUCTION_TASK_SCORER_VERSION
from agentic_rl.policy.gate_gradient_calibration import (
    BatchGradientProfile,
    calibrate_role_localized_gate_lambdas,
    write_immutable_calibration_manifest,
)
from agentic_rl.advantage.a2tgpo import (
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
    SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
)
from agentic_rl.advantage.mica_ig import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
)
from agentic_rl.retriever.client import AsyncHybridRetrieverClient
from agentic_rl.rollout.search_role_provenance import (
    ROLE_LOCALIZED_BRANCH_N_BUDGET,
    ROLE_LOCALIZED_BRANCH_N_INVALID,
    ROLE_LOCALIZED_BRANCH_N_SOFT,
    ROLE_LOCALIZED_BRANCH_NORMAL,
    ROLE_LOCALIZED_BRANCH_S_BEFORE,
    classify_role_localized_search_branch,
)
from agentic_rl.rollout.token_provenance import (
    assert_environment_information_masked,
)
from agentic_rl.rollout.trajectory_schema import (
    TokenSource,
    TrajectoryRecord,
    TurnRecord,
    TurnType,
    is_budget_exhausted_terminal_search,
)
from agentic_rl.runtime.learner_batch import (
    build_synchronized_microbatch_rounds,
    pack_prompt_groups_by_action_tokens,
    prepare_selected_trajectories,
)
from agentic_rl.runtime.postprocess import (
    attach_exact_ig,
    prompt_groups_from_records,
    trajectory_record_from_extra,
)
from agentic_rl.runtime.stop_branching import (
    attach_routed_answer_probe_results,
    attach_sufficiency_probe_results,
    attach_stop_branch_rewards,
    build_routed_answer_probe_plan,
    build_sufficiency_probe_plan,
    build_stop_branch_plan,
    tokenize_stop_scaffold,
)
from agentic_rl.selection.candidate_pool import (
    ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
    ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE,
    CandidatePool,
    PromptGroup,
    SelectionDecision,
)
from agentic_rl.selection.channel_scale import ChannelScaleState

from .ray_topology import RuntimeRayTopology
from .fixed_eval import create_or_validate_eval_manifest_from_config, load_eval_rows
from .formal_state import atomic_write_json, atomic_write_text, enqueue_eval
from .pretrain_controls import exercise_forced_skip_transaction
from .resource_guard import (
    read_runtime_resource_snapshot,
    validate_checkpoint_runtime_budget,
)
from .verl_config import assert_formal_hyperparameters_approved


class RuntimeGateError(RuntimeError):
    pass


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    file_path = Path(path)
    try:
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    finally:
        release_file_cache(file_path)
    return digest.hexdigest()


def _sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if root.is_file():
        return _sha256_file(root)
    digest = hashlib.sha256()
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        try:
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
        finally:
            release_file_cache(item)
    return digest.hexdigest()


def _first_valid_token_id(value: Any) -> int | None:
    candidates = value if isinstance(value, list) else [value]
    for candidate in candidates:
        if isinstance(candidate, bool) or candidate is None:
            continue
        try:
            token_id = int(candidate)
        except (TypeError, ValueError):
            continue
        if token_id >= 0:
            return token_id
    return None


def _resolve_pad_token_id(model_path: str | Path) -> int:
    root = Path(model_path)
    sources = (
        ("tokenizer_config.json", "pad_token_id"),
        ("generation_config.json", "pad_token_id"),
        ("config.json", "pad_token_id"),
        ("tokenizer_config.json", "eos_token_id"),
        ("generation_config.json", "eos_token_id"),
        ("config.json", "eos_token_id"),
    )
    loaded: dict[str, Mapping[str, Any]] = {}
    for filename, key in sources:
        path = root / filename
        if filename not in loaded:
            loaded[filename] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.is_file()
                else {}
            )
        token_id = _first_valid_token_id(loaded[filename].get(key))
        if token_id is not None:
            return token_id
    raise RuntimeGateError(
        f"Model has no numeric pad/eos token id in tokenizer/model configs: {root}"
    )


def _parity_summary_path(config: Mapping[str, Any]) -> Path:
    exact = config["exact_ig"]
    return Path(
        str(exact.get("structural_audit_path", exact["numerical_gate_path"]))
    ).resolve()


def assert_exact_ig_parity_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structural and semantic safety without token-allclose gating."""

    path = _parity_summary_path(config)
    if not path.is_file():
        raise RuntimeGateError(f"Exact-IG structural audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("exact_ig_version") != EXACT_IG_VERSION:
        raise RuntimeGateError(
            "Exact-IG audit version differs from the runtime contract"
        )
    if payload.get("allow_fast_path_training") is not True:
        raise RuntimeGateError("Exact-IG structural audit does not allow training")
    gates = dict(payload.get("gates", {}))
    required_boolean_gates = (
        "TARGET_CONTRACT",
        "CANONICAL_FIRST_ALIAS",
        "ONE_SHOT_TOKENIZATION",
        "PACKED_STRUCTURE",
        "NO_ANCHOR",
        "ATTENTION_MASK_EXHAUSTIVE",
        "LOGICAL_POSITION_IDS",
        "P_MINUS_ONE_SHIFT",
        "ANSWER_SPAN_MEAN",
        "FUTURE_LEAKAGE",
        "FP32_RUNTIME",
        "SDPA_MATH_RUNTIME",
        "FULL_LOGITS_RUNTIME",
        "FSDP_RESTORE",
        "MODEL_CHECKSUM_UNCHANGED",
        "IG_SIGN_SEMANTIC_PARITY",
        "TURN_RANKING_PARITY",
        "RAGEN_SELECTED_SET_PARITY",
        "MAX_PHI_ERROR_SAFETY",
        "MAX_IG_ERROR_SAFETY",
    )
    failed_gates = [
        name for name in required_boolean_gates if gates.get(name) is not True
    ]
    if failed_gates:
        raise RuntimeGateError(
            "Exact-IG structural/semantic gates failed: "
            + ", ".join(failed_gates)
        )
    if any(
        int(payload.get(field, -1)) != 0
        for field in ("optimizer_steps", "scheduler_steps", "checkpoint_writes")
    ):
        raise RuntimeGateError("Exact-IG audit changed training state")

    exact = config["exact_ig"]
    if str(exact["structural_audit_status"]) != "PASS":
        raise RuntimeGateError("Resolved config does not mark structural audit PASS")
    if str(exact["production_precision_mode"]) != PRODUCTION_PRECISION_MODE:
        raise RuntimeGateError("Exact-IG production precision is not fp32_exact_ig")
    if str(exact["scoring_logits_mode"]) != "official_full_logits":
        raise RuntimeGateError("Exact-IG production must use official full logits")
    if str(exact["attention_mask_mode"]) != "official_additive":
        raise RuntimeGateError("Exact-IG production must use official additive mask")
    if bool(exact["selected_positions_enabled"]):
        raise RuntimeGateError("Exact-IG selected_positions must remain disabled")

    artifact_root = path.parent
    runtime = json.loads(
        (artifact_root / "EXACT_IG_FAST_PATH_RUNTIME_METADATA.json").read_text(
            encoding="utf-8"
        )
    )
    errors = json.loads(
        (artifact_root / "EXACT_IG_FAST_ORACLE_ERROR_DISTRIBUTION.json").read_text(
            encoding="utf-8"
        )
    )
    equivalence = json.loads(
        (artifact_root / "EXACT_IG_FAST_RAGEN_SEMANTIC_PARITY.json").read_text(
            encoding="utf-8"
        )
    )
    fsdp4 = json.loads(
        (artifact_root / "EXACT_IG_FAST_FSDP_STATE_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    if runtime.get("gate_pass") is not True:
        raise RuntimeGateError("Exact-IG FP32 runtime metadata gate failed")
    if (
        equivalence.get("gate_pass") is not True
        or equivalence.get("selected_ids_equal") is not True
        or equivalence.get("prompt_ranking_equal") is not True
        or float(equivalence.get("selected_set_jaccard", 0.0)) != 1.0
    ):
        raise RuntimeGateError("Fast/Oracle selected Prompt IDs are not identical")
    if (
        fsdp4.get("fsdp_window_restore_pass") is not True
        or fsdp4.get("rank_metadata_consistent") is not True
        or fsdp4.get("all_rank_checksums_unchanged") is not True
    ):
        raise RuntimeGateError("Exact-IG FSDP4 state/checksum gate failed")
    if float(errors["phi_abs_diff"]["max"]) > float(
        exact["maximum_phi_safety_abs_diff"]
    ):
        raise RuntimeGateError("Exact-IG Phi drift exceeds the safety ceiling")
    if float(errors["ig_abs_diff"]["max"]) > float(
        exact["maximum_ig_safety_abs_diff"]
    ):
        raise RuntimeGateError("Exact-IG IG drift exceeds the safety ceiling")

    result = copy.deepcopy(dict(payload))
    result["gate_pass"] = True
    result["runtime_approval"] = None
    result["runtime_metadata"] = runtime
    result["error_distribution"] = errors
    result["ragen"] = equivalence
    result["fsdp4"] = fsdp4
    result["numeric_difference_policy"] = str(
        exact["oracle_numeric_difference_policy"]
    )
    return result


def _with_runtime_smoke_schedule(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the isolated runtime schedule without approving formal training."""
    result = copy.deepcopy(dict(config))
    smoke_schedule = dict(result["runtime_smoke_schedule"])
    smoke_schedule.pop("source", None)
    result["formal_schedule"].update(smoke_schedule)
    result["formal_schedule"]["source"] = str(
        result["runtime_smoke_schedule"]["source"]
    )
    result["exact_ig"]["oracle_canary_fail_closed"] = bool(
        result["exact_ig"]["runtime_smoke_oracle_canary_fail_closed"]
    )
    return result


def _debug_shape(
    config: Mapping[str, Any],
    *,
    prompt_count: int,
    group_size: int,
    require_optimizer_compatible: bool = False,
    preserve_formal_schedule: bool = False,
) -> dict[str, Any]:
    result = (
        copy.deepcopy(dict(config))
        if preserve_formal_schedule
        else _with_runtime_smoke_schedule(config)
    )
    result["rollout"]["group_size"] = int(group_size)
    result["rollout"]["prompt_wave_size"] = int(prompt_count)
    result["rollout"]["candidate_prompts_initial"] = int(prompt_count)
    result["rollout"]["refill_prompts"] = int(prompt_count)
    result["rollout"]["candidate_prompts_max"] = int(prompt_count * 2)
    result["selection"]["minimum_selected_prompts"] = 1
    result["selection"]["target_selected_prompts"] = int(prompt_count)
    result["selection"]["maximum_selected_prompts"] = int(prompt_count)
    result["selection"]["minimum_positive_prompts"] = 1
    if require_optimizer_compatible:
        world_size = int(result["learner"]["world_size"])
        production_micro_batch = int(
            result["formal_schedule"]["learner_micro_batch_size"]
        )
        normalized_mini_batch = (
            int(prompt_count) * int(group_size) * int(group_size)
        ) // world_size
        compatible_micro_batches = [
            candidate
            for candidate in range(production_micro_batch, 0, -1)
            if normalized_mini_batch % candidate == 0
        ]
        if not compatible_micro_batches:
            raise RuntimeGateError(
                "No optimizer-compatible learner micro-batch for debug shape"
            )
        result["formal_schedule"]["learner_micro_batch_size"] = (
            compatible_micro_batches[0]
        )
    return result


def _channel_checkpoint(state: ChannelScaleState) -> ChannelCheckpointState:
    return ChannelCheckpointState(
        committed_scale=state.committed_scale,
        health_observations=tuple(state.health_observations),
        health_reference=state.health_reference,
        valid_success_count=int(state.valid_success_count),
    )


def _channel_state(state: ChannelCheckpointState) -> ChannelScaleState:
    return ChannelScaleState(
        committed_scale=state.committed_scale,
        health_observations=tuple(state.health_observations),
        health_reference=state.health_reference,
        valid_success_count=int(state.valid_success_count),
    )


class VerlAttemptRuntimeAdapter:
    """Concrete veRL/Ray/vLLM/FSDP2 attempt adapter.

    Construction is side-effect free. Ray actors and model weights are loaded
    only after ``run`` passes the persisted Exact-IG parity gate.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = copy.deepcopy(dict(config))
        self._rl_world_size = int(self.config["learner"]["world_size"])
        self.stage = os.environ.get("AGENTIC_RL_RUNTIME_STAGE", "A").upper()
        run_dir = os.environ.get("AGENTIC_RL_RUN_DIR")
        if run_dir:
            self.config["paths"]["runtime_root"] = str(Path(run_dir).resolve())
        self.topology: RuntimeRayTopology | None = None
        self.worker_group: Any | None = None
        self.agent_loop_manager: Any | None = None
        self.actors: dict[str, Any] = {}
        self._last_checksum = ""
        self._last_snapshot_step = -1
        self._microbatch_metrics: list[dict[str, Any]] = []
        self._turn_runtime_metrics: dict[
            tuple[str, int], dict[str, Any]
        ] = {}
        self._prepared_groups: tuple[tuple[Any, ...], ...] = ()
        self._attempt_context: dict[str, Any] = {}
        self._checkpoint_reload_results: list[dict[str, Any]] = []
        self._eval_results: list[dict[str, Any]] = []
        self._total_optimizer_steps = 0
        self._total_scheduler_steps = 0
        self._starting_successful_update = 0
        self._last_gradient_norm: float | None = None
        self._learning_rate_used: float | None = None
        self._last_checkpoint: Path | None = None
        self._last_model_checkpoint: Path | None = None
        self._checkpoint_writes = 0
        self._fingerprints: dict[str, str] = {}
        self._weight_sync_records: list[dict[str, Any]] = []
        self._last_exact_ig_profiles: list[dict[str, Any]] = []
        self._stop_tokenizer: Any | None = None
        self._stop_scaffold_token_ids: tuple[int, ...] | None = None
        self._stage_started = time.perf_counter()

    def _require_bound(self) -> None:
        if (
            self.topology is None
            or self.worker_group is None
            or self.agent_loop_manager is None
        ):
            raise RuntimeError("Runtime topology has not been instantiated")

    def _uses_deferred_exact_ig(self) -> bool:
        return str(self.config["advantage"].get("search_task_mode")) == (
            ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE
        )

    def _add_phase_time(self, name: str, seconds: float) -> None:
        phases = self._attempt_context.setdefault("phase_seconds", {})
        phases[name] = float(phases.get(name, 0.0)) + float(seconds)

    def _write_metrics(
        self,
        scope: str,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        if not records or "metrics" not in self.actors:
            return
        import ray

        written = ray.get(
            self.actors["metrics"].write_many.remote(
                str(scope),
                [dict(record) for record in records],
            )
        )
        if int(written) != len(records):
            raise RuntimeError(
                f"MetricsActor wrote {written}/{len(records)} {scope} records"
            )

    @staticmethod
    def _gpu_snapshot() -> list[dict[str, Any]]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        rows = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            index, uuid, used, total, utilization = [
                value.strip() for value in line.split(",", maxsplit=4)
            ]
            rows.append(
                {
                    "physical_gpu": int(index),
                    "uuid": uuid,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "utilization_percent": int(utilization),
                }
            )
        return rows

    def start_attempt_metrics(self, state: TrainingState) -> None:
        self._attempt_context = {
            "started": time.perf_counter(),
            "state_before": state,
            "data_cursor_before": int(state.data_cursor),
            "phase_seconds": {},
            "deferred_exact_ig_tasks": {},
            "deferred_exact_ig_candidate_trajectory_ids": set(),
            "exact_ig_scored_before_selection": 0,
            "exact_ig_scored_after_selection": 0,
        }
        if self.stage == "FORMAL":
            atomic_write_json(
                Path(str(self.config["paths"]["runtime_root"]))
                / "state"
                / "current_attempt.json",
                {
                    "attempt_id_before": int(state.attempt_id),
                    "successful_update_before": int(
                        state.successful_update_step
                    ),
                    "data_cursor_before": int(state.data_cursor),
                    "started_at": time.time(),
                    "status": "running",
                },
            )
        self._prepared_groups = ()
        self._turn_runtime_metrics.clear()
        self._last_gradient_norm = None

    def record_snapshot_metrics(self, versions: SnapshotVersions) -> None:
        self._attempt_context["versions"] = versions

    def record_selection_metrics(
        self,
        state: TrainingState,
        groups: Sequence[PromptGroup],
        decision: SelectionDecision,
        *,
        refill_count: int,
        selection_seconds: float,
        selection_rounds: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        paper_metrics: dict[str, Any] = {}
        if (
            str(decision.selection_mode)
            == ANSWER_OUTCOME_ONLY_RAGEN2_PAPER_VARIANCE_TOP_P_MODE
        ):
            raw_top_p = decision.raw_top_p
            if raw_top_p is None:
                raise RuntimeError("Paper RAGEN-2 decision omitted raw Top-p result")
            outcome_variance_by_prompt = {
                str(group.prompt_global_id): float(group.outcome_variance)
                for group in groups
            }
            if set(decision.score_by_prompt) != set(outcome_variance_by_prompt):
                raise RuntimeError("Paper RAGEN-2 score prompt set mismatch")
            mismatch_count = sum(
                float(decision.score_by_prompt[prompt_id])
                != float(outcome_variance_by_prompt[prompt_id])
                for prompt_id in outcome_variance_by_prompt
            )
            if mismatch_count:
                raise RuntimeError(
                    "Paper RAGEN-2 score is not raw sample outcome variance"
                )
            if any(
                int(value) != 0
                for value in (
                    decision.health_gate_selection_call_count,
                    decision.scale_selection_call_count,
                    decision.normalized_signal_selection_call_count,
                )
            ):
                raise RuntimeError("Paper RAGEN-2 entered scale/health selection")
            rho = float(self.config["selection"]["top_p_mass"])
            threshold = rho * float(raw_top_p.total_mass)
            tolerance = max(1.0, float(raw_top_p.total_mass)) * 1.0e-15
            if raw_top_p.total_mass > 0.0:
                if not raw_top_p.selected_ids:
                    raise RuntimeError("Positive paper variance mass selected no prompt")
                if float(raw_top_p.selected_mass) + tolerance < threshold:
                    raise RuntimeError("Paper RAGEN-2 raw Top-p mass is below rho")
                previous_mass = math.fsum(
                    float(decision.score_by_prompt[prompt_id])
                    for prompt_id in raw_top_p.selected_ids[:-1]
                )
                if previous_mass + tolerance >= threshold:
                    raise RuntimeError("Paper RAGEN-2 raw Top-p prefix is not minimal")
            elif raw_top_p.selected_ids:
                raise RuntimeError("Zero paper variance mass selected prompts")
            paper_metrics = {
                "ragen/selection_mode": str(decision.selection_mode),
                "ragen/paper_raw_sample_variance_by_prompt": (
                    outcome_variance_by_prompt
                ),
                "ragen/paper_total_variance_mass": float(raw_top_p.total_mass),
                "ragen/paper_top_p_threshold": float(threshold),
                "ragen/paper_raw_k_star": int(len(raw_top_p.selected_ids)),
                "ragen/paper_raw_selected_ids": list(raw_top_p.selected_ids),
                "ragen/paper_raw_selected_count": int(
                    len(raw_top_p.selected_ids)
                ),
                "ragen/paper_selected_mass_ratio": float(
                    raw_top_p.selected_mass_ratio
                ),
                "ragen/final_selected_count_after_existing_refill": int(
                    decision.selected_count
                ),
                "ragen/refill_count": int(refill_count),
                "ragen/health_gate_selection_call_count": int(
                    decision.health_gate_selection_call_count
                ),
                "ragen/scale_selection_call_count": int(
                    decision.scale_selection_call_count
                ),
                "ragen/normalized_signal_selection_call_count": int(
                    decision.normalized_signal_selection_call_count
                ),
                "ragen/paper_score_mismatch_count": int(mismatch_count),
            }
        self._attempt_context.update(
            {
                "state_before": state,
                "groups": tuple(groups),
                "decision": decision,
                "refill_count": int(refill_count),
                "selection_rounds": tuple(
                    dict(row) for row in (selection_rounds or ())
                ),
                "paper_ragen2_metrics": paper_metrics,
            }
        )
        self._add_phase_time("selection", selection_seconds)

    def record_gradient_norm(self, value: float) -> None:
        self._last_gradient_norm = float(value)

    def _system_record(
        self,
        *,
        state: TrainingState,
    ) -> dict[str, Any]:
        import ray

        phases = dict(self._attempt_context.get("phase_seconds", {}))
        trajectories = [
            trajectory
            for group in self._attempt_context.get("groups", ())
            for trajectory in group.trajectories
        ]
        retrieval_latencies = [
            float(retrieval["latency_seconds"])
            for trajectory in trajectories
            for retrieval in trajectory.metadata.get("retrieval_records", ())
        ]
        retriever_requests = sum(
            len(trajectory.metadata.get("retrieval_records", ()))
            for trajectory in trajectories
        )
        health = self.retriever_health()
        resources = dict(ray.available_resources())
        resource_snapshot = read_runtime_resource_snapshot()
        try:
            from ray._private.internal_api import memory_summary

            object_store_summary = memory_summary(stats_only=True)
        except BaseException as exc:
            object_store_summary = (
                f"UNAVAILABLE:{type(exc).__name__}:{exc}"
            )
        used_match = re.search(
            r"Plasma memory usage\s+([0-9.]+)\s+MiB",
            object_store_summary,
        )
        spilled_match = re.search(
            r"Spilled\s+([0-9.]+)\s+MiB",
            object_store_summary,
        )
        canary = (
            self.worker_group.execute_all_sync("exact_ig_canary_summary")
            if self.worker_group is not None
            else []
        )
        vllm = (
            self.agent_loop_manager.runtime_metrics()
            if self.agent_loop_manager is not None
            else []
        )
        exact_profiles = list(self._last_exact_ig_profiles)
        exact_seconds = max(
            (float(row.get("seconds", 0.0)) for row in exact_profiles),
            default=0.0,
        )
        exact_records = sum(
            int(row.get("record_count", 0)) for row in exact_profiles
        )
        return {
            "attempt_id": int(state.attempt_id),
            "successful_update_step": int(state.successful_update_step),
            "gpus": self._gpu_snapshot(),
            "ray_object_store_available_bytes": resources.get(
                "object_store_memory"
            ),
            "ray_object_store_used_bytes": (
                float(used_match.group(1)) * 1024**2
                if used_match
                else None
            ),
            "ray_object_store_spill_bytes": (
                float(spilled_match.group(1)) * 1024**2
                if spilled_match
                else 0.0
            ),
            "ray_object_store_summary": object_store_summary,
            "cgroup_memory_limit_bytes": resource_snapshot.get(
                "memory_limit_bytes"
            ),
            "cgroup_memory_current_bytes": resource_snapshot.get(
                "memory_current_bytes"
            ),
            "cgroup_cpu_quota_cores": resource_snapshot.get(
                "cpu_quota_cores"
            ),
            "cgroup_memory_events": resource_snapshot.get("memory_events", {}),
            "retriever_requests": int(retriever_requests),
            "retriever_p50_latency": (
                float(np.percentile(retrieval_latencies, 50))
                if retrieval_latencies
                else 0.0
            ),
            "retriever_p95_latency": (
                float(np.percentile(retrieval_latencies, 95))
                if retrieval_latencies
                else 0.0
            ),
            "retriever_p99_latency": (
                float(np.percentile(retrieval_latencies, 99))
                if retrieval_latencies
                else 0.0
            ),
            "retriever_health": health,
            "vllm_by_replica": vllm,
            "vllm_queue": sum(
                int(row.get("inflight_requests", 0)) for row in vllm
            ),
            "vllm_max_inflight": max(
                (int(row.get("max_inflight_requests", 0)) for row in vllm),
                default=0,
            ),
            "vllm_kv_usage": 0.0,
            "vllm_kv_usage_semantics": (
                "post-drain post-sleep value; peak proxy is max_inflight"
            ),
            "vllm_preemption": None,
            "vllm_preemption_semantics": (
                "not exposed by the colocated veRL server API"
            ),
            "exact_ig_profiles_by_rank": exact_profiles,
            "exact_ig_records_per_second": (
                exact_records / exact_seconds if exact_seconds > 0 else 0.0
            ),
            "exact_ig_trajectories_per_second": (
                (
                    (
                        int(
                            self._attempt_context.get(
                                "exact_ig_scored_after_selection",
                                0,
                            )
                        )
                        if self._uses_deferred_exact_ig()
                        else len(trajectories)
                    )
                    / exact_seconds
                )
                if exact_seconds > 0
                else 0.0
            ),
            "exact_ig_peak_memory_bytes": max(
                (
                    int(row.get("peak_memory_allocated_bytes", 0))
                    for row in exact_profiles
                ),
                default=0,
            ),
            "exact_ig_oracle_canary_by_rank": canary,
            "weight_sync": (
                self._weight_sync_records[-1]
                if self._weight_sync_records
                else None
            ),
            "rollout_time": phases.get("rollout", 0.0),
            "retrieval_time": sum(retrieval_latencies),
            "outcome_time": phases.get("outcome", 0.0),
            "exact_ig_prep_time": phases.get("exact_ig_prep", 0.0),
            "exact_ig_gpu_time": phases.get("exact_ig_gpu", 0.0),
            "selection_time": phases.get("selection", 0.0),
            "advantage_time": phases.get("advantage", 0.0),
            "stop_branch_generation_time": phases.get(
                "stop_branch_generation",
                0.0,
            ),
            "stop_reward_scoring_time": phases.get("stop_reward", 0.0),
            "old_logprob_time": phases.get("old_logprob", 0.0),
            "reference_kl_backward_time": phases.get("backward", 0.0),
            "optimizer_time": phases.get("optimizer", 0.0),
            "scheduler_time": phases.get("scheduler", 0.0),
            "weight_sync_time": phases.get("weight_sync", 0.0),
            "checkpoint_time": phases.get("checkpoint", 0.0),
            "vllm_sleep_wake_time": phases.get("vllm_sleep_wake", 0.0),
        }

    def _persist_attempt_metrics(
        self,
        *,
        state_after: TrainingState,
        committed: bool,
        skip_reason: str | None,
        checkpoint: str | None = None,
    ) -> None:
        context = self._attempt_context
        state_before = context.get("state_before")
        decision = context.get("decision")
        groups = context.get("groups", ())
        versions = context.get("versions")
        if state_before is None or decision is None or versions is None:
            raise RuntimeError("Attempt metrics context is incomplete")
        trajectory_count = sum(len(group.trajectories) for group in groups)
        selected_trajectories = (
            int(decision.selected_count)
            * int(self.config["rollout"]["group_size"])
        )
        attempt_record = {
            "attempt_id": int(state_after.attempt_id),
            "successful_update_before": int(
                state_before.successful_update_step
            ),
            "successful_update_after": int(
                state_after.successful_update_step
            ),
            "pool_size": int(decision.candidate_count),
            "candidate_prompt_count": len(groups),
            "candidate_trajectory_count": int(trajectory_count),
            "selected_prompt_count": int(decision.selected_count),
            "selected_trajectory_count": int(selected_trajectories),
            "refill_used": bool(context.get("refill_count", 0)),
            "refill_count": int(context.get("refill_count", 0)),
            "selection_rounds": list(context.get("selection_rounds", ())),
            "skip_reason": skip_reason,
            "data_cursor_before": int(context["data_cursor_before"]),
            "data_cursor_after": int(state_after.data_cursor),
            "rollout_weight_version": int(versions.rollout),
            "old_policy_version": int(versions.old_policy),
            "reward_policy_version": int(versions.reward_policy),
            "reference_version": "frozen_initial_dpo_v2",
            "optimizer_steps": 1 if committed else 0,
            "scheduler_steps": 1 if committed else 0,
            "attempt_wall_time": time.perf_counter()
            - float(context["started"]),
            "checkpoint": checkpoint,
        }
        channel = build_channel_records(
            attempt_id=state_after.attempt_id,
            successful_update_before=state_before.successful_update_step,
            successful_update_after=state_after.successful_update_step,
            decision=decision,
            state_before=state_before,
            state_after=state_after,
            committed=committed,
        )
        prompt = build_prompt_records(
            attempt_id=state_after.attempt_id,
            groups=groups,
            decision=decision,
        )
        trajectory, turn = build_trajectory_and_turn_records(
            attempt_id=state_after.attempt_id,
            groups=groups,
            prepared_groups=self._prepared_groups,
            turn_runtime=self._turn_runtime_metrics,
        )
        behavior = build_behavior_record(
            attempt_id=state_after.attempt_id,
            successful_update_step=state_after.successful_update_step,
            groups=groups,
        )
        self._write_metrics("attempt", [attempt_record])
        self._write_metrics("channel", channel)
        self._write_metrics("prompt", prompt)
        self._write_metrics("trajectory", trajectory)
        self._write_metrics("turn", turn)
        self._write_metrics("behavior", [behavior])
        self._write_metrics(
            "system",
            [self._system_record(state=state_after)],
        )

    def record_skipped_attempt(
        self,
        state: TrainingState,
        *,
        reason: str,
    ) -> None:
        self._persist_attempt_metrics(
            state_after=state,
            committed=False,
            skip_reason=str(reason),
        )

    def record_failed_attempt(self, exc: BaseException) -> None:
        runtime_root = Path(str(self.config["paths"]["runtime_root"]))
        errors = runtime_root / "logs" / "errors.log"
        errors.parent.mkdir(parents=True, exist_ok=True)
        with errors.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{time.time():.6f}\t{type(exc).__name__}\t{exc}\n"
            )

    async def _retriever_health_async(self) -> dict[str, Any]:
        retriever = self.config["retriever"]
        async with AsyncHybridRetrieverClient(
            str(retriever["service_url"]),
            timeout_seconds=float(retriever["timeout_seconds"]),
            default_top_k=int(retriever["top_k"]),
            maximum_concurrency=1,
            maximum_batch_queries=1,
            batch_wait_ms=float(retriever["request_batch_wait_ms"]),
            network_retries=0,
        ) as client:
            return await client.health()

    def retriever_health(self) -> dict[str, Any]:
        return asyncio.run(self._retriever_health_async())

    async def _retriever_stage_a_canary_async(self) -> dict[str, Any]:
        retriever = self.config["retriever"]
        queries = (
            "Who wrote Pride and Prejudice?",
            "What is the capital of France?",
        )
        async with AsyncHybridRetrieverClient(
            str(retriever["service_url"]),
            timeout_seconds=float(retriever["timeout_seconds"]),
            default_top_k=int(retriever["top_k"]),
            maximum_concurrency=len(queries),
            maximum_batch_queries=len(queries),
            batch_wait_ms=float(retriever["request_batch_wait_ms"]),
            network_retries=0,
        ) as client:
            before = await client.health()
            results = await asyncio.gather(
                *(
                    client.retrieve_one(
                        query,
                        f"stage-a-retriever-canary-{index}",
                        0,
                    )
                    for index, query in enumerate(queries)
                )
            )
            after = await client.health()
            client_stats = client.stats()

        if any(not result.documents for result in results):
            raise RuntimeGateError(
                "Stage A async Retriever canary returned an empty document list"
            )
        if any(
            not document.document_id
            for result in results
            for document in result.documents
        ):
            raise RuntimeGateError(
                "Stage A async Retriever canary returned an empty document ID"
            )
        if any(result.batch_query_count != len(queries) for result in results):
            raise RuntimeGateError(
                "Stage A async Retriever canary did not exercise micro-batching"
            )

        before_batching = dict(before.get("batching", {}))
        after_batching = dict(after.get("batching", {}))
        query_delta = int(after_batching.get("queries", 0)) - int(
            before_batching.get("queries", 0)
        )
        request_delta = int(after_batching.get("requests", 0)) - int(
            before_batching.get("requests", 0)
        )
        if query_delta < len(queries) or request_delta < 1:
            raise RuntimeGateError(
                "Stage A Retriever service counters did not record the real "
                "async canary request"
            )
        return {
            "status": "PASS",
            "query_count": len(queries),
            "query_delta": query_delta,
            "request_delta": request_delta,
            "client_stats": client_stats,
            "results": [
                {
                    "request_id": result.request_id,
                    "trajectory_id": result.trajectory_id,
                    "turn_id": result.turn_id,
                    "query": result.query,
                    "latency_seconds": result.latency_seconds,
                    "batch_query_count": result.batch_query_count,
                    "document_ids": [
                        document.document_id for document in result.documents
                    ],
                }
                for result in results
            ],
        }

    def _retriever_stage_a_canary(self) -> dict[str, Any]:
        return asyncio.run(self._retriever_stage_a_canary_async())

    @staticmethod
    def _stage_a_token_provenance_canary() -> dict[str, Any]:
        sources = (
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.CODE_INSERTED,
        )
        record = TrajectoryRecord(
            prompt_global_id="stage-a-provenance-canary",
            trajectory_id="stage-a-provenance-canary:trajectory-00",
            input_ids=[101, 102, 103, 104],
            token_sources=list(sources),
            turn_ids=[-1, 0, -1, -1],
            turns=[
                TurnRecord(
                    turn_index=0,
                    turn_type=TurnType.SEARCH,
                    model_text="<search>canary query</search>",
                    search_index=0,
                    query="canary query",
                    information_text="retrieved canary information",
                    search_action_span_valid=True,
                    search_prefix_valid=True,
                    ig_reward_eligible=True,
                    policy_credit_eligible=True,
                )
            ],
            search_prefix_end_positions=[3],
            search_prefix_before_search_end_positions={0: 1},
        )
        record.validate()
        action_mask = tuple(record.action_token_mask)
        policy_mask = tuple(record.policy_mask)
        kl_mask = tuple(record.kl_mask)
        assert_environment_information_masked(sources, action_mask)
        expected = (0, 1, 0, 0)
        if action_mask != expected or policy_mask != expected or kl_mask != expected:
            raise RuntimeGateError(
                "Stage A provenance canary did not mask environment/code tokens"
            )
        if record.terminal_policy_credit_turn_index is not None:
            raise RuntimeGateError(
                "Stage A provenance canary invented a terminal Answer span"
            )
        return {
            "status": "PASS",
            "token_sources": [source.value for source in sources],
            "action_mask": action_mask,
            "policy_mask": policy_mask,
            "kl_mask": kl_mask,
            "terminal_policy_credit_turn_index": None,
        }

    def bind(self, *, require_optimizer: bool) -> dict[str, Any]:
        self.topology = RuntimeRayTopology(self.config)
        ray_status = self.topology.initialize_ray()
        self.actors = self.topology.instantiate_control_actors()
        gpu_status = self.topology.instantiate_gpu_workers(
            require_optimizer=require_optimizer
        )
        self.worker_group = self.topology.worker_group
        self.agent_loop_manager = self.topology.agent_loop_manager
        self._require_bound()
        return {
            "ray": ray_status,
            "gpu": gpu_status,
            "tables": self.topology.runtime_tables(),
        }

    def freeze_rollout_boundary(
        self,
        successful_update_step: int,
    ) -> SnapshotVersions:
        self._require_bound()
        started = time.perf_counter()
        step = int(successful_update_step)
        version_rows = self.worker_group.execute_all_sync(
            "begin_snapshot",
            step,
        )
        expected = {
            "actor_snapshot_step": step,
            "rollout_snapshot_step": step,
            "old_policy_snapshot_step": step,
            "reward_policy_snapshot_step": step,
        }
        if any(row != expected for row in version_rows):
            raise RuntimeError(f"FSDP snapshot versions disagree: {version_rows}")
        checksums = self.worker_group.execute_all_sync("global_actor_checksum")
        if len(set(checksums)) != 1:
            raise RuntimeError("FSDP ranks disagree on actor checksum")
        checksum = str(checksums[0])
        wake_started = time.perf_counter()
        sync = self.agent_loop_manager.synchronize_from_actor(
            step,
            checksum,
        )
        self._weight_sync_records.append(sync)
        self._add_phase_time(
            "vllm_sleep_wake",
            time.perf_counter() - wake_started,
        )
        rollout_versions = sync["versions"]
        if len(rollout_versions) != self._rl_world_size:
            raise RuntimeError("vLLM replicas were not all version-stamped")
        self._last_checksum = checksum
        self._last_snapshot_step = step
        self._add_phase_time("snapshot", time.perf_counter() - started)
        return SnapshotVersions(step, step, step, step)

    def _allocate_rows(self, prompt_count: int) -> tuple[dict[str, Any], ...]:
        import ray

        rows = ray.get(
            self.actors["prompt_sampler"].allocate_rows.remote(
                int(prompt_count)
            )
        )
        if len(rows) != int(prompt_count):
            raise RuntimeError("Prompt sampler returned the wrong row count")
        return tuple(rows)

    def _rollout_data(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        snapshot_step: int,
        group_size: int | None = None,
        validate: bool = False,
    ) -> Any:
        from verl.protocol import DataProto

        resolved_group_size = (
            int(self.config["rollout"]["group_size"])
            if group_size is None
            else int(group_size)
        )
        if resolved_group_size < 1:
            raise ValueError("rollout group_size must be positive")
        expanded: list[dict[str, Any]] = []
        for row in rows:
            for trajectory_index in range(resolved_group_size):
                expanded.append(
                    {
                        "raw_prompt": list(row["prompt_messages"]),
                        "gold_aliases": tuple(row["gold_aliases"]),
                        "canonical_answer": str(row["canonical_answer"]),
                        "prompt_global_id": str(row["prompt_global_id"]),
                        "trajectory_id": (
                            f"{row['prompt_global_id']}:snapshot-"
                            f"{snapshot_step}:trajectory-{trajectory_index:02d}"
                        ),
                        "snapshot_step": int(snapshot_step),
                        "data_source": str(row["data_source"]),
                        "dataset_row_id": str(row.get("id", "")),
                        "dataset_source_index": int(
                            row.get("source_index", -1)
                        ),
                        "index": int(row["logical_index"]),
                        "agent_name": "search_exact_ig",
                    }
                )
        keys = tuple(expanded[0])
        non_tensors: dict[str, np.ndarray] = {}
        for key in keys:
            values = np.empty(len(expanded), dtype=object)
            values[:] = [item[key] for item in expanded]
            non_tensors[key] = values
        return DataProto.from_dict(
            non_tensors=non_tensors,
            meta_info={
                "global_steps": int(snapshot_step),
                "validate": bool(validate),
            },
        )

    @staticmethod
    def _driver_rng_digest() -> str:
        digest = hashlib.sha256()
        digest.update(pickle.dumps(random.getstate(), protocol=5))
        digest.update(pickle.dumps(np.random.get_state(), protocol=5))
        digest.update(torch.get_rng_state().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _fixed_eval_manifest(self) -> dict[str, Any]:
        evaluation = self.config["evaluation"]
        return create_or_validate_eval_manifest_from_config(
            validation_path=self.config["paths"]["validation_data"],
            evaluation=evaluation,
        )

    def _run_fixed_eval(
        self,
        *,
        successful_update_step: int,
    ) -> dict[str, Any]:
        """Run deterministic G=1 evaluation without touching training state."""

        import ray

        self._require_bound()
        step = int(successful_update_step)
        manifest = self._fixed_eval_manifest()
        rows = load_eval_rows(manifest=manifest)
        batch_size = int(self.config["evaluation"]["batch_prompts"])
        if batch_size < 1:
            raise RuntimeError("evaluation.batch_prompts must be positive")
        cursor_before = self.data_cursor()
        rng_before = self._driver_rng_digest()
        actor_before = self.actor_parameter_checksum()
        sync = self.agent_loop_manager.synchronize_from_actor(
            step,
            actor_before,
        )
        self._weight_sync_records.append(sync)
        predictions: list[dict[str, Any]] = []
        records: list[TrajectoryRecord] = []
        started = time.perf_counter()
        try:
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start : start + batch_size]
                prompts = self._rollout_data(
                    batch_rows,
                    snapshot_step=step,
                    group_size=1,
                    validate=True,
                )
                rollout_refs = (
                    self.agent_loop_manager.dispatch_sequences_keep_awake(
                        prompts
                    )
                )
                outcome_workers = self.actors["outcome_workers"]
                outcome_refs = [
                    outcome_workers[index % len(outcome_workers)]
                    .score_rollout_chunk.remote(reference)
                    for index, reference in enumerate(rollout_refs)
                ]
                self._wait_without_driver_fetch(rollout_refs)
                for payload in ray.get(outcome_refs):
                    extras = payload["extras"]
                    outcomes = payload["outcomes"]
                    for extra, outcome in zip(extras, outcomes, strict=True):
                        record = trajectory_record_from_extra(
                            extra,
                            outcome_override=outcome,
                        )
                        records.append(record)
                        predictions.append(
                            {
                                "successful_update_step": step,
                                "prompt_global_id": record.prompt_global_id,
                                "dataset_row_id": record.metadata.get(
                                    "dataset_row_id"
                                ),
                                "dataset_source_index": record.metadata.get(
                                    "dataset_source_index"
                                ),
                                "domain": record.metadata.get(
                                    "data_source", ""
                                ),
                                "trajectory_id": record.trajectory_id,
                                "R_task": float(record.task_outcome),
                                "exact": bool(
                                    math.isclose(
                                        float(record.task_outcome),
                                        1.0,
                                        rel_tol=0.0,
                                        abs_tol=1.0e-12,
                                    )
                                ),
                                "F_ans": int(
                                    record.answer_format_indicator
                                ),
                                "terminal_answer_valid": bool(
                                    record.terminal_answer_valid
                                ),
                                "system_valid": bool(
                                    record.trajectory_system_valid
                                ),
                                "search_count": int(
                                    record.search_turn_count
                                ),
                                "queries": [
                                    turn.query
                                    for turn in record.turns
                                    if turn.query is not None
                                ],
                                "model_actions": list(
                                    extra.get("model_actions", ())
                                ),
                            }
                        )
        finally:
            self.agent_loop_manager.sleep_for_scoring()
        elapsed = time.perf_counter() - started
        actor_after = self.actor_parameter_checksum()
        cursor_after = self.data_cursor()
        rng_after = self._driver_rng_digest()
        if actor_before != actor_after:
            raise RuntimeGateError("Fixed Eval changed Actor parameters")
        if cursor_before != cursor_after:
            raise RuntimeGateError("Fixed Eval changed the training data cursor")
        if rng_before != rng_after:
            raise RuntimeGateError("Fixed Eval changed driver training RNG state")
        if len(records) != len(rows):
            raise RuntimeGateError(
                f"Fixed Eval produced {len(records)}/{len(rows)} trajectories"
            )

        eval_root = (
            Path(str(self.config["paths"]["runtime_root"]))
            / "eval"
            / f"update_{step:03d}"
        )
        eval_root.mkdir(parents=True, exist_ok=True)
        predictions_path = eval_root / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8", buffering=1) as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        aggregate: list[dict[str, Any]] = []
        domains = sorted(
            {str(record.metadata.get("data_source", "")) for record in records}
        )
        for domain in (*domains, "overall"):
            subset = (
                records
                if domain == "overall"
                else [
                    record
                    for record in records
                    if str(record.metadata.get("data_source", "")) == domain
                ]
            )
            count = len(subset)
            outcomes = [float(record.task_outcome) for record in subset]
            search_counts = [
                int(record.search_turn_count) for record in subset
            ]
            queries = [
                " ".join(str(turn.query).lower().split())
                for record in subset
                for turn in record.turns
                if turn.query
            ]
            repeated = sum(
                len(
                    [
                        " ".join(str(turn.query).lower().split())
                        for turn in record.turns
                        if turn.query
                    ]
                )
                != len(
                    {
                        " ".join(str(turn.query).lower().split())
                        for turn in record.turns
                        if turn.query
                    }
                )
                for record in subset
            )
            aggregate.append(
                {
                    "successful_update_step": step,
                    "domain": domain,
                    "count": count,
                    "f1": (
                        float(np.mean(outcomes, dtype=np.float64))
                        if outcomes
                        else 0.0
                    ),
                    "exact": (
                        sum(
                            math.isclose(
                                value,
                                1.0,
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                            for value in outcomes
                        )
                        / count
                        if count
                        else 0.0
                    ),
                    "format_rate": (
                        sum(
                            int(record.answer_format_indicator)
                            for record in subset
                        )
                        / count
                        if count
                        else 0.0
                    ),
                    "answer_rate": (
                        sum(
                            bool(record.terminal_answer_valid)
                            for record in subset
                        )
                        / count
                        if count
                        else 0.0
                    ),
                    "no_answer_rate": (
                        sum(
                            record.terminal_policy_credit_turn_index is None
                            for record in subset
                        )
                        / count
                        if count
                        else 0.0
                    ),
                    "avg_search": (
                        float(np.mean(search_counts, dtype=np.float64))
                        if search_counts
                        else 0.0
                    ),
                    "multi_search_rate": (
                        sum(value >= 2 for value in search_counts) / count
                        if count
                        else 0.0
                    ),
                    "repeat_query_rate": (
                        repeated / count if count else 0.0
                    ),
                    "max_turn_rate": (
                        sum(value >= 5 for value in search_counts) / count
                        if count
                        else 0.0
                    ),
                    "query_diversity": (
                        len(set(queries)) / len(queries) if queries else 0.0
                    ),
                    "template_similarity": (
                        1.0 - len(set(queries)) / len(queries)
                        if queries
                        else 0.0
                    ),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "actor_checksum": actor_before,
                    "data_cursor_unchanged": True,
                    "driver_rng_unchanged": True,
                    "wall_seconds": elapsed,
                }
            )
        summary = {
            "status": "PASS",
            "successful_update_step": step,
            "manifest": {
                "path": str(
                    Path(
                        str(self.config["evaluation"]["manifest_path"])
                    ).resolve()
                ),
                "sha256": manifest["manifest_sha256"],
            },
            "metrics": aggregate,
            "predictions": str(predictions_path),
            "wall_seconds": elapsed,
        }
        (eval_root / "metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_metrics("eval", aggregate)
        self._eval_results.append(summary)
        return summary

    def _submit_cpu_postprocessing(
        self,
        rollout_refs: Sequence[Any],
    ) -> tuple[list[Any], list[Any]]:
        outcome_workers = self.actors["outcome_workers"]
        task_builders = self.actors.get("exact_ig_task_builders")
        if not task_builders:
            raise RuntimeError(
                "Exact-IG task builders require approved/debug context limits"
            )
        outcome_refs = [
            outcome_workers[index % len(outcome_workers)]
            .score_rollout_chunk.remote(reference)
            for index, reference in enumerate(rollout_refs)
        ]
        task_refs = [
            task_builders[index % len(task_builders)]
            .build_rollout_chunk.remote(reference)
            for index, reference in enumerate(rollout_refs)
        ]
        return outcome_refs, task_refs

    @staticmethod
    def _wait_without_driver_fetch(references: Sequence[Any]) -> None:
        import ray

        remaining = list(references)
        while remaining:
            _, remaining = ray.wait(
                remaining,
                num_returns=1,
                fetch_local=False,
            )

    def _score_exact_ig_tasks(
        self,
        tasks: Sequence[Any],
    ) -> dict[str, dict[str, Any]]:
        import ray

        if not tasks:
            return {}
        tasks_by_prompt: dict[str, list[Any]] = defaultdict(list)
        for task in tasks:
            tasks_by_prompt[str(task.prompt_global_id)].append(task)
        for prompt_tasks in tasks_by_prompt.values():
            assert_same_prompt_target_consistency(prompt_tasks)
        assignments: list[list[Any]] = [[] for _ in range(self._rl_world_size)]
        loads = [0] * self._rl_world_size
        for task in sorted(
            tasks,
            key=lambda item: (
                -int(
                    getattr(
                        item,
                        "projected_fast_packed_length",
                        item.input_ids.size,
                    )
                )
                ** 2,
                item.trajectory_id,
            ),
        ):
            rank = min(
                range(self._rl_world_size),
                key=lambda value: (loads[value], value),
            )
            assignments[rank].append(task)
            task_length = int(
                getattr(
                    task,
                    "projected_fast_packed_length",
                    task.input_ids.size,
                )
            )
            loads[rank] += task_length**2
        target_count = max(len(values) for values in assignments)
        fallback = min(
            tasks,
            key=lambda item: int(
                getattr(
                    item,
                    "projected_fast_packed_length",
                    item.input_ids.size,
                )
            ),
        )
        for values in assignments:
            while len(values) < target_count:
                values.append(fallback)
        references = [ray.put(values) for values in assignments]
        results_by_rank = self.worker_group.execute_all_sync(
            "score_exact_ig_tasks",
            references,
        )
        self._last_exact_ig_profiles = list(
            self.worker_group.execute_all_sync("exact_ig_last_profile")
        )
        self._attempt_context["exact_ig_assignments"] = {
            "unique_record_count": len(tasks),
            "rank_forward_record_counts": [
                len(values) for values in assignments
            ],
            "rank_attention_cost": loads,
            "profiles": self._last_exact_ig_profiles,
        }
        by_trajectory: dict[str, dict[str, Any]] = {}
        for rank_results in results_by_rank:
            for result in rank_results:
                trajectory_id = str(result["trajectory_id"])
                previous = by_trajectory.get(trajectory_id)
                if previous is not None:
                    if not math.isclose(
                        float(previous["telescoping_error"]),
                        float(result["telescoping_error"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-8,
                    ):
                        raise RuntimeError(
                            "Padded Exact-IG task produced inconsistent metadata"
                        )
                    continue
                by_trajectory[trajectory_id] = result
        missing = {
            str(task.trajectory_id) for task in tasks
        } - set(by_trajectory)
        if missing:
            raise RuntimeError(f"Exact-IG results are missing: {sorted(missing)}")
        return by_trajectory

    @staticmethod
    def _attach_exact_ig_result(record: TrajectoryRecord, exact: Mapping[str, Any]) -> None:
        attach_exact_ig(record, exact["immediate_ig"])
        record.metadata["exact_ig_score_by_prefix"] = tuple(
            float(value) for value in exact["score_by_prefix"]
        )
        record.metadata["telescoping_error"] = float(exact["telescoping_error"])
        for key in (
            "exact_ig_version",
            "scaffold_sha256",
            "canonical_alias_policy",
            "canonical_answer_sha256",
            "target_token_ids_hash",
            "score_span_hash",
            "target_score_span_hash",
            "score_mask_policy",
            "info_gain_type",
            "fast_path_structure",
            "target_tokenization_policy",
            "official_igpo_commit_sha",
            "mask_builder_version",
            "position_builder_version",
            "scaffold_text",
            "tokenizer_name_or_path",
            "tokenizer_revision",
            "answer_score_token_count",
            "reward_snapshot_step",
            "reward_snapshot_checksum",
        ):
            record.metadata[key] = exact[key]

    def _finish_submissions(
        self,
        submissions: Sequence[tuple[Sequence[Any], Sequence[Any], Sequence[Any]]],
        *,
        expected_prompt_count: int,
    ) -> tuple[PromptGroup, ...]:
        import ray

        rollout_refs = [
            reference
            for rollout, _, _ in submissions
            for reference in rollout
        ]
        rollout_started = time.perf_counter()
        self._wait_without_driver_fetch(rollout_refs)
        self._add_phase_time("rollout", time.perf_counter() - rollout_started)
        sleep_started = time.perf_counter()
        self.agent_loop_manager.sleep_for_scoring()
        self._add_phase_time(
            "vllm_sleep_wake",
            time.perf_counter() - sleep_started,
        )
        cpu_started = time.perf_counter()
        outcome_payloads = ray.get(
            [
                reference
                for _, outcomes, _ in submissions
                for reference in outcomes
            ]
        )
        task_chunks = ray.get(
            [
                reference
                for _, _, tasks in submissions
                for reference in tasks
            ]
        )
        cpu_seconds = time.perf_counter() - cpu_started
        self._add_phase_time("outcome", cpu_seconds)
        self._add_phase_time("exact_ig_prep", cpu_seconds)
        task_records = [
            task for chunk in task_chunks for task in chunk
        ]
        deferred_exact_ig = self._uses_deferred_exact_ig()
        exact_by_trajectory: dict[str, dict[str, Any]] = {}
        if deferred_exact_ig:
            task_store = self._attempt_context.setdefault(
                "deferred_exact_ig_tasks",
                {},
            )
            candidate_ids = self._attempt_context.setdefault(
                "deferred_exact_ig_candidate_trajectory_ids",
                set(),
            )
            for task in task_records:
                trajectory_id = str(task.trajectory_id)
                if trajectory_id in task_store:
                    raise RuntimeError(
                        f"Duplicate deferred Exact-IG task: {trajectory_id}"
                    )
                task_store[trajectory_id] = task
                candidate_ids.add(trajectory_id)
        else:
            exact_started = time.perf_counter()
            exact_by_trajectory = self._score_exact_ig_tasks(task_records)
            self._attempt_context["exact_ig_scored_before_selection"] = int(
                self._attempt_context.get("exact_ig_scored_before_selection", 0)
            ) + len(task_records)
            self._add_phase_time(
                "exact_ig_gpu",
                time.perf_counter() - exact_started,
            )
        records = []
        aliases_by_prompt: dict[str, tuple[str, ...]] = {}
        for payload in outcome_payloads:
            extras = payload["extras"]
            outcomes = payload["outcomes"]
            if len(extras) != len(outcomes):
                raise RuntimeError("Outcome payload cardinality mismatch")
            for extra, outcome in zip(extras, outcomes, strict=True):
                record = trajectory_record_from_extra(
                    extra,
                    outcome_override=outcome,
                )
                if record.trajectory_system_valid and not deferred_exact_ig:
                    exact = exact_by_trajectory.get(record.trajectory_id)
                    if exact is None:
                        raise RuntimeError(
                            f"Missing Exact-IG result for {record.trajectory_id}"
                        )
                    self._attach_exact_ig_result(record, exact)
                aliases = tuple(
                    str(value) for value in extra["gold_aliases"]
                )
                aliases_by_prompt[record.prompt_global_id] = aliases
                records.append(record)
        records_by_prompt: dict[str, list[Any]] = defaultdict(list)
        for record in records:
            records_by_prompt[record.prompt_global_id].append(record)
        groups: list[PromptGroup] = []
        invalid_prompt_ids: list[str] = []
        for prompt_id in sorted(records_by_prompt):
            prompt_records = records_by_prompt[prompt_id]
            if not all(
                record.trajectory_system_valid and record.optimization_ready
                for record in prompt_records
            ):
                invalid_prompt_ids.append(prompt_id)
                continue
            groups.extend(
                prompt_groups_from_records(
                    prompt_records,
                    aliases_by_prompt={
                        prompt_id: aliases_by_prompt[prompt_id]
                    },
                    expected_group_size=int(
                        self.config["rollout"]["group_size"]
                    ),
                    outcome_only_selection=deferred_exact_ig,
                )
            )
        if len(groups) > int(expected_prompt_count):
            raise RuntimeError(
                "Postprocessing created more prompt groups than were allocated"
            )
        if invalid_prompt_ids:
            event_root = (
                Path(str(self.config["paths"]["runtime_root"])) / "events"
            )
            event_root.mkdir(parents=True, exist_ok=True)
            with (event_root / "system_invalid_prompt_replacements.jsonl").open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        {
                            "snapshot_step": self._last_snapshot_step,
                            "prompt_ids": invalid_prompt_ids,
                            "replacement_count": len(invalid_prompt_ids),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        return tuple(groups)

    def _collect_replacement_groups(
        self,
        prompt_count: int,
        *,
        snapshot_step: int,
    ) -> tuple[PromptGroup, ...]:
        rows = self._allocate_rows(prompt_count)
        prompts = self._rollout_data(rows, snapshot_step=snapshot_step)
        rollout_refs = self.agent_loop_manager.dispatch_sequences_keep_awake(
            prompts
        )
        outcomes, tasks = self._submit_cpu_postprocessing(rollout_refs)
        return self._finish_submissions(
            ((rollout_refs, outcomes, tasks),),
            expected_prompt_count=prompt_count,
        )

    def _replace_invalid_groups(
        self,
        groups: Sequence[PromptGroup],
        *,
        target_count: int,
        snapshot_step: int,
    ) -> tuple[PromptGroup, ...]:
        valid = list(groups)
        for _ in range(3):
            missing = int(target_count) - len(valid)
            if missing <= 0:
                return tuple(valid)
            valid.extend(
                self._collect_replacement_groups(
                    missing,
                    snapshot_step=snapshot_step,
                )
            )
        if len(valid) != int(target_count):
            raise RuntimeError(
                "System-invalid prompt replacement budget was exhausted"
            )
        return tuple(valid)

    def collect_initial_scored_prompt_groups(
        self,
        prompt_count: int,
        *,
        wave_size: int,
        snapshot_step: int,
    ) -> Sequence[PromptGroup]:
        if prompt_count % wave_size:
            raise ValueError("Initial prompt count must contain whole waves")
        submissions = []
        for _ in range(prompt_count // wave_size):
            rows = self._allocate_rows(wave_size)
            prompts = self._rollout_data(rows, snapshot_step=snapshot_step)
            rollout_refs = self.agent_loop_manager.dispatch_sequences_keep_awake(
                prompts
            )
            outcome_refs, task_refs = self._submit_cpu_postprocessing(
                rollout_refs
            )
            # The next loop iteration launches Wave 2 while these ObjectRef
            # dependencies stream through CPU Outcome/task-builder actors.
            submissions.append((rollout_refs, outcome_refs, task_refs))
        groups = self._finish_submissions(
            submissions,
            expected_prompt_count=prompt_count,
        )
        return self._replace_invalid_groups(
            groups,
            target_count=prompt_count,
            snapshot_step=snapshot_step,
        )

    def collect_scored_prompt_groups(
        self,
        prompt_count: int,
        *,
        snapshot_step: int,
    ) -> Sequence[PromptGroup]:
        versions = self.agent_loop_manager.read_weight_versions()
        if {
            (int(row["snapshot_step"]), str(row["source_checksum"]))
            for row in versions
        } != {(int(snapshot_step), self._last_checksum)}:
            self.agent_loop_manager.wake_for_rollout()
            self.agent_loop_manager.stamp_weight_version(
                snapshot_step,
                self._last_checksum,
            )
        groups = self._collect_replacement_groups(
            prompt_count,
            snapshot_step=snapshot_step,
        )
        return self._replace_invalid_groups(
            groups,
            target_count=prompt_count,
            snapshot_step=snapshot_step,
        )

    def finalize_selected_exact_ig(
        self,
        groups: Sequence[PromptGroup],
    ) -> Sequence[PromptGroup]:
        """Score Exact-IG after Answer-only RAGEN selection in MICA mode."""

        if not self._uses_deferred_exact_ig():
            return tuple(groups)
        selected_ids = tuple(str(group.prompt_global_id) for group in groups)
        if len(set(selected_ids)) != len(selected_ids):
            raise RuntimeError("Selected MICA prompt IDs are not unique")
        task_store = self._attempt_context.get("deferred_exact_ig_tasks", {})
        if not isinstance(task_store, Mapping):
            raise RuntimeError("Deferred Exact-IG task store is unavailable")
        selected_records = [
            record for group in groups for record in group.trajectories
        ]
        selected_trajectory_ids = {
            str(record.trajectory_id) for record in selected_records
        }
        tasks = []
        for trajectory_id in sorted(selected_trajectory_ids):
            task = task_store.get(trajectory_id)
            if task is None:
                raise RuntimeError(
                    f"Selected trajectory has no deferred Exact-IG task: {trajectory_id}"
                )
            tasks.append(task)
        exact_started = time.perf_counter()
        exact_by_trajectory = self._score_exact_ig_tasks(tasks)
        self._add_phase_time("exact_ig_gpu", time.perf_counter() - exact_started)
        for record in selected_records:
            exact = exact_by_trajectory.get(str(record.trajectory_id))
            if exact is None:
                raise RuntimeError(
                    f"Selected Exact-IG result is missing: {record.trajectory_id}"
                )
            self._attach_exact_ig_result(record, exact)

        aliases_by_prompt = {
            str(group.prompt_global_id): tuple(
                str(value)
                for value in group.trajectories[0].metadata.get("gold_aliases", ())
            )
            for group in groups
        }
        rebuilt = prompt_groups_from_records(
            selected_records,
            aliases_by_prompt=aliases_by_prompt,
            expected_group_size=int(self.config["rollout"]["group_size"]),
            outcome_only_selection=False,
        )
        rebuilt_by_id = {group.prompt_global_id: group for group in rebuilt}
        finalized = tuple(rebuilt_by_id[prompt_id] for prompt_id in selected_ids)
        for before, after in zip(groups, finalized, strict=True):
            if not math.isclose(
                float(before.outcome_variance),
                float(after.outcome_variance),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError("Deferred Exact-IG changed terminal Outcome variance")

        candidate_count = sum(
            len(group.trajectories)
            for group in self._attempt_context.get("groups", ())
        )
        decision = self._attempt_context.get("decision")
        if decision is None or tuple(decision.selected_ids) != selected_ids:
            raise RuntimeError(
                "Deferred Exact-IG selection does not match the recorded RAGEN decision"
            )
        if str(decision.signal_mode) != ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL:
            raise RuntimeError("Deferred Exact-IG requires Answer-only RAGEN")
        candidate_ids = {
            str(record.trajectory_id)
            for group in self._attempt_context.get("groups", ())
            for record in group.trajectories
        }
        if len(candidate_ids) != candidate_count or not candidate_ids.issubset(
            set(task_store)
        ):
            raise RuntimeError("Deferred Exact-IG candidate accounting diverged")
        selected_count = len(selected_records)
        self._attempt_context["exact_ig_scored_after_selection"] = selected_count
        self._attempt_context["deferred_exact_ig_metrics"] = {
            "ragen_signal_mode": ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
            "candidate_trajectory_count": candidate_count,
            "selected_trajectory_count": selected_count,
            "exact_ig_scored_before": 0,
            "exact_ig_scored_after": selected_count,
            "theoretical_exact_ig_reduction_ratio": (
                1.0 - selected_count / candidate_count
                if candidate_count
                else 0.0
            ),
            "selected_prompt_ids": list(selected_ids),
        }
        # Candidate task objects are no longer needed once selected records
        # carry their Exact-IG results.  Release non-selected task payloads
        # before learner packing and checkpoint commit.
        task_store.clear()
        return finalized

    def selected_microbatches(
        self,
        groups: Sequence[PromptGroup],
    ) -> Sequence[Any]:
        import ray

        self._require_bound()
        group_size = int(self.config["rollout"]["group_size"])
        advantage_started = time.perf_counter()
        sc_math_metrics: dict[str, float | int] = {}
        advantage_config = dict(self.config["advantage"])
        advantage_config["mica"] = dict(self.config.get("mica", {}))
        prepared = prepare_selected_trajectories(
            groups,
            expected_group_size=group_size,
            advantage_config=advantage_config,
            expected_policy_version=int(self._last_snapshot_step),
            expected_scorer_version=PRODUCTION_TASK_SCORER_VERSION,
            stop_continue_metrics=sc_math_metrics,
        )
        runtime_sc_metrics = dict(
            self._attempt_context.get("sc_runtime_metrics", {})
        )
        runtime_sc_metrics.update(sc_math_metrics)
        self._attempt_context["sc_metrics"] = runtime_sc_metrics
        self._prepared_groups = tuple(tuple(group) for group in prepared)
        self._validate_and_record_search_advantage_components()
        assignments = pack_prompt_groups_by_action_tokens(
            prepared,
            world_size=self._rl_world_size,
        )
        micro_batch_size = self.config["formal_schedule"].get(
            "learner_micro_batch_size"
        )
        if micro_batch_size is None:
            raise RuntimeError("learner_micro_batch_size is not approved")
        pad_token_id = _resolve_pad_token_id(
            str(self.config["paths"]["actor_model"])
        )
        role_gate = dict(self.config["advantage"].get("role_localized_gate", {}))
        role_mode = (
            str(self.config["advantage"].get("search_task_mode"))
            == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
        )
        lambda_decision = float(role_gate["lambda_decision"]) if role_mode else 0.0
        lambda_query = float(role_gate["lambda_query"]) if role_mode else 0.0
        rounds = build_synchronized_microbatch_rounds(
            assignments,
            micro_batch_size_per_rank=int(micro_batch_size),
            pad_token_id=pad_token_id,
            snapshot_step=self._last_snapshot_step,
            global_prompt_count=len(groups),
            group_size=group_size,
            action_state_chunk_size=int(
                self.config["policy"]["kl_action_state_chunk_size"]
            ),
            vocabulary_chunk_size=int(
                self.config["policy"]["kl_vocabulary_chunk_size"]
            ),
            kl_coefficient=float(self.config["policy"]["kl_coefficient"]),
            lambda_decision=lambda_decision,
            lambda_query=lambda_query,
        )
        self._add_phase_time(
            "advantage",
            time.perf_counter() - advantage_started,
        )
        sleep_started = time.perf_counter()
        self.agent_loop_manager.sleep_for_scoring()
        self._add_phase_time(
            "vllm_sleep_wake",
            time.perf_counter() - sleep_started,
        )
        old_started = time.perf_counter()
        materialized: list[tuple[dict[str, Any], ...]] = []
        for rank_payloads in rounds:
            references = [ray.put(payload) for payload in rank_payloads]
            old_by_rank = self.worker_group.execute_all_sync(
                "materialize_old_logprobs",
                references,
            )
            payloads = []
            for payload, old in zip(
                rank_payloads,
                old_by_rank,
                strict=True,
            ):
                updated = dict(payload)
                updated["old_logprobs"] = old
                payloads.append(updated)
            materialized.append(tuple(payloads))
        self._add_phase_time(
            "old_logprob",
            time.perf_counter() - old_started,
        )
        return tuple(materialized)

    def _validate_sufficiency_novelty_search_advantages(self) -> None:
        """Assert the exact production formula at the learner payload boundary."""

        lambda_outcome = float(self.config["advantage"]["lambda_outcome"])
        lambda_format = float(self.config["advantage"]["lambda_format"])
        local_ig_values: list[float] = []
        local_ig_hat_values: list[float] = []
        a_search_values: list[float] = []
        a_answer_values: list[float] = []
        sufficient_count = 0
        no_new_count = 0
        sufficient_and_no_new_count = 0
        normal_count = 0
        exact_query_repeat_count = 0
        different_query_no_new_count = 0
        state_count = 0
        answer_formula_assertion_count = 0
        searched_trajectory_count = 0
        no_search_trajectory_count = 0

        for group in self._prepared_groups:
            for item in group:
                advantage = item.advantage
                if advantage is None:
                    raise RuntimeError(
                        f"{item.record.trajectory_id}: selected trajectory has no advantage"
                    )
                search_indices = set(advantage.search_advantage)
                if search_indices:
                    searched_trajectory_count += 1
                else:
                    no_search_trajectory_count += 1
                if (
                    advantage.future_ig_sum
                    or advantage.accumulated_ig_count
                    or advantage.future_ig_rescaled
                ):
                    raise RuntimeError("Future IG entered the production Search path")
                if advantage.stop_continue_by_search_index:
                    raise RuntimeError("A_SC entered the production Search path")
                if advantage.search_task_advantage:
                    raise RuntimeError("Legacy Search task advantage is still populated")
                if (
                    set(advantage.sufficient_before_search) != search_indices
                    or set(advantage.no_new_observation) != search_indices
                    or set(advantage.search_branch_by_search_index)
                    != search_indices
                ):
                    raise RuntimeError("S/N/local-IG component coverage mismatch")
                turns_by_search = {
                    int(turn.search_index): turn
                    for turn in item.record.turns
                    if turn.turn_type is TurnType.SEARCH
                    and turn.search_index is not None
                    and turn.policy_credit_eligible
                }
                if set(turns_by_search) != search_indices:
                    raise RuntimeError("Learner Search turn coverage mismatch")
                for search_index in sorted(search_indices):
                    sufficient = advantage.sufficient_before_search[search_index]
                    no_new = advantage.no_new_observation[search_index]
                    if not isinstance(sufficient, bool) or not isinstance(no_new, bool):
                        raise RuntimeError("S and N must be binary bools")
                    local_ig_hat = float(
                        advantage.normalized_ig.get(search_index, 0.0)
                    )
                    actual = float(advantage.search_advantage[search_index])
                    expected = (
                        -1.0
                        if sufficient
                        else -1.0
                        if no_new
                        else local_ig_hat
                    )
                    if not all(
                        math.isfinite(value)
                        for value in (local_ig_hat, actual, expected)
                    ):
                        raise RuntimeError("Search advantage is non-finite")
                    if not math.isclose(
                        actual,
                        expected,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ):
                        raise RuntimeError(
                            "A_search != -1 if S else -1 if N else local_ig_hat"
                        )
                    turn = turns_by_search[search_index]
                    learner_value = float(
                        item.advantage_by_turn[int(turn.turn_index)]
                    )
                    if not math.isclose(
                        learner_value,
                        actual,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ):
                        raise RuntimeError(
                            "Validated Search advantage did not reach learner payload"
                        )
                    expected_branch = (
                        "sufficient_before_search"
                        if sufficient
                        else "no_new_observation"
                        if no_new
                        else "normalized_local_ig"
                    )
                    if (
                        advantage.search_branch_by_search_index[search_index]
                        != expected_branch
                    ):
                        raise RuntimeError("Search formula branch metadata is wrong")
                    state_count += 1
                    sufficient_count += int(sufficient)
                    no_new_count += int(no_new)
                    sufficient_and_no_new_count += int(sufficient and no_new)
                    normal_count += int(not sufficient and not no_new)
                    exact_query_repeat_count += int(turn.exact_query_repeat)
                    different_query_no_new_count += int(
                        turn.different_query_no_new_passage
                    )
                    if search_index in item.record.immediate_ig:
                        local_ig_values.append(
                            float(item.record.immediate_ig[search_index])
                        )
                    local_ig_hat_values.append(local_ig_hat)
                    a_search_values.append(actual)

                if advantage.answer_advantage is not None:
                    expected_answer = (
                        lambda_outcome * float(advantage.normalized_outcome)
                        + lambda_format
                        * float(advantage.centered_format_indicator)
                    )
                    if not math.isclose(
                        float(advantage.answer_advantage),
                        expected_answer,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ):
                        raise RuntimeError("A_answer != z_O + A_format")
                    answer_formula_assertion_count += 1
                    a_answer_values.append(float(advantage.answer_advantage))

        def summary(prefix: str, values: Sequence[float]) -> dict[str, float | int]:
            if not values:
                return {
                    f"{prefix}_count": 0,
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_std": 0.0,
                    f"{prefix}_min": 0.0,
                    f"{prefix}_max": 0.0,
                }
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_count": int(array.size),
                f"{prefix}_mean": float(array.mean()),
                f"{prefix}_std": float(array.std(ddof=0)),
                f"{prefix}_min": float(array.min()),
                f"{prefix}_max": float(array.max()),
            }

        metrics: dict[str, float | int | bool | str] = {}
        metrics.update(summary("local_ig", local_ig_values))
        metrics.update(summary("local_ig_hat", local_ig_hat_values))
        metrics.update(summary("A_search", a_search_values))
        metrics.update(summary("A_answer", a_answer_values))
        metrics.update(
            {
                "search_advantage_formula_assertion_pass": True,
                "answer_advantage_formula_assertion_pass": True,
                "advantage_component_coverage_pass": True,
                "S_count": sufficient_count,
                "S_rate": sufficient_count / state_count if state_count else 0.0,
                "N_count": no_new_count,
                "N_rate": no_new_count / state_count if state_count else 0.0,
                "S_and_N_count": sufficient_and_no_new_count,
                "normal_local_ig_branch_count": normal_count,
                "normal_local_ig_branch_rate": (
                    normal_count / state_count if state_count else 0.0
                ),
                "exact_query_repeat_count": exact_query_repeat_count,
                "exact_query_repeat_rate": (
                    exact_query_repeat_count / state_count if state_count else 0.0
                ),
                "no_new_passage_count": no_new_count,
                "no_new_passage_rate": (
                    no_new_count / state_count if state_count else 0.0
                ),
                "different_query_no_new_passage_count": (
                    different_query_no_new_count
                ),
                "different_query_no_new_passage_rate": (
                    different_query_no_new_count / state_count
                    if state_count
                    else 0.0
                ),
                "search_state_count": state_count,
                "search_z_o_entry_count": 0,
                "search_a_sc_entry_count": 0,
                "future_ig_contribution_count": 0,
                "sqrt_n_rescale_call_count": 0,
                "external_ig_multiplier_call_count": 0,
                "answer_formula_assertion_count": answer_formula_assertion_count,
                "searched_trajectory_count": searched_trajectory_count,
                "no_search_trajectory_count": no_search_trajectory_count,
                "search_advantage_formula": (
                    "-1.0 if S else -1.0 if N else normalized_local_ig"
                ),
                "outcome_fallback_to_search": False,
                "a_sc_shadow_only": True,
            }
        )
        trajectory_count = searched_trajectory_count + no_search_trajectory_count
        metrics["no_search_trajectory_rate"] = (
            no_search_trajectory_count / trajectory_count
            if trajectory_count
            else 0.0
        )
        self._attempt_context["advantage_component_metrics"] = metrics

    def _validate_probe_routed_search_advantages(self) -> None:
        """Independently recompute the new formula at the learner boundary."""

        probe_epsilon = float(self.config["advantage"]["probe_epsilon"])
        if not math.isclose(probe_epsilon, 1.0e-6, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError("Probe-routed Outcome epsilon drifted")
        lambda_outcome = float(self.config["advantage"]["lambda_outcome"])
        lambda_format = float(self.config["advantage"]["lambda_format"])
        state_count = 0
        s_before_count = 0
        s_after_count = 0
        post_count = 0
        n_count = 0
        normal_count = 0
        s_and_n_count = 0
        truncated_count = 0
        n_continued_count = 0
        z_normal_entry_count = 0
        answer_count = 0
        budget_exhausted_count = 0
        budget_exhausted_post_probe_count = 0
        budget_exhausted_o_route_nonzero_count = 0
        budget_exhausted_ig_entry_count = 0
        budget_exhausted_a_search_not_minus_one_count = 0
        normal_search_missing_post_prefix_count = 0
        cumulative_counts: list[float] = []
        d_values: list[float] = []
        delta_values: list[float] = []
        route_values: list[float] = []
        a_search_values: list[float] = []
        local_values: list[float] = []
        local_hat_values: list[float] = []
        by_index: dict[int, list[float]] = defaultdict(list)

        def probe_state(raw: Mapping[str, Any], stage: str) -> tuple[bool, float]:
            for name in (
                "parser_success",
                "no_answer",
                "output_truncated",
                "alias_aware_exact",
                "prefix_provenance_valid",
                "detached",
            ):
                if not isinstance(raw.get(name), bool):
                    raise RuntimeError(f"{stage} Probe {name} is not bool")
            if raw["prefix_provenance_valid"] is not True or raw["detached"] is not True:
                raise RuntimeError(f"{stage} Probe provenance/detach contract failed")
            sufficient = bool(
                raw["alias_aware_exact"]
                and raw["parser_success"]
                and not raw["no_answer"]
                and not raw["output_truncated"]
            )
            field = (
                "sufficient_before_search"
                if stage == "pre"
                else "sufficient_after_search"
            )
            if raw.get(field) is not sufficient:
                raise RuntimeError(f"{stage} Probe precomputed bool mismatch")
            versions = {
                int(raw[name])
                for name in (
                    "candidate_rollout_policy_version",
                    "exact_ig_policy_version",
                    "probe_policy_version",
                    "old_logprob_policy_version",
                )
            }
            if versions != {int(self._last_snapshot_step)}:
                raise RuntimeError(f"{stage} Probe policy versions mismatch")
            reward = float(raw["raw_task_reward"])
            if not math.isfinite(reward):
                raise RuntimeError(f"{stage} Probe reward is non-finite")
            return sufficient, reward

        for group in self._prepared_groups:
            for item in group:
                advantage = item.advantage
                if advantage is None:
                    raise RuntimeError("Selected trajectory has no advantage")
                if (
                    advantage.search_task_mode
                    != SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
                ):
                    raise RuntimeError("Prepared Search advantage uses wrong mode")
                if (
                    advantage.future_ig_sum
                    or advantage.accumulated_ig_count
                    or advantage.future_ig_rescaled
                ):
                    raise RuntimeError("Legacy future IG entered new mode")
                if advantage.stop_continue_by_search_index or advantage.search_task_advantage:
                    raise RuntimeError("A_SC entered the new Actor Search credit")
                all_turns = {
                    int(turn.search_index): turn
                    for turn in item.record.turns
                    if turn.turn_type is TurnType.SEARCH
                    and turn.search_index is not None
                }
                search_indices = set(advantage.search_advantage)
                routed = item.record.metadata.get("routed_answer_probes", {})
                s_before: dict[int, bool] = {}
                s_after: dict[int, bool] = {}
                pre_reward: dict[int, float] = {}
                post_reward: dict[int, float] = {}
                budget_exhausted_by_search: dict[int, bool] = {}
                for search_index in sorted(all_turns):
                    stages = routed.get(search_index, routed.get(str(search_index)))
                    if not isinstance(stages, Mapping) or not isinstance(
                        stages.get("pre"), Mapping
                    ):
                        raise RuntimeError("Learner boundary is missing pre Probe")
                    before, reward_before = probe_state(stages["pre"], "pre")
                    s_before[search_index] = before
                    pre_reward[search_index] = reward_before
                    budget_exhausted = is_budget_exhausted_terminal_search(
                        item.record,
                        search_index,
                    )
                    budget_exhausted_by_search[search_index] = budget_exhausted
                    budget_exhausted_count += int(budget_exhausted)
                    raw_post = stages.get("post")
                    if budget_exhausted:
                        if raw_post is not None:
                            budget_exhausted_post_probe_count += 1
                            raise RuntimeError(
                                "Budget-exhausted Search carried a post Probe"
                            )
                    elif before:
                        if raw_post is not None:
                            raise RuntimeError("Post Probe exists after S_before=1")
                    else:
                        if not isinstance(raw_post, Mapping):
                            normal_search_missing_post_prefix_count += 1
                            raise RuntimeError("Learner boundary is missing post Probe")
                        after, reward_after = probe_state(raw_post, "post")
                        s_after[search_index] = after
                        post_reward[search_index] = reward_after
                        post_count += 1
                        s_after_count += int(after)
                if advantage.sufficient_before_search != s_before:
                    raise RuntimeError("S_before metadata differs at learner boundary")
                if advantage.sufficient_after_search != s_after:
                    raise RuntimeError("S_after metadata differs at learner boundary")
                expected_n = {
                    index: bool(turn.no_new_observation)
                    for index, turn in all_turns.items()
                }
                if advantage.no_new_observation != expected_n:
                    raise RuntimeError("N metadata differs at learner boundary")

                for search_index in sorted(search_indices):
                    sufficient = s_before[search_index]
                    no_new = expected_n[search_index]
                    state_count += 1
                    s_before_count += int(sufficient)
                    n_count += int(no_new)
                    s_and_n_count += int(sufficient and no_new)
                    local_hat = float(
                        advantage.normalized_ig.get(search_index, 0.0)
                    )
                    local_hat_values.append(local_hat)
                    budget_exhausted = budget_exhausted_by_search[search_index]
                    if budget_exhausted and (
                        search_index in item.record.immediate_ig
                        or search_index in advantage.normalized_ig
                    ):
                        budget_exhausted_ig_entry_count += 1
                        raise RuntimeError(
                            "Budget-exhausted Search entered Exact-IG credit"
                        )
                    if search_index in item.record.immediate_ig:
                        local_values.append(
                            float(item.record.immediate_ig[search_index])
                        )
                    if sufficient:
                        expected = -1.0
                        expected_branch = "sufficient_before_search"
                    elif no_new:
                        expected = -1.0
                        expected_branch = "no_new_observation"
                    else:
                        values: list[float] = []
                        saw_n = False
                        continued_after_n = False
                        for future_index in sorted(
                            index for index in all_turns if index >= search_index
                        ):
                            if s_before[future_index]:
                                break
                            turn = all_turns[future_index]
                            valid = bool(
                                turn.policy_credit_eligible
                                and turn.ig_reward_eligible
                                and not expected_n[future_index]
                            )
                            if valid:
                                values.append(
                                    float(advantage.normalized_ig[future_index])
                                )
                                continued_after_n = continued_after_n or saw_n
                            elif expected_n[future_index]:
                                saw_n = True
                            if s_after.get(future_index, False):
                                truncated_count += 1
                                break
                        if not values:
                            raise RuntimeError("Learner cumulative IG is empty")
                        n_continued_count += int(continued_after_n)
                        d_ig_eff = float(math.fsum(values) / math.sqrt(len(values)))
                        delta = float(
                            post_reward[search_index] - pre_reward[search_index]
                        )
                        z_outcome = float(advantage.normalized_outcome)
                        if delta > probe_epsilon:
                            route = max(z_outcome, 0.0)
                        elif delta < -probe_epsilon:
                            route = min(z_outcome, 0.0)
                        else:
                            route = 0.0
                        expected = float(d_ig_eff + route)
                        expected_branch = "cumulative_ig_probe_routed_outcome"
                        if not math.isclose(
                            float(advantage.effective_cumulative_ig[search_index]),
                            d_ig_eff,
                            rel_tol=0.0,
                            abs_tol=0.0,
                        ):
                            raise RuntimeError("D_ig_eff changed before learner")
                        if int(
                            advantage.effective_cumulative_ig_count[search_index]
                        ) != len(values):
                            raise RuntimeError("D_ig_eff count changed before learner")
                        if not math.isclose(
                            float(advantage.probe_reward_delta[search_index]),
                            delta,
                            rel_tol=0.0,
                            abs_tol=0.0,
                        ):
                            raise RuntimeError("delta_probe changed before learner")
                        if not math.isclose(
                            float(advantage.routed_outcome[search_index]),
                            route,
                            rel_tol=0.0,
                            abs_tol=0.0,
                        ):
                            raise RuntimeError("O_route changed before learner")
                        normal_count += 1
                        z_normal_entry_count += int(route != 0.0)
                        cumulative_counts.append(float(len(values)))
                        d_values.append(d_ig_eff)
                        delta_values.append(delta)
                        route_values.append(route)
                        by_index[search_index].append(d_ig_eff)
                    actual = float(advantage.search_advantage[search_index])
                    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.0):
                        raise RuntimeError(
                            "A_search != S/N priority then D_ig_eff + O_route"
                        )
                    if budget_exhausted:
                        budget_exhausted_o_route_nonzero_count += int(
                            float(advantage.routed_outcome.get(search_index, 0.0))
                            != 0.0
                        )
                        budget_exhausted_a_search_not_minus_one_count += int(
                            actual != -1.0
                        )
                        if (
                            search_index in advantage.sufficient_after_search
                            or search_index in advantage.probe_reward_delta
                            or search_index in advantage.routed_outcome
                            or search_index in advantage.effective_cumulative_ig
                        ):
                            raise RuntimeError(
                                "Budget-exhausted Search entered post/Normal metadata"
                            )
                    if (
                        advantage.search_branch_by_search_index[search_index]
                        != expected_branch
                    ):
                        raise RuntimeError("Search branch metadata is incorrect")
                    turn = all_turns[search_index]
                    learner_value = float(
                        item.advantage_by_turn[int(turn.turn_index)]
                    )
                    if not math.isclose(
                        learner_value,
                        expected,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ):
                        raise RuntimeError("Expected A_search did not reach learner")
                    a_search_values.append(actual)
                if advantage.answer_advantage is not None:
                    expected_answer = (
                        lambda_outcome * float(advantage.normalized_outcome)
                        + lambda_format
                        * float(advantage.centered_format_indicator)
                    )
                    if not math.isclose(
                        float(advantage.answer_advantage),
                        expected_answer,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ):
                        raise RuntimeError("A_answer != z_O + A_format")
                    answer_count += 1

        def summary(prefix: str, values: Sequence[float]) -> dict[str, float | int]:
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_count": int(array.size),
                f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
                f"{prefix}_std": float(array.std(ddof=0)) if array.size else 0.0,
                f"{prefix}_min": float(array.min()) if array.size else 0.0,
                f"{prefix}_max": float(array.max()) if array.size else 0.0,
            }

        if any(
            (
                budget_exhausted_post_probe_count,
                budget_exhausted_o_route_nonzero_count,
                budget_exhausted_ig_entry_count,
                budget_exhausted_a_search_not_minus_one_count,
                normal_search_missing_post_prefix_count,
            )
        ):
            raise RuntimeError(
                "Budget-exhausted Search learner safety counters are non-zero"
            )

        metrics: dict[str, Any] = {
            "search_advantage_formula_assertion_pass": True,
            "answer_advantage_formula_assertion_pass": True,
            "search_state_count": state_count,
            "S_before_count": s_before_count,
            "S_before_rate": s_before_count / state_count if state_count else 0.0,
            "S_after_count": s_after_count,
            "S_after_rate": s_after_count / post_count if post_count else 0.0,
            "N_count": n_count,
            "N_rate": n_count / state_count if state_count else 0.0,
            "S_and_N_count": s_and_n_count,
            "normal_count": normal_count,
            "normal_rate": normal_count / state_count if state_count else 0.0,
            "cumulative_truncated_by_S_after_count": truncated_count,
            "N_masked_future_continued_count": n_continued_count,
            "z_O_normal_search_entry_count": z_normal_entry_count,
            "z_O_S_or_N_search_entry_count": 0,
            "A_SC_search_entry_count": 0,
            "future_IG_cross_S_after_boundary_count": 0,
            "budget_exhausted_count": budget_exhausted_count,
            "budget_exhausted_post_probe_count": (
                budget_exhausted_post_probe_count
            ),
            "budget_exhausted_o_route_nonzero_count": (
                budget_exhausted_o_route_nonzero_count
            ),
            "budget_exhausted_ig_entry_count": budget_exhausted_ig_entry_count,
            "budget_exhausted_A_search_not_minus_one_count": (
                budget_exhausted_a_search_not_minus_one_count
            ),
            "normal_search_missing_post_prefix_count": (
                normal_search_missing_post_prefix_count
            ),
            "answer_formula_assertion_count": answer_count,
            "search_advantage_formula": (
                "-1 if S_before else -1 if N else D_ig_eff + O_route"
            ),
        }
        metrics.update(summary("local_ig", local_values))
        metrics.update(summary("local_ig_hat", local_hat_values))
        metrics.update(summary("cumulative_effective_count", cumulative_counts))
        metrics.update(summary("D_ig_eff", d_values))
        metrics.update(summary("delta_probe", delta_values))
        metrics.update(summary("O_route", route_values))
        metrics.update(summary("A_search", a_search_values))
        metrics["delta_probe_positive_rate"] = (
            sum(value > probe_epsilon for value in delta_values)
            / len(delta_values)
            if delta_values
            else 0.0
        )
        metrics["delta_probe_neutral_rate"] = (
            sum(abs(value) <= probe_epsilon for value in delta_values)
            / len(delta_values)
            if delta_values
            else 0.0
        )
        metrics["delta_probe_negative_rate"] = (
            sum(value < -probe_epsilon for value in delta_values)
            / len(delta_values)
            if delta_values
            else 0.0
        )
        metrics["O_route_positive_rate"] = (
            sum(value > 0.0 for value in route_values) / len(route_values)
            if route_values
            else 0.0
        )
        metrics["O_route_zero_rate"] = (
            sum(value == 0.0 for value in route_values) / len(route_values)
            if route_values
            else 0.0
        )
        metrics["O_route_negative_rate"] = (
            sum(value < 0.0 for value in route_values) / len(route_values)
            if route_values
            else 0.0
        )
        metrics["A_search_positive_rate"] = (
            sum(value > 0.0 for value in a_search_values) / len(a_search_values)
            if a_search_values
            else 0.0
        )
        metrics["A_search_zero_rate"] = (
            sum(value == 0.0 for value in a_search_values) / len(a_search_values)
            if a_search_values
            else 0.0
        )
        metrics["A_search_negative_rate"] = (
            sum(value < 0.0 for value in a_search_values) / len(a_search_values)
            if a_search_values
            else 0.0
        )
        for search_index, values in sorted(by_index.items()):
            metrics.update(summary(f"D_ig_eff_t{search_index}", values))
        self._attempt_context["advantage_component_metrics"] = metrics

    def _validate_role_localized_gate_search_advantages(self) -> None:
        """Independently rebuild role-localized credits at the learner boundary."""

        advantage_config = dict(self.config["advantage"])
        role_config = dict(advantage_config["role_localized_gate"])
        lambda_decision = float(role_config["lambda_decision"])
        lambda_query = float(role_config["lambda_query"])
        if not (
            math.isfinite(lambda_decision)
            and math.isfinite(lambda_query)
            and 0.0 <= lambda_decision <= 1.0
            and 0.0 <= lambda_query <= 1.0
        ):
            raise RuntimeError("Calibrated role-localized lambdas are invalid")
        probe_epsilon = float(advantage_config["probe_epsilon"])
        if probe_epsilon != 1.0e-6:
            raise RuntimeError("Role-localized probe epsilon drifted")

        branch_counts: dict[str, int] = defaultdict(int)
        branch_depth_domain_counts: dict[tuple[str, int, str], int] = defaultdict(int)
        main_values: list[float] = []
        decision_values: list[float] = []
        query_values: list[float] = []
        allowed_soft_overlap_count = 0
        empty_query_count = 0
        soft_n_main_count = 0
        answer_formula_count = 0
        budget_post_probe_count = 0
        budget_ig_count = 0
        budget_main_nonzero_count = 0
        observation_mask_violation_count = 0
        decision_query_token_overlap_count = 0

        def probe_state(
            raw: Mapping[str, Any],
            *,
            stage: str,
        ) -> tuple[bool, float]:
            required_bools = (
                "parser_success",
                "no_answer",
                "output_truncated",
                "alias_aware_exact",
                "prefix_provenance_valid",
                "detached",
            )
            if any(not isinstance(raw.get(name), bool) for name in required_bools):
                raise RuntimeError(f"{stage} Probe bool metadata is invalid")
            if raw["prefix_provenance_valid"] is not True or raw["detached"] is not True:
                raise RuntimeError(f"{stage} Probe provenance/detach contract failed")
            if int(raw.get("completion_count", -1)) != 1:
                raise RuntimeError(f"{stage} Probe completion count is not one")
            versions = {
                int(raw[name])
                for name in (
                    "candidate_rollout_policy_version",
                    "exact_ig_policy_version",
                    "probe_policy_version",
                    "old_logprob_policy_version",
                )
            }
            if versions != {int(self._last_snapshot_step)}:
                raise RuntimeError(f"{stage} Probe policy version mismatch")
            if str(raw.get("task_scorer_version")) != PRODUCTION_TASK_SCORER_VERSION:
                raise RuntimeError(f"{stage} Probe task scorer version mismatch")
            sufficient = bool(
                raw["alias_aware_exact"]
                and raw["parser_success"]
                and not raw["no_answer"]
                and not raw["output_truncated"]
            )
            stored_name = (
                "sufficient_before_search"
                if stage == "pre"
                else "sufficient_after_search"
            )
            if raw.get(stored_name) is not sufficient:
                raise RuntimeError(f"{stage} Probe sufficiency bool was not recomputed")
            reward = float(raw["raw_task_reward"])
            if not math.isfinite(reward):
                raise RuntimeError(f"{stage} Probe reward is non-finite")
            return sufficient, reward

        for group in self._prepared_groups:
            for item in group:
                advantage = item.advantage
                if advantage is None or advantage.search_task_mode != (
                    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
                ):
                    raise RuntimeError("Prepared trajectory uses the wrong Search mode")
                if advantage.search_advantage != advantage.search_main_advantage:
                    raise RuntimeError("Main credit changed before learner transport")
                if advantage.stop_continue_by_search_index:
                    raise RuntimeError("A_SC entered role-localized Actor credit")
                turns = {
                    int(turn.search_index): turn
                    for turn in item.record.turns
                    if turn.turn_type is TurnType.SEARCH
                    and turn.search_index is not None
                    and turn.policy_credit_eligible
                }
                indices = set(turns)
                if not (
                    set(advantage.search_main_advantage)
                    == set(advantage.search_decision_advantage)
                    == set(advantage.search_query_advantage)
                    == set(advantage.search_branch_by_search_index)
                    == indices
                ):
                    raise RuntimeError("Role-localized credit coverage mismatch")
                routed = item.record.metadata.get("routed_answer_probes", {})
                pre_sufficient: dict[int, bool] = {}
                post_sufficient: dict[int, bool] = {}
                pre_reward: dict[int, float] = {}
                post_reward: dict[int, float] = {}
                branches: dict[int, str] = {}
                for search_index, turn in sorted(turns.items()):
                    stages = routed.get(search_index, routed.get(str(search_index)))
                    if not isinstance(stages, Mapping) or not isinstance(
                        stages.get("pre"), Mapping
                    ):
                        raise RuntimeError("Role-localized pre Probe is missing")
                    before, reward_before = probe_state(stages["pre"], stage="pre")
                    branch = classify_role_localized_search_branch(
                        retrieval_budget_exhausted=bool(
                            turn.retrieval_budget_exhausted
                        ),
                        model_search_invalid=bool(turn.model_search_invalid),
                        sufficient_before_search=before,
                        retriever_executed=bool(turn.retriever_executed),
                        no_new_observation=turn.no_new_observation,
                    )
                    if branch != turn.branch_type or branch != (
                        advantage.search_branch_by_search_index[search_index]
                    ):
                        raise RuntimeError("Search branch priority/metadata mismatch")
                    pre_sufficient[search_index] = before
                    pre_reward[search_index] = reward_before
                    branches[search_index] = branch
                    raw_post = stages.get("post")
                    if branch in {
                        ROLE_LOCALIZED_BRANCH_N_BUDGET,
                        ROLE_LOCALIZED_BRANCH_N_INVALID,
                        ROLE_LOCALIZED_BRANCH_S_BEFORE,
                    }:
                        if raw_post is not None:
                            if branch == ROLE_LOCALIZED_BRANCH_N_BUDGET:
                                budget_post_probe_count += 1
                            raise RuntimeError(f"{branch} cannot carry a post Probe")
                    else:
                        if not isinstance(raw_post, Mapping):
                            raise RuntimeError("Main-eligible Search lacks post Probe")
                        after, reward_after = probe_state(raw_post, stage="post")
                        post_sufficient[search_index] = after
                        post_reward[search_index] = reward_after

                if advantage.sufficient_before_search != pre_sufficient:
                    raise RuntimeError("S_before changed before learner")
                if advantage.sufficient_after_search != post_sufficient:
                    raise RuntimeError("S_after changed before learner")

                expected_decision_by_turn: dict[int, float] = {}
                expected_query_by_turn: dict[int, float] = {}
                expected_decision_mask = [0] * len(item.record.input_ids)
                expected_query_mask = [0] * len(item.record.input_ids)
                domain = str(item.record.metadata.get("data_source", "unknown"))
                for search_index, turn in sorted(turns.items()):
                    branch = branches[search_index]
                    branch_counts[branch] += 1
                    branch_depth_domain_counts[(domain, search_index, branch)] += 1
                    expected_decision = (
                        -1.0
                        if branch == ROLE_LOCALIZED_BRANCH_N_BUDGET
                        else -0.5
                        if branch in {
                            ROLE_LOCALIZED_BRANCH_N_INVALID,
                            ROLE_LOCALIZED_BRANCH_S_BEFORE,
                        }
                        else 0.0
                    )
                    query_start, query_end = map(int, turn.query_token_span)
                    query_count = query_end - query_start
                    raw_ig = item.record.immediate_ig.get(search_index)
                    duplicate_gate = bool(
                        branch == ROLE_LOCALIZED_BRANCH_N_SOFT
                        and turn.exact_query_repeat
                        and int(turn.new_passage_count) == 0
                        and raw_ig is not None
                        and float(raw_ig) <= 0.0
                    )
                    expected_query = (
                        -0.5
                        if branch == ROLE_LOCALIZED_BRANCH_N_INVALID
                        and query_count > 0
                        else -0.25
                        if duplicate_gate
                        else 0.0
                    )
                    if branch == ROLE_LOCALIZED_BRANCH_N_INVALID and query_count == 0:
                        empty_query_count += 1
                    if branch in {
                        ROLE_LOCALIZED_BRANCH_N_SOFT,
                        ROLE_LOCALIZED_BRANCH_NORMAL,
                    }:
                        values: list[float] = []
                        for future_index in sorted(
                            value for value in turns if value >= search_index
                        ):
                            future_branch = branches[future_index]
                            if future_branch == ROLE_LOCALIZED_BRANCH_S_BEFORE:
                                break
                            if future_branch in {
                                ROLE_LOCALIZED_BRANCH_N_SOFT,
                                ROLE_LOCALIZED_BRANCH_NORMAL,
                            }:
                                future_turn = turns[future_index]
                                if not (
                                    future_turn.policy_credit_eligible
                                    and future_turn.ig_reward_eligible
                                    and future_turn.main_credit_eligible
                                    and future_index in advantage.normalized_ig
                                ):
                                    raise RuntimeError("Main IG eligibility mismatch")
                                values.append(
                                    float(advantage.normalized_ig[future_index])
                                )
                            if post_sufficient.get(future_index, False):
                                break
                        if not values:
                            raise RuntimeError("D_ig_eff has no eligible IG values")
                        expected_d = float(
                            math.fsum(values) / math.sqrt(len(values))
                        )
                        expected_delta = float(
                            post_reward[search_index] - pre_reward[search_index]
                        )
                        z_outcome = float(advantage.normalized_outcome)
                        expected_route = (
                            max(z_outcome, 0.0)
                            if expected_delta > probe_epsilon
                            else min(z_outcome, 0.0)
                            if expected_delta < -probe_epsilon
                            else 0.0
                        )
                        expected_main = float(expected_d + expected_route)
                        if not math.isclose(
                            float(advantage.effective_cumulative_ig[search_index]),
                            expected_d,
                            rel_tol=0.0,
                            abs_tol=0.0,
                        ) or int(
                            advantage.effective_cumulative_ig_count[search_index]
                        ) != len(values):
                            raise RuntimeError("D_ig_eff changed before learner")
                        if float(
                            advantage.probe_reward_delta[search_index]
                        ) != expected_delta or float(
                            advantage.routed_outcome[search_index]
                        ) != expected_route:
                            raise RuntimeError("Probe Outcome routing changed")
                        soft_n_main_count += int(
                            branch == ROLE_LOCALIZED_BRANCH_N_SOFT
                        )
                    else:
                        expected_main = 0.0
                        if branch == ROLE_LOCALIZED_BRANCH_N_BUDGET:
                            budget_main_nonzero_count += int(
                                float(
                                    advantage.search_main_advantage[search_index]
                                )
                                != 0.0
                            )
                            budget_ig_count += int(
                                search_index in item.record.immediate_ig
                                or search_index
                                in advantage.effective_cumulative_ig
                            )
                    actual_main = float(
                        advantage.search_main_advantage[search_index]
                    )
                    actual_decision = float(
                        advantage.search_decision_advantage[search_index]
                    )
                    actual_query = float(
                        advantage.search_query_advantage[search_index]
                    )
                    if (actual_main, actual_decision, actual_query) != (
                        expected_main,
                        expected_decision,
                        expected_query,
                    ):
                        raise RuntimeError("Role-localized Credit formula mismatch")
                    turn_id = int(turn.turn_index)
                    if float(item.advantage_by_turn[turn_id]) != expected_main:
                        raise RuntimeError("A_main did not reach Main learner input")
                    if expected_decision:
                        expected_decision_by_turn[turn_id] = expected_decision
                        start, end = map(int, turn.decision_token_span)
                        for position in range(start, end):
                            expected_decision_mask[position] = 1
                    if expected_query:
                        expected_query_by_turn[turn_id] = expected_query
                        for position in range(query_start, query_end):
                            expected_query_mask[position] = 1
                    if expected_query == -0.25:
                        allowed_soft_overlap_count += 1
                    if turn.observation_token_span is not None:
                        start, end = map(int, turn.observation_token_span)
                        violations = sum(
                            int(item.record.policy_mask[position])
                            for position in range(start, end)
                        )
                        observation_mask_violation_count += violations
                        if violations:
                            raise RuntimeError("Observation entered policy mask")
                    main_values.append(actual_main)
                    decision_values.append(actual_decision)
                    query_values.append(actual_query)

                if item.decision_advantage_by_turn != expected_decision_by_turn:
                    raise RuntimeError("Decision credit did not reach learner payload")
                if item.query_advantage_by_turn != expected_query_by_turn:
                    raise RuntimeError("Query credit did not reach learner payload")
                if tuple(expected_decision_mask) != item.decision_token_mask:
                    raise RuntimeError("Decision learner mask differs from D span")
                if tuple(expected_query_mask) != item.query_token_mask:
                    raise RuntimeError("Query learner mask differs from Q span")
                item_decision_query_overlap_count = sum(
                    int(left and right)
                    for left, right in zip(
                        item.decision_token_mask,
                        item.query_token_mask,
                        strict=True,
                    )
                )
                decision_query_token_overlap_count += (
                    item_decision_query_overlap_count
                )
                if item_decision_query_overlap_count:
                    raise RuntimeError("Decision and Query token masks overlap")
                if advantage.answer_advantage is not None:
                    expected_answer = (
                        float(advantage_config["lambda_outcome"])
                        * float(advantage.normalized_outcome)
                        + float(advantage_config["lambda_format"])
                        * float(advantage.centered_format_indicator)
                    )
                    if not math.isclose(
                        float(advantage.answer_advantage),
                        expected_answer,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ):
                        raise RuntimeError("A_answer != z_O + A_format")
                    answer_formula_count += 1

        if any(
            (
                budget_post_probe_count,
                budget_ig_count,
                budget_main_nonzero_count,
                observation_mask_violation_count,
            )
        ):
            raise RuntimeError("Role-localized hard safety counters are non-zero")

        def summary(prefix: str, values: Sequence[float]) -> dict[str, Any]:
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_count": int(array.size),
                f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
                f"{prefix}_std": float(array.std(ddof=0)) if array.size else 0.0,
                f"{prefix}_nonzero_rate": (
                    float(np.count_nonzero(array)) / float(array.size)
                    if array.size
                    else 0.0
                ),
            }

        metrics: dict[str, Any] = {
            "search_advantage_formula_assertion_pass": True,
            "answer_advantage_formula_assertion_pass": True,
            "answer_formula_assertion_count": answer_formula_count,
            "role_gate/lambda_decision": lambda_decision,
            "role_gate/lambda_query": lambda_query,
            "role_gate/soft_n_reentered_main_count": soft_n_main_count,
            "role_gate/empty_query_without_query_span_count": empty_query_count,
            "role_gate/allowed_soft_duplicate_main_query_overlap_count": (
                allowed_soft_overlap_count
            ),
            "role_gate/nonzero_decision_and_query_same_token_count": (
                decision_query_token_overlap_count
            ),
            "role_gate/unexpected_nonzero_main_gate_overlap_count": 0,
            "role_gate/observation_policy_mask_violation_count": (
                observation_mask_violation_count
            ),
            "role_gate/budget_post_probe_count": budget_post_probe_count,
            "role_gate/budget_ig_entry_count": budget_ig_count,
            "role_gate/budget_main_nonzero_count": budget_main_nonzero_count,
        }
        for branch, count in sorted(branch_counts.items()):
            metrics[f"role_gate/branch_{branch}_count"] = count
        for (domain, depth, branch), count in sorted(
            branch_depth_domain_counts.items()
        ):
            metrics[
                f"role_gate/branch/{domain}/t{depth}/{branch}_count"
            ] = count
        metrics.update(summary("role_gate/A_main", main_values))
        metrics.update(summary("role_gate/A_decision", decision_values))
        metrics.update(summary("role_gate/A_query", query_values))
        self._attempt_context["advantage_component_metrics"] = metrics

    def _validate_mica_search_advantages(self) -> None:
        """Independently rebuild MICA credit at the learner boundary."""

        mica_config = dict(self.config.get("mica", {}))
        gamma = float(mica_config.get("gamma", float("nan")))
        alpha = float(mica_config.get("alpha", float("nan")))
        epsilon = float(self.config["advantage"]["normalization_epsilon"])
        variance_tolerance = float(
            self.config["advantage"]["zero_variance_tolerance"]
        )
        if gamma != 1.0 or alpha != 0.5:
            raise RuntimeError("MICA V1 gamma/alpha contract drifted")
        if str(mica_config.get("normalization_scope")) != "prompt_search_depth":
            raise RuntimeError("MICA normalization scope drifted")
        if str(mica_config.get("singleton_fallback")) != (
            "normalized_terminal_outcome"
        ):
            raise RuntimeError("MICA singleton fallback contract drifted")

        raw_values: list[float] = []
        return_values: list[float] = []
        local_values: list[float] = []
        ret_values: list[float] = []
        search_values: list[float] = []
        peer_counts: list[float] = []
        singleton_z_values: list[float] = []
        singleton_lengths: list[float] = []
        singleton_count = 0
        missing_count = 0
        loc_zero_variance_count = 0
        ret_zero_variance_count = 0
        answer_count = 0
        observation_mask_violation_count = 0
        role_gate_actor_loss_count = 0
        routed_outcome_entry_count = 0
        normal_terminal_outcome_entry_count = 0
        by_depth: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        def close(actual: float, expected: float, label: str) -> None:
            if not math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"MICA learner-boundary mismatch for {label}: "
                    f"{actual!r} != {expected!r}"
                )

        for prepared_group in self._prepared_groups:
            raw_by_trajectory = [
                {
                    int(index): float(value)
                    for index, value in item.record.immediate_ig.items()
                }
                for item in prepared_group
            ]
            returns_by_trajectory = []
            for raw in raw_by_trajectory:
                ordered = sorted(raw)
                returns_by_trajectory.append(
                    {
                        index: float(
                            math.fsum(
                                raw[future_index]
                                for future_index in ordered
                                if future_index >= index
                            )
                        )
                        for index in ordered
                    }
                )
            local_peers: dict[int, list[float]] = defaultdict(list)
            return_peers: dict[int, list[float]] = defaultdict(list)
            for raw, returns in zip(
                raw_by_trajectory,
                returns_by_trajectory,
                strict=True,
            ):
                for index, value in raw.items():
                    local_peers[index].append(value)
                    return_peers[index].append(returns[index])

            for trajectory_index, item in enumerate(prepared_group):
                advantage = item.advantage
                if advantage is None:
                    raise RuntimeError("MICA selected trajectory has no advantage")
                if advantage.search_task_mode != (
                    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE
                ):
                    raise RuntimeError("MICA mode was lost before the learner")
                if any(
                    (
                        advantage.stop_continue_by_search_index,
                        advantage.sufficient_before_search,
                        advantage.sufficient_after_search,
                        advantage.no_new_observation,
                        advantage.effective_cumulative_ig,
                        advantage.effective_cumulative_ig_count,
                        advantage.probe_reward_delta,
                        advantage.routed_outcome,
                        advantage.search_main_advantage,
                        advantage.search_decision_advantage,
                        advantage.search_query_advantage,
                    )
                ):
                    raise RuntimeError("Legacy Search credit entered MICA mode")
                if any(
                    (
                        item.decision_advantage_by_turn,
                        item.query_advantage_by_turn,
                        item.decision_token_mask,
                        item.query_token_mask,
                        item.decision_turn_ids,
                        item.query_turn_ids,
                    )
                ):
                    role_gate_actor_loss_count += 1
                    raise RuntimeError("A_decision/A_query entered MICA actor loss")
                if item.record.metadata.get("sufficiency_probes") or item.record.metadata.get(
                    "routed_answer_probes"
                ):
                    raise RuntimeError("Diagnostic Answer probes ran in MICA mode")
                for source, mask in zip(
                    item.record.token_sources,
                    item.record.policy_mask,
                    strict=True,
                ):
                    if source is not TokenSource.MODEL and bool(mask):
                        observation_mask_violation_count += 1
                        raise RuntimeError("Observation/prompt token entered MICA loss")

                expected_answer = (
                    float(self.config["advantage"]["lambda_outcome"])
                    * float(advantage.normalized_outcome)
                    + float(self.config["advantage"]["lambda_format"])
                    * float(advantage.centered_format_indicator)
                )
                if advantage.answer_policy_credit_eligible:
                    if advantage.answer_advantage is None:
                        raise RuntimeError("MICA Answer advantage is missing")
                    close(
                        float(advantage.answer_advantage),
                        expected_answer,
                        "A_answer",
                    )
                    answer_count += 1

                turns_by_search = {
                    int(turn.search_index): turn
                    for turn in item.record.turns
                    if turn.turn_type is TurnType.SEARCH
                    and turn.search_index is not None
                    and turn.policy_credit_eligible
                }
                if set(advantage.search_advantage) != set(turns_by_search):
                    raise RuntimeError("MICA Search credit coverage mismatch")
                raw = raw_by_trajectory[trajectory_index]
                returns = returns_by_trajectory[trajectory_index]
                for search_index, turn in sorted(turns_by_search.items()):
                    if not turn.ig_reward_eligible:
                        expected_search = 0.0
                        if search_index in raw or search_index in returns:
                            raise RuntimeError("Undefined Exact-IG was fabricated")
                        if advantage.mica_peer_count.get(search_index) != 0:
                            raise RuntimeError("Undefined Exact-IG increased peer count")
                        if advantage.mica_singleton_fallback.get(search_index):
                            raise RuntimeError("Undefined Exact-IG used singleton fallback")
                        if search_index not in advantage.mica_ig_missing_reason:
                            raise RuntimeError("Undefined Exact-IG reason was not recorded")
                        missing_count += 1
                    else:
                        local_array = np.asarray(
                            local_peers[search_index],
                            dtype=np.float64,
                        )
                        return_array = np.asarray(
                            return_peers[search_index],
                            dtype=np.float64,
                        )
                        peer_count = int(local_array.size)
                        if peer_count != int(return_array.size) or peer_count < 1:
                            raise RuntimeError("MICA peer group is invalid")
                        loc_mean = float(local_array.mean())
                        loc_std = float(local_array.std(ddof=0))
                        ret_mean = float(return_array.mean())
                        ret_std = float(return_array.std(ddof=0))
                        close(
                            advantage.mica_ig_return[search_index],
                            returns[search_index],
                            "G_IG",
                        )
                        if advantage.mica_peer_count.get(search_index) != peer_count:
                            raise RuntimeError("MICA peer count drifted")
                        for actual, expected, label in (
                            (advantage.mica_loc_mean[search_index], loc_mean, "mu_loc"),
                            (advantage.mica_loc_std[search_index], loc_std, "sigma_loc"),
                            (advantage.mica_ret_mean[search_index], ret_mean, "mu_ret"),
                            (advantage.mica_ret_std[search_index], ret_std, "sigma_ret"),
                        ):
                            close(actual, expected, label)
                        if peer_count == 1:
                            if not advantage.mica_singleton_fallback.get(search_index):
                                raise RuntimeError("Singleton Outcome fallback was skipped")
                            if search_index in advantage.mica_local_advantage or (
                                search_index in advantage.mica_return_advantage
                            ):
                                raise RuntimeError("Singleton computed relative MICA channels")
                            expected_search = float(advantage.normalized_outcome)
                            singleton_count += 1
                            singleton_z_values.append(expected_search)
                            by_depth[search_index]["singleton"].append(1.0)
                            by_depth[search_index]["singleton_Z_O"].append(
                                expected_search
                            )
                        else:
                            if advantage.mica_singleton_fallback.get(search_index):
                                raise RuntimeError("Outcome fallback entered a peer group")
                            expected_local = (
                                0.0
                                if loc_std * loc_std <= variance_tolerance
                                else (raw[search_index] - loc_mean) / (loc_std + epsilon)
                            )
                            expected_return = (
                                0.0
                                if ret_std * ret_std <= variance_tolerance
                                else (returns[search_index] - ret_mean)
                                / (ret_std + epsilon)
                            )
                            close(
                                advantage.mica_local_advantage[search_index],
                                expected_local,
                                "A_loc",
                            )
                            close(
                                advantage.mica_return_advantage[search_index],
                                expected_return,
                                "A_ret",
                            )
                            expected_search = 0.5 * expected_return + 0.5 * expected_local
                            loc_zero_variance_count += int(
                                loc_std * loc_std <= variance_tolerance
                            )
                            ret_zero_variance_count += int(
                                ret_std * ret_std <= variance_tolerance
                            )
                            local_values.append(expected_local)
                            ret_values.append(expected_return)
                            by_depth[search_index]["A_loc"].append(expected_local)
                            by_depth[search_index]["A_ret"].append(expected_return)
                            by_depth[search_index]["singleton"].append(0.0)
                        raw_values.append(raw[search_index])
                        return_values.append(returns[search_index])
                        peer_counts.append(float(peer_count))
                        by_depth[search_index]["peer_count"].append(
                            float(peer_count)
                        )
                        by_depth[search_index]["raw_ig"].append(raw[search_index])
                        by_depth[search_index]["ig_return"].append(
                            returns[search_index]
                        )
                    actual_search = float(advantage.search_advantage[search_index])
                    close(actual_search, expected_search, "A_search")
                    close(
                        item.advantage_by_turn[int(turn.turn_index)],
                        expected_search,
                        "learner A_search",
                    )
                    search_values.append(actual_search)
                    by_depth[search_index]["A_search"].append(actual_search)
                if advantage.mica_singleton_consecutive_length:
                    singleton_lengths.append(
                        float(advantage.mica_singleton_consecutive_length)
                    )

        def summarize(prefix: str, values: Sequence[float]) -> dict[str, float | int]:
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_count": int(array.size),
                f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
                f"{prefix}_std": float(array.std(ddof=0)) if array.size else 0.0,
            }

        state_count = len(search_values)
        singleton_positive_count = sum(
            value > 0.0 for value in singleton_z_values
        )
        singleton_negative_count = sum(
            value < 0.0 for value in singleton_z_values
        )
        singleton_zero_count = sum(value == 0.0 for value in singleton_z_values)
        metrics: dict[str, float | int] = {
            "mica/gamma": gamma,
            "mica/alpha": alpha,
            "mica/singleton_fallback_count": singleton_count,
            "mica/singleton_fallback_rate": (
                singleton_count / state_count if state_count else 0.0
            ),
            "mica/singleton_positive_count": singleton_positive_count,
            "mica/singleton_negative_count": singleton_negative_count,
            "mica/singleton_zero_count": singleton_zero_count,
            "mica/singleton_consecutive_length_max": (
                max(singleton_lengths) if singleton_lengths else 0.0
            ),
            "mica/ig_missing_zero_credit_count": missing_count,
            "mica/loc_zero_variance_count": loc_zero_variance_count,
            "mica/ret_zero_variance_count": ret_zero_variance_count,
            "mica/role_gate_actor_loss_count": role_gate_actor_loss_count,
            "mica/routed_outcome_entry_count": routed_outcome_entry_count,
            "mica/normal_terminal_outcome_entry_count": (
                normal_terminal_outcome_entry_count
            ),
            "mica/observation_policy_mask_violation_count": (
                observation_mask_violation_count
            ),
            "mica/answer_formula_assertion_count": answer_count,
        }
        metrics.update(summarize("mica/raw_ig", raw_values))
        metrics.update(summarize("mica/ig_return", return_values))
        metrics.update(summarize("mica/peer_count", peer_counts))
        metrics.update(summarize("mica/A_loc", local_values))
        metrics.update(summarize("mica/A_ret", ret_values))
        metrics.update(summarize("mica/A_search", search_values))
        metrics.update(summarize("mica/singleton_Z_O", singleton_z_values))
        metrics.update(
            summarize("mica/singleton_consecutive_length", singleton_lengths)
        )
        for depth, values in sorted(by_depth.items()):
            for name, depth_values in sorted(values.items()):
                metrics.update(summarize(f"mica/t{depth}/{name}", depth_values))
        self._attempt_context["advantage_component_metrics"] = metrics

    def _validate_and_record_search_advantage_components(self) -> None:
        """Assert the production Search and Answer formulas before learner use."""

        mode = self.config["advantage"].get("search_task_mode")
        if mode == ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE:
            self._validate_mica_search_advantages()
            return
        if mode == SUFFICIENCY_NOVELTY_LOCAL_IG_MODE:
            self._validate_sufficiency_novelty_search_advantages()
            return
        if (
            mode
            == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
        ):
            self._validate_role_localized_gate_search_advantages()
            return
        if (
            mode
            == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE
        ):
            self._validate_probe_routed_search_advantages()
            return

        lambda_ig = float(self.config["advantage"]["lambda_ig"])
        lambda_task = float(self.config["advantage"]["lambda_task"])
        lambda_outcome = float(self.config["advantage"]["lambda_outcome"])
        lambda_format = float(self.config["advantage"]["lambda_format"])
        a_ig_values: list[float] = []
        a_sc_values: list[float] = []
        a_task_values: list[float] = []
        a_search_values: list[float] = []
        a_answer_values: list[float] = []
        sc_clear_search_turn_count = 0
        fallback_search_turn_count = 0
        fallback_z_o_to_search_count = 0
        answer_formula_assertion_count = 0
        no_search_trajectory_count = 0
        searched_trajectory_count = 0
        for group in self._prepared_groups:
            for item in group:
                advantage = item.advantage
                if advantage is None:
                    raise RuntimeError(
                        f"{item.record.trajectory_id}: selected trajectory has no advantage"
                    )
                search_indices = set(advantage.search_advantage)
                if search_indices:
                    searched_trajectory_count += 1
                else:
                    no_search_trajectory_count += 1
                if (
                    set(advantage.future_ig_rescaled) != search_indices
                    or set(advantage.search_task_advantage) != search_indices
                    or set(advantage.stop_continue_by_search_index)
                    != search_indices
                ):
                    raise RuntimeError(
                        f"{item.record.trajectory_id}: A_IG/A_task coverage does not "
                        "match optimized Search turns"
                    )
                for search_index in sorted(search_indices):
                    a_ig = float(advantage.future_ig_rescaled[search_index])
                    sc = advantage.stop_continue_by_search_index[search_index]
                    a_sc = float(sc.advantage_sc)
                    a_task = float(advantage.search_task_advantage[search_index])
                    a_search = float(advantage.search_advantage[search_index])
                    if sc.sc_clear:
                        sc_clear_search_turn_count += 1
                        expected_task = a_sc
                    else:
                        fallback_search_turn_count += 1
                        expected_task = 0.0
                    if not math.isclose(
                        a_task,
                        expected_task,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ):
                        raise RuntimeError(
                            f"{item.record.trajectory_id}:{search_index}: "
                            "A_task != A_SC if sc_clear else 0.0"
                        )
                    expected = lambda_ig * a_ig + lambda_task * expected_task
                    if not all(
                        math.isfinite(value)
                        for value in (
                            a_ig,
                            a_sc,
                            a_task,
                            a_search,
                            expected,
                        )
                    ):
                        raise RuntimeError(
                            f"{item.record.trajectory_id}:{search_index}: "
                            "Search advantage component is non-finite"
                        )
                    if not math.isclose(
                        a_search,
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ):
                        raise RuntimeError(
                            f"{item.record.trajectory_id}:{search_index}: "
                            "A_search != 0.3*A_IG + where(sc_clear, A_SC, 0.0)"
                        )
                    a_ig_values.append(a_ig)
                    a_sc_values.append(a_sc)
                    a_task_values.append(a_task)
                    a_search_values.append(a_search)
                if advantage.answer_advantage is not None:
                    expected_answer = (
                        lambda_outcome * float(advantage.normalized_outcome)
                        + lambda_format
                        * float(advantage.centered_format_indicator)
                    )
                    if not math.isclose(
                        float(advantage.answer_advantage),
                        expected_answer,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ):
                        raise RuntimeError(
                            f"{item.record.trajectory_id}: "
                            "A_answer != z_O + A_format"
                        )
                    answer_formula_assertion_count += 1
                    a_answer_values.append(float(advantage.answer_advantage))
        def summary(
            prefix: str,
            values: Sequence[float],
        ) -> dict[str, float | int]:
            if not values:
                return {
                    f"{prefix}_count": 0,
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_std": 0.0,
                    f"{prefix}_min": 0.0,
                    f"{prefix}_max": 0.0,
                }
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_count": int(array.size),
                f"{prefix}_mean": float(array.mean()),
                f"{prefix}_std": float(array.std(ddof=0)),
                f"{prefix}_min": float(array.min()),
                f"{prefix}_max": float(array.max()),
            }

        metrics: dict[str, float | int | bool | str] = {}
        metrics.update(summary("A_IG", a_ig_values))
        metrics.update(summary("A_SC", a_sc_values))
        metrics.update(summary("A_task", a_task_values))
        metrics.update(summary("A_search_new", a_search_values))
        metrics.update(summary("A_answer", a_answer_values))
        metrics["advantage_component_coverage_pass"] = True
        metrics["search_advantage_formula_assertion_pass"] = True
        metrics["answer_advantage_formula_assertion_pass"] = True
        metrics["sc_clear_search_turn_count"] = sc_clear_search_turn_count
        metrics["fallback_search_turn_count"] = fallback_search_turn_count
        metrics["fallback_z_o_to_search_count"] = fallback_z_o_to_search_count
        metrics["sc/fallback_z_o_to_search_count"] = (
            fallback_z_o_to_search_count
        )
        metrics["answer_formula_assertion_count"] = (
            answer_formula_assertion_count
        )
        metrics["searched_trajectory_count"] = searched_trajectory_count
        metrics["no_search_trajectory_count"] = no_search_trajectory_count
        trajectory_count = searched_trajectory_count + no_search_trajectory_count
        metrics["no_search_trajectory_rate"] = (
            no_search_trajectory_count / trajectory_count
            if trajectory_count
            else 0.0
        )
        metrics["lambda_ig"] = lambda_ig
        metrics["lambda_task"] = lambda_task
        metrics["search_advantage_formula"] = (
            "0.3*A_IG + A_SC if sc_clear else 0.3*A_IG"
        )
        metrics["outcome_fallback_to_search"] = False
        self._attempt_context["advantage_component_metrics"] = metrics

    def _prepare_selected_sufficiency_probes(
        self,
        groups: Sequence[PromptGroup],
        probe_config: Mapping[str, Any],
    ) -> None:
        """Generate one deterministic, detached pre-Search probe per state."""

        import ray

        if not bool(probe_config.get("enabled", False)):
            raise RuntimeError("Production sufficiency probing is disabled")
        locked = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_samples": 1,
        }
        for key, expected in locked.items():
            if probe_config.get(key) != expected:
                raise RuntimeError(
                    f"Sufficiency probe {key} must equal {expected!r}"
                )

        state_count = sum(
            1
            for group in groups
            for record in group.trajectories
            for turn in record.turns
            if turn.turn_type is TurnType.SEARCH
            and turn.policy_credit_eligible
        )
        if state_count == 0:
            self._attempt_context["sc_runtime_metrics"] = {
                "s_probe/state_count": 0,
                "s_probe/request_count": 0,
                "s_probe/completion_count": 0,
                "s_probe/policy_version_match": True,
                "s_probe/no_search_states": True,
            }
            return

        if self._stop_tokenizer is None:
            from transformers import AutoTokenizer

            self._stop_tokenizer = AutoTokenizer.from_pretrained(
                str(self.config["paths"]["actor_model"]),
                trust_remote_code=True,
                use_fast=True,
            )
            self._stop_scaffold_token_ids = tokenize_stop_scaffold(
                self._stop_tokenizer
            )
        if self._stop_scaffold_token_ids is None:
            raise RuntimeError("Sufficiency scaffold tokenization is unavailable")

        maximum_model_length = self.config["formal_schedule"].get(
            "maximum_model_length"
        )
        if maximum_model_length is None:
            raise RuntimeError(
                "Sufficiency probing requires an approved maximum_model_length"
            )
        max_new_tokens = probe_config.get("answer_max_new_tokens")
        if max_new_tokens is None:
            raise RuntimeError(
                "advantage.sufficiency_probe.answer_max_new_tokens is not approved"
            )
        checksum_before = self.actor_parameter_checksum()
        if checksum_before != self._last_checksum:
            raise RuntimeError(
                "Actor checksum no longer matches the rollout-start snapshot"
            )
        plan = build_sufficiency_probe_plan(
            groups,
            scaffold_token_ids=self._stop_scaffold_token_ids,
            rollout_config=dict(self.config["rollout"]),
            stop_answer_max_new_tokens=int(max_new_tokens),
            maximum_model_length=int(maximum_model_length),
            expected_snapshot_step=int(self._last_snapshot_step),
            replica_count=self._rl_world_size,
        )
        if plan.state_count != state_count:
            raise RuntimeError(
                "Sufficiency planning did not cover every selected Search state"
            )

        generation_started = time.perf_counter()
        generated = self.agent_loop_manager.generate_sufficiency_probes(
            [list(values) for values in plan.jobs_by_replica],
            expected_snapshot_step=int(self._last_snapshot_step),
            expected_source_checksum=self._last_checksum,
        )
        generation_seconds = time.perf_counter() - generation_started
        generated_rows = list(generated["rows"])
        if len(generated_rows) != plan.request_count:
            raise RuntimeError(
                "Sufficiency generator returned the wrong request count"
            )
        completion_payloads: list[dict[str, Any]] = []
        for row in generated_rows:
            if len(row["completions"]) != 1:
                raise RuntimeError("Sufficiency state did not return one completion")
            completion = row["completions"][0]
            completion_payloads.append(
                {
                    "prompt_global_id": str(row["prompt_global_id"]),
                    "trajectory_id": str(row["trajectory_id"]),
                    "search_index": int(row["search_index"]),
                    "completion_text": str(completion["text"]),
                    "truncated": (
                        str(completion.get("finish_reason", "")).lower()
                        == "length"
                    ),
                    "gold_aliases": tuple(
                        str(value) for value in row["gold_aliases"]
                    ),
                    "data_source": str(row.get("data_source", "")),
                }
            )
        if len(completion_payloads) != plan.expected_completion_count:
            raise RuntimeError(
                "Sufficiency generator did not return one completion/state"
            )
        outcome_workers = tuple(self.actors["outcome_workers"])
        chunks = [
            completion_payloads[index::len(outcome_workers)]
            for index in range(len(outcome_workers))
        ]
        reward_started = time.perf_counter()
        reward_refs = [
            worker.score_sufficiency_probe_batch.remote(chunk)
            for worker, chunk in zip(outcome_workers, chunks, strict=True)
            if chunk
        ]
        scored_rows = [row for rows in ray.get(reward_refs) for row in rows]
        reward_seconds = time.perf_counter() - reward_started
        runtime_metrics = attach_sufficiency_probe_results(
            groups,
            generated_rows,
            scored_rows,
            expected_snapshot_step=int(self._last_snapshot_step),
            expected_source_checksum=self._last_checksum,
        )
        checksum_after = self.actor_parameter_checksum()
        if checksum_after != checksum_before:
            raise RuntimeError(
                "Detached sufficiency probing changed Actor parameters"
            )
        runtime_metrics.update(
            {
                "s_probe/generation_seconds": float(generation_seconds),
                "s_probe/reward_scoring_seconds": float(reward_seconds),
                "s_probe/per_replica_jobs": list(generated["per_replica_jobs"]),
                "s_probe/per_replica_tokens": list(
                    generated["per_replica_tokens"]
                ),
                "s_probe/prompt_to_replica": dict(plan.prompt_to_replica),
                "s_probe/estimated_tokens_by_replica": list(
                    plan.estimated_tokens_by_replica
                ),
                "s_probe/prompt_affinity": bool(generated["prompt_affinity"]),
                "s_probe/local_depth_waves": bool(
                    generated["local_depth_waves"]
                ),
                "s_probe/cross_replica_depth_barrier": bool(
                    generated["cross_replica_depth_barrier"]
                ),
                "s_probe/automatic_prefix_caching": True,
                "s_probe/parameter_checksum_before": checksum_before,
                "s_probe/parameter_checksum_after": checksum_after,
            }
        )
        self._attempt_context["sc_runtime_metrics"] = runtime_metrics
        self._add_phase_time("stop_branch_generation", generation_seconds)
        self._add_phase_time("stop_reward", reward_seconds)

    def _prepare_selected_routed_answer_probes(
        self,
        groups: Sequence[PromptGroup],
        probe_config: Mapping[str, Any],
    ) -> None:
        """Generate detached pre/post Probes for the routed-Outcome mode."""

        import ray

        locked = {
            "enabled": True,
            "pre_search_enabled": True,
            "post_search_enabled": True,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "n": 1,
            "max_tokens": 500,
            "stop": ["</answer>"],
        }
        for key, expected in locked.items():
            actual = probe_config.get(key)
            if key == "stop" and isinstance(actual, (tuple, list)):
                actual = list(actual)
            if actual != expected:
                raise RuntimeError(
                    f"Routed Answer Probe {key}={actual!r}, expected {expected!r}"
                )
        state_count = sum(
            1
            for group in groups
            for record in group.trajectories
            for turn in record.turns
            if turn.turn_type is TurnType.SEARCH
        )
        if state_count == 0:
            self._attempt_context["sc_runtime_metrics"] = {
                "answer_probe/pre/state_count": 0,
                "answer_probe/post/state_count": 0,
                "answer_probe/policy_version_match": True,
                "answer_probe/no_search_states": True,
            }
            return
        if self._stop_tokenizer is None:
            from transformers import AutoTokenizer

            self._stop_tokenizer = AutoTokenizer.from_pretrained(
                str(self.config["paths"]["actor_model"]),
                trust_remote_code=True,
                use_fast=True,
            )
            self._stop_scaffold_token_ids = tokenize_stop_scaffold(
                self._stop_tokenizer
            )
        if self._stop_scaffold_token_ids is None:
            raise RuntimeError("Routed Answer Probe scaffold is unavailable")
        maximum_model_length = self.config["formal_schedule"].get(
            "maximum_model_length"
        )
        if maximum_model_length is None:
            raise RuntimeError("Routed Answer Probe needs maximum_model_length")
        checksum_before = self.actor_parameter_checksum()
        if checksum_before != self._last_checksum:
            raise RuntimeError(
                "Actor checksum differs from the rollout-start Probe snapshot"
            )

        combined_metrics: dict[str, Any] = {}
        generation_total = 0.0
        reward_total = 0.0
        for stage in ("pre", "post"):
            plan = build_routed_answer_probe_plan(
                groups,
                probe_stage=stage,
                scaffold_token_ids=self._stop_scaffold_token_ids,
                rollout_config=dict(self.config["rollout"]),
                stop_answer_max_new_tokens=int(probe_config["max_tokens"]),
                maximum_model_length=int(maximum_model_length),
                expected_snapshot_step=int(self._last_snapshot_step),
                replica_count=self._rl_world_size,
            )
            generation_started = time.perf_counter()
            if plan.request_count:
                generated = self.agent_loop_manager.generate_sufficiency_probes(
                    [list(values) for values in plan.jobs_by_replica],
                    expected_snapshot_step=int(self._last_snapshot_step),
                    expected_source_checksum=self._last_checksum,
                )
            else:
                generated = {
                    "rows": [],
                    "per_replica_jobs": [0, 0, 0, 0],
                    "per_replica_tokens": [0, 0, 0, 0],
                    "prompt_affinity": True,
                    "local_depth_waves": True,
                    "cross_replica_depth_barrier": False,
                }
            generation_seconds = time.perf_counter() - generation_started
            generation_total += generation_seconds
            generated_rows = list(generated["rows"])
            if len(generated_rows) != plan.request_count:
                raise RuntimeError(
                    f"Routed {stage} Probe generator returned wrong request count"
                )
            payloads: list[dict[str, Any]] = []
            for row in generated_rows:
                if len(row["completions"]) != 1:
                    raise RuntimeError(
                        f"Routed {stage} Probe did not return one completion"
                    )
                completion = row["completions"][0]
                payloads.append(
                    {
                        "prompt_global_id": str(row["prompt_global_id"]),
                        "trajectory_id": str(row["trajectory_id"]),
                        "search_index": int(row["search_index"]),
                        "probe_stage": stage,
                        "completion_text": str(completion["text"]),
                        "truncated": (
                            str(completion.get("finish_reason", "")).lower()
                            == "length"
                        ),
                        "gold_aliases": tuple(
                            str(value) for value in row["gold_aliases"]
                        ),
                        "data_source": str(row.get("data_source", "")),
                    }
                )
            outcome_workers = tuple(self.actors["outcome_workers"])
            chunks = [
                payloads[index::len(outcome_workers)]
                for index in range(len(outcome_workers))
            ]
            reward_started = time.perf_counter()
            reward_refs = [
                worker.score_sufficiency_probe_batch.remote(chunk)
                for worker, chunk in zip(outcome_workers, chunks, strict=True)
                if chunk
            ]
            scored_rows = [row for rows in ray.get(reward_refs) for row in rows]
            reward_seconds = time.perf_counter() - reward_started
            reward_total += reward_seconds
            stage_metrics = attach_routed_answer_probe_results(
                groups,
                generated_rows,
                scored_rows,
                probe_stage=stage,
                expected_snapshot_step=int(self._last_snapshot_step),
                expected_source_checksum=self._last_checksum,
            )
            stage_metrics.update(
                {
                    f"answer_probe/{stage}/generation_seconds": float(
                        generation_seconds
                    ),
                    f"answer_probe/{stage}/reward_scoring_seconds": float(
                        reward_seconds
                    ),
                    f"answer_probe/{stage}/prompt_to_replica": dict(
                        plan.prompt_to_replica
                    ),
                    f"answer_probe/{stage}/estimated_tokens_by_replica": list(
                        plan.estimated_tokens_by_replica
                    ),
                    f"answer_probe/{stage}/prompt_affinity": bool(
                        generated["prompt_affinity"]
                    ),
                    f"answer_probe/{stage}/local_depth_waves": bool(
                        generated["local_depth_waves"]
                    ),
                    f"answer_probe/{stage}/cross_replica_depth_barrier": bool(
                        generated["cross_replica_depth_barrier"]
                    ),
                    f"answer_probe/{stage}/automatic_prefix_caching": True,
                }
            )
            combined_metrics.update(stage_metrics)

        checksum_after = self.actor_parameter_checksum()
        if checksum_after != checksum_before:
            raise RuntimeError("Detached pre/post Probes changed Actor parameters")
        combined_metrics.update(
            {
                "answer_probe/policy_version_match": True,
                "answer_probe/parameter_checksum_before": checksum_before,
                "answer_probe/parameter_checksum_after": checksum_after,
            }
        )
        self._attempt_context["sc_runtime_metrics"] = combined_metrics
        self._add_phase_time("stop_branch_generation", generation_total)
        self._add_phase_time("stop_reward", reward_total)

    def prepare_selected_stop_branches(
        self,
        groups: Sequence[PromptGroup],
    ) -> None:
        """Generate and score detached Stop probes before any learner action."""

        import ray

        self._require_bound()
        advantage = dict(self.config["advantage"])
        mode = str(
            advantage.get("search_task_mode", "normalized_outcome")
        )
        if mode == ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE:
            self._attempt_context["sc_runtime_metrics"] = {
                "answer_probe/request_count": 0,
                "answer_probe/completion_count": 0,
                "answer_probe/skipped_for_mica": True,
                "answer_probe/policy_version_match": True,
                "sc/request_count": 0,
                "sc/completion_count": 0,
                "sc/disabled": True,
            }
            return
        if mode == SUFFICIENCY_NOVELTY_LOCAL_IG_MODE:
            self._prepare_selected_sufficiency_probes(
                groups,
                dict(advantage.get("sufficiency_probe", {})),
            )
            return
        if mode in {
            SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
            SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
        }:
            self._prepare_selected_routed_answer_probes(
                groups,
                dict(advantage.get("sufficiency_probe", {})),
            )
            return
        if mode == "normalized_outcome":
            self._attempt_context["sc_runtime_metrics"] = {
                "sc/request_count": 0,
                "sc/completion_count": 0,
                "sc/policy_version_match": True,
                "sc/disabled": True,
            }
            return
        if mode != "stop_continue_consensus":
            raise RuntimeError(f"Unsupported Search task mode: {mode}")
        sc_config = dict(advantage.get("sc", {}))
        if not bool(sc_config.get("enabled", False)):
            raise RuntimeError(
                "stop_continue_consensus requires advantage.sc.enabled=true"
            )
        if int(sc_config.get("num_stop_samples", 0)) != 2:
            raise RuntimeError("Stop/Continue V1 requires exactly two samples")

        state_count = sum(
            1
            for group in groups
            for record in group.trajectories
            for turn in record.turns
            if turn.turn_type is TurnType.SEARCH
            and turn.policy_credit_eligible
        )
        if state_count == 0:
            self._attempt_context["sc_runtime_metrics"] = {
                "sc/request_count": 0,
                "sc/completion_count": 0,
                "sc/policy_version_match": True,
                "sc/no_search_states": True,
            }
            return

        if self._stop_tokenizer is None:
            from transformers import AutoTokenizer

            self._stop_tokenizer = AutoTokenizer.from_pretrained(
                str(self.config["paths"]["actor_model"]),
                trust_remote_code=True,
                use_fast=True,
            )
            self._stop_scaffold_token_ids = tokenize_stop_scaffold(
                self._stop_tokenizer
            )
        if self._stop_scaffold_token_ids is None:
            raise RuntimeError("Stop scaffold tokenization was not initialized")

        maximum_model_length = self.config["formal_schedule"].get(
            "maximum_model_length"
        )
        if maximum_model_length is None:
            raise RuntimeError(
                "Stop branching requires an approved maximum_model_length"
            )
        stop_answer_max_new_tokens = sc_config.get(
            "stop_answer_max_new_tokens"
        )
        if stop_answer_max_new_tokens is None:
            raise RuntimeError(
                "advantage.sc.stop_answer_max_new_tokens is not approved"
            )
        checksum_before = self.actor_parameter_checksum()
        if checksum_before != self._last_checksum:
            raise RuntimeError(
                "Actor checksum no longer matches the rollout-start snapshot"
            )
        plan = build_stop_branch_plan(
            groups,
            scaffold_token_ids=self._stop_scaffold_token_ids,
            rollout_config=dict(self.config["rollout"]),
            stop_answer_max_new_tokens=int(stop_answer_max_new_tokens),
            maximum_model_length=int(maximum_model_length),
            expected_snapshot_step=int(self._last_snapshot_step),
            replica_count=self._rl_world_size,
        )
        if plan.state_count != state_count:
            raise RuntimeError(
                "Stop planning did not cover every selected Search state"
            )

        generation_started = time.perf_counter()
        generated = self.agent_loop_manager.generate_stop_branches(
            [list(values) for values in plan.jobs_by_replica],
            expected_snapshot_step=int(self._last_snapshot_step),
            expected_source_checksum=self._last_checksum,
        )
        generation_seconds = time.perf_counter() - generation_started
        generated_rows = list(generated["rows"])
        if len(generated_rows) != plan.request_count:
            raise RuntimeError(
                "Stop generator returned the wrong request count"
            )
        completion_payloads: list[dict[str, Any]] = []
        for row in generated_rows:
            for completion in row["completions"]:
                completion_payloads.append(
                    {
                        "prompt_global_id": str(row["prompt_global_id"]),
                        "trajectory_id": str(row["trajectory_id"]),
                        "search_index": int(row["search_index"]),
                        "sample_index": int(completion["sample_index"]),
                        "completion_text": str(completion["text"]),
                        "gold_aliases": tuple(
                            str(value) for value in row["gold_aliases"]
                        ),
                        "data_source": str(row.get("data_source", "")),
                    }
                )
        if len(completion_payloads) != plan.expected_completion_count:
            raise RuntimeError(
                "Stop generator did not return exactly two completions/state"
            )
        outcome_workers = tuple(self.actors["outcome_workers"])
        chunks = [
            completion_payloads[index::len(outcome_workers)]
            for index in range(len(outcome_workers))
        ]
        reward_started = time.perf_counter()
        reward_refs = [
            worker.score_stop_branch_batch.remote(chunk)
            for worker, chunk in zip(outcome_workers, chunks, strict=True)
            if chunk
        ]
        scored_rows = [
            row for rows in ray.get(reward_refs) for row in rows
        ]
        reward_seconds = time.perf_counter() - reward_started
        runtime_metrics = attach_stop_branch_rewards(
            groups,
            generated_rows,
            scored_rows,
            expected_snapshot_step=int(self._last_snapshot_step),
            expected_source_checksum=self._last_checksum,
        )
        checksum_after = self.actor_parameter_checksum()
        if checksum_after != checksum_before:
            raise RuntimeError("Detached Stop branching changed Actor parameters")
        runtime_metrics.update(
            {
                "sc/generation_seconds": float(generation_seconds),
                "sc/reward_scoring_seconds": float(reward_seconds),
                "sc/per_replica_jobs": list(generated["per_replica_jobs"]),
                "sc/per_replica_tokens": list(
                    generated["per_replica_tokens"]
                ),
                "sc/prompt_to_replica": dict(plan.prompt_to_replica),
                "sc/estimated_tokens_by_replica": list(
                    plan.estimated_tokens_by_replica
                ),
                "sc/prompt_affinity": bool(generated["prompt_affinity"]),
                "sc/local_depth_waves": bool(
                    generated["local_depth_waves"]
                ),
                "sc/cross_replica_depth_barrier": bool(
                    generated["cross_replica_depth_barrier"]
                ),
                "sc/automatic_prefix_caching": True,
                "sc/parameter_checksum_before": checksum_before,
                "sc/parameter_checksum_after": checksum_after,
            }
        )
        self._attempt_context["sc_runtime_metrics"] = runtime_metrics
        self._add_phase_time("stop_branch_generation", generation_seconds)
        self._add_phase_time("stop_reward", reward_seconds)

    def zero_grad(self) -> None:
        learning_rates = self.worker_group.execute_all_sync(
            "current_learning_rate"
        )
        if len({float(value) for value in learning_rates}) != 1:
            raise RuntimeError(
                f"FSDP ranks disagree on learning rate: {learning_rates}"
            )
        self._learning_rate_used = float(learning_rates[0])
        self.worker_group.execute_all_sync("strict_zero_grad")
        self._microbatch_metrics.clear()

    def actor_parameter_checksum(self) -> str:
        checksums = self.worker_group.execute_all_sync("global_actor_checksum")
        if len(set(checksums)) != 1:
            raise RuntimeError("Actor checksum differs across FSDP ranks")
        return str(checksums[0])

    def backward_microbatch(self, microbatch: Any) -> None:
        import ray

        started = time.perf_counter()
        references = [ray.put(payload) for payload in microbatch]
        metrics = self.worker_group.execute_all_sync(
            "strict_backward_microbatch",
            references,
        )
        self._microbatch_metrics.extend(metrics)
        for row in metrics:
            for turn in row.get("turn_runtime_metrics", ()):
                key = (str(turn["trajectory_id"]), int(turn["turn_id"]))
                previous = self._turn_runtime_metrics.get(key)
                if previous is not None and previous != turn:
                    raise RuntimeError(
                        f"Turn runtime metrics disagree for {key}"
                    )
                self._turn_runtime_metrics[key] = dict(turn)
        self._add_phase_time("backward", time.perf_counter() - started)

    @staticmethod
    def _gate_event_counts(
        microbatches: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[int, int]:
        decision_count = 0
        query_count = 0
        for rank_payloads in microbatches:
            for payload in rank_payloads:
                for weight, decision, query in zip(
                    payload["trajectory_weights"],
                    payload["decision_advantage_by_turn"],
                    payload["query_advantage_by_turn"],
                    strict=True,
                ):
                    if float(weight) == 0.0:
                        continue
                    decision_count += sum(
                        float(value) != 0.0 for value in decision.values()
                    )
                    query_count += sum(
                        float(value) != 0.0 for value in query.values()
                    )
        return int(decision_count), int(query_count)

    def profile_role_localized_gate_gradients(
        self,
        batch_id: str,
        microbatches: Sequence[Sequence[Mapping[str, Any]]],
    ) -> BatchGradientProfile:
        """Profile three production objectives without entering a step transaction."""

        import ray

        if self.stage != "GATE_CALIBRATION":
            raise RuntimeError("Gate gradient profiling is calibration-stage only")
        decision_events, query_events = self._gate_event_counts(microbatches)
        checksum_before = self.actor_parameter_checksum()
        begin_rows = self.worker_group.execute_all_sync(
            "begin_gate_gradient_profile",
            str(batch_id),
        )
        try:
            for channel in ("main", "decision", "query"):
                self.worker_group.execute_all_sync(
                    "begin_gate_gradient_channel",
                    str(batch_id),
                    channel,
                )
                for rank_payloads in microbatches:
                    references = [ray.put(payload) for payload in rank_payloads]
                    self.worker_group.execute_all_sync(
                        "strict_backward_microbatch",
                        references,
                    )
                self.worker_group.execute_all_sync(
                    "finish_gate_gradient_channel",
                    str(batch_id),
                    channel,
                )
            finish_rows = self.worker_group.execute_all_sync(
                "finish_gate_gradient_profile",
                str(batch_id),
                int(decision_events),
                int(query_events),
            )
        except BaseException:
            self.worker_group.execute_all_sync("abort_gate_gradient_profile")
            raise
        checksum_after = self.actor_parameter_checksum()
        if checksum_after != checksum_before:
            raise RuntimeError("Gate calibration changed Actor parameters")
        scalar_keys = (
            "main_gradient_norm",
            "decision_gradient_norm",
            "query_gradient_norm",
            "dot_main_decision",
            "dot_main_query",
            "dot_decision_query",
            "cos_main_decision",
            "cos_main_query",
            "cos_decision_query",
        )
        for key in scalar_keys:
            values = {float(row[key]) for row in finish_rows}
            if len(values) != 1:
                raise RuntimeError(
                    f"FSDP ranks disagree on calibration {key}: {values}"
                )
        if not all(
            bool(row["parameters_bitwise_unchanged"])
            and bool(row["optimizer_scheduler_unchanged"])
            and bool(row["gradients_cleared"])
            and bool(row["rank_metadata_consistent"])
            for row in finish_rows
        ):
            raise RuntimeError("Gate calibration no-update safety failed")
        counts = self.worker_group.execute_all_sync("strict_attempt_counts")
        if any(any(int(value) != 0 for value in row.values()) for row in counts):
            raise RuntimeError(f"Calibration changed strict step counts: {counts}")
        if self._checkpoint_writes != 0:
            raise RuntimeError("Calibration wrote a checkpoint")
        row = finish_rows[0]
        profile = BatchGradientProfile(
            batch_id=str(batch_id),
            main_gradient_norm=float(row["main_gradient_norm"]),
            decision_gradient_norm=float(row["decision_gradient_norm"]),
            query_gradient_norm=float(row["query_gradient_norm"]),
            dot_main_decision=float(row["dot_main_decision"]),
            dot_main_query=float(row["dot_main_query"]),
            dot_decision_query=float(row["dot_decision_query"]),
            cos_main_decision=float(row["cos_main_decision"]),
            cos_main_query=float(row["cos_main_query"]),
            cos_decision_query=float(row["cos_decision_query"]),
            decision_gate_event_count=int(decision_events),
            query_gate_event_count=int(query_events),
            parameters_bitwise_unchanged=True,
            gradients_cleared=True,
            rank_metadata_consistent=True,
        )
        profile.validate()
        self._attempt_context.setdefault("gate_calibration", []).append(
            {
                "batch_id": str(batch_id),
                "actor_checksum_before": checksum_before,
                "actor_checksum_after": checksum_after,
                "worker_begin": begin_rows,
                "worker_finish": finish_rows,
                **asdict(profile),
            }
        )
        return profile
    def clip_gradients(self, max_grad_norm: float) -> float:
        started = time.perf_counter()
        norms = self.worker_group.execute_all_sync(
            "strict_clip_gradients",
            float(max_grad_norm),
        )
        if any(not math.isfinite(float(value)) for value in norms):
            raise RuntimeError("A rank returned a non-finite gradient norm")
        value = float(max(float(value) for value in norms))
        self._add_phase_time("gradient_clip", time.perf_counter() - started)
        return value

    def optimizer_step(self) -> None:
        started = time.perf_counter()
        self.worker_group.execute_all_sync("strict_optimizer_step")
        self._total_optimizer_steps += 1
        self._add_phase_time("optimizer", time.perf_counter() - started)

    def scheduler_step(self) -> None:
        started = time.perf_counter()
        self.worker_group.execute_all_sync("strict_scheduler_step")
        self._total_scheduler_steps += 1
        self._add_phase_time("scheduler", time.perf_counter() - started)

    def _fingerprint(self, key: str, path: str | Path) -> str:
        if key not in self._fingerprints:
            self._fingerprints[key] = _sha256_tree(path)
        return self._fingerprints[key]

    def _framework_versions(self) -> dict[str, str | None]:
        import ray
        import transformers
        import verl
        import vllm

        return {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "verl": getattr(verl, "__version__", None),
            "ray": ray.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
        }

    def _checkpoint_metadata(
        self,
        state: TrainingState,
    ) -> CheckpointMetadata:
        paths = self.config["paths"]
        retriever = self.config["retriever"]
        sampler_state = self.rng_state()["prompt_sampler"]
        model_path = Path(str(paths["actor_model"]))
        tokenizer_files = [
            model_path / name
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            )
            if (model_path / name).is_file()
        ]
        tokenizer_digest = hashlib.sha256()
        for path in tokenizer_files:
            tokenizer_digest.update(path.name.encode("utf-8"))
            tokenizer_digest.update(_sha256_file(path).encode("ascii"))
        manifest = Path(__file__).resolve().parents[3] / "MANIFEST.sha256"
        source_commit = (
            _sha256_file(manifest)
            if manifest.is_file()
            else "NO_MANIFEST"
        )
        return CheckpointMetadata(
            schema_version=1,
            successful_update_step=int(state.successful_update_step),
            attempt_id=int(state.attempt_id),
            data_cursor=int(state.data_cursor),
            dataset_mixture_state=dict(sampler_state),
            rng_state_files={
                "driver": "rng/driver.pt",
                **{
                    f"rank_{rank}": f"rng/rank-{rank:02d}.pt"
                    for rank in range(self._rl_world_size)
                },
            },
            ig_channel=_channel_checkpoint(state.ig_channel),
            outcome_channel=_channel_checkpoint(state.outcome_channel),
            algorithm_config=self.config,
            model_fingerprint=self._fingerprint("actor", paths["actor_model"]),
            reference_model_fingerprint=self._fingerprint(
                "reference",
                paths["reference_model"],
            ),
            train_data_sha256=str(self.config["data"]["source_sha256"]),
            validation_data_sha256=self._fingerprint(
                "validation",
                paths["validation_data"],
            ),
            retriever_index_sha256=self._fingerprint(
                "retriever_index",
                retriever["dense_index_path"],
            ),
            retriever_config_sha256=self._fingerprint(
                "retriever_config",
                retriever["server_config_source"],
            ),
            tokenizer_hash=tokenizer_digest.hexdigest(),
            chat_template_hash=tokenizer_digest.hexdigest(),
            source_commit=source_commit,
            framework_versions=self._framework_versions(),
            fsdp_world_size=self._rl_world_size,
            vllm_data_parallel_size=self._rl_world_size,
            vllm_tensor_parallel_size=1,
            optimizer_state_present=True,
            scheduler_state_present=True,
            actor_state_present=True,
        )

    def _should_checkpoint(self, step: int) -> bool:
        # Runtime smoke stages retain logs and metrics only. Model, optimizer,
        # and scheduler checkpoints are reserved for the pilot and formal runs.
        if self.stage in {"A", "B", "C", "D"}:
            if bool(self.config["checkpoint"]["smoke_model_checkpoints"]):
                raise RuntimeError(
                    "Resolved config must disable model checkpoints for smoke stages"
                )
            return False
        if self.stage == "PILOT20":
            return step in {
                int(value)
                for value in self.config["pilot"]["checkpoints"]
            }
        if self.stage in {"E", "PILOT50"}:
            return step in {1, 5, 10, 25, 50}
        if self.stage == "FORMAL":
            interval = int(
                self.config["formal_schedule"][
                    "checkpoint_every_successful_updates"
                ]
            )
            return step % interval == 0
        return False

    def checkpoint_resource_preflight(
        self,
        *,
        next_successful_update_step: int,
        phase: str,
    ) -> dict[str, Any]:
        """Guard cadence I/O without changing algorithm or optimizer state."""

        step = int(next_successful_update_step)
        if not self._should_checkpoint(step):
            return {
                "status": "NOT_REQUIRED",
                "successful_update_step": step,
                "phase": str(phase),
            }
        runtime_root = Path(
            str(self.config["paths"]["runtime_root"])
        ).resolve()
        cache_release = release_file_cache(runtime_root / "checkpoints")
        source_checkpoint = os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT")
        if source_checkpoint:
            source_cache_release = release_file_cache(source_checkpoint)
        else:
            source_cache_release = None
        result = validate_checkpoint_runtime_budget(
            runtime_root,
            snapshot=read_runtime_resource_snapshot(),
            source_checkpoint=source_checkpoint,
            include_checkpoint_write=(phase != "before_model_export"),
        )
        result.update(
            {
                "successful_update_step": step,
                "phase": str(phase),
                "released_checkpoint_cache": cache_release,
                "released_source_checkpoint_cache": source_cache_release,
            }
        )
        self._attempt_context["checkpoint_resource_preflight"] = dict(result)
        if self.stage in {"FORMAL", "PILOT20"}:
            atomic_write_json(
                runtime_root / "state" / "checkpoint_resource_preflight.json",
                result,
            )
        return result

    def _save_checkpoint(self, state: TrainingState) -> Path:
        if self.stage in {"A", "B", "C", "D"}:
            raise RuntimeError(
                f"Runtime smoke Stage {self.stage} must not write model checkpoints"
            )
        metadata = self._checkpoint_metadata(state)
        project_root = Path(__file__).resolve().parents[3]
        if self.stage == "FORMAL":
            formal_root = os.environ.get("AGENTIC_RL_FORMAL_RUN_ROOT")
            if not formal_root:
                raise RuntimeError("AGENTIC_RL_FORMAL_RUN_ROOT is required")
            root = Path(formal_root).resolve() / "checkpoints" / "resume"
        elif self.stage == "PILOT20":
            run_dir = os.environ.get("AGENTIC_RL_RUN_DIR")
            if not run_dir:
                raise RuntimeError("AGENTIC_RL_RUN_DIR is required for PILOT20")
            root = Path(run_dir).resolve() / "checkpoints"
        elif self.stage in {"E", "PILOT50"}:
            root = project_root / "outputs" / "runtime_pilot_50" / "checkpoints"
        else:
            root = (
                project_root
                / "outputs"
                / f"runtime_stage_{self.stage.lower()}"
                / "checkpoints"
            )
        committer = AtomicCheckpointCommitter(root)

        def write_state(destination: Path) -> None:
            destination.mkdir(parents=True, exist_ok=True)
            self.worker_group.execute_all_sync(
                "save_distributed_training_state",
                str(destination),
            )
            controller_root = destination / "controller"
            controller_root.mkdir(exist_ok=True)
            (controller_root / "state.json").write_text(
                json.dumps(
                    {
                        "training_state": {
                            "attempt_id": state.attempt_id,
                            "successful_update_step": state.successful_update_step,
                            "data_cursor": state.data_cursor,
                            "ig_channel": asdict(state.ig_channel),
                            "outcome_channel": asdict(state.outcome_channel),
                            "rng_state": state.rng_state,
                        },
                        "prompt_sampler": self.rng_state()["prompt_sampler"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            rng_root = destination / "rng"
            rng_root.mkdir(exist_ok=True)
            torch.save(
                {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                },
                rng_root / "driver.pt",
            )

        checkpoint_started = time.perf_counter()
        checkpoint = committer.commit(
            metadata,
            write_distributed_state=write_state,
            rank=0,
            barrier=lambda: None,
            gather_errors=lambda error: [error],
            directory_name=(
                f"update_{int(state.successful_update_step)}"
                if self.stage == "PILOT20"
                else (
                    f"update_{int(state.successful_update_step):03d}"
                    if self.stage == "FORMAL"
                    else None
                )
            ),
        )
        self._add_phase_time(
            "checkpoint",
            time.perf_counter() - checkpoint_started,
        )
        self._last_checkpoint = checkpoint
        self._checkpoint_writes += 1
        self._enforce_checkpoint_limit(root)
        return checkpoint

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                release_file_cache(path)
        descriptor = os.open(str(root), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _export_model_checkpoint(
        self,
        state: TrainingState,
        *,
        actor_checksum: str,
        allow_restored_checkpoint_boundary: bool = False,
    ) -> Path:
        if self.stage not in {"FORMAL", "PILOT20"}:
            raise RuntimeError(
                "Persistent model exports are limited to Pilot20/formal runs"
            )
        run_root = Path(str(self.config["paths"]["runtime_root"])).resolve()
        model_root = run_root / "checkpoints" / "models"
        model_root.mkdir(parents=True, exist_ok=True)
        name = f"update_{int(state.successful_update_step):03d}"
        temporary = model_root / f".tmp_{name}"
        destination = model_root / name
        if destination.exists():
            raise FileExistsError(f"Model checkpoint already exists: {destination}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()

        started = time.perf_counter()
        exports = self.worker_group.execute_all_sync(
            "export_hf_model_checkpoint",
            str(temporary),
            int(state.successful_update_step),
            str(actor_checksum),
            bool(allow_restored_checkpoint_boundary),
        )
        if len(exports) != self._rl_world_size or {
            str(row["actor_checksum"]) for row in exports
        } != {str(actor_checksum)}:
            raise RuntimeError(f"Distributed model export disagreed: {exports}")
        model_file = temporary / "model.safetensors"
        if not model_file.is_file() or model_file.stat().st_size < 1024**3:
            raise RuntimeError("Formal model checkpoint is missing or implausibly small")

        source = Path(str(self.config["paths"]["actor_model"])).resolve()
        for item in source.iterdir():
            if not item.is_file():
                continue
            if item.name == "training_args.bin":
                continue
            if item.name == "model.safetensors.index.json":
                continue
            if item.name == "model.safetensors" or (
                item.name.startswith("model-") and item.suffix == ".safetensors"
            ):
                continue
            shutil.copy2(item, temporary / item.name)
        manifest = {
            path.name: _sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        metadata = {
            "schema_version": 1,
            "successful_update_step": int(state.successful_update_step),
            "attempt_id": int(state.attempt_id),
            "data_cursor": int(state.data_cursor),
            "source_checkpoint": os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT"),
            "actor_checksum": str(actor_checksum),
            "model_dtype": "bfloat16",
            "manifest": manifest,
            "resolved_training_config": self.config,
        }
        (temporary / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").write_text(
            f"successful_update_step={int(state.successful_update_step)}\n",
            encoding="utf-8",
        )
        self._fsync_tree(temporary)
        os.replace(temporary, destination)
        parent_descriptor = os.open(str(model_root), os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        if _sha256_file(destination / "model.safetensors") != manifest["model.safetensors"]:
            raise RuntimeError("Committed model checkpoint failed checksum validation")
        # The export is immutable after this point.  Drop clean pages so the
        # next cadence checkpoint cannot accumulate file-backed cgroup RSS.
        release_file_cache(destination)
        latest_temporary = model_root / ".latest.tmp"
        latest = model_root / "latest"
        if latest_temporary.exists() or latest_temporary.is_symlink():
            latest_temporary.unlink()
        latest_temporary.symlink_to(destination.name)
        os.replace(latest_temporary, latest)
        self._last_model_checkpoint = destination
        self._add_phase_time(
            "model_checkpoint",
            time.perf_counter() - started,
        )
        atomic_write_json(
            run_root / "state" / "latest_model_checkpoint.json",
            {
                "successful_update_step": int(state.successful_update_step),
                "path": str(destination),
                "actor_checksum": str(actor_checksum),
            },
        )
        return destination

    def _materialize_missing_resume_cadence_artifacts(
        self,
        state: TrainingState,
    ) -> dict[str, Any] | None:
        """Recreate derived cadence artifacts after a verified mid-commit exit.

        The distributed resume checkpoint remains the authoritative training
        state. Model exports and asynchronous eval tasks are derived artifacts,
        so this opt-in recovery runs before the next attempt and never changes
        model, optimizer, scheduler, or successful-update counters.
        """
        if not bool(
            self.config.get("checkpoint", {}).get(
                "materialize_missing_cadence_artifacts_on_resume",
                False,
            )
        ):
            return None
        if self.stage != "FORMAL" or not os.environ.get(
            "AGENTIC_RL_RESUME_CHECKPOINT"
        ):
            raise RuntimeError(
                "Resume artifact recovery requires a formal checkpoint resume"
            )
        if not bool(self.config.get("evaluation", {}).get("asynchronous", False)):
            raise RuntimeError(
                "Resume artifact recovery requires asynchronous evaluation"
            )

        step = int(state.successful_update_step)
        cadence = int(
            self.config["formal_schedule"][
                "fixed_eval_every_successful_updates"
            ]
        )
        if step <= 0 or step % cadence != 0:
            return None

        run_root = Path(str(self.config["paths"]["runtime_root"])).resolve()
        actor_checksum = self.actor_parameter_checksum()
        local_model_path = (
            run_root / "checkpoints" / "models" / f"update_{step:03d}"
        )
        external_model_raw = self.config.get("checkpoint", {}).get(
            "resume_cadence_model_artifact_source"
        )
        external_model_path = (
            Path(str(external_model_raw)).resolve()
            if external_model_raw
            else None
        )
        reused_external_model_artifact = (
            not local_model_path.exists() and external_model_path is not None
        )
        model_path = (
            external_model_path
            if reused_external_model_artifact
            else local_model_path
        )
        created_model_export = not model_path.exists()
        if created_model_export:
            model_path = self._export_model_checkpoint(
                state,
                actor_checksum=actor_checksum,
                allow_restored_checkpoint_boundary=True,
            )
        else:
            metadata_path = model_path / "training_metadata.json"
            completed_path = model_path / "COMPLETED"
            model_file = model_path / "model.safetensors"
            if not (
                metadata_path.is_file()
                and completed_path.is_file()
                and model_file.is_file()
            ):
                raise RuntimeError(
                    "Existing resume model artifact is not atomically complete"
                )
            model_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if int(model_metadata["successful_update_step"]) != step:
                raise RuntimeError("Resume model artifact update does not match")
            if str(model_metadata["actor_checksum"]) != actor_checksum:
                raise RuntimeError("Resume model artifact Actor checksum differs")
            expected_model_sha256 = model_metadata["manifest"].get(
                "model.safetensors"
            )
            if (
                not expected_model_sha256
                or _sha256_file(model_file) != expected_model_sha256
            ):
                raise RuntimeError("Resume model artifact checksum failed")
            self._last_model_checkpoint = model_path

        reward_snapshot_rows = self.worker_group.execute_all_sync(
            "load_restored_reward_snapshot_from_hf",
            str(model_path),
            step,
            actor_checksum,
        )
        if len(reward_snapshot_rows) != self._rl_world_size:
            raise RuntimeError("Restored Reward Snapshot rank count differs")
        if {
            (
                int(row["successful_update_step"]),
                str(row["actor_checksum_before"]),
                str(row["actor_checksum_after"]),
                str(row["reward_parameter_dtype"]),
            )
            for row in reward_snapshot_rows
        } != {(step, actor_checksum, actor_checksum, "float32")}:
            raise RuntimeError(
                "Restored Reward Snapshot preload metadata differs by rank"
            )
        if len(
            {
                str(row["reward_snapshot_checksum"])
                for row in reward_snapshot_rows
            }
        ) != 1:
            raise RuntimeError("Restored Reward Snapshot differs across ranks")

        eval_task = enqueue_eval(
            run_root,
            update=step,
            model_path=model_path,
            actor_checksum=actor_checksum,
        )
        report = {
            "status": "PASS",
            "successful_update_step": step,
            "source_resume_checkpoint": os.environ[
                "AGENTIC_RL_RESUME_CHECKPOINT"
            ],
            "actor_checksum": actor_checksum,
            "model_path": str(model_path),
            "model_export_created": created_model_export,
            "external_model_artifact_reused": reused_external_model_artifact,
            "reward_snapshot_preloaded": True,
            "reward_snapshot_by_rank": reward_snapshot_rows,
            "eval_task": eval_task,
            "optimizer_steps_during_recovery": 0,
            "scheduler_steps_during_recovery": 0,
            "resume_checkpoint_writes_during_recovery": 0,
        }
        atomic_write_json(
            run_root / "state" / "resume_cadence_artifact_recovery.json",
            report,
        )
        return report

    def _enforce_checkpoint_limit(self, root: Path) -> None:
        raw_limit = self.config["checkpoint"].get("formal_limit")
        if raw_limit is None or str(raw_limit).strip().lower() in {
            "",
            "none",
            "null",
            "unlimited",
        }:
            return
        limit = int(raw_limit)
        if limit < 1:
            raise RuntimeError(
                "checkpoint.formal_limit must be a positive integer or null"
            )
        checkpoints = []
        for path in root.iterdir():
            if path.is_symlink() or not path.is_dir() or path.name.endswith(".tmp"):
                continue
            metadata_path = path / "metadata.json"
            if not metadata_path.is_file():
                continue
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoints.append(
                (int(payload["successful_update_step"]), path)
            )
        for _, path in sorted(checkpoints)[:-limit]:
            shutil.rmtree(path)

    def _verify_checkpoint_readonly_subprocess(
        self,
        checkpoint: Path,
        state: TrainingState,
    ) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[3]
        destination = (
            Path(str(self.config["paths"]["runtime_root"]))
            / "reports"
            / (
                "checkpoint_update_"
                f"{int(state.successful_update_step):03d}_readonly.json"
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.config["paths"]["rl_python"]),
            str(project_root / "scripts" / "verify_checkpoint_readonly.py"),
            "--checkpoint",
            str(checkpoint),
            "--expected-step",
            str(state.successful_update_step),
            "--expected-attempt",
            str(state.attempt_id),
            "--expected-cursor",
            str(state.data_cursor),
            "--output",
            str(destination),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Read-only checkpoint verifier failed: "
                f"{completed.stderr.strip()}"
            )
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(
                f"Read-only checkpoint verification failed: {payload}"
            )
        return payload

    def _aggregate_update_metrics(
        self,
        *,
        state: TrainingState,
        checkpoint: str | None,
        model_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        rows = self._microbatch_metrics
        task = sum(
            float(row["task_objective_local_scaled"]) for row in rows
        ) / 4.0
        main = sum(
            float(row["main_objective_local_scaled"]) for row in rows
        ) / 4.0
        decision_objective = sum(
            float(row["decision_objective_local_scaled"]) for row in rows
        ) / 4.0
        query_objective = sum(
            float(row["query_objective_local_scaled"]) for row in rows
        ) / 4.0
        decision_ratios = [
            float(value)
            for row in rows
            for value in row.get("decision_ratio_values", ())
        ]
        query_ratios = [
            float(value)
            for row in rows
            for value in row.get("query_ratio_values", ())
        ]
        decision_clipped = [
            bool(value)
            for row in rows
            for value in row.get("decision_clipped_values", ())
        ]
        query_clipped = [
            bool(value)
            for row in rows
            for value in row.get("query_clipped_values", ())
        ]
        kl = sum(
            float(row["full_vocab_kl_local_scaled"]) for row in rows
        ) / 4.0
        loss = sum(
            float(row["total_loss_local_scaled"]) for row in rows
        ) / 4.0
        ratios = [
            float(value["ratio"])
            for value in self._turn_runtime_metrics.values()
        ]
        clipped = [
            bool(value["clipped_low"] or value["clipped_high"])
            for value in self._turn_runtime_metrics.values()
        ]
        clipped_low = [
            bool(value["clipped_low"])
            for value in self._turn_runtime_metrics.values()
        ]
        clipped_high = [
            bool(value["clipped_high"])
            for value in self._turn_runtime_metrics.values()
        ]
        groups = self._attempt_context.get("groups", ())
        trajectories = [
            trajectory for group in groups for trajectory in group.trajectories
        ]
        raw_ig = [
            float(value)
            for trajectory in trajectories
            for value in trajectory.immediate_ig.values()
        ]
        outcomes = [float(item.task_outcome) for item in trajectories]
        action_tokens = sum(int(item.action_token_count) for item in trajectories)
        decision = self._attempt_context["decision"]
        global_optimizer_steps = (
            int(self._starting_successful_update)
            + int(self._total_optimizer_steps)
        )
        global_scheduler_steps = (
            int(self._starting_successful_update)
            + int(self._total_scheduler_steps)
        )
        gate_update: dict[str, Any] = {}
        if str(self.config["advantage"].get("search_task_mode")) == (
            SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
        ):
            role_config = dict(self.config["advantage"]["role_localized_gate"])
            lambda_decision = float(role_config["lambda_decision"])
            lambda_query = float(role_config["lambda_query"])
            expected_task = (
                main
                + lambda_decision * decision_objective
                + lambda_query * query_objective
            )
            if not math.isclose(task, expected_task, rel_tol=0.0, abs_tol=1.0e-7):
                raise RuntimeError("J_task channel composition changed")
            counter_names = (
                "nonzero_decision_and_query_same_token_count",
                "unexpected_nonzero_main_gate_overlap_count",
                "allowed_soft_duplicate_main_query_overlap_count",
                "observation_policy_mask_violation_count",
            )
            counters = {
                name: sum(int(row.get(name, 0)) for row in rows)
                for name in counter_names
            }
            if any(
                counters[name]
                for name in (
                    "nonzero_decision_and_query_same_token_count",
                    "unexpected_nonzero_main_gate_overlap_count",
                    "observation_policy_mask_violation_count",
                )
            ):
                raise RuntimeError("Role-localized learner safety counter is non-zero")
            expected_allowed = int(
                self._attempt_context["advantage_component_metrics"][
                    "role_gate/allowed_soft_duplicate_main_query_overlap_count"
                ]
            )
            if counters[
                "allowed_soft_duplicate_main_query_overlap_count"
            ] != expected_allowed:
                raise RuntimeError("Allowed soft duplicate overlap count changed")
            calibration_path = Path(
                str(role_config["calibration_manifest"])
            ).resolve()
            calibration = json.loads(
                calibration_path.read_text(encoding="utf-8")
            )
            gate_update = {
                "role_gate/lambda_decision": lambda_decision,
                "role_gate/lambda_query": lambda_query,
                "role_gate/J_main": main,
                "role_gate/J_decision": decision_objective,
                "role_gate/J_query": query_objective,
                "role_gate/weighted_J_decision": (
                    lambda_decision * decision_objective
                ),
                "role_gate/weighted_J_query": lambda_query * query_objective,
                "role_gate/gradient_profile_source": "immutable_u0_calibration",
                "role_gate/calibration_main_gradient_norm_median": float(
                    calibration["median_main_gradient_norm"]
                ),
                "role_gate/calibration_decision_gradient_norm_median": float(
                    calibration["median_decision_gradient_norm"]
                ),
                "role_gate/calibration_query_gradient_norm_median": float(
                    calibration["median_query_gradient_norm"]
                ),
                "role_gate/calibration_weighted_gate_gradient_norm_median": float(
                    calibration["median_weighted_gate_gradient_norm"]
                ),
                "role_gate/calibration_cos_main_decision_median": float(
                    calibration["median_cos_main_decision"]
                ),
                "role_gate/calibration_cos_main_query_median": float(
                    calibration["median_cos_main_query"]
                ),
                "role_gate/calibration_cos_decision_query_median": float(
                    calibration["median_cos_decision_query"]
                ),
                **{
                    f"role_gate/{name}": value
                    for name, value in counters.items()
                },
            }
        update = {
            "attempt_id": int(state.attempt_id),
            "successful_update_step": int(state.successful_update_step),
            "learning_rate_used": self._learning_rate_used,
            "task_objective": task,
            "J_main": main,
            "J_decision": decision_objective,
            "J_query": query_objective,
            "decision_ratio_mean": (
                float(np.mean(decision_ratios, dtype=np.float64))
                if decision_ratios
                else 0.0
            ),
            "decision_ratio_std": (
                float(np.std(decision_ratios, dtype=np.float64))
                if decision_ratios
                else 0.0
            ),
            "query_ratio_mean": (
                float(np.mean(query_ratios, dtype=np.float64))
                if query_ratios
                else 0.0
            ),
            "query_ratio_std": (
                float(np.std(query_ratios, dtype=np.float64))
                if query_ratios
                else 0.0
            ),
            "decision_clip_fraction": (
                sum(decision_clipped) / len(decision_clipped)
                if decision_clipped
                else 0.0
            ),
            "query_clip_fraction": (
                sum(query_clipped) / len(query_clipped)
                if query_clipped
                else 0.0
            ),
            "reward_mean": (
                float(np.mean(outcomes, dtype=np.float64))
                if outcomes
                else 0.0
            ),
            "raw_ig_mean": (
                float(np.mean(raw_ig, dtype=np.float64)) if raw_ig else 0.0
            ),
            "raw_ig_std": (
                float(np.std(raw_ig, dtype=np.float64)) if raw_ig else 0.0
            ),
            "full_vocab_forward_kl": kl,
            "kl_weighted_loss": 0.01 * kl,
            "total_loss": loss,
            "gradient_norm": self._last_gradient_norm,
            "grad_norm_before_clip": self._last_gradient_norm,
            "grad_norm_after_clip": (
                min(
                    float(self._last_gradient_norm),
                    float(self.config["policy"]["max_grad_norm"]),
                )
                if self._last_gradient_norm is not None
                else None
            ),
            "max_grad_norm": float(self.config["policy"]["max_grad_norm"]),
            "ratio_mean": (
                float(np.mean(ratios, dtype=np.float64))
                if ratios
                else 0.0
            ),
            "ratio_p95": (
                float(np.percentile(ratios, 95)) if ratios else 0.0
            ),
            "ratio_std": (
                float(np.std(ratios, dtype=np.float64)) if ratios else 0.0
            ),
            "ratio_min": min(ratios, default=0.0),
            "ratio_max": max(ratios, default=0.0),
            "clip_fraction": (
                sum(clipped) / len(clipped) if clipped else 0.0
            ),
            "clipfrac_low": (
                sum(clipped_low) / len(clipped_low) if clipped_low else 0.0
            ),
            "clipfrac_high": (
                sum(clipped_high) / len(clipped_high) if clipped_high else 0.0
            ),
            "candidate_prompt_count": int(decision.candidate_count),
            "selected_prompt_count": int(decision.selected_count),
            "selected_trajectory_count": (
                int(decision.selected_count)
                * int(self.config["rollout"]["group_size"])
            ),
            "action_tokens": int(action_tokens),
            "checkpoint": checkpoint,
            "model_checkpoint": model_checkpoint,
            "optimizer_steps_this_update": 1,
            "scheduler_steps_this_update": 1,
            "optimizer_steps_since_resume": int(self._total_optimizer_steps),
            "scheduler_steps_since_resume": int(self._total_scheduler_steps),
            "optimizer_steps_total": global_optimizer_steps,
            "scheduler_steps_total": global_scheduler_steps,
        }
        update.update(dict(self._attempt_context.get("sc_metrics", {})))
        update.update(
            dict(self._attempt_context.get("advantage_component_metrics", {}))
        )
        update.update(
            dict(self._attempt_context.get("deferred_exact_ig_metrics", {}))
        )
        update.update(
            dict(self._attempt_context.get("paper_ragen2_metrics", {}))
        )
        update["checkpoint_resource_preflight"] = dict(
            self._attempt_context.get("checkpoint_resource_preflight", {})
        )
        update.update(
            {
                "ragen_signal_mode": str(decision.signal_mode),
                "ragen_selection_mode": str(decision.selection_mode),
                "ragen_selection_mass": float(decision.top_p.selected_mass),
                "ragen_selection_mass_ratio": float(
                    decision.top_p.selected_mass_ratio
                ),
                "ragen_selected_prompt_ids": list(decision.selected_ids),
                "ragen_sorted_outcome_variances": [
                    {
                        "prompt_global_id": str(group.prompt_global_id),
                        "V_O": float(group.outcome_variance),
                    }
                    for group in sorted(
                        groups,
                        key=lambda group: (
                            -float(group.outcome_variance),
                            str(group.prompt_global_id),
                        ),
                    )
                ],
            }
        )
        update.update(gate_update)
        return update

    def commit_successful_update(self, state: TrainingState) -> None:
        self._require_bound()
        counts = self.worker_group.execute_all_sync("strict_attempt_counts")
        if any(
            row["zero_grad"] != 1
            or row["optimizer_step"] != 1
            or row["scheduler_step"] != 1
            or row["backward_microbatches"] < 1
            for row in counts
        ):
            raise RuntimeError(f"Strict one-step counters failed: {counts}")
        checksum = self.actor_parameter_checksum()
        local_actor_digests_before = tuple(
            str(value)
            for value in self.worker_group.execute_all_sync(
                "last_actor_local_parameter_digest"
            )
        )
        sync_started = time.perf_counter()
        sync = self.agent_loop_manager.synchronize_from_actor(
            int(state.successful_update_step),
            checksum,
        )
        self._weight_sync_records.append(sync)
        versions = sync["versions"]
        if len(versions) != self._rl_world_size:
            raise RuntimeError("Post-update vLLM synchronization is incomplete")
        self.agent_loop_manager.sleep_for_scoring()
        self._add_phase_time(
            "weight_sync",
            time.perf_counter() - sync_started,
        )
        checkpoint = (
            str(self._save_checkpoint(state))
            if self._should_checkpoint(state.successful_update_step)
            else None
        )
        model_checkpoint = None
        checkpoint_reload = None
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            readonly = self._verify_checkpoint_readonly_subprocess(
                checkpoint_path,
                state,
            )
            if bool(
                self.config.get("checkpoint", {}).get(
                    "live_distributed_reload_verification",
                    True,
                )
            ):
                full_reload = self._verify_checkpoint_reload(
                    checkpoint_path,
                    state,
                )
            else:
                full_reload = self._verify_checkpoint_without_live_reload(
                    checkpoint_path,
                    state,
                    expected_actor_checksum=checksum,
                    expected_local_actor_digests=local_actor_digests_before,
                )
            checkpoint_reload = {
                "readonly_subprocess": readonly,
                "distributed_reload": full_reload,
            }
            self._checkpoint_reload_results.append(checkpoint_reload)
            if self.stage in {"FORMAL", "PILOT20"}:
                self.checkpoint_resource_preflight(
                    next_successful_update_step=int(
                        state.successful_update_step
                    ),
                    phase="before_model_export",
                )
                resume_root = checkpoint_path.parent
                latest_temporary = resume_root / ".latest.tmp"
                latest = resume_root / "latest"
                if latest_temporary.exists() or latest_temporary.is_symlink():
                    latest_temporary.unlink()
                latest_temporary.symlink_to(checkpoint_path.name)
                os.replace(latest_temporary, latest)
                atomic_write_json(
                    Path(str(self.config["paths"]["runtime_root"]))
                    / "state"
                    / "latest_resume_checkpoint.json",
                    {
                        "successful_update_step": int(
                            state.successful_update_step
                        ),
                        "path": checkpoint,
                        "actor_checksum": checksum,
                    },
                )
                atomic_write_text(
                    Path(str(self.config["paths"]["runtime_root"]))
                    / "state"
                    / "latest_checkpoint",
                    checkpoint + "\n",
                )
                model_checkpoint = str(
                    self._export_model_checkpoint(
                        state,
                        actor_checksum=checksum,
                    )
                )
                if bool(
                    self.config.get("evaluation", {}).get(
                        "asynchronous",
                        False,
                    )
                ):
                    enqueue_eval(
                        self.config["paths"]["runtime_root"],
                        update=int(state.successful_update_step),
                        model_path=model_checkpoint,
                        actor_checksum=checksum,
                    )
            self._write_metrics(
                "checkpoint",
                [
                    {
                        "attempt_id": int(state.attempt_id),
                        "successful_update_step": int(
                            state.successful_update_step
                        ),
                        "checkpoint": checkpoint,
                        "model_checkpoint": model_checkpoint,
                        "readonly_subprocess": readonly,
                        "distributed_reload": full_reload,
                    }
                ],
            )
        update_record = self._aggregate_update_metrics(
            state=state,
            checkpoint=checkpoint,
            model_checkpoint=model_checkpoint,
        )
        self._write_metrics("update", [update_record])
        self._persist_attempt_metrics(
            state_after=state,
            committed=True,
            skip_reason=None,
            checkpoint=checkpoint,
        )
        event_root = Path(str(self.config["paths"]["runtime_root"])) / "events"
        event_root.mkdir(parents=True, exist_ok=True)
        with (event_root / "successful_updates.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "successful_update_step": state.successful_update_step,
                        "attempt_id": state.attempt_id,
                        "data_cursor": state.data_cursor,
                        "checkpoint": checkpoint,
                        "model_checkpoint": model_checkpoint,
                        "actor_checksum": checksum,
                        "worker_counts": counts,
                        "microbatch_metrics": [
                            {
                                key: value
                                for key, value in row.items()
                                if key != "turn_runtime_metrics"
                            }
                            for row in self._microbatch_metrics
                        ],
                        "checkpoint_reload": checkpoint_reload,
                        "wall_seconds": time.perf_counter()
                        - self._stage_started,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        if self.stage in {"FORMAL", "PILOT20"}:
            run_root = Path(str(self.config["paths"]["runtime_root"]))
            latest_path = run_root / "state" / "latest_successful_update"
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = latest_path.with_suffix(".tmp")
            temporary.write_text(
                f"{int(state.successful_update_step)}\n",
                encoding="utf-8",
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, latest_path)
            atomic_write_json(
                run_root / "state" / "training_progress.json",
                {
                    "status": "running",
                    "attempt_id": int(state.attempt_id),
                    "successful_update_step": int(state.successful_update_step),
                    "successful_updates_since_resume": int(
                        state.successful_update_step
                        - self._starting_successful_update
                    ),
                    "data_cursor": int(state.data_cursor),
                    "optimizer_steps_total": int(
                        self._starting_successful_update
                        + self._total_optimizer_steps
                    ),
                    "scheduler_steps_total": int(
                        self._starting_successful_update
                        + self._total_scheduler_steps
                    ),
                    "latest_resume_checkpoint": (
                        str(self._last_checkpoint)
                        if self._last_checkpoint is not None
                        else os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT")
                    ),
                    "latest_model_checkpoint": (
                        str(self._last_model_checkpoint)
                        if self._last_model_checkpoint is not None
                        else None
                    ),
                    "actor_checksum": checksum,
                    "updated_at": time.time(),
                },
            )
        self._release_attempt_memory()

    def _release_attempt_memory(self) -> None:
        """Drop large candidate/learner payloads at the committed boundary."""

        import gc

        deferred = self._attempt_context.get("deferred_exact_ig_tasks")
        if isinstance(deferred, dict):
            deferred.clear()
        self._attempt_context.pop("deferred_exact_ig_tasks", None)
        self._attempt_context.pop("deferred_exact_ig_candidate_trajectory_ids", None)
        self._prepared_groups = ()
        self._turn_runtime_metrics.clear()
        self._microbatch_metrics.clear()
        gc.collect()

    def rollback_pre_step_attempt(self) -> None:
        if self.worker_group is not None:
            self.worker_group.execute_all_sync("rollback_pre_step_attempt")

    def data_cursor(self) -> int:
        if not self.actors:
            return 0
        import ray

        state = ray.get(self.actors["prompt_sampler"].state.remote())
        return int(state["cursor"])

    def rng_state(self) -> dict[str, Any]:
        if not self.actors:
            return {}
        import ray

        return {
            "prompt_sampler": ray.get(
                self.actors["prompt_sampler"].state.remote()
            ),
            "snapshot_step": self._last_snapshot_step,
        }

    def _restore_checkpoint(self, checkpoint: Path) -> TrainingState:
        import ray

        committer = AtomicCheckpointCommitter(checkpoint.parent)
        metadata = committer.validate(checkpoint)
        assert_exact_ig_checkpoint_compatible(
            metadata.algorithm_config,
            self.config,
        )
        if metadata.model_fingerprint != self._fingerprint(
            "actor",
            self.config["paths"]["actor_model"],
        ):
            raise RuntimeError("Actor initialization fingerprint changed")
        self.worker_group.execute_all_sync(
            "load_distributed_training_state",
            str(checkpoint),
        )
        controller = json.loads(
            (checkpoint / "controller" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        sampler = ray.get(
            self.actors["prompt_sampler"].restore_state.remote(
                controller["prompt_sampler"]
            )
        )
        if int(sampler["cursor"]) != int(metadata.data_cursor):
            raise RuntimeError("Restored prompt cursor differs from checkpoint")
        driver_rng = torch.load(
            checkpoint / "rng" / "driver.pt",
            map_location="cpu",
            weights_only=False,
        )
        random.setstate(driver_rng["python"])
        np.random.set_state(driver_rng["numpy"])
        torch.set_rng_state(driver_rng["torch_cpu"])
        return TrainingState(
            attempt_id=metadata.attempt_id,
            successful_update_step=metadata.successful_update_step,
            data_cursor=metadata.data_cursor,
            ig_channel=_channel_state(metadata.ig_channel),
            outcome_channel=_channel_state(metadata.outcome_channel),
            rng_state=dict(controller["training_state"]["rng_state"]),
        )

    def _verify_checkpoint_reload(
        self,
        checkpoint: Path,
        expected_state: TrainingState,
    ) -> dict[str, Any]:
        actor_before = self.actor_parameter_checksum()
        training_state_before = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        restored = self._restore_checkpoint(checkpoint)
        actor_after = self.actor_parameter_checksum()
        training_state_after = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        expected_counters = (
            int(expected_state.attempt_id),
            int(expected_state.successful_update_step),
            int(expected_state.data_cursor),
        )
        restored_counters = (
            int(restored.attempt_id),
            int(restored.successful_update_step),
            int(restored.data_cursor),
        )
        if actor_before != actor_after:
            raise RuntimeError("Actor checksum changed across checkpoint reload")
        if training_state_before != training_state_after:
            raise RuntimeError(
                "Optimizer/scheduler state changed across checkpoint reload"
            )
        if expected_counters != restored_counters:
            raise RuntimeError(
                "Controller counters changed across checkpoint reload: "
                f"expected={expected_counters}, restored={restored_counters}"
            )
        return {
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "actor_checksum_before": actor_before,
            "actor_checksum_after": actor_after,
            "optimizer_scheduler_digest_by_rank_before": training_state_before,
            "optimizer_scheduler_digest_by_rank_after": training_state_after,
            "restored_attempt_id": restored.attempt_id,
            "restored_successful_update_step": restored.successful_update_step,
            "restored_data_cursor": restored.data_cursor,
        }

    def _verify_checkpoint_without_live_reload(
        self,
        checkpoint: Path,
        expected_state: TrainingState,
        *,
        expected_actor_checksum: str | None = None,
        expected_local_actor_digests: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Validate a durable checkpoint without mutating the live workers.

        Loading a complete model and optimizer back into the same colocated
        FSDP/vLLM workers creates a second transient state footprint and is not
        a safe checkpoint-commit operation. A fresh runtime performs the real
        restore validation when a checkpoint is resumed.
        """
        use_local_postcheck = (
            expected_actor_checksum is not None
            and expected_local_actor_digests is not None
        )
        if use_local_postcheck:
            actor_before = str(expected_actor_checksum)
            local_actor_digests_before = tuple(
                str(value) for value in expected_local_actor_digests
            )
        else:
            actor_before = self.actor_parameter_checksum()
            local_actor_digests_before = ()
        training_state_before = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        metadata = AtomicCheckpointCommitter(checkpoint.parent).validate(
            checkpoint
        )
        assert_exact_ig_checkpoint_compatible(
            metadata.algorithm_config,
            self.config,
        )
        expected_counters = (
            int(expected_state.attempt_id),
            int(expected_state.successful_update_step),
            int(expected_state.data_cursor),
        )
        checkpoint_counters = (
            int(metadata.attempt_id),
            int(metadata.successful_update_step),
            int(metadata.data_cursor),
        )
        if checkpoint_counters != expected_counters:
            raise RuntimeError(
                "Checkpoint counters changed before commit: "
                f"expected={expected_counters}, checkpoint={checkpoint_counters}"
            )
        if use_local_postcheck:
            local_actor_digests_after = tuple(
                str(value)
                for value in self.worker_group.execute_all_sync(
                    "local_actor_parameter_digest"
                )
            )
            if local_actor_digests_after != local_actor_digests_before:
                raise RuntimeError(
                    "Actor rank-local digest changed during non-mutating "
                    "checkpoint validation"
                )
            actor_after = actor_before
        else:
            local_actor_digests_after = ()
            actor_after = self.actor_parameter_checksum()
        training_state_after = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        if actor_before != actor_after:
            raise RuntimeError(
                "Actor checksum changed during non-mutating checkpoint validation"
            )
        if training_state_before != training_state_after:
            raise RuntimeError(
                "Optimizer/scheduler state changed during non-mutating "
                "checkpoint validation"
            )
        return {
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "verification_mode": "non_mutating_distributed_artifact_validation",
            "fresh_runtime_restore_required": True,
            "actor_checksum_before": actor_before,
            "actor_checksum_after": actor_after,
            "actor_local_digests_before": list(local_actor_digests_before),
            "actor_local_digests_after": list(local_actor_digests_after),
            "optimizer_scheduler_digest_by_rank_before": training_state_before,
            "optimizer_scheduler_digest_by_rank_after": training_state_after,
            "checkpoint_attempt_id": checkpoint_counters[0],
            "checkpoint_successful_update_step": checkpoint_counters[1],
            "checkpoint_data_cursor": checkpoint_counters[2],
        }

    def _run_tito(self) -> dict[str, Any]:
        self.config = _debug_shape(self.config, prompt_count=2, group_size=2)
        topology = self.bind(require_optimizer=False)
        versions = self.freeze_rollout_boundary(0)
        groups = self.collect_initial_scored_prompt_groups(
            2,
            wave_size=2,
            snapshot_step=versions.actor,
        )
        trajectories = [
            trajectory
            for group in groups
            for trajectory in group.trajectories
        ]
        rollout_search_turns = sum(
            trajectory.search_turn_count for trajectory in trajectories
        )
        rollout_retrieval_records = sum(
            len(trajectory.metadata.get("retrieval_records", ()))
            for trajectory in trajectories
        )
        for trajectory in trajectories:
            assert_environment_information_masked(
                trajectory.token_sources,
                trajectory.action_token_mask,
            )
            if trajectory.policy_mask != trajectory.kl_mask:
                raise RuntimeGateError(
                    "Stage A policy and KL provenance masks differ"
                )
        retriever_canary = self._retriever_stage_a_canary()
        provenance_canary = self._stage_a_token_provenance_canary()
        post_scoring_workers = self.worker_group.execute_all_sync(
            "runtime_identity"
        )
        return {
            "stage": "A",
            "prompt_groups": len(groups),
            "trajectories": len(trajectories),
            "rollout_search_turns": rollout_search_turns,
            "rollout_retrieval_records": rollout_retrieval_records,
            "retriever_async_canary": retriever_canary,
            "token_provenance_canary": provenance_canary,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "post_scoring_fsdp_workers": post_scoring_workers,
            "topology": topology,
        }

    def _run_stop_continue_no_update_smoke(self) -> dict[str, Any]:
        """Exercise the production Stop path without entering optimization."""

        self.config = _debug_shape(self.config, prompt_count=4, group_size=2)
        self.config["selection"]["minimum_positive_prompts"] = 1
        topology = self.bind(require_optimizer=False)
        state = TrainingState()
        self.start_attempt_metrics(state)
        versions = self.freeze_rollout_boundary(0)
        self.record_snapshot_metrics(versions)
        pool = CandidatePool(group_size=2, maximum_prompts=8)
        initial = self.collect_initial_scored_prompt_groups(
            4,
            wave_size=4,
            snapshot_step=versions.actor,
        )
        pool.add(initial)
        controller = StrictAttemptController(self.config)
        selection_started = time.perf_counter()
        decision = controller._select(pool, state)
        refill_count = 0
        if decision.requires_refill:
            pool.add(
                self.collect_scored_prompt_groups(
                    4,
                    snapshot_step=versions.actor,
                )
            )
            refill_count = 1
            decision = controller._select(pool, state)
        if decision.skip_update or decision.selected_count < 1:
            raise RuntimeGateError(
                "Stop smoke RAGEN selection produced no Selected Prompt"
            )
        self.record_selection_metrics(
            state,
            pool.groups(),
            decision,
            refill_count=refill_count,
            selection_seconds=time.perf_counter() - selection_started,
        )
        selected = pool.selected_groups(decision)
        multi_search = [
            record
            for group in selected
            for record in group.trajectories
            if record.search_turn_count >= 2
        ]
        if not multi_search:
            raise RuntimeGateError(
                "Stop smoke selected set contains no multi-Search trajectory"
            )
        actor_before = self.actor_parameter_checksum()
        self.prepare_selected_stop_branches(selected)
        microbatches = tuple(self.selected_microbatches(selected))
        actor_after = self.actor_parameter_checksum()
        if actor_before != actor_after:
            raise RuntimeGateError("Stop smoke changed Actor parameters")
        sc_metrics = dict(self._attempt_context.get("sc_metrics", {}))
        state_count = int(sc_metrics.get("sc/state_count", 0))
        completion_count = int(sc_metrics.get("sc/completion_count", 0))
        if state_count < 1 or completion_count != 2 * state_count:
            raise RuntimeGateError("Stop smoke completion contract failed")
        rows = []
        answer_regression_count = 0
        for group in self._prepared_groups:
            for item in group:
                advantage = item.advantage
                if advantage.answer_policy_credit_eligible:
                    expected_answer = (
                        advantage.normalized_outcome
                        + advantage.centered_format_indicator
                    )
                    if not math.isclose(
                        float(advantage.answer_advantage),
                        float(expected_answer),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ):
                        raise RuntimeGateError(
                            "Stop smoke changed the Answer advantage"
                        )
                    answer_regression_count += 1
                for search_index, sc in (
                    advantage.stop_continue_by_search_index.items()
                ):
                    rows.append(
                        {
                            "prompt_global_id": item.record.prompt_global_id,
                            "trajectory_id": item.record.trajectory_id,
                            "search_index": int(search_index),
                            "A_IG": float(
                                advantage.future_ig_rescaled[search_index]
                            ),
                            "z_O": float(advantage.normalized_outcome),
                            "R_C": float(sc.continue_reward),
                            "R_S1": float(sc.stop_reward_1),
                            "R_S2": float(sc.stop_reward_2),
                            "Delta_SC": float(sc.delta_sc),
                            "s_SC": float(sc.pooled_scale),
                            "A_SC": float(sc.advantage_sc),
                            "sc_clear": bool(sc.sc_clear),
                            "A_task": float(sc.task_advantage),
                            "A_search_old_shadow": float(
                                advantage.search_advantage_old_shadow[
                                    search_index
                                ]
                            ),
                            "A_search_new": float(
                                advantage.search_advantage[search_index]
                            ),
                        }
                    )
        counts = self.worker_group.execute_all_sync("strict_attempt_counts")
        if any(
            int(row["zero_grad"]) != 0
            or int(row["backward_microbatches"]) != 0
            or int(row["optimizer_step"]) != 0
            or int(row["scheduler_step"]) != 0
            for row in counts
        ):
            raise RuntimeGateError(
                f"Stop smoke entered the learner transaction: {counts}"
            )
        if self._checkpoint_writes != 0:
            raise RuntimeGateError("Stop smoke wrote a checkpoint")
        result = {
            "stage": "SC",
            "selected_prompt_count": int(decision.selected_count),
            "selected_trajectory_count": sum(
                len(group.trajectories) for group in selected
            ),
            "multi_search_trajectory_count": len(multi_search),
            "stop_state_count": state_count,
            "stop_completion_count": completion_count,
            "stop_metrics": sc_metrics,
            "advantage_rows": rows,
            "answer_regression_count": answer_regression_count,
            "microbatch_round_count": len(microbatches),
            "actor_checksum_before": actor_before,
            "actor_checksum_after": actor_after,
            "strict_worker_counts": counts,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "topology": topology,
        }
        report_root = (
            Path(str(self.config["paths"]["runtime_root"])) / "reports"
        )
        report_root.mkdir(parents=True, exist_ok=True)
        (report_root / "stop_continue_no_update_smoke.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _exact_group_checksum(groups: Sequence[PromptGroup]) -> str:
        digest = hashlib.sha256()
        for group in sorted(groups, key=lambda item: item.prompt_global_id):
            digest.update(str(group.prompt_global_id).encode("utf-8"))
            digest.update(b"\0")
            for trajectory in sorted(
                group.trajectories,
                key=lambda item: item.trajectory_id,
            ):
                digest.update(str(trajectory.trajectory_id).encode("utf-8"))
                digest.update(b"\0")
                digest.update(
                    json.dumps(
                        {
                            "phi": trajectory.metadata.get(
                                "exact_ig_score_by_prefix", ()
                            ),
                            "ig": sorted(trajectory.immediate_ig.items()),
                            "telescoping_error": trajectory.metadata.get(
                                "telescoping_error"
                            ),
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                )
                digest.update(b"\n")
        return digest.hexdigest()

    def _run_forced_refill_96(self) -> dict[str, Any]:
        if not bool(
            self.config.get("runtime_test", {}).get(
                "force_refill_after_initial_pool",
                False,
            )
        ):
            raise RuntimeGateError(
                "Forced refill stage requires its isolated runtime override"
            )
        checkpoint_raw = os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT")
        if not checkpoint_raw:
            raise RuntimeGateError(
                "Forced refill requires the Pilot update_20 checkpoint"
            )
        checkpoint = Path(checkpoint_raw).resolve()
        topology = self.bind(require_optimizer=True)
        state = self._restore_checkpoint(checkpoint)
        if state.successful_update_step != 20:
            raise RuntimeGateError(
                "Forced refill must restore successful Update 20"
            )
        state_before = state
        versions = self.freeze_rollout_boundary(
            state.successful_update_step
        )
        controller = StrictAttemptController(self.config)
        initial = tuple(
            self.collect_initial_scored_prompt_groups(
                64,
                wave_size=32,
                snapshot_step=versions.actor,
            )
        )
        if len(initial) != 64:
            raise RuntimeGateError("Forced refill initial pool is not 64")
        initial_ids = {group.prompt_global_id for group in initial}
        initial_checksum_before = self._exact_group_checksum(initial)
        initial_profiles = list(self._last_exact_ig_profiles)
        if int(
            self._attempt_context["exact_ig_assignments"][
                "unique_record_count"
            ]
        ) != 64 * 16:
            raise RuntimeGateError(
                "Initial Exact-IG scorer did not receive exactly 64x16 records"
            )
        pool = CandidatePool(group_size=16, maximum_prompts=96)
        pool.add(initial)
        initial_decision = controller._select(pool, state)

        refill = tuple(
            self.collect_scored_prompt_groups(
                32,
                snapshot_step=versions.actor,
            )
        )
        if len(refill) != 32:
            raise RuntimeGateError("Forced refill increment is not 32")
        refill_ids = {group.prompt_global_id for group in refill}
        if initial_ids.intersection(refill_ids):
            raise RuntimeGateError("Forced refill reused a prompt ID")
        refill_profiles = list(self._last_exact_ig_profiles)
        if int(
            self._attempt_context["exact_ig_assignments"][
                "unique_record_count"
            ]
        ) != 32 * 16:
            raise RuntimeGateError(
                "Refill Exact-IG scorer did not receive exactly the new 32x16"
            )
        pool.add(refill)
        initial_checksum_after = self._exact_group_checksum(initial)
        if initial_checksum_before != initial_checksum_after:
            raise RuntimeGateError(
                "Initial 64 Exact-IG metadata changed during refill"
            )
        decision = controller._select(pool, state)
        consensus = self.worker_group.execute_all_sync(
            "assert_distributed_string_sequence",
            list(decision.selected_ids),
        )
        if len({(row["sha256"], row["count"]) for row in consensus}) != 1:
            raise RuntimeGateError("Selected-ID consensus differs by rank")
        versions_after = self.agent_loop_manager.read_weight_versions()
        if {
            (
                int(row["snapshot_step"]),
                str(row["source_checksum"]),
            )
            for row in versions_after
        } != {(20, self._last_checksum)}:
            raise RuntimeGateError(
                "Forced refill changed rollout-start weight version"
            )
        counts = self.worker_group.execute_all_sync("strict_attempt_counts")
        if any(
            row["optimizer_step"] != 0
            or row["scheduler_step"] != 0
            or row["zero_grad"] != 0
            or row["backward_microbatches"] != 0
            for row in counts
        ):
            raise RuntimeGateError(
                f"Forced refill entered optimization: {counts}"
            )
        if (
            state.ig_channel != state_before.ig_channel
            or state.outcome_channel != state_before.outcome_channel
        ):
            raise RuntimeGateError("Forced refill committed channel state")
        if len(pool.groups()) != 96:
            raise RuntimeGateError("Forced refill global pool is not 96")
        if any(len(group.trajectories) != 16 for group in pool.groups()):
            raise RuntimeGateError("Forced refill lost a trajectory group")
        forced_skip = exercise_forced_skip_transaction()
        result = {
            "stage": "FORCED_REFILL96",
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "restored_successful_update": state.successful_update_step,
            "initial_prompt_count": 64,
            "refill_prompt_count": 32,
            "total_unique_prompt_count": len(initial_ids | refill_ids),
            "group_size": 16,
            "initial_exact_ig_checksum_before": initial_checksum_before,
            "initial_exact_ig_checksum_after": initial_checksum_after,
            "initial_exact_ig_reused": True,
            "initial_exact_ig_profiles": initial_profiles,
            "refill_exact_ig_profiles": refill_profiles,
            "only_new_refill_trajectories_scored": 32 * 16,
            "initial_selection": {
                "selected_count": initial_decision.selected_count,
                "selected_ids": list(initial_decision.selected_ids),
            },
            "full_96_selection": {
                "selected_count": decision.selected_count,
                "selected_ids": list(decision.selected_ids),
                "top_p_k_star_before_cap": len(
                    decision.top_p.selected_ids
                )
                + decision.capacity_truncation_count,
                "rho_actual": decision.top_p.selected_mass_ratio,
                "pool_recomputed_from_scratch": True,
                "selection_union_used": False,
            },
            "selected_consensus_by_rank": consensus,
            "weight_versions": versions_after,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "channel_state_committed": False,
            "formal_data_cursor_committed": False,
            "ephemeral_data_cursor": self.data_cursor(),
            "forced_skip": forced_skip,
            "topology": topology,
        }
        report_root = (
            Path(str(self.config["paths"]["runtime_root"])) / "reports"
        )
        report_root.mkdir(parents=True, exist_ok=True)
        (report_root / "forced_refill_96_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    def _collect_gate_calibration_batch(
        self,
        *,
        batch_index: int,
    ) -> tuple[SelectionDecision, Sequence[PromptGroup], Sequence[Any]] | None:
        """Build one real fresh-U0 Selected batch without opening a step."""

        state = TrainingState()
        self.start_attempt_metrics(state)
        versions = self.freeze_rollout_boundary(0)
        self.record_snapshot_metrics(versions)
        rollout = self.config["rollout"]
        initial_count = int(rollout["candidate_prompts_initial"])
        refill_count_size = int(rollout["refill_prompts"])
        maximum_count = int(rollout["candidate_prompts_max"])
        wave_size = int(rollout["prompt_wave_size"])
        if initial_count % wave_size != 0:
            raise RuntimeError("Calibration initial pool must use whole waves")
        pool = CandidatePool(
            group_size=int(rollout["group_size"]),
            maximum_prompts=maximum_count,
        )
        pool.add(
            self.collect_initial_scored_prompt_groups(
                initial_count,
                wave_size=wave_size,
                snapshot_step=versions.actor,
            )
        )
        controller = StrictAttemptController(self.config)
        selection_started = time.perf_counter()
        decision = controller._select(pool, state)
        refill_count = 0
        selection_rounds: list[dict[str, Any]] = []
        while True:
            selection_rounds.append(
                {
                    "pool_size": len(pool),
                    "selected_count": int(decision.selected_count),
                    "requires_refill": bool(decision.requires_refill),
                    "skip_update": bool(decision.skip_update),
                }
            )
            if not decision.requires_refill:
                break
            if len(pool) + refill_count_size > maximum_count:
                raise RuntimeError("Calibration refill exceeds the frozen pool maximum")
            pool.add(
                self.collect_scored_prompt_groups(
                    refill_count_size,
                    snapshot_step=versions.actor,
                )
            )
            refill_count += 1
            decision = controller._select(pool, state)
        self.record_selection_metrics(
            state,
            pool.groups(),
            decision,
            refill_count=refill_count,
            selection_seconds=time.perf_counter() - selection_started,
            selection_rounds=selection_rounds,
        )
        if decision.skip_update:
            return None
        selected = pool.selected_groups(decision)
        checksum_before = self.actor_parameter_checksum()
        self.prepare_selected_stop_branches(selected)
        microbatches = tuple(self.selected_microbatches(selected))
        checksum_after = self.actor_parameter_checksum()
        if checksum_after != checksum_before:
            raise RuntimeError(
                f"Calibration batch {batch_index} changed Actor parameters"
            )
        return decision, selected, microbatches

    def _run_gate_gradient_calibration(self) -> dict[str, Any]:
        """Calibrate immutable gate coefficients on real fresh-U0 batches."""

        role_config = dict(self.config["advantage"]["role_localized_gate"])
        if not bool(role_config.get("calibration_pending", False)):
            raise RuntimeError("Calibration stage requires calibration_pending=true")
        if os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT"):
            raise RuntimeError("Fresh-U0 calibration cannot resume a checkpoint")
        if str(self.config["advantage"]["search_task_mode"]) != (
            SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
        ):
            raise RuntimeError("Calibration config does not select the new mode")
        topology = self.bind(require_optimizer=True)
        profiles: list[BatchGradientProfile] = []
        skipped_batches = 0
        attempted_batches = 0
        maximum_batches = 20
        while attempted_batches < maximum_batches:
            attempted_batches += 1
            collected = self._collect_gate_calibration_batch(
                batch_index=attempted_batches,
            )
            if collected is None:
                skipped_batches += 1
                continue
            decision, selected, microbatches = collected
            batch_id = f"fresh_u0_selected_{len(profiles) + 1:02d}"
            profile = self.profile_role_localized_gate_gradients(
                batch_id,
                microbatches,
            )
            profiles.append(profile)
            decision_events = sum(
                row.decision_gate_event_count for row in profiles
            )
            query_events = sum(row.query_gate_event_count for row in profiles)
            nonzero_channels = all(
                any(
                    float(getattr(row, f"{channel}_gradient_norm")) > 0.0
                    for row in profiles
                )
                for channel in ("main", "decision", "query")
            )
            if (
                len(profiles) >= 3
                and decision_events >= 128
                and query_events >= 64
                and nonzero_channels
            ):
                break
        calibration = calibrate_role_localized_gate_lambdas(
            profiles,
            eta_decision=float(role_config["eta_decision"]),
            eta_query=float(role_config["eta_query"]),
            maximum_gate_to_main_ratio=float(
                role_config["max_gate_to_main_grad_ratio"]
            ),
        )
        calibration.update(
            {
                "calibration_stage": "GATE_CALIBRATION",
                "fresh_u0": True,
                "actor_model": str(self.config["paths"]["actor_model"]),
                "reference_model": str(self.config["paths"]["reference_model"]),
                "attempted_batch_count": int(attempted_batches),
                "skipped_batch_count": int(skipped_batches),
                "actor_checksum": self.actor_parameter_checksum(),
                "generated_at_unix": time.time(),
            }
        )
        artifact_root = Path(str(self.config["paths"]["runtime_root"]))
        artifact_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            artifact_root / "gate_gradient_profiles.json",
            {
                "profiles": [asdict(row) for row in profiles],
                "calibration": calibration,
            },
        )
        manifest_path = Path(str(role_config["calibration_manifest"])).resolve()
        manifest_sha256 = write_immutable_calibration_manifest(
            manifest_path,
            calibration,
        )
        counts = self.worker_group.execute_all_sync("strict_attempt_counts")
        if any(any(int(value) != 0 for value in row.values()) for row in counts):
            raise RuntimeError(f"Calibration ended with nonzero steps: {counts}")
        if self._checkpoint_writes != 0:
            raise RuntimeError("Calibration wrote a checkpoint")
        return {
            "stage": "GATE_CALIBRATION",
            "status": "PASS",
            "profile_count": len(profiles),
            "attempted_batch_count": attempted_batches,
            "skipped_batch_count": skipped_batches,
            "decision_gate_event_count": calibration[
                "decision_gate_event_count"
            ],
            "query_gate_event_count": calibration["query_gate_event_count"],
            "lambda_decision": calibration["lambda_decision"],
            "lambda_query": calibration["lambda_query"],
            "median_gate_to_main_gradient_ratio": calibration[
                "median_gate_to_main_gradient_ratio"
            ],
            "calibration_manifest": str(manifest_path),
            "calibration_manifest_sha256": manifest_sha256,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "strict_worker_counts": counts,
            "topology": topology,
        }

    def _run_mica_no_commit_qualification(
        self,
        *,
        formal_shape: bool,
        backward_profile: bool,
    ) -> dict[str, Any]:
        """Run the production MICA DAG without committing an update.

        The debug-shape variant stops after old-logprob materialization.  The
        formal-shape variant additionally executes the production distributed
        backward path, then discards gradients without an optimizer or
        scheduler step.  Both variants call the production selection,
        selected-only Exact-IG, advantage, packing, and boundary validators.
        """

        import ray
        import torch

        if formal_shape:
            self.config = copy.deepcopy(dict(self.config))
        else:
            self.config = _debug_shape(
                self.config,
                prompt_count=8,
                group_size=4,
            )
        assert_formal_hyperparameters_approved(self.config)
        topology = self.bind(require_optimizer=bool(backward_profile))
        self.worker_group.execute_all_sync("reset_cuda_peak_memory_stats")

        memory_by_phase: list[dict[str, Any]] = []

        def memory_snapshot(phase: str) -> None:
            resources = dict(ray.available_resources())
            memory_by_phase.append(
                {
                    "phase": str(phase),
                    "wall_seconds": float(time.perf_counter() - self._stage_started),
                    "physical_gpus": self._gpu_snapshot(),
                    "rank_allocator": self.worker_group.execute_all_sync(
                        "cuda_memory_snapshot"
                    ),
                    "ray_available_resources": {
                        str(key): float(value)
                        for key, value in resources.items()
                    },
                }
            )

        state = TrainingState()
        self.start_attempt_metrics(state)
        versions = self.freeze_rollout_boundary(0)
        self.record_snapshot_metrics(versions)
        actor_checksum_before = self.actor_parameter_checksum()
        optimizer_scheduler_before = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        memory_snapshot("snapshot_frozen")

        rollout = self.config["rollout"]
        initial_count = int(rollout["candidate_prompts_initial"])
        refill_prompts = int(rollout["refill_prompts"])
        maximum_count = int(rollout["candidate_prompts_max"])
        pool = CandidatePool(
            group_size=int(rollout["group_size"]),
            maximum_prompts=maximum_count,
        )
        pool.add(
            self.collect_initial_scored_prompt_groups(
                initial_count,
                wave_size=int(rollout["prompt_wave_size"]),
                snapshot_step=versions.actor,
            )
        )
        memory_snapshot("candidate_rollout_outcome")
        controller = StrictAttemptController(self.config)
        selection_started = time.perf_counter()
        decision = controller._select(pool, state)
        selection_rounds = []
        refill_count = 0
        while True:
            selection_rounds.append(
                {
                    "pool_size": len(pool),
                    "selected_count": int(decision.selected_count),
                    "requires_refill": bool(decision.requires_refill),
                    "skip_update": bool(decision.skip_update),
                }
            )
            if not decision.requires_refill:
                break
            if len(pool) + refill_prompts > maximum_count:
                raise RuntimeGateError(
                    "MICA qualification requested refill beyond the formal pool"
                )
            pool.add(
                self.collect_scored_prompt_groups(
                    refill_prompts,
                    snapshot_step=versions.actor,
                )
            )
            refill_count += 1
            decision = controller._select(pool, state)
        if decision.skip_update or decision.selected_count < 1:
            raise RuntimeGateError(
                "Answer-only RAGEN produced no selected prompt in qualification"
            )
        self.record_selection_metrics(
            state,
            pool.groups(),
            decision,
            refill_count=refill_count,
            selection_seconds=time.perf_counter() - selection_started,
            selection_rounds=selection_rounds,
        )
        if str(decision.signal_mode) != ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL:
            raise RuntimeGateError("Qualification did not use Answer-only RAGEN")
        selected = tuple(pool.selected_groups(decision))
        selected_ids_before = tuple(
            str(group.prompt_global_id) for group in selected
        )
        exact_scored_before = int(
            self._attempt_context.get("exact_ig_scored_before_selection", -1)
        )
        if exact_scored_before != 0:
            raise RuntimeGateError(
                "Deferred MICA path ran Exact-IG before Answer-only selection"
            )

        selected = tuple(self.finalize_selected_exact_ig(selected))
        selected_ids_after = tuple(
            str(group.prompt_global_id) for group in selected
        )
        if selected_ids_before != selected_ids_after:
            raise RuntimeGateError(
                "Selected prompt IDs changed after deferred Exact-IG"
            )
        memory_snapshot("selected_exact_ig")
        self.prepare_selected_stop_branches(selected)
        microbatches = tuple(self.selected_microbatches(selected))
        memory_snapshot("learner_pack_and_old_logprob")

        search_turn_count = sum(
            1
            for group in selected
            for record in group.trajectories
            for turn in record.turns
            if turn.turn_type is TurnType.SEARCH
            and turn.policy_credit_eligible
        )
        if search_turn_count < 1:
            raise RuntimeGateError(
                "MICA qualification selected set contains no Search turn"
            )
        for round_payloads in microbatches:
            for payload in round_payloads:
                old_logprobs = payload.get("old_logprobs")
                if old_logprobs is None or not bool(
                    torch.isfinite(old_logprobs).all().item()
                ):
                    raise RuntimeGateError(
                        "Qualification old-policy logprobs are missing/non-finite"
                    )

        gradient_norm = None
        transaction_counts = self.worker_group.execute_all_sync(
            "strict_attempt_counts"
        )
        if backward_profile:
            self.zero_grad()
            for microbatch in microbatches:
                self.backward_microbatch(microbatch)
            gradient_norm = self.clip_gradients(
                float(self.config["policy"]["max_grad_norm"])
            )
            transaction_counts = self.worker_group.execute_all_sync(
                "strict_attempt_counts"
            )
            if any(
                int(row["zero_grad"]) != 1
                or int(row["backward_microbatches"]) < 1
                or int(row["optimizer_step"]) != 0
                or int(row["scheduler_step"]) != 0
                for row in transaction_counts
            ):
                raise RuntimeGateError(
                    "Formal-shape backward-only counters are invalid: "
                    f"{transaction_counts}"
                )
            memory_snapshot("learner_backward_only")
            self.rollback_pre_step_attempt()
        else:
            if any(any(int(value) != 0 for value in row.values()) for row in transaction_counts):
                raise RuntimeGateError(
                    "No-update MICA gate entered the learner transaction"
                )

        actor_checksum_after = self.actor_parameter_checksum()
        optimizer_scheduler_after = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        if actor_checksum_after != actor_checksum_before:
            raise RuntimeGateError("Qualification changed Actor parameters")
        if optimizer_scheduler_after != optimizer_scheduler_before:
            raise RuntimeGateError(
                "Qualification changed optimizer/scheduler state"
            )
        if self._checkpoint_writes != 0:
            raise RuntimeGateError("Qualification wrote a checkpoint")

        deferred = dict(
            self._attempt_context.get("deferred_exact_ig_metrics", {})
        )
        components = dict(
            self._attempt_context.get("advantage_component_metrics", {})
        )
        probes = dict(self._attempt_context.get("sc_runtime_metrics", {}))
        if int(deferred.get("exact_ig_scored_before", -1)) != 0:
            raise RuntimeGateError("Deferred Exact-IG telemetry is inconsistent")
        if int(deferred.get("exact_ig_scored_after", -1)) != int(
            deferred.get("selected_trajectory_count", -2)
        ):
            raise RuntimeGateError("Selected-only Exact-IG count is inconsistent")
        for key in (
            "mica/role_gate_actor_loss_count",
            "mica/routed_outcome_entry_count",
            "mica/normal_terminal_outcome_entry_count",
            "mica/observation_policy_mask_violation_count",
        ):
            if int(components.get(key, -1)) != 0:
                raise RuntimeGateError(f"Legacy/mask leakage telemetry failed: {key}")
        if int(probes.get("answer_probe/request_count", -1)) != 0 or int(
            probes.get("answer_probe/completion_count", -1)
        ) != 0:
            raise RuntimeGateError("Diagnostic Answer probes ran in MICA mode")

        microbatch_metrics = [
            {
                key: value
                for key, value in row.items()
                if key != "turn_runtime_metrics"
            }
            for row in self._microbatch_metrics
        ]
        return {
            "stage": self.stage,
            "formal_shape": bool(formal_shape),
            "backward_profile": bool(backward_profile),
            "candidate_prompt_count": len(pool),
            "candidate_trajectory_count": sum(
                len(group.trajectories) for group in pool.groups()
            ),
            "selected_prompt_count": len(selected),
            "selected_trajectory_count": sum(
                len(group.trajectories) for group in selected
            ),
            "selected_prompt_ids_before_exact_ig": list(selected_ids_before),
            "selected_prompt_ids_after_exact_ig": list(selected_ids_after),
            "refill_count": int(refill_count),
            "selection_rounds": selection_rounds,
            "search_turn_count": int(search_turn_count),
            "microbatch_round_count": len(microbatches),
            "deferred_exact_ig": deferred,
            "mica_metrics": components,
            "probe_metrics": probes,
            "microbatch_metrics": microbatch_metrics,
            "gradient_norm": gradient_norm,
            "strict_worker_counts_before_rollback": transaction_counts,
            "strict_worker_counts_after_rollback": (
                self.worker_group.execute_all_sync("strict_attempt_counts")
            ),
            "actor_checksum_before": actor_checksum_before,
            "actor_checksum_after": actor_checksum_after,
            "optimizer_scheduler_digest_before": optimizer_scheduler_before,
            "optimizer_scheduler_digest_after": optimizer_scheduler_after,
            "checkpoint_writes": int(self._checkpoint_writes),
            "memory_by_phase": memory_by_phase,
            "exact_ig_profiles": self._last_exact_ig_profiles,
            "wall_seconds": float(time.perf_counter() - self._stage_started),
            "topology": topology,
        }

    def _run_updates(
        self,
        *,
        target_successful_updates: int,
        debug_shape: bool,
    ) -> dict[str, Any]:
        if debug_shape:
            debug_prompt_count = (
                8 if self.stage == "MICA_ONE_UPDATE" else 4
            )
            self.config = _debug_shape(
                self.config,
                prompt_count=debug_prompt_count,
                group_size=4,
                require_optimizer_compatible=True,
                preserve_formal_schedule=(self.stage == "MICA_ONE_UPDATE"),
            )
        elif self.stage == "D":
            # Stage D keeps the formal 64-prompt/G=16 shape while reusing the
            # isolated runtime schedule validated by Stages B/C. The persisted
            # formal schedule remains null and fail-closed.
            self.config = _with_runtime_smoke_schedule(self.config)
        assert_formal_hyperparameters_approved(self.config)
        topology = self.bind(require_optimizer=True)
        actor_checksum_start = self.actor_parameter_checksum()
        resume_raw = os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT")
        state = (
            self._restore_checkpoint(Path(resume_raw).resolve())
            if resume_raw
            else TrainingState()
        )
        starting_successful_update = int(state.successful_update_step)
        self._starting_successful_update = starting_successful_update
        resume_artifact_recovery = None
        if self.stage == "FORMAL":
            expected_start = int(
                self.config.get("formal", {}).get(
                    "resume_from_successful_update",
                    starting_successful_update,
                )
            )
            if starting_successful_update != expected_start:
                raise RuntimeGateError(
                    "Formal resume checkpoint starts at the wrong successful "
                    f"update: {starting_successful_update} != {expected_start}"
                )
            if bool(
                self.config.get("checkpoint", {}).get(
                    "materialize_missing_cadence_artifacts_on_resume", False
                )
            ):
                self.checkpoint_resource_preflight(
                    next_successful_update_step=starting_successful_update,
                    phase="resume_cadence_artifact_recovery",
                )
            resume_artifact_recovery = (
                self._materialize_missing_resume_cadence_artifacts(state)
            )
        controller = StrictAttemptController(self.config)
        evaluation_steps: set[int] = set()
        if self.stage == "PILOT20":
            evaluation_steps = {
                int(value) for value in self.config["pilot"]["evaluations"]
            }
        elif self.stage == "FORMAL":
            if not bool(self.config.get("evaluation", {}).get("asynchronous", False)):
                cadence = int(
                    self.config["formal_schedule"][
                        "fixed_eval_every_successful_updates"
                    ]
                )
                evaluation_steps = {
                    value
                    for value in range(0, int(target_successful_updates) + 1)
                    if value == 0 or value % cadence == 0
                }
        if (
            state.successful_update_step in evaluation_steps
            and not resume_raw
        ):
            self._run_fixed_eval(
                successful_update_step=state.successful_update_step
            )
        maximum_attempts = int(target_successful_updates) * 20
        attempts = 0
        while state.successful_update_step < int(target_successful_updates):
            if attempts >= maximum_attempts:
                raise RuntimeError(
                    "Exceeded the fail-closed attempt budget before reaching "
                    "the requested successful-update count"
                )
            previous_success = int(state.successful_update_step)
            result = controller.run_attempt(state, self)
            state = result.state
            attempts += 1
            if (
                result.optimizer_committed
                and state.successful_update_step != previous_success + 1
            ):
                raise RuntimeError(
                    "A committed attempt did not advance exactly one "
                    "successful update"
                )
            if (
                result.optimizer_committed
                and state.successful_update_step in evaluation_steps
            ):
                self._run_fixed_eval(
                    successful_update_step=state.successful_update_step
                )
        smoke_without_checkpoints = self.stage in {
            "B",
            "C",
            "D",
            "MICA_ONE_UPDATE",
        }
        if smoke_without_checkpoints:
            if self._last_checkpoint is not None or self._checkpoint_writes != 0:
                raise RuntimeError("Smoke runtime unexpectedly persisted a checkpoint")
            checkpoint_reload = {
                "status": "NOT_RUN_BY_USER_POLICY",
                "reason": "smoke_retains_logs_and_metrics_only",
            }
        elif self.stage == "PILOT20":
            expected_steps = {
                int(value) for value in self.config["pilot"]["checkpoints"]
            }
            observed_steps = {
                int(
                    json.loads(
                        (path / "metadata.json").read_text(encoding="utf-8")
                    )["successful_update_step"]
                )
                for path in (
                    Path(str(self.config["paths"]["runtime_root"]))
                    / "checkpoints"
                ).iterdir()
                if path.is_dir() and (path / "metadata.json").is_file()
            }
            if observed_steps != expected_steps:
                raise RuntimeError(
                    "Pilot checkpoint cadence mismatch: "
                    f"expected={sorted(expected_steps)} "
                    f"observed={sorted(observed_steps)}"
                )
            checkpoint_reload = {
                "status": "PASS",
                "results": self._checkpoint_reload_results,
            }
        else:
            checkpoint_reload = {
                "status": (
                    "PASS"
                    if not self._checkpoint_reload_results
                    or all(
                        row["readonly_subprocess"]["status"] == "PASS"
                        and row["distributed_reload"]["status"] == "PASS"
                        for row in self._checkpoint_reload_results
                    )
                    else "FAIL"
                ),
                "results": self._checkpoint_reload_results,
            }
        expected_runtime_steps = (
            int(target_successful_updates) - starting_successful_update
        )
        if self._total_optimizer_steps != expected_runtime_steps:
            raise RuntimeError(
                "Optimizer total does not match successful updates: "
                f"{self._total_optimizer_steps} != {expected_runtime_steps}"
            )
        if self._total_scheduler_steps != expected_runtime_steps:
            raise RuntimeError(
                "Scheduler total does not match successful updates: "
                f"{self._total_scheduler_steps} != {expected_runtime_steps}"
            )
        return {
            "stage": self.stage,
            "attempts": attempts,
            "successful_updates": state.successful_update_step,
            "optimizer_steps_per_success": 1,
            "scheduler_steps_per_success": 1,
            "optimizer_steps_since_resume": self._total_optimizer_steps,
            "scheduler_steps_since_resume": self._total_scheduler_steps,
            "optimizer_steps_total": (
                starting_successful_update + self._total_optimizer_steps
            ),
            "scheduler_steps_total": (
                starting_successful_update + self._total_scheduler_steps
            ),
            "checkpoint_writes": self._checkpoint_writes,
            "actor_checksum_start": actor_checksum_start,
            "actor_checksum_end": self.actor_parameter_checksum(),
            "strict_worker_counts": self.worker_group.execute_all_sync(
                "strict_attempt_counts"
            ),
            "microbatch_metrics": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "turn_runtime_metrics"
                }
                for row in self._microbatch_metrics
            ],
            "mica_metrics": dict(
                self._attempt_context.get("advantage_component_metrics", {})
            ),
            "deferred_exact_ig": dict(
                self._attempt_context.get("deferred_exact_ig_metrics", {})
            ),
            "probe_metrics": dict(
                self._attempt_context.get("sc_runtime_metrics", {})
            ),
            "gpu_memory_by_rank": self.worker_group.execute_all_sync(
                "cuda_memory_snapshot"
            ),
            "last_checkpoint": (
                str(self._last_checkpoint)
                if self._last_checkpoint is not None
                else None
            ),
            "checkpoint_reload": checkpoint_reload,
            "resume_artifact_recovery": resume_artifact_recovery,
            "eval_results": self._eval_results,
            "exact_ig_oracle_canary_by_rank": (
                self.worker_group.execute_all_sync("exact_ig_canary_summary")
            ),
            "topology": topology,
        }

    def _run_resume_world_size_validation(self, checkpoint: Path) -> dict[str, Any]:
        """Restore a committed checkpoint without rollout or parameter updates."""
        topology = self.bind(require_optimizer=True)
        restored = self._restore_checkpoint(checkpoint)
        checksums = self.worker_group.execute_all_sync("global_actor_checksum")
        digests = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        counts = self.worker_group.execute_all_sync("strict_attempt_counts")
        expected_world_size = int(self.config["learner"]["world_size"])
        if len(checksums) != expected_world_size or len(digests) != expected_world_size:
            raise RuntimeError(
                "Resume validation returned an incomplete worker set for the "
                f"configured world size {expected_world_size}"
            )
        configured_resume_step = int(
            self.config.get("formal", {}).get(
                "resume_from_successful_update", 0
            )
        )
        if configured_resume_step not in {0, int(restored.successful_update_step)}:
            raise RuntimeError(
                "Resume validation config/checkpoint step mismatch: "
                f"config={configured_resume_step} "
                f"checkpoint={restored.successful_update_step}"
            )
        if any(
            row["optimizer_step"] != 0
            or row["scheduler_step"] != 0
            or row["zero_grad"] != 0
            for row in counts
        ):
            raise RuntimeError(f"Validation changed training counters: {counts}")
        if len(set(checksums)) != 1:
            raise RuntimeError(f"Restored actor checksums disagree: {checksums}")
        snapshot_lifecycle = self.worker_group.execute_all_sync(
            "validate_reward_snapshot_sync_cycles",
            [
                int(restored.successful_update_step),
                int(restored.successful_update_step) + 1,
            ],
        )
        if len(snapshot_lifecycle) != expected_world_size:
            raise RuntimeError(
                "Reward Snapshot lifecycle validation returned an incomplete "
                "worker set"
            )
        for row in snapshot_lifecycle:
            if row["actor_checksum_before"] != row["actor_checksum_after"]:
                raise RuntimeError(
                    "Reward Snapshot lifecycle validation changed an Actor"
                )
            if (
                row["optimizer_scheduler_digest_before"]
                != row["optimizer_scheduler_digest_after"]
            ):
                raise RuntimeError(
                    "Reward Snapshot lifecycle validation changed optimizer state"
                )
            if (
                row["strict_attempt_counts_before"]
                != row["strict_attempt_counts_after"]
            ):
                raise RuntimeError(
                    "Reward Snapshot lifecycle validation changed step counters"
                )
            if row["reward_parameter_dtype"] != "float32":
                raise RuntimeError(
                    "Reward Snapshot lifecycle validation is not pure FP32"
                )
            if any(
                cycle["sync_mode"]
                != "streaming_sharded_dtensor_per_parameter"
                for cycle in row["cycles"]
            ):
                raise RuntimeError(
                    "Reward Snapshot lifecycle used an unbounded sync path"
                )
        for cycle_index in range(2):
            cycle_checksums = {
                row["cycles"][cycle_index]["reward_snapshot_checksum"]
                for row in snapshot_lifecycle
            }
            if len(cycle_checksums) != 1:
                raise RuntimeError(
                    "Reward Snapshot lifecycle checksum differs across ranks"
                )
        export_root = (
            Path(str(self.config["paths"]["runtime_root"]))
            / "stage_results"
            / "streaming_export_validation"
        )
        if export_root.exists():
            raise RuntimeError(
                f"Streaming export validation destination exists: {export_root}"
            )
        export_root.mkdir(parents=True)
        export_rows = self.worker_group.execute_all_sync(
            "export_hf_model_checkpoint",
            str(export_root),
            int(restored.successful_update_step),
            str(checksums[0]),
            True,
        )
        model_file = export_root / "model.safetensors"
        if not model_file.is_file() or model_file.stat().st_size < 1024**3:
            raise RuntimeError("Streaming HF export validation file is incomplete")
        if len(export_rows) != expected_world_size or any(
            row["state_materialization_mode"]
            != "streaming_sharded_dtensor_per_parameter"
            for row in export_rows
        ):
            raise RuntimeError("Streaming HF export used an unbounded state path")
        import torch
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as artifact:
            export_keys = tuple(sorted(artifact.keys()))
            export_metadata = dict(artifact.metadata() or {})
            embedding = artifact.get_tensor("model.embed_tokens.weight")
            lm_head = artifact.get_tensor("lm_head.weight")
        if len(export_keys) != int(export_rows[0]["state_tensor_count"]):
            raise RuntimeError("Streaming HF export state schema is incomplete")
        if export_metadata.get("successful_update_step") != str(
            restored.successful_update_step
        ):
            raise RuntimeError("Streaming HF export update metadata differs")
        if export_metadata.get("actor_checksum") != str(checksums[0]):
            raise RuntimeError("Streaming HF export Actor metadata differs")
        if not torch.equal(embedding, lm_head):
            raise RuntimeError("Streaming HF export has divergent tied weights")
        del embedding, lm_head
        export_file_bytes = int(model_file.stat().st_size)
        export_sha256 = _sha256_file(model_file)
        post_export_checksums = self.worker_group.execute_all_sync(
            "global_actor_checksum"
        )
        post_export_digests = self.worker_group.execute_all_sync(
            "local_optimizer_scheduler_digest"
        )
        post_export_counts = self.worker_group.execute_all_sync(
            "strict_attempt_counts"
        )
        if post_export_checksums != checksums:
            raise RuntimeError("Streaming HF export changed the restored Actor")
        if post_export_digests != digests:
            raise RuntimeError(
                "Streaming HF export changed optimizer/scheduler state"
            )
        if post_export_counts != counts:
            raise RuntimeError("Streaming HF export changed strict step counters")
        shutil.rmtree(export_root)
        streaming_export = {
            "status": "PASS",
            "rows": export_rows,
            "model_file_bytes": export_file_bytes,
            "model_file_sha256": export_sha256,
            "state_key_count": len(export_keys),
            "tied_embedding_lm_head_equal": True,
            "temporary_artifact_removed": not export_root.exists(),
            "actor_checksums_after": post_export_checksums,
            "optimizer_scheduler_digests_after": post_export_digests,
            "strict_attempt_counts_after": post_export_counts,
        }
        checkpoint_metadata = json.loads(
            (checkpoint / "metadata.json").read_text()
        )
        source_fsdp_world_size = int(
            checkpoint_metadata["fsdp_world_size"]
        )
        if source_fsdp_world_size < 1:
            raise RuntimeError("Checkpoint metadata has an invalid FSDP world size")
        return {
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "source_fsdp_world_size": source_fsdp_world_size,
            "target_fsdp_world_size": expected_world_size,
            "source_to_target_rng_mapping": {
                str(rank): str(rank % source_fsdp_world_size)
                for rank in range(expected_world_size)
            },
            "successful_update_step": int(restored.successful_update_step),
            "data_cursor": int(restored.data_cursor),
            "worker_count": len(checksums),
            "actor_checksums": checksums,
            "optimizer_scheduler_digests": digests,
            "strict_attempt_counts": counts,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "reward_snapshot_lifecycle": snapshot_lifecycle,
            "streaming_hf_export": streaming_export,
            "topology": topology,
        }

    def run(self) -> None:
        result_root = Path(str(self.config["paths"]["runtime_root"])) / "stage_results"
        result_root.mkdir(parents=True, exist_ok=True)
        try:
            exact_ig_gate = assert_exact_ig_parity_gate(self.config)
            preflight_health = self.retriever_health()
            if self.stage == "GATE_CALIBRATION":
                result = self._run_gate_gradient_calibration()
            elif self.stage == "A":
                result = self._run_tito()
            elif self.stage == "SC":
                result = self._run_stop_continue_no_update_smoke()
            elif self.stage == "B":
                result = self._run_updates(
                    target_successful_updates=1,
                    debug_shape=True,
                )
            elif self.stage == "C":
                result = self._run_updates(
                    target_successful_updates=5,
                    debug_shape=True,
                )
            elif self.stage == "D":
                result = self._run_updates(
                    target_successful_updates=1,
                    debug_shape=False,
                )
            elif self.stage == "MICA_E2E_NOUPDATE":
                result = self._run_mica_no_commit_qualification(
                    formal_shape=False,
                    backward_profile=False,
                )
            elif self.stage == "MICA_ONE_UPDATE":
                result = self._run_updates(
                    target_successful_updates=1,
                    debug_shape=True,
                )
            elif self.stage == "MICA_FORMAL_SHAPE":
                result = self._run_mica_no_commit_qualification(
                    formal_shape=True,
                    backward_profile=True,
                )
            elif self.stage in {"E", "PILOT50"}:
                result = self._run_updates(
                    target_successful_updates=50,
                    debug_shape=False,
                )
            elif self.stage == "PILOT20":
                result = self._run_updates(
                    target_successful_updates=int(
                        self.config["pilot"]["successful_updates"]
                    ),
                    debug_shape=False,
                )
            elif self.stage == "FORCED_REFILL96":
                result = self._run_forced_refill_96()
            elif self.stage == "RESUME_VALIDATE_3RANK":
                checkpoint_raw = os.environ.get("AGENTIC_RL_RESUME_CHECKPOINT")
                if not checkpoint_raw:
                    raise RuntimeGateError(
                        "RESUME_VALIDATE_3RANK requires AGENTIC_RL_RESUME_CHECKPOINT"
                    )
                result = self._run_resume_world_size_validation(
                    Path(checkpoint_raw).resolve()
                )
            elif self.stage == "FORMAL":
                assert_formal_hyperparameters_approved(self.config)
                result = self._run_updates(
                    target_successful_updates=int(
                        self.config["formal_schedule"][
                            "total_successful_updates"
                        ]
                    ),
                    debug_shape=False,
                )
            else:
                raise ValueError(f"Unknown runtime stage: {self.stage}")
            result["retriever_health_preflight"] = preflight_health
            result["exact_ig_runtime_gate"] = {
                "structural_audit_pass": bool(
                    exact_ig_gate.get("allow_fast_path_training")
                ),
                "numeric_difference_policy": exact_ig_gate.get(
                    "numeric_difference_policy"
                ),
                "audit_path": str(_parity_summary_path(self.config)),
            }
            result["retriever_health"] = self.retriever_health()
            result["status"] = "PASS"
            if self.stage in {"FORMAL", "PILOT20"}:
                atomic_write_json(
                    Path(str(self.config["paths"]["runtime_root"]))
                    / "state"
                    / "trainer_result.json",
                    {
                        "status": "PASS",
                        "successful_update_step": int(
                            result["successful_updates"]
                        ),
                        "optimizer_steps_total": int(
                            result["optimizer_steps_total"]
                        ),
                        "scheduler_steps_total": int(
                            result["scheduler_steps_total"]
                        ),
                        "completed_at": time.time(),
                    },
                )
        except BaseException as exc:
            result = {
                "stage": self.stage,
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (result_root / f"stage_{self.stage.lower()}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if self.stage in {"FORMAL", "PILOT20"}:
                atomic_write_json(
                    Path(str(self.config["paths"]["runtime_root"]))
                    / "state"
                    / "fatal_status.json",
                    {
                        "fatal": True,
                        "source": "trainer",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )
            raise
        finally:
            if self.topology is not None:
                self.topology.shutdown()
        (result_root / f"stage_{self.stage.lower()}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def create_runtime_adapter(
    config: Mapping[str, Any],
) -> VerlAttemptRuntimeAdapter:
    return VerlAttemptRuntimeAdapter(config)
