from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentic_rl.exact_ig.precision_policy import (
    precision_runtime_metadata,
    production_precision_policy,
)
from agentic_rl.exact_ig.sequential_oracle import (
    sequential_teacher_forced_oracle,
)
from agentic_rl.exact_ig.vectorized_scorer import (
    OFFICIAL_FULL_LOGITS,
    VectorizedExactIGScorer,
)
from agentic_rl.exact_ig.target_schema import (
    ANSWER_SCAFFOLD_TEXT,
    CANONICAL_ALIAS_POLICY,
    EXACT_IG_VERSION,
    FAST_PATH_STRUCTURE,
    INFO_GAIN_TYPE,
    PRODUCTION_PRECISION_MODE,
    SCORE_MASK_POLICY,
    exact_ig_tokenizer_identity,
)
from agentic_rl.advantage.a2tgpo import (
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE,
)
from agentic_rl.advantage.mica_ig import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
)
from agentic_rl.policy.gate_gradient_calibration import (
    global_gradient_profile_from_shards,
    parameter_shard_sha256,
)
from agentic_rl.policy.reference_kl import actor_to_reference_full_vocab_kl
from agentic_rl.policy.strict_onpolicy_loss import (
    a2tgpo_adaptive_turn_objective,
    fixed_gate_turn_objective,
)
from agentic_rl.policy.turn_ratio import compute_turn_ratios

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


def successful_update_warmup_factor(
    completed_scheduler_steps: int,
    warmup_successful_updates: int,
) -> float:
    if warmup_successful_updates <= 0:
        raise ValueError("warmup_successful_updates must be positive")
    return min(
        float(completed_scheduler_steps + 1)
        / float(warmup_successful_updates),
        1.0,
    )


def _uniform_sample_indices(
    numel: int,
    *,
    maximum_samples: int,
    device: Any,
) -> Any:
    import torch

    count = min(int(maximum_samples), int(numel))
    if count <= 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    if count == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)
    # Float32 linspace can round a large final index up to numel. Integer
    # arithmetic keeps every checksum sample strictly in [0, numel - 1].
    ordinal = torch.arange(count, dtype=torch.int64, device=device)
    return torch.div(
        ordinal * (int(numel) - 1),
        count - 1,
        rounding_mode="floor",
    )


def _causal_state_mask(policy_mask: Any) -> Any:
    import torch

    mask = policy_mask.bool()
    if mask.ndim != 2:
        raise ValueError("policy_mask must have shape [batch, sequence]")
    if bool(mask[:, 0].any().detach().cpu().item()):
        raise ValueError("A token at sequence index zero has no causal logit state")
    result = torch.zeros_like(mask)
    result[:, :-1] = mask[:, 1:]
    return result


def _update_state_digest(digest: Any, value: Any) -> None:
    import torch

    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(f"sequence:{len(value)}\0".encode("ascii"))
        for item in value:
            _update_state_digest(digest, item)
        return
    if torch.is_tensor(value):
        local = value.detach()
        if hasattr(local, "to_local"):
            local = local.to_local()
        flat = local.reshape(-1)
        digest.update(f"tensor:{tuple(local.shape)}:{local.dtype}\0".encode("ascii"))
        if flat.numel():
            indices = _uniform_sample_indices(
                flat.numel(),
                maximum_samples=64,
                device=flat.device,
            )
            sample = flat.index_select(0, indices).cpu().contiguous().numpy()
            digest.update(sample.tobytes())
        return
    digest.update(f"{type(value).__qualname__}:{value!r}\0".encode("utf-8"))


def _update_sampled_state_digest_entry(
    digest: Any,
    name: str,
    value: Any,
) -> None:
    """Add one state-dict entry using the Reward Snapshot digest contract."""
    import torch

    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    if not isinstance(value, torch.Tensor):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\n")
        return
    tensor = value.detach().cpu()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    flat = tensor.reshape(-1)
    if flat.numel():
        indices = _uniform_sample_indices(
            flat.numel(),
            maximum_samples=32,
            device=flat.device,
        )
        digest.update(flat.index_select(0, indices).float().numpy().tobytes())
    digest.update(b"\n")


def _to_builtin_optimizer_metadata(value: Any) -> Any:
    """Remove OmegaConf containers from optimizer checkpoint metadata."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(value, ListConfig):
        return tuple(
            _to_builtin_optimizer_metadata(item)
            for item in value
        )
    if isinstance(value, DictConfig):
        return {
            key: _to_builtin_optimizer_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_to_builtin_optimizer_metadata(item) for item in value)
    if isinstance(value, list):
        return [_to_builtin_optimizer_metadata(item) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _to_builtin_optimizer_metadata(item)
            for key, item in value.items()
        }
    return value


def _load_safetensors_state_dict_with_tied_key_validation(
    model: Any,
    filename: Path,
) -> int:
    """Strictly load an exported state dict that contains explicit tied keys.

    ``safetensors.torch.load_model`` assumes a tied model artifact stores only
    one member of each shared-storage group. Our FSDP export intentionally
    clones every state-dict key so the HF artifact is self-contained. Validate
    those explicit aliases before using PyTorch's strict state-dict loader.
    """
    import torch
    from safetensors.torch import load_file

    exported_state = load_file(filename, device="cpu", backend="mmap")
    model_state = model.state_dict()
    expected_keys = set(model_state)
    exported_keys = set(exported_state)
    missing = sorted(expected_keys - exported_keys)
    unexpected = sorted(exported_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Restored Reward Snapshot state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    storage_aliases: dict[
        tuple[str, int, int, int, tuple[int, ...], tuple[int, ...]],
        list[str],
    ] = {}
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        alias_key = (
            str(tensor.device),
            int(storage.data_ptr()),
            int(storage.nbytes()),
            int(tensor.storage_offset()),
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.stride()),
        )
        storage_aliases.setdefault(alias_key, []).append(name)

    tied_groups = 0
    for names in storage_aliases.values():
        if len(names) < 2:
            continue
        tied_groups += 1
        reference_name = names[0]
        reference = exported_state[reference_name]
        for alias_name in names[1:]:
            alias = exported_state[alias_name]
            if not torch.equal(reference, alias):
                raise RuntimeError(
                    "Restored Reward Snapshot has divergent tied weights: "
                    f"{reference_name!r} != {alias_name!r}"
                )

    incompatible = model.load_state_dict(exported_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Restored Reward Snapshot state mismatch after strict load: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return tied_groups


def _classify_exact_ig_canary(
    *,
    token_allclose: bool,
    phi_allclose: bool,
    ig_allclose: bool,
    finite: bool,
    target_coverage: bool,
    canonical_answer_agreement: bool,
    non_ambiguous_sign_agreement: bool,
    turn_ranking_agreement: bool,
    token_error: float,
    phi_error: float,
    ig_error: float,
    telescoping_error: float,
    telemetry_token_error: float,
    telemetry_phi_error: float,
    telemetry_ig_error: float,
    maximum_phi_safety_error: float,
    maximum_ig_safety_error: float,
    maximum_telescoping_error: float,
    observed_p99_ig_error: float | None = None,
    calibration_p99_ig_error: float | None = None,
    enforce_p99_drift: bool = False,
) -> tuple[bool, bool]:
    """Separate shape-dependent telemetry from structural/semantic safety.

    Token-level Fast/Oracle allclose is intentionally not a stop condition.
    Different packed matrix shapes can produce finite FP32 drift while preserving
    the Exact-IG structure and downstream decision. Catastrophic Phi/IG drift,
    non-finite values, or semantic changes remain fail-closed.
    """

    if maximum_phi_safety_error <= 0 or maximum_ig_safety_error <= 0:
        raise ValueError("Exact-IG numerical safety limits must be positive")
    if (
        enforce_p99_drift
        and (
            observed_p99_ig_error is None
            or calibration_p99_ig_error is None
            or calibration_p99_ig_error <= 0
        )
    ):
        raise ValueError("P99 drift enforcement requires a positive calibration")
    threshold_exceeded = (
        not token_allclose
        or not phi_allclose
        or not ig_allclose
        or token_error > telemetry_token_error
        or phi_error > telemetry_phi_error
        or ig_error > telemetry_ig_error
        or telescoping_error > maximum_telescoping_error
    )
    safety_failure = (
        not finite
        or not target_coverage
        or not canonical_answer_agreement
        or not non_ambiguous_sign_agreement
        or not turn_ranking_agreement
        or phi_error > maximum_phi_safety_error
        or ig_error > maximum_ig_safety_error
        or telescoping_error > maximum_telescoping_error
        or (
            enforce_p99_drift
            and float(observed_p99_ig_error)
            > 2.0 * float(calibration_p99_ig_error)
        )
    )
    return threshold_exceeded, safety_failure


def _write_exact_ig_canary_failure(
    *,
    task: Any,
    scored: Any,
    oracle: Any,
    token_error: float,
    phi_error: float,
    ig_error: float,
) -> Path:
    """Persist a replayable token-level diagnostic, never model state."""
    root = (
        Path(__file__).resolve().parents[3]
        / "runtime"
        / "diagnostics"
        / "exact_ig_canary"
    )
    root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        f"{task.prompt_global_id}\0{task.trajectory_id}".encode("utf-8")
    ).hexdigest()[:16]
    destination = root / f"failure-{identity}.json"

    payload = {
        "schema": "exact_ig_canary_replay_official_offset_fp32_no_anchor_v4",
        "exact_ig_version": EXACT_IG_VERSION,
        "prompt_global_id": str(task.prompt_global_id),
        "trajectory_id": str(task.trajectory_id),
        "original_input_ids": [
            int(value)
            for value in task.input_ids[: task.original_token_count].tolist()
        ],
        "original_attention_mask": [
            int(value) for value in task.original_attention_mask.tolist()
        ],
        "original_position_ids": [
            int(value) for value in task.original_position_ids.tolist()
        ],
        "prefix_end_positions": [
            int(value) for value in task.prefix_end_positions
        ],
        "canonical_answer": task.canonical_answer,
        "canonical_answer_sha256": task.canonical_answer_hash,
        "canonical_alias_policy": CANONICAL_ALIAS_POLICY,
        "token_max_abs_error": float(token_error),
        "target_token_ids": [
            int(value) for value in task.canonical_target.token_ids
        ],
        "target_token_ids_hash": task.target_token_ids_hash,
        "answer_token_ids": [
            int(value) for value in task.canonical_target.answer_token_ids
        ],
        "answer_token_start": int(
            task.canonical_target.answer_token_start
        ),
        "answer_token_end": int(task.canonical_target.answer_token_end),
        "target_score_mask": [
            bool(value) for value in task.canonical_target.score_mask
        ],
        "score_mask_policy": SCORE_MASK_POLICY,
        "info_gain_type": INFO_GAIN_TYPE,
        "fast_path_structure": FAST_PATH_STRUCTURE,
        "maximum_extended_sequence_length": int(
            task.maximum_extended_sequence_length
        ),
        "maximum_position_id_exclusive": int(
            task.maximum_position_id_exclusive
        ),
        "fast": {
            "score_by_prefix": [float(value) for value in scored.score_by_prefix],
            "immediate_ig": [float(value) for value in scored.immediate_ig],
            "score_token_ids_by_prefix": [
                [int(value) for value in token_ids]
                for token_ids in scored.score_token_ids_by_prefix
            ],
            "answer_token_log_probs_by_prefix": [
                [float(value) for value in row]
                for row in scored.answer_token_log_probs_by_prefix
            ],
            "runtime_metadata": dict(scored.runtime_metadata),
        },
        "oracle": {
            "score_by_prefix": [float(value) for value in oracle.score_by_prefix],
            "immediate_ig": [float(value) for value in oracle.immediate_ig],
            "score_token_ids_by_prefix": [
                [int(value) for value in token_ids]
                for token_ids in oracle.score_token_ids_by_prefix
            ],
            "token_scores": [
                {
                    "prefix_index": int(item.prefix_index),
                    "physical_token_index": int(item.physical_token_index),
                    "token_id": int(item.token_id),
                    "decoded_token": item.decoded_token,
                    "predicting_logit_index": int(item.predicting_logit_index),
                    "score_mask": bool(item.score_mask),
                    "token_log_prob": float(item.token_log_prob),
                }
                for item in oracle.token_scores
            ],
            "answer_token_log_probs_by_prefix": [
                [float(value) for value in row]
                for row in oracle.answer_token_log_probs_by_prefix
            ],
            "runtime_metadata": dict(oracle.runtime_metadata),
        },
        "phi_max_abs_error": float(phi_error),
        "ig_max_abs_error": float(ig_error),
        "contains_model_or_optimizer_state": False,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


class StrictOnPolicyFSDP2Worker(AsyncActorRolloutRefWorker):
    """veRL FSDP2 worker with the frozen custom one-step objective.

    The stock veRL PPO update is deliberately not called. This worker uses
    veRL for FSDP2 initialization, hybrid vLLM weight synchronization and
    distributed process management, while the actual loss is the audited
    A2TGPO/full-vocabulary-KL objective.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().init_model()
        if str(self.config.actor.strategy) != "fsdp2":
            raise RuntimeError("Strict worker requires actor.strategy=fsdp2")
        configured_world_size = int(self.config.actor.fsdp_config.fsdp_size)
        if int(self.world_size) != configured_world_size:
            raise RuntimeError(
                "Strict worker world size disagrees with configured FSDP size: "
                f"runtime={self.world_size}, configured={configured_world_size}"
            )
        if str(self.role) != "actor_rollout_ref":
            raise RuntimeError("Strict worker requires actor_rollout_ref role")
        for key, value in tuple(self.actor_optimizer.defaults.items()):
            self.actor_optimizer.defaults[key] = (
                _to_builtin_optimizer_metadata(value)
            )
        for group in self.actor_optimizer.param_groups:
            for key, value in tuple(group.items()):
                if key != "params":
                    group[key] = _to_builtin_optimizer_metadata(value)
        scheduler_mode = str(
            getattr(self.config, "project_scheduler_mode", "")
        )
        if scheduler_mode:
            if scheduler_mode != "successful_update_constant_with_warmup":
                raise RuntimeError(
                    f"Unsupported project scheduler mode: {scheduler_mode}"
                )
            warmup_steps = int(
                getattr(
                    self.config,
                    "project_warmup_successful_updates",
                    0,
                )
            )
            if warmup_steps <= 0:
                raise RuntimeError(
                    "Project successful-update warmup must be positive"
                )
            # veRL's stock constant scheduler starts at LR=0. The frozen Pilot
            # contract requires Update 1 to use 1/warmup of base LR, so use a
            # successful-update-indexed LambdaLR whose constructor sets that
            # first non-zero rate and whose step is called only after commits.
            self.actor_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.actor_optimizer,
                lr_lambda=lambda completed: successful_update_warmup_factor(
                    int(completed),
                    warmup_steps,
                ),
            )
        for parameter in self.ref_module_fsdp.parameters():
            parameter.requires_grad_(False)
        self.ref_module_fsdp.eval()
        self._snapshot_step = -1
        self._zero_grad_calls = 0
        self._backward_calls = 0
        self._optimizer_steps = 0
        self._scheduler_steps = 0
        self._attempt_optimizer_committed = False
        self._restored_checkpoint_step: int | None = None
        self._restored_checkpoint_source: str | None = None
        self._gate_calibration_channel: str | None = None
        self._gate_calibration_gradients: dict[str, list[Any]] = {}
        self._gate_calibration_event_counts: dict[str, int] = {}
        self._gate_calibration_batch_id: str | None = None
        self._gate_calibration_parameter_hash = ""
        self._gate_calibration_optimizer_digest = ""
        self._exact_ig_precision_mode = str(
            getattr(
                self.config,
                "exact_ig_precision_mode",
                PRODUCTION_PRECISION_MODE,
            )
        )
        if self._exact_ig_precision_mode != PRODUCTION_PRECISION_MODE:
            raise RuntimeError(
                "The runtime supports only project-locked pure FP32 Exact-IG"
            )
        if str(getattr(self.config, "exact_ig_version", "")) != EXACT_IG_VERSION:
            raise RuntimeError("Worker received an incompatible Exact-IG version")
        if str(
            getattr(self.config, "exact_ig_info_gain_type", "")
        ) != INFO_GAIN_TYPE:
            raise RuntimeError("Worker permits only log_prob_diff Exact-IG")
        model_path = str(self.config.model.path)
        self._actor_parameter_dtypes = {
            parameter.dtype
            for parameter in self.actor_module_fsdp.parameters()
            if parameter.is_floating_point()
        }
        if not self._actor_parameter_dtypes:
            raise RuntimeError(
                "Actor snapshot exposes no auditable floating parameters"
            )
        self._reward_parameter_dtype = torch.float32
        self._reward_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self._reward_model.requires_grad_(False)
        self._reward_model.eval()
        if any(
            parameter.requires_grad
            for parameter in self._reward_model.parameters()
        ):
            raise RuntimeError("Exact-IG reward scorer must be permanently frozen")
        if {
            parameter.dtype
            for parameter in self._reward_model.parameters()
            if parameter.is_floating_point()
        } != {torch.float32}:
            raise RuntimeError(
                "Exact-IG Reward Snapshot must load entirely in float32"
            )
        self._reward_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self._reward_tokenizer.pad_token_id is None:
            self._reward_tokenizer.pad_token_id = (
                self._reward_tokenizer.eos_token_id
            )
        (
            self._reward_tokenizer_name,
            self._reward_tokenizer_revision,
        ) = exact_ig_tokenizer_identity(self._reward_tokenizer)
        self._reward_snapshot_step = -1
        self._reward_snapshot_checksum = ""
        self._reward_source_checksum = ""
        self._last_reward_snapshot_sync: dict[str, Any] = {}
        # Populated by global_actor_checksum.  The commit verifier can reuse
        # this rank-local digest instead of repeating an all_gather_object
        # immediately after a large checkpoint write.
        self._last_actor_local_parameter_digest: str | None = None
        self._reward_canary_checks = 0
        self._reward_canary_token_max = 0.0
        self._reward_canary_threshold_exceeded = 0
        self._reward_canary_hard_failures = 0
        self._reward_canary_phi_max = 0.0
        self._reward_canary_ig_max = 0.0
        self._reward_canary_ig_errors: list[float] = []
        self._reward_canary_numeric_ambiguous = 0
        self._reward_canary_non_ambiguous = 0
        self._reward_canary_diagnostics: list[str] = []
        self._last_exact_ig_profile: dict[str, Any] = {}
        return self.runtime_identity()

    def runtime_identity(self) -> dict[str, Any]:
        import os
        import torch

        identity = {
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "local_rank": int(os.environ.get("LOCAL_RANK", self.rank)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device": int(torch.cuda.current_device()),
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "actor_strategy": str(self.config.actor.strategy),
            "actor_reshard_after_forward": bool(
                self.config.actor.fsdp_config.reshard_after_forward
            ),
            "reference_reshard_after_forward": bool(
                self.config.ref.fsdp_config.reshard_after_forward
            ),
            "exact_ig_production_precision_mode": self._exact_ig_precision_mode,
            "exact_ig_version": EXACT_IG_VERSION,
            "exact_ig_info_gain_type": INFO_GAIN_TYPE,
            "exact_ig_canonical_alias_policy": CANONICAL_ALIAS_POLICY,
            "exact_ig_score_mask_policy": SCORE_MASK_POLICY,
            "exact_ig_fast_path_structure": FAST_PATH_STRUCTURE,
            "exact_ig_oracle_canary_rate": float(
                getattr(self.config, "exact_ig_oracle_canary_rate", 0.0)
            ),
            "exact_ig_oracle_canary_fail_closed": bool(
                getattr(
                    self.config,
                    "exact_ig_oracle_canary_fail_closed",
                    False,
                )
            ),
        }
        policy = production_precision_policy(self._exact_ig_precision_mode)
        identity.update(precision_runtime_metadata(self._reward_model, policy))
        identity.update(
            {
                "reward_snapshot_step": int(self._reward_snapshot_step),
                "reward_snapshot_checksum": self._reward_snapshot_checksum,
                "reward_source_checksum": self._reward_source_checksum,
                "reward_scorer_independent": True,
                "reward_scorer_requires_grad": False,
                "actor_snapshot_parameter_dtypes": sorted(
                    str(dtype).removeprefix("torch.")
                    for dtype in self._actor_parameter_dtypes
                ),
                "reward_scorer_device": str(
                    next(self._reward_model.parameters()).device
                ),
                "exact_ig_scoring_logits_mode": str(
                    getattr(
                        self.config,
                        "exact_ig_scoring_logits_mode",
                        OFFICIAL_FULL_LOGITS,
                    )
                ),
                "exact_ig_attention_mask_mode": str(
                    getattr(
                        self.config,
                        "exact_ig_attention_mask_mode",
                        "official_additive",
                    )
                ),
            }
        )
        return identity

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def exact_ig_canary_summary(self) -> dict[str, Any]:
        ordered_errors = sorted(self._reward_canary_ig_errors)
        if ordered_errors:
            p99_index = max(
                0,
                math.ceil(0.99 * len(ordered_errors)) - 1,
            )
            observed_p99 = float(ordered_errors[p99_index])
        else:
            observed_p99 = 0.0
        return {
            "rank": int(self.rank),
            "checks": int(self._reward_canary_checks),
            "threshold_exceeded": int(self._reward_canary_threshold_exceeded),
            "hard_failures": int(self._reward_canary_hard_failures),
            "phi_max_abs_error": float(self._reward_canary_phi_max),
            "ig_max_abs_error": float(self._reward_canary_ig_max),
            "ig_p99_abs_error": observed_p99,
            "numeric_ambiguous_ig_count": int(
                self._reward_canary_numeric_ambiguous
            ),
            "non_ambiguous_ig_count": int(
                self._reward_canary_non_ambiguous
            ),
            "gate_policy": "structural_semantic_with_numeric_telemetry",
            "fail_closed": bool(
                getattr(
                    self.config,
                    "exact_ig_oracle_canary_fail_closed",
                    False,
                )
            ),
            "diagnostics": tuple(self._reward_canary_diagnostics),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def exact_ig_last_profile(self) -> dict[str, Any]:
        return dict(self._last_exact_ig_profile)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def assert_distributed_string_sequence(
        self,
        values: Sequence[str],
    ) -> dict[str, Any]:
        import torch.distributed as dist

        normalized = tuple(str(value) for value in values)
        digest = hashlib.sha256(
            "\n".join(normalized).encode("utf-8")
        ).hexdigest()
        gathered: list[tuple[str, int] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, (digest, len(normalized)))
        if len(set(gathered)) != 1:
            raise RuntimeError(
                f"Distributed selected-ID consensus failed: {gathered}"
            )
        return {
            "rank": int(self.rank),
            "count": len(normalized),
            "sha256": digest,
        }

    @staticmethod
    def _sampled_state_digest(state: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name in sorted(state):
            _update_sampled_state_digest_entry(digest, name, state[name])
        return digest.hexdigest()

    def _streaming_actor_state(self) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Return sharded FSDP2 state without materializing the full model."""
        import torch.distributed as dist
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        self._reshard_all_actor_modules()
        state = get_model_state_dict(
            self.actor_module_fsdp,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=False,
                strict=True,
            ),
        )
        names = tuple(sorted(str(name) for name in state))
        schema_digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        gathered: list[tuple[str, int] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, (schema_digest, len(names)))
        if len(set(gathered)) != 1:
            raise RuntimeError(
                "FSDP2 sharded Actor state schema differs across ranks: "
                f"{gathered}"
            )
        return state, names

    @staticmethod
    def _materialize_streamed_actor_tensor(name: str, value: Any) -> Any:
        """Collectively materialize exactly one Actor state tensor."""
        import torch
        from torch.distributed.tensor import DTensor

        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                "FSDP2 sharded Actor state contains a non-tensor value: "
                f"{name}"
            )
        tensor = value.full_tensor() if isinstance(value, DTensor) else value
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Failed to materialize Actor tensor {name}")
        return tensor

    def _sync_reward_snapshot(self, step: int) -> str:
        import torch
        import torch.distributed as dist

        self._reward_model.to("cpu")
        actor_checksum_before = self.global_actor_checksum()
        sharded_state, state_names = self._streaming_actor_state()
        reward_state = self._reward_model.state_dict()
        if tuple(sorted(reward_state)) != state_names:
            raise RuntimeError(
                "FSDP2 Actor/Reward snapshot state-dict schemas differ"
            )

        supported_dtypes = (
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        )
        source_digest_builder = hashlib.sha256()
        reward_digest_builder = hashlib.sha256()
        maximum_materialized_tensor_bytes = 0
        floating_tensor_count = 0
        for name in state_names:
            value = sharded_state.pop(name)
            materialized = self._materialize_streamed_actor_tensor(name, value)
            maximum_materialized_tensor_bytes = max(
                maximum_materialized_tensor_bytes,
                int(materialized.numel() * materialized.element_size()),
            )
            received_tensor = materialized.detach().contiguous().cpu()
            if received_tensor.dtype not in supported_dtypes:
                raise RuntimeError(
                    "Unsupported FSDP2 snapshot dtype "
                    f"{received_tensor.dtype} for {name}"
                )
            target = reward_state[name]
            converted = (
                received_tensor.to(self._reward_parameter_dtype)
                if received_tensor.is_floating_point()
                else received_tensor
            )
            if tuple(converted.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Reward Snapshot shape mismatch for {name}: "
                    f"actor={tuple(converted.shape)} reward={tuple(target.shape)}"
                )
            if converted.dtype != target.dtype:
                raise RuntimeError(
                    f"Reward Snapshot dtype mismatch for {name}: "
                    f"converted={converted.dtype} reward={target.dtype}"
                )
            _update_sampled_state_digest_entry(
                source_digest_builder,
                name,
                received_tensor,
            )
            _update_sampled_state_digest_entry(
                reward_digest_builder,
                name,
                converted,
            )
            with torch.no_grad():
                target.copy_(converted)
            floating_tensor_count += int(converted.is_floating_point())
            del value, materialized, received_tensor, converted, target

        if sharded_state:
            raise RuntimeError("FSDP2 streaming snapshot left unconsumed entries")
        del sharded_state, reward_state
        source_digest = source_digest_builder.hexdigest()
        expected_reward_digest = reward_digest_builder.hexdigest()
        reward_digest = self._sampled_state_digest(
            self._reward_model.state_dict()
        )
        if expected_reward_digest != reward_digest:
            raise RuntimeError("Synchronized reward snapshot checksum differs")
        wire_device = torch.device("cuda", torch.cuda.current_device())
        digest_tensor = torch.tensor(
            list(bytes.fromhex(reward_digest)),
            dtype=torch.uint8,
            device=wire_device,
        )
        gathered_digest = [torch.empty_like(digest_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_digest, digest_tensor)
        if any(not torch.equal(digest_tensor, item) for item in gathered_digest):
            raise RuntimeError("Reward snapshot differs across ranks")
        actor_checksum_after = self.global_actor_checksum()
        if actor_checksum_after != actor_checksum_before:
            raise RuntimeError("Reward Snapshot synchronization changed the Actor")
        self._reward_model.eval()
        self._reward_snapshot_step = int(step)
        self._reward_snapshot_checksum = reward_digest
        self._reward_source_checksum = source_digest
        self._last_reward_snapshot_sync = {
            "rank": int(self.rank),
            "successful_update_step": int(step),
            "sync_mode": "streaming_sharded_dtensor_per_parameter",
            "state_tensor_count": len(state_names),
            "floating_tensor_count": int(floating_tensor_count),
            "maximum_materialized_tensor_bytes": int(
                maximum_materialized_tensor_bytes
            ),
            "actor_checksum_before": actor_checksum_before,
            "actor_checksum_after": actor_checksum_after,
            "source_sampled_digest": source_digest,
            "reward_snapshot_checksum": reward_digest,
        }
        torch.cuda.empty_cache()
        return reward_digest

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_snapshot(self, successful_update_step: int) -> dict[str, int]:
        step = int(successful_update_step)
        self._snapshot_step = step
        self._zero_grad_calls = 0
        self._backward_calls = 0
        self._optimizer_steps = 0
        self._scheduler_steps = 0
        self._attempt_optimizer_committed = False
        # A verified resume may preload the independent FP32 Reward Snapshot
        # from the immutable HF artifact exported at that exact checkpoint.
        # Avoid immediately rematerializing a full FSDP2 state dict after DCP
        # restore; that Torch 2.8 boundary can terminate a Ray worker natively.
        if self._reward_snapshot_step != step:
            self._sync_reward_snapshot(step)
        return {
            "actor_snapshot_step": step,
            "rollout_snapshot_step": step,
            "old_policy_snapshot_step": step,
            "reward_policy_snapshot_step": step,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def validate_reward_snapshot_sync_cycles(
        self,
        successful_update_steps: Sequence[int],
    ) -> dict[str, Any]:
        """Exercise consecutive production snapshot syncs without an update."""
        import torch

        steps = tuple(int(value) for value in successful_update_steps)
        if len(steps) < 2 or any(value < 0 for value in steps):
            raise ValueError("Snapshot lifecycle validation requires two valid steps")
        if self._restored_checkpoint_step != steps[0]:
            raise RuntimeError(
                "Snapshot lifecycle validation must start at the restored step"
            )
        counts_before = self.strict_attempt_counts()
        optimizer_before = self.local_optimizer_scheduler_digest()
        actor_before = self.global_actor_checksum()
        cycles: list[dict[str, Any]] = []
        for step in steps:
            self._sync_reward_snapshot(step)
            cycles.append(dict(self._last_reward_snapshot_sync))
        actor_after = self.global_actor_checksum()
        optimizer_after = self.local_optimizer_scheduler_digest()
        counts_after = self.strict_attempt_counts()
        if actor_after != actor_before:
            raise RuntimeError("Snapshot lifecycle validation changed Actor state")
        if optimizer_after != optimizer_before:
            raise RuntimeError(
                "Snapshot lifecycle validation changed optimizer/scheduler state"
            )
        if counts_after != counts_before:
            raise RuntimeError("Snapshot lifecycle validation changed step counters")
        if {
            parameter.dtype
            for parameter in self._reward_model.parameters()
            if parameter.is_floating_point()
        } != {torch.float32}:
            raise RuntimeError("Snapshot lifecycle validation lost FP32 Reward state")
        return {
            "rank": int(self.rank),
            "steps": list(steps),
            "cycles": cycles,
            "actor_checksum_before": actor_before,
            "actor_checksum_after": actor_after,
            "optimizer_scheduler_digest_before": optimizer_before,
            "optimizer_scheduler_digest_after": optimizer_after,
            "strict_attempt_counts_before": counts_before,
            "strict_attempt_counts_after": counts_after,
            "reward_parameter_dtype": "float32",
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_restored_reward_snapshot_from_hf(
        self,
        model_root: str,
        successful_update_step: int,
        expected_actor_checksum: str,
    ) -> dict[str, Any]:
        """Preload Exact-IG's FP32 Reward Snapshot at a restored boundary.

        The HF artifact is a derived, immutable representation of the same
        committed Actor checkpoint. Loading it directly avoids a second FSDP2
        full-state materialization in the freshly restored worker process.
        """
        import torch
        root = Path(model_root).resolve()
        metadata_path = root / "training_metadata.json"
        model_path = root / "model.safetensors"
        completed_path = root / "COMPLETED"
        if not (metadata_path.is_file() and model_path.is_file() and completed_path.is_file()):
            raise RuntimeError("Restored Reward Snapshot artifact is incomplete")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        step = int(successful_update_step)
        if int(metadata["successful_update_step"]) != step:
            raise RuntimeError("Restored Reward Snapshot update differs")
        if str(metadata["actor_checksum"]) != str(expected_actor_checksum):
            raise RuntimeError("Restored Reward Snapshot Actor checksum differs")
        if self._restored_checkpoint_step != step:
            raise RuntimeError("Reward Snapshot preload is not at the restored step")
        if self._restored_checkpoint_source is None:
            raise RuntimeError("Reward Snapshot preload requires a restored checkpoint")

        actor_before = self.global_actor_checksum()
        if actor_before != str(expected_actor_checksum):
            raise RuntimeError("Live restored Actor differs before Reward preload")
        tied_groups_validated = _load_safetensors_state_dict_with_tied_key_validation(
            self._reward_model,
            model_path,
        )
        self._reward_model.requires_grad_(False)
        self._reward_model.eval()
        reward_dtypes = {
            parameter.dtype
            for parameter in self._reward_model.parameters()
            if parameter.is_floating_point()
        }
        if reward_dtypes != {torch.float32}:
            raise RuntimeError("Restored Reward Snapshot is not pure FP32")
        if any(parameter.requires_grad for parameter in self._reward_model.parameters()):
            raise RuntimeError("Restored Reward Snapshot is not frozen")
        reward_digest = self._sampled_state_digest(self._reward_model.state_dict())
        actor_after = self.global_actor_checksum()
        if actor_after != actor_before:
            raise RuntimeError("Reward Snapshot preload changed the restored Actor")
        self._reward_snapshot_step = step
        self._reward_snapshot_checksum = reward_digest
        self._reward_source_checksum = str(expected_actor_checksum)
        return {
            "rank": int(self.rank),
            "successful_update_step": step,
            "actor_checksum_before": actor_before,
            "actor_checksum_after": actor_after,
            "reward_snapshot_checksum": reward_digest,
            "reward_parameter_dtype": "float32",
            "tied_state_groups_validated": int(tied_groups_validated),
            "reward_snapshot_source": str(model_path),
            "restored_checkpoint_source": self._restored_checkpoint_source,
        }

    def _local_parameter_digest(self) -> str:
        import torch

        digest = hashlib.sha256()
        for name, parameter in self.actor_module_fsdp.named_parameters():
            local = parameter.detach()
            if hasattr(local, "to_local"):
                local = local.to_local()
            flat = local.reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(local.shape)).encode("ascii"))
            digest.update(str(local.dtype).encode("ascii"))
            if flat.numel():
                indices = _uniform_sample_indices(
                    flat.numel(),
                    maximum_samples=32,
                    device=flat.device,
                )
                sample = flat.index_select(0, indices).float().cpu().numpy()
                digest.update(sample.tobytes())
        return digest.hexdigest()

    def _reshard_all_actor_modules(self) -> int:
        """Put every composable-FSDP actor module in its canonical sharded state."""
        count = 0
        # FSDP2's FSDPModule.reshard() is explicitly non-recursive. Children
        # must be resharded before the root so checksums do not depend on the
        # preceding forward/backward materialization state.
        for module in reversed(tuple(self.actor_module_fsdp.modules())):
            reshard = getattr(module, "reshard", None)
            if callable(reshard):
                reshard()
                count += 1
        if count < 1:
            raise RuntimeError("Actor has no FSDP2 modules to reshard")
        return count

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def global_actor_checksum(self) -> str:
        import torch.distributed as dist

        self._reshard_all_actor_modules()
        local = self._local_parameter_digest()
        self._last_actor_local_parameter_digest = local
        gathered: list[str | None] = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        digest = hashlib.sha256()
        for rank, value in enumerate(gathered):
            digest.update(f"{rank}:{value}\n".encode("ascii"))
        return digest.hexdigest()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def last_actor_local_parameter_digest(self) -> str:
        """Return the local digest captured by the last global checksum."""

        if self._last_actor_local_parameter_digest is None:
            raise RuntimeError("No actor checksum has captured a local digest")
        return self._last_actor_local_parameter_digest

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def local_actor_parameter_digest(self) -> str:
        """Reshard and digest this rank without a cross-rank object gather."""

        self._reshard_all_actor_modules()
        digest = self._local_parameter_digest()
        self._last_actor_local_parameter_digest = digest
        return digest

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def local_optimizer_scheduler_digest(self) -> str:
        digest = hashlib.sha256()
        _update_state_digest(digest, self.actor_optimizer.state_dict())
        _update_state_digest(digest, self.actor_lr_scheduler.state_dict())
        return digest.hexdigest()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def score_exact_ig_tasks(self, tasks: Sequence[Any]) -> list[dict[str, Any]]:
        import torch
        import torch.distributed as dist

        if self._exact_ig_precision_mode != PRODUCTION_PRECISION_MODE:
            raise RuntimeError("Only project-locked pure FP32 Exact-IG may run")
        if self._reward_snapshot_step != self._snapshot_step:
            raise RuntimeError("Reward scorer is not at the rollout snapshot")
        policy = production_precision_policy(self._exact_ig_precision_mode)
        actual_parameter_dtype = str(
            next(self._reward_model.parameters()).dtype
        ).removeprefix("torch.")
        if actual_parameter_dtype != "float32":
            raise RuntimeError(
                "Exact-IG Reward Snapshot is not float32: "
                f"actual={actual_parameter_dtype}"
            )
        count = torch.tensor(
            [len(tasks)],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        minimum = count.clone()
        maximum = count.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(minimum.item()) != int(maximum.item()):
            raise RuntimeError(
                "Every FSDP rank must execute the same number of Exact-IG forwards"
            )
        unique_tasks: dict[str, Any] = {}
        for task in tasks:
            trajectory_id = str(task.trajectory_id)
            previous = unique_tasks.get(trajectory_id)
            if previous is not None:
                if previous.target_bundle_hash != task.target_bundle_hash:
                    raise RuntimeError(
                        "A padded Exact-IG duplicate changed its canonical target"
                    )
                continue
            unique_tasks[trajectory_id] = task
        scorer = VectorizedExactIGScorer.for_production_mode(
            self._exact_ig_precision_mode,
            padding_token_id=int(self._reward_tokenizer.pad_token_id),
            tokenizer=self._reward_tokenizer,
            scoring_logits_mode=str(
                getattr(
                    self.config,
                    "exact_ig_scoring_logits_mode",
                    OFFICIAL_FULL_LOGITS,
                )
            ),
            attention_mask_mode=str(
                getattr(
                    self.config,
                    "exact_ig_attention_mask_mode",
                    "official_additive",
                )
            ),
        )
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        token_count = sum(
            int(
                getattr(
                    task,
                    "projected_fast_packed_length",
                    task.input_ids.size,
                )
            )
            for task in unique_tasks.values()
        )
        attention_cost = sum(
            int(
                getattr(
                    task,
                    "projected_fast_packed_length",
                    task.input_ids.size,
                )
            )
            ** 2
            for task in unique_tasks.values()
        )
        self._reward_model.to(device)
        self._reward_model.eval()
        output: list[dict[str, Any]] = []
        try:
            scored_by_trajectory = scorer.score_many(
                self._reward_model,
                tuple(unique_tasks.values()),
                device,
                max_records_per_forward=int(
                    getattr(
                        self.config,
                        "exact_ig_max_records_per_forward",
                        1,
                    )
                ),
                max_attention_cost_per_batch=getattr(
                    self.config,
                    "exact_ig_max_attention_cost_per_batch",
                    None,
                ),
                max_extended_tokens_per_batch=getattr(
                    self.config,
                    "exact_ig_max_extended_tokens_per_batch",
                    None,
                ),
                max_full_logits_bytes=getattr(
                    self.config,
                    "exact_ig_max_full_logits_bytes",
                    None,
                ),
                max_selected_logits_bytes=getattr(
                    self.config,
                    "exact_ig_max_selected_logits_bytes",
                    None,
                ),
            )
            for task in unique_tasks.values():
                scored = scored_by_trajectory[str(task.trajectory_id)]
                canary_payload = None
                canary_rate = float(
                    getattr(self.config, "exact_ig_oracle_canary_rate", 0.0)
                )
                sample_value = int.from_bytes(
                    hashlib.sha256(
                        str(task.trajectory_id).encode("utf-8")
                    ).digest()[:8],
                    byteorder="big",
                ) / float(2**64)
                if canary_rate > 0 and sample_value < canary_rate:
                    oracle = sequential_teacher_forced_oracle(
                        model=self._reward_model,
                        tokenizer=self._reward_tokenizer,
                        full_trajectory_input_ids=task.input_ids[
                            : task.original_token_count
                        ],
                        original_attention_mask=task.original_attention_mask,
                        original_position_ids=task.original_position_ids,
                        prefix_end_positions=task.prefix_end_positions,
                        canonical_answer=task.canonical_answer,
                        encoded_target=task.canonical_target,
                        device=device,
                        precision_policy=policy,
                    )
                    token_error = max(
                        (
                            abs(float(fast_value) - float(reference_value))
                            for fast_row, reference_row in zip(
                                scored.answer_token_log_probs_by_prefix,
                                oracle.answer_token_log_probs_by_prefix,
                                strict=True,
                            )
                            for fast_value, reference_value in zip(
                                fast_row,
                                reference_row,
                                strict=True,
                            )
                        ),
                        default=0.0,
                    )
                    phi_error = max(
                        (
                            abs(float(fast) - float(reference))
                            for fast, reference in zip(
                                scored.score_by_prefix,
                                oracle.score_by_prefix,
                                strict=True,
                            )
                        ),
                        default=0.0,
                    )
                    ig_error = max(
                        (
                            abs(float(left) - float(right))
                            for left, right in zip(
                                scored.immediate_ig,
                                oracle.immediate_ig,
                                strict=True,
                            )
                        ),
                        default=0.0,
                    )
                    self._reward_canary_checks += 1
                    self._reward_canary_token_max = max(
                        self._reward_canary_token_max,
                        float(token_error),
                    )
                    self._reward_canary_phi_max = max(
                        self._reward_canary_phi_max,
                        float(phi_error),
                    )
                    self._reward_canary_ig_max = max(
                        self._reward_canary_ig_max,
                        float(ig_error),
                    )
                    parity_rtol = float(
                        getattr(self.config, "exact_ig_parity_rtol", 1.0e-5)
                    )
                    parity_atol = float(
                        getattr(self.config, "exact_ig_parity_atol", 2.0e-5)
                    )
                    token_allclose = all(
                        torch.allclose(
                            torch.tensor(tuple(fast_row), dtype=torch.float32),
                            torch.tensor(
                                tuple(reference_row),
                                dtype=torch.float32,
                            ),
                            rtol=parity_rtol,
                            atol=parity_atol,
                        )
                        for fast_row, reference_row in zip(
                            scored.answer_token_log_probs_by_prefix,
                            oracle.answer_token_log_probs_by_prefix,
                            strict=True,
                        )
                    )
                    phi_allclose = all(
                        torch.isclose(
                            torch.tensor(float(fast)),
                            torch.tensor(float(reference)),
                            rtol=parity_rtol,
                            atol=parity_atol,
                        ).item()
                        for fast, reference in zip(
                            scored.score_by_prefix,
                            oracle.score_by_prefix,
                            strict=True,
                        )
                    )
                    ig_allclose = all(
                        torch.isclose(
                            torch.tensor(float(fast)),
                            torch.tensor(float(reference)),
                            rtol=parity_rtol,
                            atol=parity_atol,
                        ).item()
                        for fast, reference in zip(
                            scored.immediate_ig,
                            oracle.immediate_ig,
                            strict=True,
                        )
                    )
                    target_coverage = (
                        len(scored.score_by_prefix) == task.prefix_count
                        and len(oracle.score_by_prefix) == task.prefix_count
                        and scored.score_token_ids_by_prefix
                        == oracle.score_token_ids_by_prefix
                        and all(
                            token_ids
                            == task.canonical_target.answer_token_ids
                            for token_ids in scored.score_token_ids_by_prefix
                        )
                        and oracle.scored_answer_token_count
                        == (
                            task.prefix_count
                            * task.canonical_target.answer_token_count
                        )
                        and scored.target_score_span_hash
                        == oracle.score_span_hash
                    )
                    canonical_answer_agreement = (
                        scored.canonical_answer == oracle.canonical_answer
                        == task.canonical_answer
                        and scored.canonical_answer_sha256
                        == oracle.canonical_answer_sha256
                        == task.canonical_answer_hash
                    )
                    numeric_ambiguity_epsilon = float(
                        getattr(
                            self.config,
                            "exact_ig_numeric_ambiguity_epsilon",
                            0.0,
                        )
                    )
                    ambiguity_flags = tuple(
                        abs(float(oracle_value)) <= numeric_ambiguity_epsilon
                        for oracle_value in oracle.immediate_ig
                    )
                    sign_agreement = all(
                        ambiguous
                        or (
                            float(fast_value) > 0
                            and float(oracle_value) > 0
                        )
                        or (
                            float(fast_value) < 0
                            and float(oracle_value) < 0
                        )
                        or (
                            float(fast_value) == 0
                            and float(oracle_value) == 0
                        )
                        for fast_value, oracle_value, ambiguous in zip(
                            scored.immediate_ig,
                            oracle.immediate_ig,
                            ambiguity_flags,
                            strict=True,
                        )
                    )
                    self._reward_canary_numeric_ambiguous += sum(
                        bool(value) for value in ambiguity_flags
                    )
                    self._reward_canary_non_ambiguous += sum(
                        not bool(value) for value in ambiguity_flags
                    )
                    fast_turn_ranking = tuple(
                        sorted(
                            range(len(scored.immediate_ig)),
                            key=lambda index: (
                                -float(scored.immediate_ig[index]),
                                index,
                            ),
                        )
                    )
                    oracle_turn_ranking = tuple(
                        sorted(
                            range(len(oracle.immediate_ig)),
                            key=lambda index: (
                                -float(oracle.immediate_ig[index]),
                                index,
                            ),
                        )
                    )
                    turn_ranking_agreement = (
                        fast_turn_ranking == oracle_turn_ranking
                    )
                    finite = all(
                        math.isfinite(float(value))
                        for value in (
                            *scored.score_by_prefix,
                            *scored.immediate_ig,
                            *oracle.score_by_prefix,
                            *oracle.immediate_ig,
                        )
                    )
                    telemetry_token_error = float(
                        getattr(
                            self.config,
                            "exact_ig_maximum_token_log_prob_abs_diff",
                            2.0e-5,
                        )
                    )
                    telemetry_phi_error = float(
                        getattr(
                            self.config,
                            "exact_ig_maximum_phi_abs_diff",
                            2.0e-5,
                        )
                    )
                    telemetry_ig_error = float(
                        getattr(
                            self.config,
                            "exact_ig_maximum_ig_abs_diff",
                            2.0e-5,
                        )
                    )
                    maximum_phi_safety_error = float(
                        getattr(
                            self.config,
                            "exact_ig_maximum_phi_safety_abs_diff",
                            1.0e-3,
                        )
                    )
                    maximum_ig_safety_error = float(
                        getattr(
                            self.config,
                            "exact_ig_maximum_ig_safety_abs_diff",
                            1.0e-3,
                        )
                    )
                    maximum_telescoping_error = float(
                        getattr(
                            self.config,
                            "exact_ig_maximum_telescoping_error",
                            1.0e-10,
                        )
                    )
                    observed_telescoping_error = max(
                        abs(float(scored.telescoping_error)),
                        abs(float(oracle.telescoping_error)),
                    )
                    self._reward_canary_ig_errors.append(float(ig_error))
                    ordered_ig_errors = sorted(self._reward_canary_ig_errors)
                    observed_p99_ig_error = float(
                        ordered_ig_errors[
                            max(
                                0,
                                math.ceil(0.99 * len(ordered_ig_errors)) - 1,
                            )
                        ]
                    )
                    calibration_p99_ig_error = float(
                        getattr(
                            self.config,
                            "exact_ig_calibration_p99_ig_abs_diff",
                            0.0,
                        )
                    )
                    minimum_p99_samples = int(
                        getattr(
                            self.config,
                            "exact_ig_minimum_canary_samples_for_p99",
                            8,
                        )
                    )
                    enforce_p99_drift = (
                        calibration_p99_ig_error > 0
                        and len(self._reward_canary_ig_errors)
                        >= minimum_p99_samples
                    )
                    threshold_exceeded, hard_failure = (
                        _classify_exact_ig_canary(
                            token_allclose=token_allclose,
                            phi_allclose=phi_allclose,
                            ig_allclose=ig_allclose,
                            finite=finite,
                            target_coverage=target_coverage,
                            canonical_answer_agreement=(
                                canonical_answer_agreement
                            ),
                            non_ambiguous_sign_agreement=sign_agreement,
                            turn_ranking_agreement=(
                                turn_ranking_agreement
                            ),
                            token_error=token_error,
                            phi_error=phi_error,
                            ig_error=ig_error,
                            telescoping_error=(
                                observed_telescoping_error
                            ),
                            telemetry_token_error=telemetry_token_error,
                            telemetry_phi_error=telemetry_phi_error,
                            telemetry_ig_error=telemetry_ig_error,
                            maximum_phi_safety_error=(
                                maximum_phi_safety_error
                            ),
                            maximum_ig_safety_error=(
                                maximum_ig_safety_error
                            ),
                            maximum_telescoping_error=(
                                maximum_telescoping_error
                            ),
                            observed_p99_ig_error=observed_p99_ig_error,
                            calibration_p99_ig_error=(
                                calibration_p99_ig_error
                            ),
                            enforce_p99_drift=enforce_p99_drift,
                        )
                    )
                    diagnostic = None
                    if threshold_exceeded:
                        self._reward_canary_threshold_exceeded += 1
                        diagnostic = _write_exact_ig_canary_failure(
                            task=task,
                            scored=scored,
                            oracle=oracle,
                            token_error=token_error,
                            phi_error=phi_error,
                            ig_error=ig_error,
                        )
                        self._reward_canary_diagnostics.append(str(diagnostic))
                    if hard_failure:
                        self._reward_canary_hard_failures += 1
                        if diagnostic is None:
                            diagnostic = _write_exact_ig_canary_failure(
                                task=task,
                                scored=scored,
                                oracle=oracle,
                                token_error=token_error,
                                phi_error=phi_error,
                                ig_error=ig_error,
                            )
                            self._reward_canary_diagnostics.append(
                                str(diagnostic)
                            )
                        if bool(
                            getattr(
                                self.config,
                                "exact_ig_oracle_canary_fail_closed",
                                False,
                            )
                        ):
                            raise RuntimeError(
                                "Exact-IG Oracle canary failed: "
                                f"trajectory_id={task.trajectory_id} "
                                f"prefix_count={task.prefix_count} "
                                f"extended_tokens={task.input_ids.size} "
                                f"token_error={token_error} "
                                f"phi_error={phi_error} ig_error={ig_error} "
                                f"diagnostic={diagnostic}"
                            )
                    canary_payload = {
                        "token_max_abs_error": token_error,
                        "phi_max_abs_error": phi_error,
                        "ig_max_abs_error": ig_error,
                        "telescoping_max_abs_error": (
                            observed_telescoping_error
                        ),
                        "parity_rtol": parity_rtol,
                        "parity_atol": parity_atol,
                        "telemetry_token_abs_error_limit": telemetry_token_error,
                        "telemetry_phi_abs_error_limit": telemetry_phi_error,
                        "telemetry_ig_abs_error_limit": telemetry_ig_error,
                        "maximum_phi_safety_abs_diff": (
                            maximum_phi_safety_error
                        ),
                        "maximum_ig_safety_abs_diff": (
                            maximum_ig_safety_error
                        ),
                        "numeric_ambiguity_epsilon": (
                            numeric_ambiguity_epsilon
                        ),
                        "numeric_ambiguous_flags": ambiguity_flags,
                        "observed_p99_ig_abs_error": observed_p99_ig_error,
                        "calibration_p99_ig_abs_error": (
                            calibration_p99_ig_error
                        ),
                        "p99_drift_enforced": enforce_p99_drift,
                        "numeric_difference_is_telemetry": True,
                        "token_allclose": token_allclose,
                        "phi_allclose": phi_allclose,
                        "ig_allclose": ig_allclose,
                        "target_coverage": target_coverage,
                        "canonical_answer_agreement": (
                            canonical_answer_agreement
                        ),
                        "sign_agreement": sign_agreement,
                        "turn_ranking_agreement": turn_ranking_agreement,
                        "fast_score_by_prefix": tuple(
                            float(value) for value in scored.score_by_prefix
                        ),
                        "oracle_score_by_prefix": tuple(
                            float(value) for value in oracle.score_by_prefix
                        ),
                        "fast_immediate_ig": tuple(
                            float(value) for value in scored.immediate_ig
                        ),
                        "oracle_immediate_ig": tuple(
                            float(value) for value in oracle.immediate_ig
                        ),
                        "finite": finite,
                        "hard_failure": hard_failure,
                        "status": (
                            "HARD_FAIL"
                            if hard_failure
                            else (
                                "SOFT_WARNING"
                                if threshold_exceeded
                                else "PASS"
                            )
                        ),
                        "diagnostic": (
                            str(diagnostic) if diagnostic is not None else None
                        ),
                    }
                output.append(
                    {
                        "prompt_global_id": task.prompt_global_id,
                        "trajectory_id": task.trajectory_id,
                        "score_by_prefix": scored.score_by_prefix,
                        "immediate_ig": scored.immediate_ig,
                        "telescoping_error": scored.telescoping_error,
                        "exact_ig_version": scored.exact_ig_version,
                        "scaffold_sha256": scored.scaffold_sha256,
                        "canonical_alias_policy": (
                            scored.canonical_alias_policy
                        ),
                        "canonical_answer_sha256": (
                            scored.canonical_answer_sha256
                        ),
                        "target_token_ids_hash": (
                            scored.target_token_ids_hash
                        ),
                        "score_span_hash": scored.score_span_hash,
                        "target_score_span_hash": (
                            scored.target_score_span_hash
                        ),
                        "score_mask_policy": scored.score_mask_policy,
                        "info_gain_type": scored.info_gain_type,
                        "fast_path_structure": scored.fast_path_structure,
                        "target_tokenization_policy": (
                            scored.target_tokenization_policy
                        ),
                        "official_igpo_commit_sha": (
                            scored.official_igpo_commit_sha
                        ),
                        "mask_builder_version": scored.mask_builder_version,
                        "position_builder_version": (
                            scored.position_builder_version
                        ),
                        "scaffold_text": ANSWER_SCAFFOLD_TEXT,
                        "tokenizer_name_or_path": self._reward_tokenizer_name,
                        "tokenizer_revision": self._reward_tokenizer_revision,
                        "answer_score_token_count": (
                            task.canonical_target.answer_token_count
                        ),
                        "answer_char_start": int(
                            task.canonical_target.answer_char_start
                        ),
                        "answer_char_end": int(
                            task.canonical_target.answer_char_end
                        ),
                        "answer_token_start": int(
                            task.canonical_target.answer_token_start
                        ),
                        "answer_token_end": int(
                            task.canonical_target.answer_token_end
                        ),
                        "left_boundary_crossing": bool(
                            task.canonical_target.left_boundary_crossing
                        ),
                        "right_boundary_crossing": bool(
                            task.canonical_target.right_boundary_crossing
                        ),
                        "boundary_crossing_any": bool(
                            task.canonical_target.boundary_crossing_any
                        ),
                        "full_target_token_ids_sha256": (
                            task.canonical_target.full_target_token_ids_sha256
                        ),
                        "answer_span_token_ids_sha256": (
                            task.canonical_target.answer_span_token_ids_sha256
                        ),
                        "exact_ig_execution_path": scored.execution_path,
                        "scoring_logits_mode": scored.scoring_logits_mode,
                        "runtime_precision_metadata": dict(
                            scored.runtime_metadata
                        ),
                        "reward_snapshot_step": self._reward_snapshot_step,
                        "reward_snapshot_checksum": (
                            self._reward_snapshot_checksum
                        ),
                        "oracle_canary": canary_payload,
                    }
                )
        finally:
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            self._last_exact_ig_profile = {
                "rank": int(self.rank),
                "record_count": len(unique_tasks),
                "padded_dispatch_record_count": len(tasks),
                "extended_token_count": int(token_count),
                "attention_cost": int(attention_cost),
                "micro_batches": [
                    profile.as_dict()
                    for profile in scorer.last_microbatch_profiles
                ],
                "seconds": float(seconds),
                "records_per_second": (
                    len(unique_tasks) / seconds if seconds > 0 else 0.0
                ),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
            self._reward_model.to("cpu")
            torch.cuda.empty_cache()
        return output

    def _forward_action_hidden(
        self,
        model: Any,
        model_inputs: Mapping[str, Any],
        *,
        inference: bool,
    ) -> Any:
        import torch

        head = getattr(model, "lm_head", None)
        if head is None:
            raise TypeError("Expected a causal LM with an lm_head")
        inputs = dict(model_inputs)
        inputs["use_cache"] = False
        original_forward = head.forward

        def return_hidden_states(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return hidden_states

        head.forward = return_hidden_states
        try:
            if inference:
                with torch.no_grad():
                    hidden = model(**inputs).logits.detach()
            else:
                hidden = model(**inputs).logits
        finally:
            head.forward = original_forward
        hidden_size = int(getattr(model.config, "hidden_size"))
        if hidden.shape[-1] != hidden_size:
            raise RuntimeError(
                "FSDP root hidden-state projection bypass returned an invalid "
                f"last dimension: actual={hidden.shape[-1]}, expected={hidden_size}"
            )
        return hidden

    def _action_logprobs(
        self,
        hidden: Any,
        input_ids: Any,
        policy_mask: Any,
        *,
        requires_grad: bool,
    ) -> Any:
        import torch

        head = getattr(self.actor_module_fsdp, "lm_head", None)
        if head is None:
            raise TypeError("Expected a causal LM with an lm_head")
        state_mask = _causal_state_mask(policy_mask)
        action_hidden = hidden[state_mask]
        targets = input_ids[policy_mask.bool()]
        logits = head(action_hidden)
        logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)
        if requires_grad and not logprobs.requires_grad:
            raise RuntimeError("Current action logprobs lost gradients")
        return logprobs

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def materialize_old_logprobs(self, microbatch: Mapping[str, Any]) -> Any:
        import torch

        device = torch.device("cuda", torch.cuda.current_device())
        input_ids = microbatch["input_ids"].to(device)
        attention_mask = microbatch["attention_mask"].to(device)
        position_ids = microbatch["position_ids"].to(device)
        policy_mask = microbatch["policy_mask"].to(device).bool()
        try:
            with torch.no_grad():
                hidden = self._forward_action_hidden(
                    self.actor_module_fsdp,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                    },
                    inference=True,
                )
                values = self._action_logprobs(
                    hidden,
                    input_ids,
                    policy_mask,
                    requires_grad=False,
                ).detach()
        finally:
            self._reshard_all_actor_modules()
        aligned = torch.zeros(
            policy_mask.shape,
            dtype=values.dtype,
            device=device,
        )
        aligned[policy_mask] = values
        return aligned.cpu()

    def _trainable_actor_parameters(self) -> tuple[Any, ...]:
        parameters = tuple(
            parameter
            for parameter in self.actor_module_fsdp.parameters()
            if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError("Actor exposes no trainable calibration parameters")
        return parameters

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_gate_gradient_profile(self, batch_id: str) -> dict[str, Any]:
        """Open one fresh-U0, no-update gradient calibration transaction."""

        if int(self._snapshot_step) != 0:
            raise RuntimeError("Gate calibration is restricted to fresh U0")
        if any(
            (
                self._zero_grad_calls,
                self._backward_calls,
                self._optimizer_steps,
                self._scheduler_steps,
            )
        ) or self._attempt_optimizer_committed:
            raise RuntimeError("Gate calibration cannot share a learner transaction")
        if self._gate_calibration_batch_id is not None:
            raise RuntimeError("A gate calibration profile is already active")
        normalized_batch_id = str(batch_id)
        if not normalized_batch_id:
            raise ValueError("Calibration batch_id cannot be empty")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._reshard_all_actor_modules()
        parameters = self._trainable_actor_parameters()
        self._gate_calibration_batch_id = normalized_batch_id
        self._gate_calibration_channel = None
        self._gate_calibration_gradients = {}
        self._gate_calibration_event_counts = {}
        self._gate_calibration_parameter_hash = parameter_shard_sha256(parameters)
        self._gate_calibration_optimizer_digest = (
            self.local_optimizer_scheduler_digest()
        )
        return {
            "rank": int(self.rank),
            "batch_id": normalized_batch_id,
            "parameter_shard_sha256_before": self._gate_calibration_parameter_hash,
            "optimizer_scheduler_sha256_before": (
                self._gate_calibration_optimizer_digest
            ),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_gate_gradient_channel(self, batch_id: str, channel: str) -> None:
        if str(batch_id) != self._gate_calibration_batch_id:
            raise RuntimeError("Gate calibration batch identity mismatch")
        if self._gate_calibration_channel is not None:
            raise RuntimeError("A gate calibration channel is already active")
        if channel not in {"main", "decision", "query"}:
            raise ValueError(f"Unsupported calibration channel: {channel}")
        if channel in self._gate_calibration_gradients:
            raise RuntimeError(f"Calibration channel {channel} already completed")
        if any(
            (
                self._zero_grad_calls,
                self._backward_calls,
                self._optimizer_steps,
                self._scheduler_steps,
            )
        ):
            raise RuntimeError("Calibration modified strict attempt counters")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._gate_calibration_channel = str(channel)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_gate_gradient_channel(
        self,
        batch_id: str,
        channel: str,
    ) -> dict[str, Any]:
        import torch

        if str(batch_id) != self._gate_calibration_batch_id:
            raise RuntimeError("Gate calibration batch identity mismatch")
        if str(channel) != self._gate_calibration_channel:
            raise RuntimeError("Gate calibration channel identity mismatch")
        gradients: list[Any] = []
        local_squared_norm = 0.0
        for parameter in self._trainable_actor_parameters():
            parameter_local = parameter.detach()
            if hasattr(parameter_local, "to_local"):
                parameter_local = parameter_local.to_local()
            gradient = parameter.grad
            if gradient is None:
                local_gradient = torch.zeros_like(parameter_local)
            else:
                local_gradient = gradient.detach()
                if hasattr(local_gradient, "to_local"):
                    local_gradient = local_gradient.to_local()
                if local_gradient.shape != parameter_local.shape:
                    raise RuntimeError("Calibration gradient shard shape mismatch")
            cpu_gradient = local_gradient.float().contiguous().cpu().clone()
            gradients.append(cpu_gradient)
            flat_gradient = cpu_gradient.reshape(-1)
            for start in range(0, flat_gradient.numel(), 1_048_576):
                chunk = flat_gradient[start : start + 1_048_576].double()
                local_squared_norm += float(torch.dot(chunk, chunk).item())
        self._gate_calibration_gradients[str(channel)] = gradients
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._gate_calibration_channel = None
        if any(
            parameter.grad is not None
            for parameter in self._trainable_actor_parameters()
        ):
            raise RuntimeError("Calibration channel gradients were not cleared")
        return {
            "rank": int(self.rank),
            "batch_id": str(batch_id),
            "channel": str(channel),
            "local_gradient_norm": math.sqrt(max(local_squared_norm, 0.0)),
            "parameter_shard_count": len(gradients),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_gate_gradient_profile(
        self,
        batch_id: str,
        decision_gate_event_count: int,
        query_gate_event_count: int,
    ) -> dict[str, Any]:
        import torch.distributed as dist

        if str(batch_id) != self._gate_calibration_batch_id:
            raise RuntimeError("Gate calibration batch identity mismatch")
        if self._gate_calibration_channel is not None:
            raise RuntimeError("Calibration channel remains active")
        if set(self._gate_calibration_gradients) != {
            "main",
            "decision",
            "query",
        }:
            raise RuntimeError("Calibration did not collect all three channels")
        profile = global_gradient_profile_from_shards(
            self._gate_calibration_gradients["main"],
            self._gate_calibration_gradients["decision"],
            self._gate_calibration_gradients["query"],
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._reshard_all_actor_modules()
        parameters = self._trainable_actor_parameters()
        parameter_hash_after = parameter_shard_sha256(parameters)
        optimizer_digest_after = self.local_optimizer_scheduler_digest()
        gradients_cleared = all(parameter.grad is None for parameter in parameters)
        local_metadata = {
            "parameter_names": tuple(
                name
                for name, parameter in self.actor_module_fsdp.named_parameters()
                if parameter.requires_grad
            ),
            "parameter_dtypes": tuple(str(parameter.dtype) for parameter in parameters),
            "parameter_count": len(parameters),
        }
        gathered_metadata: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_metadata, local_metadata)
        rank_metadata_consistent = len(
            {json.dumps(row, sort_keys=True) for row in gathered_metadata}
        ) == 1
        local_safety = {
            "parameters_bitwise_unchanged": (
                parameter_hash_after == self._gate_calibration_parameter_hash
            ),
            "optimizer_scheduler_unchanged": (
                optimizer_digest_after == self._gate_calibration_optimizer_digest
            ),
            "gradients_cleared": gradients_cleared,
            "strict_counts_zero": not any(
                (
                    self._zero_grad_calls,
                    self._backward_calls,
                    self._optimizer_steps,
                    self._scheduler_steps,
                )
            ),
        }
        gathered_safety: list[dict[str, bool] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_safety, local_safety)
        all_safety = all(
            row is not None and all(bool(value) for value in row.values())
            for row in gathered_safety
        )
        result = {
            "batch_id": str(batch_id),
            **profile,
            "decision_gate_event_count": int(decision_gate_event_count),
            "query_gate_event_count": int(query_gate_event_count),
            "parameters_bitwise_unchanged": bool(all_safety),
            "optimizer_scheduler_unchanged": all(
                bool(row and row["optimizer_scheduler_unchanged"])
                for row in gathered_safety
            ),
            "gradients_cleared": all(
                bool(row and row["gradients_cleared"])
                for row in gathered_safety
            ),
            "rank_metadata_consistent": bool(rank_metadata_consistent),
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "parameter_shard_sha256_before": self._gate_calibration_parameter_hash,
            "parameter_shard_sha256_after": parameter_hash_after,
        }
        self._gate_calibration_batch_id = None
        self._gate_calibration_gradients = {}
        self._gate_calibration_event_counts = {}
        self._gate_calibration_parameter_hash = ""
        self._gate_calibration_optimizer_digest = ""
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def abort_gate_gradient_profile(self) -> None:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._gate_calibration_channel = None
        self._gate_calibration_batch_id = None
        self._gate_calibration_gradients = {}
        self._gate_calibration_event_counts = {}
        self._gate_calibration_parameter_hash = ""
        self._gate_calibration_optimizer_digest = ""
        if any(
            (
                self._zero_grad_calls,
                self._backward_calls,
                self._optimizer_steps,
                self._scheduler_steps,
            )
        ):
            raise RuntimeError("Calibration abort observed a strict-step side effect")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def strict_zero_grad(self) -> None:
        if self._zero_grad_calls != 0:
            raise RuntimeError("zero_grad may be called only once per attempt")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._zero_grad_calls = 1

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def strict_backward_microbatch(
        self,
        microbatch: Mapping[str, Any],
    ) -> dict[str, Any]:
        import torch

        calibration_channel = self._gate_calibration_channel
        if calibration_channel is None:
            if self._zero_grad_calls != 1:
                raise RuntimeError("Backward requires exactly one prior zero_grad")
            if self._optimizer_steps or self._scheduler_steps:
                raise RuntimeError(
                    "Parameters changed before all micro-batches completed"
                )
        else:
            if self._gate_calibration_batch_id is None:
                raise RuntimeError("Calibration channel has no active profile")
            if any(
                (
                    self._zero_grad_calls,
                    self._backward_calls,
                    self._optimizer_steps,
                    self._scheduler_steps,
                )
            ):
                raise RuntimeError("Calibration entered the strict-step transaction")
        if int(microbatch["snapshot_step"]) != self._snapshot_step:
            raise RuntimeError("Micro-batch snapshot version mismatch")
        device = torch.device("cuda", torch.cuda.current_device())
        input_ids = microbatch["input_ids"].to(device)
        attention_mask = microbatch["attention_mask"].to(device)
        position_ids = microbatch["position_ids"].to(device)
        policy_mask = microbatch["policy_mask"].to(device).bool()
        turn_ids = microbatch["turn_ids"].to(device)
        old_logprobs = microbatch["old_logprobs"].to(device).detach()
        if old_logprobs.requires_grad:
            raise RuntimeError("Old-policy logprobs must be detached")

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        actor_hidden = self._forward_action_hidden(
            self.actor_module_fsdp,
            model_inputs,
            inference=False,
        )
        reference_hidden = self._forward_action_hidden(
            self.ref_module_fsdp,
            model_inputs,
            inference=True,
        )
        actor_head = getattr(self.actor_module_fsdp, "lm_head")
        reference_head = getattr(self.ref_module_fsdp, "lm_head")
        vocabulary_chunk_size = int(microbatch["vocabulary_chunk_size"])
        action_state_chunk_size = int(microbatch["action_state_chunk_size"])
        global_prompt_count = int(microbatch["global_prompt_count"])
        group_size = int(microbatch["group_size"])
        kl_coefficient = float(microbatch["kl_coefficient"])
        if kl_coefficient != 0.01:
            raise RuntimeError("Strict runtime locks kl_coefficient=0.01")
        search_task_mode = str(
            microbatch.get("search_task_mode", "normalized_outcome")
        )
        role_localized_mode = (
            search_task_mode
            == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_ROLE_LOCALIZED_GATE_MODE
        )
        mica_mode = (
            search_task_mode
            == ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE
        )
        lambda_decision = float(microbatch.get("lambda_decision", 0.0))
        lambda_query = float(microbatch.get("lambda_query", 0.0))
        if role_localized_mode:
            if not (
                math.isfinite(lambda_decision)
                and math.isfinite(lambda_query)
                and 0.0 <= lambda_decision <= 1.0
                and 0.0 <= lambda_query <= 1.0
            ):
                raise RuntimeError("Role-localized gate lambdas are invalid")
        elif lambda_decision != 0.0 or lambda_query != 0.0:
            raise RuntimeError("Legacy modes cannot carry gate coefficients")
        if mica_mode:
            if bool(microbatch["decision_token_mask"].any().item()) or bool(
                microbatch["query_token_mask"].any().item()
            ):
                raise RuntimeError("MICA mode cannot carry role-gate token masks")
            if any(microbatch["decision_advantage_by_turn"]) or any(
                microbatch["query_advantage_by_turn"]
            ):
                raise RuntimeError("MICA mode cannot carry A_decision/A_query")
        trajectory_weights = [
            float(value) for value in microbatch["trajectory_weights"]
        ]
        if len(trajectory_weights) != input_ids.shape[0]:
            raise ValueError("trajectory_weights must align with the batch")
        if any(value not in {0.0, 1.0} for value in trajectory_weights):
            raise ValueError("trajectory_weights are binary real/dummy markers")
        world_scale = float(self.world_size) / float(
            global_prompt_count * group_size
        )

        local_task_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_main_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_decision_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_query_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_kl_sum = torch.zeros((), dtype=torch.float32, device=device)
        ratios: list[float] = []
        clipped: list[float] = []
        decision_ratios: list[float] = []
        query_ratios: list[float] = []
        decision_clipped: list[float] = []
        query_clipped: list[float] = []
        nonzero_decision_and_query_same_token_count = 0
        unexpected_nonzero_main_gate_overlap_count = 0
        allowed_soft_duplicate_main_query_overlap_count = 0
        observation_policy_mask_violation_count = 0
        turn_runtime_metrics: list[dict[str, Any]] = []
        for batch_index in range(input_ids.shape[0]):
            trajectory_weight = trajectory_weights[batch_index]
            row_mask = policy_mask[batch_index]
            token_count = int(row_mask.sum().detach().cpu().item())
            if token_count < 1:
                raise RuntimeError("Every optimized trajectory needs action tokens")
            state_mask = _causal_state_mask(row_mask.unsqueeze(0))[0]
            actor_action_hidden = actor_hidden[batch_index][state_mask]
            reference_action_hidden = reference_hidden[batch_index][state_mask]
            targets = input_ids[batch_index][row_mask]
            current_blocks: list[Any] = []
            kl_blocks: list[Any] = []
            for start in range(0, token_count, action_state_chunk_size):
                end = min(start + action_state_chunk_size, token_count)
                actor_logits = actor_head(actor_action_hidden[start:end])
                with torch.no_grad():
                    reference_logits = reference_head(
                        reference_action_hidden[start:end]
                    ).detach()
                current_blocks.append(
                    torch.log_softmax(actor_logits.float(), dim=-1).gather(
                        1,
                        targets[start:end].unsqueeze(1),
                    ).squeeze(1)
                )
                kl_blocks.append(
                    actor_to_reference_full_vocab_kl(
                        actor_logits,
                        reference_logits,
                        vocabulary_chunk_size=vocabulary_chunk_size,
                    )
                )
            current = torch.cat(current_blocks)
            row_turn_ids = turn_ids[batch_index][row_mask]
            if role_localized_mode:
                labels = microbatch["labels"].to(device)[batch_index]
                expected_labels = torch.where(
                    row_mask,
                    input_ids[batch_index],
                    torch.full_like(input_ids[batch_index], -100),
                )
                if not bool(torch.equal(labels, expected_labels)):
                    observation_policy_mask_violation_count += 1
                    raise RuntimeError("Observation/prompt tokens entered learner labels")
                full_decision_mask = microbatch["decision_token_mask"].to(device)[
                    batch_index
                ].bool()
                full_query_mask = microbatch["query_token_mask"].to(device)[
                    batch_index
                ].bool()
                if bool((full_decision_mask & full_query_mask).any().item()):
                    nonzero_decision_and_query_same_token_count += int(
                        (full_decision_mask & full_query_mask).sum().item()
                    )
                    raise RuntimeError("Decision and Query token masks overlap")
                if bool((full_decision_mask & ~row_mask).any().item()) or bool(
                    (full_query_mask & ~row_mask).any().item()
                ):
                    raise RuntimeError("Gate token mask escaped the policy mask")
                row_decision_mask = full_decision_mask[row_mask]
                row_query_mask = full_query_mask[row_mask]
            ratio_by_turn = compute_turn_ratios(
                current,
                old_logprobs[batch_index][row_mask],
                torch.ones_like(row_turn_ids, dtype=torch.bool),
                row_turn_ids,
                expected_turn_ids=microbatch["expected_turn_ids"][batch_index],
            )
            objective = a2tgpo_adaptive_turn_objective(
                ratio_by_turn,
                microbatch["advantage_by_turn"][batch_index],
                microbatch["normalized_ig_by_turn"][batch_index],
                answer_turn_ids=microbatch["answer_turn_ids"][batch_index],
            )
            token_objectives = torch.stack(
                [
                    objective.objective_by_turn[int(turn_id)]
                    for turn_id in row_turn_ids.detach().cpu().tolist()
                ]
            )
            main_trajectory_objective = token_objectives.mean()
            if role_localized_mode:
                decision_advantages = microbatch["decision_advantage_by_turn"][
                    batch_index
                ]
                query_advantages = microbatch["query_advantage_by_turn"][batch_index]
                decision_ratio_by_turn = compute_turn_ratios(
                    current,
                    old_logprobs[batch_index][row_mask],
                    row_decision_mask,
                    row_turn_ids,
                    expected_turn_ids=microbatch["decision_turn_ids"][batch_index],
                )
                query_ratio_by_turn = compute_turn_ratios(
                    current,
                    old_logprobs[batch_index][row_mask],
                    row_query_mask,
                    row_turn_ids,
                    expected_turn_ids=microbatch["query_turn_ids"][batch_index],
                )
                decision_objective = fixed_gate_turn_objective(
                    decision_ratio_by_turn,
                    decision_advantages,
                )
                query_objective = fixed_gate_turn_objective(
                    query_ratio_by_turn,
                    query_advantages,
                )
                search_turn_count = int(
                    microbatch["search_turn_counts"][batch_index]
                )
                event_denominator = float(max(search_turn_count, 1))
                zero = current.sum() * 0.0
                decision_trajectory_objective = (
                    sum(decision_objective.objective_by_turn.values(), zero)
                    / event_denominator
                )
                query_trajectory_objective = (
                    sum(query_objective.objective_by_turn.values(), zero)
                    / event_denominator
                )
                if trajectory_weight:
                    main_advantages = microbatch["advantage_by_turn"][batch_index]
                    for turn_id, gate_advantage in decision_advantages.items():
                        if float(gate_advantage) != 0.0 and float(
                            main_advantages[int(turn_id)]
                        ) != 0.0:
                            unexpected_nonzero_main_gate_overlap_count += 1
                    for turn_id, gate_advantage in query_advantages.items():
                        gate_value = float(gate_advantage)
                        if gate_value == 0.0:
                            continue
                        main_value = float(main_advantages[int(turn_id)])
                        if gate_value == -0.25:
                            allowed_soft_duplicate_main_query_overlap_count += 1
                        elif main_value != 0.0:
                            unexpected_nonzero_main_gate_overlap_count += 1
                    if unexpected_nonzero_main_gate_overlap_count:
                        raise RuntimeError("Unexpected nonzero Main/Gate overlap")
                trajectory_task_objective = (
                    main_trajectory_objective
                    + lambda_decision * decision_trajectory_objective
                    + lambda_query * query_trajectory_objective
                )
                local_main_sum = (
                    local_main_sum
                    + trajectory_weight * main_trajectory_objective
                )
                local_decision_sum = (
                    local_decision_sum
                    + trajectory_weight * decision_trajectory_objective
                )
                local_query_sum = (
                    local_query_sum
                    + trajectory_weight * query_trajectory_objective
                )
                if trajectory_weight:
                    decision_ratios.extend(
                        float(value.detach().cpu().item())
                        for value in decision_objective.ratio_by_turn.values()
                    )
                    query_ratios.extend(
                        float(value.detach().cpu().item())
                        for value in query_objective.ratio_by_turn.values()
                    )
                    decision_clipped.extend(
                        float(value)
                        for value in decision_objective.clipped_by_turn.values()
                    )
                    query_clipped.extend(
                        float(value)
                        for value in query_objective.clipped_by_turn.values()
                    )
            else:
                trajectory_task_objective = main_trajectory_objective
                local_main_sum = (
                    local_main_sum
                    + trajectory_weight * main_trajectory_objective
                )
            local_task_sum = (
                local_task_sum + trajectory_weight * trajectory_task_objective
            )
            local_kl_sum = (
                local_kl_sum + trajectory_weight * torch.cat(kl_blocks).mean()
            )
            if trajectory_weight:
                ratios.extend(
                    float(value.detach().cpu().item())
                    for value in objective.ratio_by_turn.values()
                )
                clipped.extend(
                    float(value) for value in objective.clipped_by_turn.values()
                )
                for turn_id, ratio in objective.ratio_by_turn.items():
                    ratio_value = float(ratio.detach().cpu().item())
                    lower = float(objective.lower_bound_by_turn[turn_id])
                    upper = float(objective.upper_bound_by_turn[turn_id])
                    turn_runtime_metrics.append(
                        {
                            "prompt_global_id": str(
                                microbatch["prompt_global_ids"][batch_index]
                            ),
                            "trajectory_id": str(
                                microbatch["trajectory_ids"][batch_index]
                            ),
                            "turn_id": int(turn_id),
                            "ratio": ratio_value,
                            "clip_scale": float(
                                objective.clip_scale_by_turn[turn_id]
                            ),
                            "clip_lower": lower,
                            "clip_upper": upper,
                            "clipped_low": bool(ratio_value < lower),
                            "clipped_high": bool(ratio_value > upper),
                        }
                    )

        task_objective = local_task_sum * world_scale
        main_objective = local_main_sum * world_scale
        decision_objective_value = local_decision_sum * world_scale
        query_objective_value = local_query_sum * world_scale
        kl_objective = local_kl_sum * world_scale
        if calibration_channel is None:
            loss = -task_objective + kl_coefficient * kl_objective
        else:
            if not role_localized_mode:
                raise RuntimeError(
                    "Gate calibration requires the role-localized production mode"
                )
            channel_objectives = {
                "main": main_objective,
                "decision": decision_objective_value,
                "query": query_objective_value,
            }
            loss = -channel_objectives[str(calibration_channel)]
        if not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError("Strict objective produced NaN/Inf")
        loss.backward()
        if calibration_channel is None:
            self._backward_calls += 1
        return {
            "task_objective_local_scaled": float(
                task_objective.detach().cpu().item()
            ),
            "main_objective_local_scaled": float(
                main_objective.detach().cpu().item()
            ),
            "decision_objective_local_scaled": float(
                decision_objective_value.detach().cpu().item()
            ),
            "query_objective_local_scaled": float(
                query_objective_value.detach().cpu().item()
            ),
            "full_vocab_kl_local_scaled": float(
                kl_objective.detach().cpu().item()
            ),
            "total_loss_local_scaled": float(loss.detach().cpu().item()),
            "ratio_mean": float(sum(ratios) / max(1, len(ratios))),
            "clip_fraction": float(sum(clipped) / max(1, len(clipped))),
            "decision_ratio_mean": float(
                sum(decision_ratios) / max(1, len(decision_ratios))
            ),
            "query_ratio_mean": float(
                sum(query_ratios) / max(1, len(query_ratios))
            ),
            "decision_clip_fraction": float(
                sum(decision_clipped) / max(1, len(decision_clipped))
            ),
            "query_clip_fraction": float(
                sum(query_clipped) / max(1, len(query_clipped))
            ),
            "decision_ratio_values": decision_ratios,
            "query_ratio_values": query_ratios,
            "decision_clipped_values": decision_clipped,
            "query_clipped_values": query_clipped,
            "lambda_decision": lambda_decision,
            "lambda_query": lambda_query,
            "nonzero_decision_and_query_same_token_count": (
                nonzero_decision_and_query_same_token_count
            ),
            "unexpected_nonzero_main_gate_overlap_count": (
                unexpected_nonzero_main_gate_overlap_count
            ),
            "allowed_soft_duplicate_main_query_overlap_count": (
                allowed_soft_duplicate_main_query_overlap_count
            ),
            "observation_policy_mask_violation_count": (
                observation_policy_mask_violation_count
            ),
            "turn_runtime_metrics": turn_runtime_metrics,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def current_learning_rate(self) -> float:
        values = {
            float(group["lr"]) for group in self.actor_optimizer.param_groups
        }
        if len(values) != 1:
            raise RuntimeError(f"Optimizer parameter groups disagree on LR: {values}")
        return values.pop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def strict_clip_gradients(self, max_grad_norm: float) -> float:
        from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_

        if self._backward_calls < 1 or self._optimizer_steps:
            raise RuntimeError("Gradient clipping is outside the backward boundary")
        norm = fsdp2_clip_grad_norm_(
            self.actor_module_fsdp.parameters(),
            max_norm=float(max_grad_norm),
        )
        value = float(norm.detach().cpu().item())
        if not __import__("math").isfinite(value):
            raise RuntimeError("Gradient norm is NaN/Inf")
        return value

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def strict_optimizer_step(self) -> None:
        if self._zero_grad_calls != 1 or self._backward_calls < 1:
            raise RuntimeError("optimizer.step requires accumulated gradients")
        if self._optimizer_steps:
            raise RuntimeError("optimizer.step may execute only once")
        self.actor_optimizer.step()
        self._optimizer_steps = 1
        self._attempt_optimizer_committed = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def strict_scheduler_step(self) -> None:
        if self._optimizer_steps != 1 or self._scheduler_steps:
            raise RuntimeError("scheduler.step must follow the unique optimizer.step")
        self.actor_lr_scheduler.step()
        self._scheduler_steps = 1

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def strict_attempt_counts(self) -> dict[str, int]:
        return {
            "zero_grad": int(self._zero_grad_calls),
            "backward_microbatches": int(self._backward_calls),
            "optimizer_step": int(self._optimizer_steps),
            "scheduler_step": int(self._scheduler_steps),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_cuda_peak_memory_stats(self) -> dict[str, int]:
        """Reset per-rank CUDA peaks for an isolated runtime qualification."""

        import torch

        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
        return {
            "rank": int(self.rank),
            "device": int(torch.cuda.current_device()),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def cuda_memory_snapshot(self) -> dict[str, int]:
        """Return allocator telemetry without changing model or optimizer state."""

        import torch

        device = torch.device("cuda", torch.cuda.current_device())
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "rank": int(self.rank),
            "device": int(torch.cuda.current_device()),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rollback_pre_step_attempt(self) -> None:
        if self._attempt_optimizer_committed:
            raise RuntimeError(
                "Post-step state cannot be rolled back in place; process restart is required"
            )
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._zero_grad_calls = 0
        self._backward_calls = 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_hf_model_checkpoint(
        self,
        destination: str,
        successful_update_step: int,
        expected_actor_checksum: str,
        allow_restored_checkpoint_boundary: bool = False,
    ) -> dict[str, Any]:
        """Export the committed Actor as an inference-only HF safetensors file.

        Every rank participates in DTensor materialization collectives. Only
        rank zero retains CPU tensors and writes the file. The method never
        changes Actor parameters, optimizer state, scheduler state, or dtype.
        """
        import os

        import torch
        import torch.distributed as dist
        from safetensors.torch import save_file

        step = int(successful_update_step)
        if step <= 0:
            raise ValueError("Exported successful update must be positive")
        committed_boundary = (
            self._optimizer_steps == 1 and self._scheduler_steps == 1
        )
        restored_boundary = (
            bool(allow_restored_checkpoint_boundary)
            and self._zero_grad_calls == 0
            and self._backward_calls == 0
            and self._optimizer_steps == 0
            and self._scheduler_steps == 0
            and self._attempt_optimizer_committed is False
            and self._restored_checkpoint_step == step
            and self._restored_checkpoint_source is not None
        )
        if not committed_boundary and not restored_boundary:
            raise RuntimeError(
                "Model export requires a committed strict update or an "
                "explicitly verified restored-checkpoint boundary"
            )
        before_dtype = {
            str(parameter.dtype)
            for parameter in self.actor_module_fsdp.parameters()
            if parameter.is_floating_point()
        }
        before_checksum = self.global_actor_checksum()
        if before_checksum != str(expected_actor_checksum):
            raise RuntimeError("Actor checksum changed before model export")

        sharded_state, state_names = self._streaming_actor_state()
        export_state: dict[str, torch.Tensor] = {}
        maximum_materialized_tensor_bytes = 0
        for name in state_names:
            value = sharded_state.pop(name)
            tensor = self._materialize_streamed_actor_tensor(name, value)
            maximum_materialized_tensor_bytes = max(
                maximum_materialized_tensor_bytes,
                int(tensor.numel() * tensor.element_size()),
            )
            if self.rank == 0:
                detached = tensor.detach()
                if detached.is_floating_point():
                    detached = detached.to(dtype=torch.bfloat16)
                # Clone breaks tied storage explicitly; safetensors then emits
                # a self-contained state dict loadable by AutoModel.
                export_state[name] = detached.cpu().contiguous().clone()
            del value, tensor
        if sharded_state:
            raise RuntimeError("FSDP2 streaming export left unconsumed entries")
        del sharded_state
        dist.barrier()
        torch.cuda.empty_cache()

        output = Path(destination) / "model.safetensors"
        write_error: str | None = None
        if self.rank == 0:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    export_state,
                    str(output),
                    metadata={
                        "format": "pt",
                        "successful_update_step": str(step),
                        "actor_checksum": before_checksum,
                    },
                )
                with output.open("rb") as handle:
                    os.fsync(handle.fileno())
            except BaseException as exc:
                write_error = repr(exc)
        gathered_errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(gathered_errors, write_error)
        errors = [value for value in gathered_errors if value]
        if errors:
            raise RuntimeError("HF model export failed: " + " | ".join(errors))
        del export_state
        dist.barrier()

        after_dtype = {
            str(parameter.dtype)
            for parameter in self.actor_module_fsdp.parameters()
            if parameter.is_floating_point()
        }
        after_checksum = self.global_actor_checksum()
        if before_dtype != after_dtype:
            raise RuntimeError("Actor dtype changed during model export")
        if after_checksum != before_checksum:
            raise RuntimeError("Actor checksum changed during model export")
        return {
            "rank": int(self.rank),
            "successful_update_step": step,
            "actor_checksum": after_checksum,
            "export_boundary": (
                "committed_strict_update"
                if committed_boundary
                else "restored_checkpoint_zero_step"
            ),
            "restored_checkpoint_source": (
                self._restored_checkpoint_source if restored_boundary else None
            ),
            "actor_dtypes": sorted(after_dtype),
            "model_file": str(output) if self.rank == 0 else None,
            "model_file_bytes": (
                int(output.stat().st_size)
                if self.rank == 0 and output.is_file()
                else None
            ),
            "state_materialization_mode": (
                "streaming_sharded_dtensor_per_parameter"
            ),
            "state_tensor_count": len(state_names),
            "maximum_materialized_tensor_bytes": int(
                maximum_materialized_tensor_bytes
            ),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_distributed_training_state(self, destination: str) -> None:
        import random
        from pathlib import Path

        import numpy as np
        import torch

        from agentic_rl.checkpoint.fsdp2_dcp import save_fsdp2_training_state

        if self._optimizer_steps != 1 or self._scheduler_steps != 1:
            raise RuntimeError(
                "Only a complete optimizer/scheduler boundary can be checkpointed"
            )
        save_fsdp2_training_state(
            destination,
            model=self.actor_module_fsdp,
            optimizer=self.actor_optimizer,
            scheduler_state=self.actor_lr_scheduler.state_dict(),
        )
        rng_root = Path(destination) / "rng"
        rng_root.mkdir(exist_ok=True)
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state(
                    torch.cuda.current_device()
                ),
                "trainer_cuda": self.torch_random_states,
                "rollout_cuda": self.gen_random_states,
            },
            rng_root / f"rank-{int(self.rank):02d}.pt",
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_distributed_training_state(self, source: str) -> None:
        import random
        from pathlib import Path

        import numpy as np
        import torch

        from agentic_rl.checkpoint.fsdp2_dcp import load_fsdp2_training_state

        load_fsdp2_training_state(
            source,
            model=self.actor_module_fsdp,
            optimizer=self.actor_optimizer,
            scheduler=self.actor_lr_scheduler,
        )
        source_metadata = json.loads(
            (Path(source) / "metadata.json").read_text(encoding="utf-8")
        )
        source_world_size = int(source_metadata["fsdp_world_size"])
        source_rank = int(self.rank) % source_world_size
        payload = torch.load(
            Path(source) / "rng" / f"rank-{source_rank:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        random.setstate(payload["python"])
        np.random.set_state(payload["numpy"])
        torch.set_rng_state(payload["torch_cpu"])
        torch.cuda.set_rng_state(
            payload["torch_cuda"],
            torch.cuda.current_device(),
        )
        self.torch_random_states = payload["trainer_cuda"]
        self.gen_random_states = payload["rollout_cuda"]
        self._restored_checkpoint_step = int(
            source_metadata["successful_update_step"]
        )
        self._restored_checkpoint_source = str(Path(source).resolve())
