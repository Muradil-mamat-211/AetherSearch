from __future__ import annotations

import hashlib
import inspect
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .alias_reduce import immediate_ig_from_prefix_scores, telescoping_error
from .precision_policy import (
    ExactIGPrecisionPolicy,
    assert_fp32_exact_ig_runtime,
    exact_ig_precision_context,
    production_precision_policy,
)
from .task_builder import (
    ExactIGTask,
    SequentialExactIGTask,
    VectorizedExactIGTask,
)
from .target_schema import (
    CANONICAL_ALIAS_POLICY,
    EXACT_IG_VERSION,
    FAST_PATH_STRUCTURE,
    INFO_GAIN_TYPE,
    MASK_BUILDER_VERSION,
    OFFICIAL_IGPO_COMMIT_SHA,
    POSITION_BUILDER_VERSION,
    SCAFFOLD_SHA256,
    SCORE_MASK_POLICY,
    TARGET_TOKENIZATION_POLICY,
)


OFFICIAL_FULL_LOGITS = "official_full_logits"
SELECTED_POSITIONS = "selected_positions"
OFFICIAL_ADDITIVE_MASK = "official_additive"
BOOLEAN_4D_MASK = "boolean_4d"


@dataclass(frozen=True)
class ExactIGResult:
    score_by_prefix: tuple[float, ...]
    immediate_ig: tuple[float, ...]
    telescoping_error: float
    canonical_answer: str
    canonical_answer_sha256: str
    score_span_hash: str
    target_score_span_hash: str
    target_token_ids_hash: str
    score_token_ids_by_prefix: tuple[tuple[int, ...], ...]
    answer_token_log_probs_by_prefix: tuple[tuple[float, ...], ...]
    execution_path: str
    scoring_logits_mode: str
    runtime_metadata: Mapping[str, Any]
    exact_ig_version: str = EXACT_IG_VERSION
    scaffold_sha256: str = SCAFFOLD_SHA256
    canonical_alias_policy: str = CANONICAL_ALIAS_POLICY
    score_mask_policy: str = SCORE_MASK_POLICY
    info_gain_type: str = INFO_GAIN_TYPE
    fast_path_structure: str = FAST_PATH_STRUCTURE
    target_tokenization_policy: str = TARGET_TOKENIZATION_POLICY
    official_igpo_commit_sha: str = OFFICIAL_IGPO_COMMIT_SHA
    mask_builder_version: str = MASK_BUILDER_VERSION
    position_builder_version: str = POSITION_BUILDER_VERSION


@dataclass(frozen=True)
class ExactIGBatchEstimate:
    batch_size: int
    packed_lengths: tuple[int, ...]
    max_packed_length: int
    sum_length_squared: int
    padded_attention_cost: int
    padded_token_count: int
    gt_copy_count: int
    answer_score_position_count: int
    selected_position_union_count: int
    full_logits_estimated_bytes: int
    selected_logits_estimated_bytes: int
    structural_mask_estimated_bytes: int


@dataclass(frozen=True)
class ExactIGMicroBatchProfile:
    execution_mode: str
    batch_size: int
    packed_lengths: tuple[int, ...]
    max_packed_length: int
    sum_length_squared: int
    padded_attention_cost: int
    padding_ratio: float
    gt_copy_count: int
    answer_score_position_count: int
    selected_position_union_count: int
    full_logits_estimated_bytes: int
    selected_logits_estimated_bytes: int
    structural_mask_estimated_bytes: int
    actual_peak_allocated_bytes: int | None = None
    actual_peak_reserved_bytes: int | None = None
    boolean_mask_bytes_before_conversion: int = 0
    additive_mask_bytes_after_conversion: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }


class SingleFastTaskBudgetExceeded(ValueError):
    def __init__(
        self,
        task: VectorizedExactIGTask,
        estimate: ExactIGBatchEstimate,
        reasons: Sequence[str],
    ) -> None:
        self.task = task
        self.estimate = estimate
        self.reasons = tuple(str(reason) for reason in reasons)
        super().__init__(
            "single Fast Exact-IG task exceeds hard budget: "
            f"trajectory_id={task.trajectory_id}; reasons={self.reasons}"
        )


def _answer_logit_positions(task: VectorizedExactIGTask) -> tuple[int, ...]:
    return tuple(
        position
        for span in task.score_spans
        for position in span.logit_positions
    )


def estimate_exact_ig_batch(
    tasks: Sequence[VectorizedExactIGTask],
    *,
    vocabulary_size: int,
    logits_element_size: int,
    structural_mask_element_size: int,
) -> ExactIGBatchEstimate:
    if not tasks:
        raise ValueError("Cannot estimate an empty Exact-IG batch")
    if vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be positive")
    if logits_element_size <= 0 or structural_mask_element_size <= 0:
        raise ValueError("Element sizes must be positive")
    lengths = tuple(int(task.input_ids.size) for task in tasks)
    maximum = max(lengths)
    union = {
        int(position)
        for task in tasks
        for position in _answer_logit_positions(task)
    }
    batch_size = len(tasks)
    answer_count = sum(
        len(_answer_logit_positions(task))
        for task in tasks
    )
    return ExactIGBatchEstimate(
        batch_size=batch_size,
        packed_lengths=lengths,
        max_packed_length=maximum,
        sum_length_squared=sum(length * length for length in lengths),
        padded_attention_cost=batch_size * maximum * maximum,
        padded_token_count=batch_size * maximum,
        gt_copy_count=sum(task.prefix_count for task in tasks),
        answer_score_position_count=answer_count,
        selected_position_union_count=len(union),
        full_logits_estimated_bytes=(
            batch_size * maximum * vocabulary_size * logits_element_size
        ),
        selected_logits_estimated_bytes=(
            batch_size * len(union) * vocabulary_size * logits_element_size
        ),
        structural_mask_estimated_bytes=(
            batch_size
            * maximum
            * maximum
            * structural_mask_element_size
        ),
    )


def _budget_violations(
    estimate: ExactIGBatchEstimate,
    *,
    max_records_per_forward: int,
    max_attention_cost_per_batch: int | None,
    max_extended_tokens_per_batch: int | None,
    max_full_logits_bytes: int | None,
    max_selected_logits_bytes: int | None,
) -> tuple[str, ...]:
    checks = (
        (
            estimate.batch_size > max_records_per_forward,
            "max_records_per_forward",
        ),
        (
            max_attention_cost_per_batch is not None
            and estimate.padded_attention_cost > max_attention_cost_per_batch,
            "max_attention_cost_per_batch",
        ),
        (
            max_extended_tokens_per_batch is not None
            and estimate.padded_token_count > max_extended_tokens_per_batch,
            "max_extended_tokens_per_batch",
        ),
        (
            max_full_logits_bytes is not None
            and estimate.full_logits_estimated_bytes > max_full_logits_bytes,
            "max_full_logits_bytes",
        ),
        (
            max_selected_logits_bytes is not None
            and estimate.selected_logits_estimated_bytes
            > max_selected_logits_bytes,
            "max_selected_logits_bytes",
        ),
    )
    return tuple(name for failed, name in checks if failed)


def pack_exact_ig_microbatches(
    tasks: Sequence[VectorizedExactIGTask],
    *,
    max_records_per_forward: int,
    max_attention_cost_per_batch: int | None,
    max_extended_tokens_per_batch: int | None,
    max_full_logits_bytes: int | None = None,
    max_selected_logits_bytes: int | None = None,
    vocabulary_size: int = 1,
    logits_element_size: int = 4,
    structural_mask_element_size: int = 4,
) -> tuple[tuple[VectorizedExactIGTask, ...], ...]:
    if max_records_per_forward <= 0:
        raise ValueError("max_records_per_forward must be positive")
    for name, value in (
        ("max_attention_cost_per_batch", max_attention_cost_per_batch),
        ("max_extended_tokens_per_batch", max_extended_tokens_per_batch),
        ("max_full_logits_bytes", max_full_logits_bytes),
        ("max_selected_logits_bytes", max_selected_logits_bytes),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive or null")
    ordered = sorted(
        tasks,
        key=lambda task: (int(task.input_ids.size), str(task.trajectory_id)),
    )
    batches: list[tuple[VectorizedExactIGTask, ...]] = []
    current: list[VectorizedExactIGTask] = []
    for task in ordered:
        task.validate()
        single_estimate = estimate_exact_ig_batch(
            (task,),
            vocabulary_size=vocabulary_size,
            logits_element_size=logits_element_size,
            structural_mask_element_size=structural_mask_element_size,
        )
        single_reasons = _budget_violations(
            single_estimate,
            max_records_per_forward=max_records_per_forward,
            max_attention_cost_per_batch=max_attention_cost_per_batch,
            max_extended_tokens_per_batch=max_extended_tokens_per_batch,
            max_full_logits_bytes=max_full_logits_bytes,
            max_selected_logits_bytes=max_selected_logits_bytes,
        )
        if single_reasons:
            raise SingleFastTaskBudgetExceeded(
                task,
                single_estimate,
                single_reasons,
            )
        candidate = (*current, task)
        candidate_estimate = estimate_exact_ig_batch(
            candidate,
            vocabulary_size=vocabulary_size,
            logits_element_size=logits_element_size,
            structural_mask_element_size=structural_mask_element_size,
        )
        reasons = _budget_violations(
            candidate_estimate,
            max_records_per_forward=max_records_per_forward,
            max_attention_cost_per_batch=max_attention_cost_per_batch,
            max_extended_tokens_per_batch=max_extended_tokens_per_batch,
            max_full_logits_bytes=max_full_logits_bytes,
            max_selected_logits_bytes=max_selected_logits_bytes,
        )
        if current and reasons:
            batches.append(tuple(current))
            current = [task]
        else:
            current = list(candidate)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def selected_positions_capability(model: Any) -> dict[str, Any]:
    import transformers

    signature = inspect.signature(model.forward)
    parameter = signature.parameters.get("logits_to_keep")
    forward = getattr(type(model).forward, "__wrapped__", type(model).forward)
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        source = repr(signature)
    return {
        "transformers_version": transformers.__version__,
        "forward_signature": str(signature),
        "forward_source_sha256": hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest(),
        "logits_to_keep_present": parameter is not None,
        "tensor_indices_declared": (
            parameter is not None
            and "Tensor" in str(parameter.annotation)
        ),
        "supported_by_signature": (
            parameter is not None
            and "Tensor" in str(parameter.annotation)
        ),
    }


def _hidden_dtype_probe_modules(model: Any) -> tuple[Any | None, Any | None]:
    """Return the first and last decoder layers without retaining activations."""

    candidates = tuple(
        module
        for module in model.modules()
        if type(module).__name__.lower().endswith("decoderlayer")
    )
    if not candidates:
        return None, None
    return candidates[0], candidates[-1]


def _output_dtype(value: Any) -> str | None:
    tensor = value[0] if isinstance(value, (tuple, list)) and value else value
    dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype).removeprefix("torch.")


class VectorizedExactIGScorer:
    def __init__(
        self,
        *,
        precision_policy: ExactIGPrecisionPolicy | None = None,
        padding_token_id: int = 0,
        tokenizer: Any | None = None,
        scoring_logits_mode: str = OFFICIAL_FULL_LOGITS,
        attention_mask_mode: str = OFFICIAL_ADDITIVE_MASK,
    ) -> None:
        if scoring_logits_mode not in {
            OFFICIAL_FULL_LOGITS,
            SELECTED_POSITIONS,
        }:
            raise ValueError("Unsupported Exact-IG scoring_logits_mode")
        if attention_mask_mode not in {
            OFFICIAL_ADDITIVE_MASK,
            BOOLEAN_4D_MASK,
        }:
            raise ValueError("Unsupported Exact-IG attention_mask_mode")
        self.precision_policy = precision_policy
        self.padding_token_id = int(padding_token_id)
        self.tokenizer = tokenizer
        self.scoring_logits_mode = scoring_logits_mode
        self.attention_mask_mode = attention_mask_mode
        self.last_microbatch_profiles: tuple[ExactIGMicroBatchProfile, ...] = ()
        self.last_runtime_metadata: dict[str, Any] = {}

    @classmethod
    def for_production_mode(
        cls,
        mode: str,
        *,
        padding_token_id: int = 0,
        tokenizer: Any | None = None,
        scoring_logits_mode: str = OFFICIAL_FULL_LOGITS,
        attention_mask_mode: str = OFFICIAL_ADDITIVE_MASK,
    ) -> "VectorizedExactIGScorer":
        return cls(
            precision_policy=production_precision_policy(mode),
            padding_token_id=padding_token_id,
            tokenizer=tokenizer,
            scoring_logits_mode=scoring_logits_mode,
            attention_mask_mode=attention_mask_mode,
        )

    @staticmethod
    def _profile(
        tasks: Sequence[VectorizedExactIGTask],
        *,
        estimate: ExactIGBatchEstimate,
        execution_mode: str,
        actual_peak_allocated_bytes: int | None = None,
        actual_peak_reserved_bytes: int | None = None,
        boolean_mask_bytes_before_conversion: int = 0,
        additive_mask_bytes_after_conversion: int = 0,
    ) -> ExactIGMicroBatchProfile:
        actual_tokens = sum(estimate.packed_lengths)
        padded_tokens = estimate.batch_size * estimate.max_packed_length
        return ExactIGMicroBatchProfile(
            execution_mode=execution_mode,
            batch_size=estimate.batch_size,
            packed_lengths=estimate.packed_lengths,
            max_packed_length=estimate.max_packed_length,
            sum_length_squared=estimate.sum_length_squared,
            padded_attention_cost=estimate.padded_attention_cost,
            padding_ratio=(
                1.0 - actual_tokens / padded_tokens if padded_tokens else 0.0
            ),
            gt_copy_count=estimate.gt_copy_count,
            answer_score_position_count=(
                estimate.answer_score_position_count
            ),
            selected_position_union_count=(
                estimate.selected_position_union_count
            ),
            full_logits_estimated_bytes=estimate.full_logits_estimated_bytes,
            selected_logits_estimated_bytes=(
                estimate.selected_logits_estimated_bytes
            ),
            structural_mask_estimated_bytes=(
                estimate.structural_mask_estimated_bytes
            ),
            actual_peak_allocated_bytes=actual_peak_allocated_bytes,
            actual_peak_reserved_bytes=actual_peak_reserved_bytes,
            boolean_mask_bytes_before_conversion=(
                boolean_mask_bytes_before_conversion
            ),
            additive_mask_bytes_after_conversion=(
                additive_mask_bytes_after_conversion
            ),
        )

    def score_batch(
        self,
        model: Any,
        tasks: Sequence[VectorizedExactIGTask],
        device: Any,
    ) -> tuple[ExactIGResult, ...]:
        import torch
        import torch.nn.functional as F

        if not tasks:
            return ()
        for task in tasks:
            task.validate()
        if self.scoring_logits_mode == SELECTED_POSITIONS:
            capability = selected_positions_capability(model)
            if not capability["supported_by_signature"]:
                raise RuntimeError(
                    "selected_positions requested but model.forward does not "
                    "declare Tensor logits_to_keep"
                )
        else:
            capability = selected_positions_capability(model)

        batch_size = len(tasks)
        maximum_length = max(int(task.input_ids.size) for task in tasks)
        input_ids = torch.full(
            (batch_size, maximum_length),
            self.padding_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.zeros(
            (batch_size, maximum_length),
            dtype=torch.long,
            device=device,
        )
        boolean_attention_mask = torch.zeros(
            (batch_size, maximum_length, maximum_length),
            dtype=torch.bool,
            device=device,
        )
        for batch_index, task in enumerate(tasks):
            length = int(task.input_ids.size)
            input_ids[batch_index, :length] = torch.as_tensor(
                task.input_ids.copy(),
                dtype=torch.long,
                device=device,
            )
            position_ids[batch_index, :length] = torch.as_tensor(
                task.position_ids.copy(),
                dtype=torch.long,
                device=device,
            )
            boolean_attention_mask[batch_index, :length, :length] = (
                torch.as_tensor(
                    task.attention_mask.copy(),
                    dtype=torch.bool,
                    device=device,
                )
            )
            padded = torch.arange(length, maximum_length, device=device)
            boolean_attention_mask[batch_index, padded, padded] = True

        boolean_bytes = (
            boolean_attention_mask.numel()
            * boolean_attention_mask.element_size()
        )
        if self.attention_mask_mode == BOOLEAN_4D_MASK:
            attention_mask = boolean_attention_mask.unsqueeze(1)
            additive_bytes = 0
        else:
            attention_mask = torch.where(
                boolean_attention_mask,
                torch.tensor(0.0, dtype=torch.float32, device=device),
                torch.tensor(-10000.0, dtype=torch.float32, device=device),
            ).unsqueeze(1)
            additive_bytes = attention_mask.numel() * attention_mask.element_size()
            del boolean_attention_mask

        union_positions = tuple(
            sorted(
                {
                    int(position)
                    for task in tasks
                    for position in _answer_logit_positions(task)
                }
            )
        )
        union_tensor = torch.as_tensor(
            union_positions,
            dtype=torch.long,
            device=device,
        )
        compressed_row = {
            position: row
            for row, position in enumerate(union_positions)
        }
        observed_hidden_dtypes: dict[str, str | None] = {
            "first": None,
            "last": None,
        }
        probe_handles: list[Any] = []
        first_probe, last_probe = _hidden_dtype_probe_modules(model)
        if first_probe is not None:
            probe_handles.append(
                first_probe.register_forward_hook(
                    lambda _module, _inputs, output: observed_hidden_dtypes.__setitem__(
                        "first",
                        _output_dtype(output),
                    )
                )
            )
        if last_probe is not None and last_probe is not first_probe:
            probe_handles.append(
                last_probe.register_forward_hook(
                    lambda _module, _inputs, output: observed_hidden_dtypes.__setitem__(
                        "last",
                        _output_dtype(output),
                    )
                )
            )
        model_was_training = bool(getattr(model, "training", False))
        model.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        policy_context = (
            exact_ig_precision_context(model, self.precision_policy)
            if self.precision_policy is not None
            else nullcontext()
        )
        try:
            with torch.no_grad(), policy_context:
                forward_kwargs: dict[str, Any] = {}
                if self.scoring_logits_mode == SELECTED_POSITIONS:
                    forward_kwargs["logits_to_keep"] = union_tensor
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    **forward_kwargs,
                )
                temperature = (
                    float(self.precision_policy.temperature)
                    if self.precision_policy is not None
                    else 1.0
                )
                if temperature != 1.0:
                    raise ValueError("Official Exact-IG temperature is fixed to 1.0")
                logits = outputs.logits / temperature
                observed_backend = str(
                    getattr(
                        getattr(model, "config", None),
                        "_attn_implementation",
                        "unknown",
                    )
                )
                observed_cuda_state = {
                    "matmul_allow_tf32": (
                        bool(torch.backends.cuda.matmul.allow_tf32)
                        if device.type == "cuda"
                        else None
                    ),
                    "cudnn_allow_tf32": (
                        bool(torch.backends.cudnn.allow_tf32)
                        if device.type == "cuda"
                        else None
                    ),
                }
                if self.precision_policy is not None:
                    assert_fp32_exact_ig_runtime(
                        model=model,
                        policy=self.precision_policy,
                        logits=logits,
                    )
                if (
                    self.scoring_logits_mode == SELECTED_POSITIONS
                    and logits.shape[1] != len(union_positions)
                ):
                    raise RuntimeError(
                        "Model ignored or misapplied Tensor logits_to_keep"
                    )
                results: list[ExactIGResult] = []
                actual_log_probs_dtype: str | None = None
                for batch_index, task in enumerate(tasks):
                    prefix_scores: list[float] = []
                    score_token_ids: list[tuple[int, ...]] = []
                    answer_log_probs_by_prefix: list[tuple[float, ...]] = []
                    for span in task.score_spans:
                        if self.scoring_logits_mode == OFFICIAL_FULL_LOGITS:
                            full_rows = torch.arange(
                                span.segment_start - 1,
                                span.segment_end - 1,
                                dtype=torch.long,
                                device=device,
                            )
                            full_targets = input_ids[
                                batch_index,
                                span.segment_start : span.segment_end,
                            ]
                            full_target_log_probs = F.log_softmax(
                                logits[batch_index].index_select(0, full_rows),
                                dim=-1,
                            ).gather(
                                dim=-1,
                                index=full_targets.unsqueeze(-1),
                            ).squeeze(-1)
                            answer_start = (
                                task.canonical_target.answer_token_start
                            )
                            answer_end = task.canonical_target.answer_token_end
                            token_log_probs = full_target_log_probs[
                                answer_start:answer_end
                            ]
                            targets = full_targets[answer_start:answer_end]
                        else:
                            rows = tuple(
                                compressed_row[int(position)]
                                for position in span.logit_positions
                            )
                            row_tensor = torch.as_tensor(
                                rows,
                                dtype=torch.long,
                                device=device,
                            )
                            targets = torch.as_tensor(
                                span.answer_token_ids,
                                dtype=torch.long,
                                device=device,
                            )
                            selected_logits = logits[batch_index].index_select(
                                0,
                                row_tensor,
                            )
                            token_log_probs = F.log_softmax(
                                selected_logits,
                                dim=-1,
                            ).gather(
                                dim=-1,
                                index=targets.unsqueeze(-1),
                            ).squeeze(-1)
                        if tuple(
                            int(value)
                            for value in targets.detach().cpu().tolist()
                        ) != span.answer_token_ids:
                            raise RuntimeError(
                                "Fast Exact-IG answer slice changed target token IDs"
                            )
                        if self.precision_policy is not None:
                            assert_fp32_exact_ig_runtime(
                                model=model,
                                policy=self.precision_policy,
                                logits=logits,
                                log_probs=token_log_probs,
                            )
                        actual_log_probs_dtype = str(
                            token_log_probs.dtype
                        ).removeprefix("torch.")
                        score = token_log_probs.mean()
                        if not bool(torch.isfinite(score).item()):
                            raise RuntimeError(
                                "Fast Exact-IG produced a non-finite Phi"
                            )
                        prefix_scores.append(float(score.cpu().item()))
                        score_token_ids.append(span.answer_token_ids)
                        answer_log_probs_by_prefix.append(
                            tuple(
                                float(value)
                                for value in token_log_probs.cpu().tolist()
                            )
                        )
                    score_tuple = tuple(prefix_scores)
                    immediate = immediate_ig_from_prefix_scores(score_tuple)
                    runtime_metadata = {
                        "actual_model_parameter_dtype": str(
                            next(model.parameters()).dtype
                        ).removeprefix("torch."),
                        "actual_logits_dtype": str(
                            logits.dtype
                        ).removeprefix("torch."),
                        "actual_log_probs_dtype": actual_log_probs_dtype,
                        "autocast_enabled": bool(
                            self.precision_policy is not None
                            and self.precision_policy.autocast_enabled
                            and device.type == "cuda"
                        ),
                        "autocast_dtype": (
                            self.precision_policy.autocast_dtype
                            if self.precision_policy is not None
                            else None
                        ),
                        "attention_backend": (
                            f"{self.precision_policy.attention_implementation}:"
                            f"{self.precision_policy.sdpa_backend or 'native'}"
                            if self.precision_policy is not None
                            else "native"
                        ),
                        "model_config_attn_implementation": observed_backend,
                        "tf32_matmul_enabled": observed_cuda_state[
                            "matmul_allow_tf32"
                        ],
                        "tf32_cudnn_enabled": observed_cuda_state[
                            "cudnn_allow_tf32"
                        ],
                        "actual_hidden_dtype_first_layer": (
                            observed_hidden_dtypes["first"]
                        ),
                        "actual_hidden_dtype_last_layer": (
                            observed_hidden_dtypes["last"]
                        ),
                        "temperature": temperature,
                        "allow_tf32": (
                            bool(self.precision_policy.allow_tf32)
                            if self.precision_policy is not None
                            else None
                        ),
                        "allow_bf16_reduced_precision_reduction": (
                            bool(
                                self.precision_policy
                                .allow_bf16_reduced_precision_reduction
                            )
                            if self.precision_policy is not None
                            else None
                        ),
                        "allow_fp16_reduced_precision_reduction": (
                            bool(
                                self.precision_policy
                                .allow_fp16_reduced_precision_reduction
                            )
                            if self.precision_policy is not None
                            else None
                        ),
                        "float32_matmul_precision": (
                            "highest"
                            if self.precision_policy is not None
                            else None
                        ),
                        "scoring_logits_mode": self.scoring_logits_mode,
                        "attention_mask_mode": self.attention_mask_mode,
                        "attention_mask_dtype": str(
                            attention_mask.dtype
                        ).removeprefix("torch."),
                        "attention_mask_shape": tuple(
                            int(value) for value in attention_mask.shape
                        ),
                        "attention_mask_masked_value": (
                            None
                            if self.attention_mask_mode == BOOLEAN_4D_MASK
                            else -10000.0
                        ),
                        "batch_size": batch_size,
                        "batch_trajectory_ids": tuple(
                            str(item.trajectory_id) for item in tasks
                        ),
                        "packed_lengths": tuple(
                            int(item.input_ids.size) for item in tasks
                        ),
                        "padded_max_length": maximum_length,
                        "padding_ratio": (
                            1.0
                            - sum(int(item.input_ids.size) for item in tasks)
                            / (batch_size * maximum_length)
                        ),
                        "prefix_count": int(task.prefix_count),
                        "answer_token_count": int(
                            task.canonical_target.answer_token_count
                        ),
                        "selected_positions_capability": capability,
                    }
                    results.append(
                        ExactIGResult(
                            score_by_prefix=score_tuple,
                            immediate_ig=immediate,
                            telescoping_error=telescoping_error(
                                score_tuple,
                                immediate,
                            ),
                            canonical_answer=task.canonical_answer,
                            canonical_answer_sha256=task.canonical_answer_hash,
                            score_span_hash=task.score_span_hash,
                            target_score_span_hash=(
                                task.canonical_target.score_span_hash
                            ),
                            target_token_ids_hash=task.target_token_ids_hash,
                            score_token_ids_by_prefix=tuple(score_token_ids),
                            answer_token_log_probs_by_prefix=tuple(
                                answer_log_probs_by_prefix
                            ),
                            execution_path="official_fast",
                            scoring_logits_mode=self.scoring_logits_mode,
                            runtime_metadata=runtime_metadata,
                        )
                    )
                del outputs, logits
        finally:
            for handle in probe_handles:
                handle.remove()
            if model_was_training:
                model.train()
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        )
        vocabulary_size = int(
            getattr(
                getattr(model, "config", None),
                "vocab_size",
                getattr(model, "vocabulary_size", 1),
            )
        )
        logits_element_size = next(model.parameters()).element_size()
        mask_element_size = (
            1 if self.attention_mask_mode == BOOLEAN_4D_MASK else 4
        )
        estimate = estimate_exact_ig_batch(
            tasks,
            vocabulary_size=vocabulary_size,
            logits_element_size=logits_element_size,
            structural_mask_element_size=mask_element_size,
        )
        self.last_microbatch_profiles = (
            self._profile(
                tasks,
                estimate=estimate,
                execution_mode=self.scoring_logits_mode,
                actual_peak_allocated_bytes=peak_allocated,
                actual_peak_reserved_bytes=peak_reserved,
                boolean_mask_bytes_before_conversion=boolean_bytes,
                additive_mask_bytes_after_conversion=additive_bytes,
            ),
        )
        results = [
            replace(
                result,
                runtime_metadata={
                    **dict(result.runtime_metadata),
                    "actual_peak_allocated_bytes": peak_allocated,
                    "actual_peak_reserved_bytes": peak_reserved,
                },
            )
            for result in results
        ]
        self.last_runtime_metadata = dict(results[0].runtime_metadata)
        return tuple(results)

    def score(
        self,
        model: Any,
        task: VectorizedExactIGTask,
        device: Any,
    ) -> ExactIGResult:
        return self.score_batch(model, (task,), device)[0]

    @staticmethod
    def _sequential_budget_reasons(
        task: ExactIGTask,
        *,
        vocabulary_size: int,
        logits_element_size: int,
        max_attention_cost_per_batch: int | None,
        max_extended_tokens_per_batch: int | None,
        max_full_logits_bytes: int | None,
        max_selected_logits_bytes: int | None,
    ) -> tuple[str, ...]:
        reasons: set[str] = set()
        target_length = len(task.canonical_target.token_ids)
        answer_count = task.canonical_target.answer_token_count
        for prefix_end in task.prefix_end_positions:
            length = int(prefix_end) + target_length
            if length > task.maximum_extended_sequence_length:
                reasons.add("maximum_extended_sequence_length")
            if (
                int(task.original_position_ids[prefix_end - 1])
                + target_length
                >= task.maximum_position_id_exclusive
            ):
                reasons.add("maximum_position_id_exclusive")
            if (
                max_attention_cost_per_batch is not None
                and length * length > max_attention_cost_per_batch
            ):
                reasons.add("max_attention_cost_per_batch")
            if (
                max_extended_tokens_per_batch is not None
                and length > max_extended_tokens_per_batch
            ):
                reasons.add("max_extended_tokens_per_batch")
            if (
                max_full_logits_bytes is not None
                and length * vocabulary_size * logits_element_size
                > max_full_logits_bytes
            ):
                reasons.add("max_full_logits_bytes")
            if (
                max_selected_logits_bytes is not None
                and answer_count * vocabulary_size * logits_element_size
                > max_selected_logits_bytes
            ):
                reasons.add("max_selected_logits_bytes")
        return tuple(sorted(reasons))

    def _score_sequential_fallback(
        self,
        model: Any,
        task: ExactIGTask,
        device: Any,
    ) -> ExactIGResult:
        if self.tokenizer is None or self.precision_policy is None:
            raise RuntimeError(
                "Sequential fallback requires tokenizer and official precision policy"
            )
        from .sequential_oracle import sequential_teacher_forced_oracle

        oracle = sequential_teacher_forced_oracle(
            model=model,
            tokenizer=self.tokenizer,
            full_trajectory_input_ids=task.input_ids[
                : task.original_token_count
            ],
            original_attention_mask=task.original_attention_mask,
            original_position_ids=task.original_position_ids,
            prefix_end_positions=task.prefix_end_positions,
            canonical_answer=task.canonical_answer,
            encoded_target=task.canonical_target,
            device=device,
            precision_policy=self.precision_policy,
        )
        return ExactIGResult(
            score_by_prefix=oracle.score_by_prefix,
            immediate_ig=oracle.immediate_ig,
            telescoping_error=oracle.telescoping_error,
            canonical_answer=oracle.canonical_answer,
            canonical_answer_sha256=oracle.canonical_answer_sha256,
            score_span_hash=task.score_span_hash,
            target_score_span_hash=oracle.score_span_hash,
            target_token_ids_hash=task.target_token_ids_hash,
            score_token_ids_by_prefix=oracle.score_token_ids_by_prefix,
            answer_token_log_probs_by_prefix=(
                oracle.answer_token_log_probs_by_prefix
            ),
            execution_path="official_sequential_fallback",
            scoring_logits_mode=OFFICIAL_FULL_LOGITS,
            runtime_metadata={
                **dict(oracle.runtime_metadata),
                "fallback_reason": "single_fast_task_budget_exceeded",
            },
        )

    def score_many(
        self,
        model: Any,
        tasks: Sequence[ExactIGTask],
        device: Any,
        *,
        max_records_per_forward: int,
        max_attention_cost_per_batch: int | None,
        max_extended_tokens_per_batch: int | None,
        max_full_logits_bytes: int | None = None,
        max_selected_logits_bytes: int | None = None,
    ) -> dict[str, ExactIGResult]:
        vocabulary_size = int(
            getattr(
                getattr(model, "config", None),
                "vocab_size",
                getattr(model, "vocabulary_size", 1),
            )
        )
        logits_element_size = next(model.parameters()).element_size()
        mask_element_size = (
            1 if self.attention_mask_mode == BOOLEAN_4D_MASK else 4
        )
        fast_tasks: list[VectorizedExactIGTask] = []
        fallback_tasks: list[ExactIGTask] = []
        for task in tasks:
            task.validate()
            if isinstance(task, SequentialExactIGTask):
                sequential_reasons = self._sequential_budget_reasons(
                    task,
                    vocabulary_size=vocabulary_size,
                    logits_element_size=logits_element_size,
                    max_attention_cost_per_batch=max_attention_cost_per_batch,
                    max_extended_tokens_per_batch=max_extended_tokens_per_batch,
                    max_full_logits_bytes=max_full_logits_bytes,
                    max_selected_logits_bytes=max_selected_logits_bytes,
                )
                if sequential_reasons:
                    raise RuntimeError(
                        "Exact-IG task cannot be represented by Fast Path and "
                        "its Sequential prefixes exceed hard budgets: "
                        f"trajectory_id={task.trajectory_id}; "
                        f"sequential={sequential_reasons}"
                    )
                fallback_tasks.append(task)
                continue
            estimate = estimate_exact_ig_batch(
                (task,),
                vocabulary_size=vocabulary_size,
                logits_element_size=logits_element_size,
                structural_mask_element_size=mask_element_size,
            )
            reasons = _budget_violations(
                estimate,
                max_records_per_forward=max_records_per_forward,
                max_attention_cost_per_batch=max_attention_cost_per_batch,
                max_extended_tokens_per_batch=max_extended_tokens_per_batch,
                max_full_logits_bytes=max_full_logits_bytes,
                max_selected_logits_bytes=max_selected_logits_bytes,
            )
            if reasons:
                sequential_reasons = self._sequential_budget_reasons(
                    task,
                    vocabulary_size=vocabulary_size,
                    logits_element_size=logits_element_size,
                    max_attention_cost_per_batch=max_attention_cost_per_batch,
                    max_extended_tokens_per_batch=max_extended_tokens_per_batch,
                    max_full_logits_bytes=max_full_logits_bytes,
                    max_selected_logits_bytes=max_selected_logits_bytes,
                )
                if sequential_reasons:
                    raise RuntimeError(
                        "Exact-IG task exceeds both Fast and Sequential hard "
                        f"budgets: trajectory_id={task.trajectory_id}; "
                        f"fast={reasons}; sequential={sequential_reasons}"
                    )
                fallback_tasks.append(task)
            else:
                fast_tasks.append(task)

        batches = pack_exact_ig_microbatches(
            fast_tasks,
            max_records_per_forward=max_records_per_forward,
            max_attention_cost_per_batch=max_attention_cost_per_batch,
            max_extended_tokens_per_batch=max_extended_tokens_per_batch,
            max_full_logits_bytes=max_full_logits_bytes,
            max_selected_logits_bytes=max_selected_logits_bytes,
            vocabulary_size=vocabulary_size,
            logits_element_size=logits_element_size,
            structural_mask_element_size=mask_element_size,
        )
        by_trajectory: dict[str, ExactIGResult] = {}
        profiles: list[ExactIGMicroBatchProfile] = []
        for batch in batches:
            batch_results = self.score_batch(model, batch, device)
            profiles.extend(self.last_microbatch_profiles)
            for task, result in zip(batch, batch_results, strict=True):
                if task.trajectory_id in by_trajectory:
                    raise RuntimeError(
                        f"Duplicate Exact-IG trajectory: {task.trajectory_id}"
                    )
                by_trajectory[task.trajectory_id] = result
        for task in fallback_tasks:
            result = self._score_sequential_fallback(model, task, device)
            by_trajectory[task.trajectory_id] = result
            sequence_lengths = tuple(
                int(prefix_end) + len(task.canonical_target.token_ids)
                for prefix_end in task.prefix_end_positions
            )
            maximum = max(sequence_lengths)
            profiles.append(
                ExactIGMicroBatchProfile(
                    execution_mode="official_sequential_fallback",
                    batch_size=1,
                    packed_lengths=sequence_lengths,
                    max_packed_length=maximum,
                    sum_length_squared=sum(
                        length * length for length in sequence_lengths
                    ),
                    padded_attention_cost=max(
                        length * length for length in sequence_lengths
                    ),
                    padding_ratio=0.0,
                    gt_copy_count=task.prefix_count,
                    answer_score_position_count=(
                        task.prefix_count
                        * task.canonical_target.answer_token_count
                    ),
                    selected_position_union_count=(
                        task.canonical_target.answer_token_count
                    ),
                    full_logits_estimated_bytes=(
                        maximum
                        * vocabulary_size
                        * logits_element_size
                    ),
                    selected_logits_estimated_bytes=(
                        task.canonical_target.answer_token_count
                        * vocabulary_size
                        * logits_element_size
                    ),
                    structural_mask_estimated_bytes=maximum,
                )
            )
        self.last_microbatch_profiles = tuple(profiles)
        return by_trajectory
