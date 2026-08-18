from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .alias_reduce import immediate_ig_from_prefix_scores, telescoping_error
from .precision_policy import (
    ExactIGPrecisionPolicy,
    assert_fp32_exact_ig_runtime,
    exact_ig_precision_context,
)
from .target_schema import (
    EncodedExactIGTarget,
    encode_exact_ig_target,
    select_canonical_answer,
)


@dataclass(frozen=True)
class OracleTokenScore:
    prefix_index: int
    physical_token_index: int
    token_id: int
    decoded_token: str
    predicting_logit_index: int
    score_mask: bool
    token_log_prob: float


@dataclass(frozen=True)
class SequentialOracleResult:
    score_by_prefix: tuple[float, ...]
    immediate_ig: tuple[float, ...]
    canonical_answer: str
    canonical_answer_sha256: str
    score_span_hash: str
    target_token_ids: tuple[int, ...]
    answer_token_range: tuple[int, int]
    answer_token_count: int
    scored_answer_token_count: int
    score_token_ids_by_prefix: tuple[tuple[int, ...], ...]
    answer_token_log_probs_by_prefix: tuple[tuple[float, ...], ...]
    full_target_log_probs_by_prefix: tuple[tuple[float, ...], ...]
    token_scores: tuple[OracleTokenScore, ...]
    telescoping_error: float
    runtime_metadata: Mapping[str, Any]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def sequential_teacher_forced_oracle(
    *,
    model: Any,
    tokenizer: Any,
    full_trajectory_input_ids: Sequence[int],
    original_attention_mask: Sequence[int],
    prefix_end_positions: Sequence[int],
    canonical_answer: str,
    device: Any,
    precision_policy: ExactIGPrecisionPolicy,
    original_position_ids: Sequence[int] | None = None,
    encoded_target: EncodedExactIGTarget | None = None,
) -> SequentialOracleResult:
    """Independent one-prefix standard causal teacher-forcing reference."""

    import torch

    original = np.asarray(full_trajectory_input_ids, dtype=np.int64)
    attention = np.asarray(original_attention_mask, dtype=np.int64)
    endpoints = tuple(int(value) for value in prefix_end_positions)
    if original.ndim != 1 or original.size == 0:
        raise ValueError("Oracle trajectory input must be a non-empty vector")
    if attention.shape != original.shape:
        raise ValueError("Oracle attention mask must align with trajectory input")
    if not endpoints or tuple(sorted(set(endpoints))) != endpoints:
        raise ValueError("Oracle prefix endpoints must be strictly increasing")
    if any(value <= 0 or value > original.size for value in endpoints):
        raise ValueError("Oracle prefix endpoint lies outside the trajectory")
    if any(attention[value - 1] != 1 for value in endpoints):
        raise ValueError("Oracle prefix must end on a non-padding token")
    if original_position_ids is None:
        positions = (np.cumsum(attention, dtype=np.int64) - 1).clip(min=0)
    else:
        positions = np.asarray(original_position_ids, dtype=np.int64)
    if positions.shape != original.shape:
        raise ValueError("Oracle position IDs must align with trajectory input")

    answer = select_canonical_answer(canonical_answer)
    target = encoded_target or encode_exact_ig_target(tokenizer, answer)
    if target.canonical_answer != answer:
        raise RuntimeError(
            "Sequential Oracle target does not match the fixed canonical answer"
        )
    target_ids = np.asarray(target.token_ids, dtype=np.int64)
    answer_local_positions = np.flatnonzero(
        np.asarray(target.score_mask, dtype=np.bool_)
    ).astype(np.int64)
    if answer_local_positions.size != target.answer_token_count:
        raise RuntimeError("Oracle answer score mask does not match target encoding")

    scores: list[float] = []
    token_details: list[OracleTokenScore] = []
    score_token_ids: list[tuple[int, ...]] = []
    answer_log_probs_by_prefix: list[tuple[float, ...]] = []
    full_log_probs_by_prefix: list[tuple[float, ...]] = []
    actual_logits_dtype: str | None = None
    actual_log_probs_dtype: str | None = None
    model_was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad(), exact_ig_precision_context(
            model,
            precision_policy,
        ):
            for prefix_index, prefix_end in enumerate(endpoints):
                combined_ids = np.concatenate(
                    (original[:prefix_end], target_ids)
                )
                combined_attention = np.concatenate(
                    (
                        attention[:prefix_end],
                        np.ones(target_ids.size, dtype=np.int64),
                    )
                )
                last_position = int(positions[prefix_end - 1])
                combined_positions = np.concatenate(
                    (
                        positions[:prefix_end],
                        np.arange(
                            last_position + 1,
                            last_position + 1 + target_ids.size,
                            dtype=np.int64,
                        ),
                    )
                )
                input_tensor = torch.as_tensor(
                    combined_ids,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                output = model(
                    input_ids=input_tensor,
                    attention_mask=torch.as_tensor(
                        combined_attention,
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0),
                    position_ids=torch.as_tensor(
                        combined_positions,
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0),
                    use_cache=False,
                )
                logits = output.logits / float(precision_policy.temperature)
                assert_fp32_exact_ig_runtime(
                    model=model,
                    policy=precision_policy,
                    logits=logits,
                )
                actual_logits_dtype = str(logits.dtype).removeprefix("torch.")
                target_start = int(prefix_end)
                target_end = target_start + int(target_ids.size)
                target_pred_logits = logits[
                    :,
                    target_start - 1 : target_end - 1,
                    :,
                ]
                expected_full = torch.as_tensor(
                    target.token_ids,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                full_log_probs = torch.nn.functional.log_softmax(
                    target_pred_logits,
                    dim=-1,
                ).gather(
                    dim=-1,
                    index=expected_full.unsqueeze(-1),
                ).squeeze(0).squeeze(-1)
                assert_fp32_exact_ig_runtime(
                    model=model,
                    policy=precision_policy,
                    logits=logits,
                    log_probs=full_log_probs,
                )
                actual_log_probs_dtype = str(
                    full_log_probs.dtype
                ).removeprefix("torch.")
                answer_log_probs = full_log_probs[
                    target.answer_token_start : target.answer_token_end
                ]
                if answer_log_probs.numel() != target.answer_token_count:
                    raise RuntimeError(
                        "Oracle official answer range has the wrong token count"
                    )
                score = answer_log_probs.mean()
                if not bool(torch.isfinite(score).item()):
                    raise RuntimeError("Oracle produced a non-finite Phi")
                scores.append(float(score.detach().cpu().item()))
                score_token_ids.append(target.answer_token_ids)
                answer_values = tuple(
                    float(value)
                    for value in answer_log_probs.detach().cpu().tolist()
                )
                full_values = tuple(
                    float(value)
                    for value in full_log_probs.detach().cpu().tolist()
                )
                answer_log_probs_by_prefix.append(answer_values)
                full_log_probs_by_prefix.append(full_values)
                physical_answer_positions = torch.as_tensor(
                    prefix_end + answer_local_positions,
                    dtype=torch.long,
                    device=device,
                )
                predicting_positions = physical_answer_positions - 1
                for physical, logit_index, token_id, log_prob in zip(
                    physical_answer_positions.detach().cpu().tolist(),
                    predicting_positions.detach().cpu().tolist(),
                    target.answer_token_ids,
                    answer_values,
                    strict=True,
                ):
                    token_details.append(
                        OracleTokenScore(
                            prefix_index=prefix_index,
                            physical_token_index=int(physical),
                            token_id=int(token_id),
                            decoded_token=_decode_token(tokenizer, int(token_id)),
                            predicting_logit_index=int(logit_index),
                            score_mask=True,
                            token_log_prob=float(log_prob),
                        )
                    )
                del (
                    output,
                    logits,
                    target_pred_logits,
                    full_log_probs,
                    answer_log_probs,
                )
    finally:
        if model_was_training:
            model.train()

    expected_answer_tokens = len(endpoints) * target.answer_token_count
    if len(token_details) != expected_answer_tokens:
        raise RuntimeError("Oracle did not score every answer token exactly once")
    score_tuple = tuple(scores)
    immediate = immediate_ig_from_prefix_scores(score_tuple)
    return SequentialOracleResult(
        score_by_prefix=score_tuple,
        immediate_ig=immediate,
        canonical_answer=answer,
        canonical_answer_sha256=target.canonical_answer_sha256,
        score_span_hash=target.score_span_hash,
        target_token_ids=target.token_ids,
        answer_token_range=(
            target.answer_token_start,
            target.answer_token_end,
        ),
        answer_token_count=target.answer_token_count,
        scored_answer_token_count=len(token_details),
        score_token_ids_by_prefix=tuple(score_token_ids),
        answer_token_log_probs_by_prefix=tuple(answer_log_probs_by_prefix),
        full_target_log_probs_by_prefix=tuple(full_log_probs_by_prefix),
        token_scores=tuple(token_details),
        telescoping_error=telescoping_error(score_tuple, immediate),
        runtime_metadata={
            "actual_model_parameter_dtype": str(
                next(model.parameters()).dtype
            ).removeprefix("torch."),
            "actual_logits_dtype": actual_logits_dtype,
            "actual_log_probs_dtype": actual_log_probs_dtype,
            "autocast_enabled": bool(
                precision_policy.autocast_enabled
                and getattr(device, "type", str(device).split(":")[0]) == "cuda"
            ),
            "autocast_dtype": precision_policy.autocast_dtype,
            "attention_backend": (
                f"{precision_policy.attention_implementation}:"
                f"{precision_policy.sdpa_backend or 'native'}"
            ),
            "temperature": float(precision_policy.temperature),
            "allow_tf32": bool(precision_policy.allow_tf32),
            "allow_bf16_reduced_precision_reduction": bool(
                precision_policy.allow_bf16_reduced_precision_reduction
            ),
            "allow_fp16_reduced_precision_reduction": bool(
                precision_policy.allow_fp16_reduced_precision_reduction
            ),
            "float32_matmul_precision": "highest",
            "scoring_logits_mode": "official_full_logits",
        },
    )
