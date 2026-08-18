from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EXACT_IG_VERSION = "exact_ig_official_offset_fp32_no_anchor_v4"
OFFICIAL_IGPO_COMMIT_SHA = "64165e2741ed8801f977948c8128080ce87b4101"
ANSWER_SCAFFOLD_TEXT = (
    "<think>The retrieved evidence now supports the answer.</think><answer>"
)
TARGET_SCHEMA_PREFIX = ANSWER_SCAFFOLD_TEXT
TARGET_SCHEMA_SUFFIX = "</answer>"
DEFAULT_TARGET_TEMPLATE = ANSWER_SCAFFOLD_TEXT + "{answer}" + TARGET_SCHEMA_SUFFIX
CANONICAL_ALIAS_POLICY = "first"
SCORE_MASK_POLICY = "igpo_official_answer_covering_span"
INFO_GAIN_TYPE = "log_prob_diff"
FAST_PATH_STRUCTURE = "official_no_anchor"
TARGET_TOKENIZATION_POLICY = "official_full_string_single_tokenization"
ANSWER_SPAN_RESOLUTION_POLICY = "igpo_official_offset_covering"
PRODUCTION_PRECISION_MODE = "fp32_exact_ig"
MASK_BUILDER_VERSION = "exact_ig_structural_mask_official_no_anchor_v4"
POSITION_BUILDER_VERSION = "exact_ig_logical_positions_official_no_anchor_v4"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


SCAFFOLD_SHA256 = _sha256_text(ANSWER_SCAFFOLD_TEXT)


@dataclass(frozen=True)
class EncodedExactIGTarget:
    canonical_answer: str
    rendered_text: str
    token_ids: tuple[int, ...]
    answer_token_start: int
    answer_token_end: int
    answer_token_ids: tuple[int, ...]
    score_mask: tuple[bool, ...]
    span_resolution: str
    offset_mapping: tuple[tuple[int, int], ...]
    answer_char_start: int
    answer_char_end: int
    left_boundary_crossing: bool
    right_boundary_crossing: bool
    boundary_crossing_any: bool
    token_ids_hash: str
    full_target_token_ids_sha256: str
    answer_span_token_ids_sha256: str
    canonical_answer_sha256: str
    score_span_hash: str

    @property
    def answer_token_count(self) -> int:
        return len(self.answer_token_ids)


def token_ids_hash(token_ids: Sequence[int]) -> str:
    serialized = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def select_canonical_answer(value: Any) -> str:
    """Select one immutable Exact-IG answer without fallback or alias switching."""

    if not isinstance(value, str) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        answer = value
    else:
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            raise ValueError(
                "Exact-IG ground truth must be a string or an ordered answer sequence"
            )
        if len(value) < 1:
            raise ValueError("Exact-IG answer sequence is empty")
        first = value[0]
        if not isinstance(first, str):
            raise ValueError("Exact-IG aliases[0] must be a string")
        answer = first
    if not answer.strip():
        raise ValueError("Exact-IG canonical answer is empty")
    return answer


def render_exact_ig_target(
    canonical_answer: str,
    *,
    target_template: str = DEFAULT_TARGET_TEMPLATE,
) -> str:
    if target_template != DEFAULT_TARGET_TEMPLATE:
        raise ValueError("Corrected Exact-IG locks one target scaffold")
    answer = select_canonical_answer(canonical_answer)
    return ANSWER_SCAFFOLD_TEXT + answer + TARGET_SCHEMA_SUFFIX


def _as_token_id_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Exact-IG tokenizer unexpectedly returned a batch")
        value = value[0]
    return list(value)


def _as_offset_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        value
        and isinstance(value[0], (list, tuple))
        and value[0]
        and isinstance(value[0][0], (list, tuple))
    ):
        if len(value) != 1:
            raise ValueError("Exact-IG tokenizer unexpectedly returned a batch")
        value = value[0]
    return list(value)


def _tokenize_complete_target_once(
    tokenizer: Any,
    text: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError(
            "Exact-IG requires one complete-target tokenization with offset_mapping"
        ) from exc
    ids = tuple(int(item) for item in _as_token_id_list(encoded["input_ids"]))
    if not ids:
        raise ValueError("Exact-IG tokenization is empty")
    if "offset_mapping" not in encoded:
        raise ValueError("Exact-IG tokenizer did not return offset_mapping")
    raw_offsets = _as_offset_list(encoded["offset_mapping"])
    mapped = tuple((int(start), int(end)) for start, end in raw_offsets)
    if len(mapped) != len(ids):
        raise ValueError("Exact-IG token IDs and offsets do not align")
    return ids, mapped


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _official_answer_token_range(
    offsets: tuple[tuple[int, int], ...],
    answer_char_start: int,
    answer_char_end: int,
) -> tuple[int, int]:
    """Reproduce IGPO's answer-covering offset loop exactly."""

    token_start: int | None = None
    token_end: int | None = None
    for token_index, (char_start, char_end) in enumerate(offsets):
        if token_start is None and char_end > answer_char_start:
            token_start = token_index
        if char_start < answer_char_end and char_end > 0:
            token_end = token_index + 1
    if token_start is None:
        token_start = len(offsets)
    if token_end is None:
        token_end = len(offsets)
    return token_start, token_end


def encode_exact_ig_target(
    tokenizer: Any,
    canonical_answer: str,
    *,
    target_template: str = DEFAULT_TARGET_TEMPLATE,
) -> EncodedExactIGTarget:
    answer = select_canonical_answer(canonical_answer)
    rendered = render_exact_ig_target(answer, target_template=target_template)
    token_ids, offsets = _tokenize_complete_target_once(tokenizer, rendered)
    if _decode(tokenizer, token_ids) != rendered:
        raise ValueError(
            "Exact-IG complete target token IDs do not decode to rendered_target"
        )
    answer_char_start = len(ANSWER_SCAFFOLD_TEXT)
    answer_char_end = answer_char_start + len(answer)
    answer_start, answer_end = _official_answer_token_range(
        offsets,
        answer_char_start,
        answer_char_end,
    )
    if not 0 <= answer_start < answer_end <= len(token_ids):
        raise ValueError(
            "IGPO official offset algorithm produced an empty or invalid answer span"
        )
    span_resolution = ANSWER_SPAN_RESOLUTION_POLICY
    answer_token_ids = token_ids[answer_start:answer_end]
    if not answer_token_ids:
        raise ValueError("Exact-IG canonical answer has no scoreable token")
    left_offset = offsets[answer_start]
    right_offset = offsets[answer_end - 1]
    left_boundary_crossing = (
        left_offset[0] < answer_char_start < left_offset[1]
    )
    right_boundary_crossing = (
        right_offset[0] < answer_char_end < right_offset[1]
    )
    score_mask = tuple(
        answer_start <= index < answer_end for index in range(len(token_ids))
    )
    if sum(score_mask) != len(answer_token_ids):
        raise RuntimeError("Exact-IG answer score mask cardinality is inconsistent")
    span_digest = hashlib.sha256()
    span_digest.update(str(answer_start).encode("ascii"))
    span_digest.update(b":")
    span_digest.update(str(answer_end).encode("ascii"))
    span_digest.update(b":")
    span_digest.update(token_ids_hash(answer_token_ids).encode("ascii"))
    return EncodedExactIGTarget(
        canonical_answer=answer,
        rendered_text=rendered,
        token_ids=token_ids,
        answer_token_start=answer_start,
        answer_token_end=answer_end,
        answer_token_ids=answer_token_ids,
        score_mask=score_mask,
        span_resolution=span_resolution,
        offset_mapping=offsets,
        answer_char_start=answer_char_start,
        answer_char_end=answer_char_end,
        left_boundary_crossing=left_boundary_crossing,
        right_boundary_crossing=right_boundary_crossing,
        boundary_crossing_any=(
            left_boundary_crossing or right_boundary_crossing
        ),
        token_ids_hash=token_ids_hash(token_ids),
        full_target_token_ids_sha256=token_ids_hash(token_ids),
        answer_span_token_ids_sha256=token_ids_hash(answer_token_ids),
        canonical_answer_sha256=_sha256_text(answer),
        score_span_hash=span_digest.hexdigest(),
    )


def exact_ig_schema_hash(tokenizer: Any, canonical_answer: str) -> str:
    target = encode_exact_ig_target(tokenizer, canonical_answer)
    digest = hashlib.sha256()
    for value in (
        EXACT_IG_VERSION,
        DEFAULT_TARGET_TEMPLATE,
        CANONICAL_ALIAS_POLICY,
        SCORE_MASK_POLICY,
        INFO_GAIN_TYPE,
        FAST_PATH_STRUCTURE,
        TARGET_TOKENIZATION_POLICY,
        PRODUCTION_PRECISION_MODE,
        target.canonical_answer_sha256,
        target.token_ids_hash,
        target.score_span_hash,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def exact_ig_tokenizer_identity(tokenizer: Any) -> tuple[str, str]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    revision = getattr(tokenizer, "revision", None)
    if revision is None and isinstance(init_kwargs, Mapping):
        revision = init_kwargs.get("revision")
    return (
        str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__)),
        str(revision or "unknown"),
    )


def exact_ig_static_metadata(tokenizer: Any, canonical_answer: str) -> dict[str, Any]:
    target = encode_exact_ig_target(tokenizer, canonical_answer)
    tokenizer_name, tokenizer_revision = exact_ig_tokenizer_identity(tokenizer)
    return {
        "exact_ig_version": EXACT_IG_VERSION,
        "scaffold_text": ANSWER_SCAFFOLD_TEXT,
        "scaffold_sha256": SCAFFOLD_SHA256,
        "canonical_alias_policy": CANONICAL_ALIAS_POLICY,
        "canonical_answer_sha256": target.canonical_answer_sha256,
        "score_mask_policy": SCORE_MASK_POLICY,
        "info_gain_type": INFO_GAIN_TYPE,
        "fast_path_structure": FAST_PATH_STRUCTURE,
        "target_tokenization_policy": TARGET_TOKENIZATION_POLICY,
        "answer_span_resolution_policy": ANSWER_SPAN_RESOLUTION_POLICY,
        "production_precision_mode": PRODUCTION_PRECISION_MODE,
        "official_igpo_commit_sha": OFFICIAL_IGPO_COMMIT_SHA,
        "tokenizer_name_or_path": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "score_span_hash": target.score_span_hash,
        "answer_char_start": target.answer_char_start,
        "answer_char_end": target.answer_char_end,
        "answer_token_start": target.answer_token_start,
        "answer_token_end": target.answer_token_end,
        "answer_token_count": target.answer_token_count,
        "left_boundary_crossing": target.left_boundary_crossing,
        "right_boundary_crossing": target.right_boundary_crossing,
        "boundary_crossing_any": target.boundary_crossing_any,
        "full_target_token_ids_sha256": target.full_target_token_ids_sha256,
        "answer_span_token_ids_sha256": target.answer_span_token_ids_sha256,
        "mask_builder_version": MASK_BUILDER_VERSION,
        "position_builder_version": POSITION_BUILDER_VERSION,
        "target_token_ids_hash": target.token_ids_hash,
        "span_resolution": target.span_resolution,
    }


def assert_exact_ig_checkpoint_compatible(
    checkpoint_algorithm_config: Mapping[str, Any],
    current_algorithm_config: Mapping[str, Any],
) -> None:
    """Reject state created with a different Exact-IG reward definition."""

    checkpoint_exact = checkpoint_algorithm_config.get("exact_ig")
    current_exact = current_algorithm_config.get("exact_ig")
    if not isinstance(checkpoint_exact, Mapping):
        raise RuntimeError(
            "Checkpoint has no auditable Exact-IG config; state reuse is forbidden"
        )
    if not isinstance(current_exact, Mapping):
        raise RuntimeError("Current runtime has no Exact-IG configuration")
    required = {
        "exact_ig_version": EXACT_IG_VERSION,
        "official_igpo_commit_sha": OFFICIAL_IGPO_COMMIT_SHA,
        "scaffold_text": ANSWER_SCAFFOLD_TEXT,
        "scaffold_sha256": SCAFFOLD_SHA256,
        "target_template": DEFAULT_TARGET_TEMPLATE,
        "target_schema_prefix": TARGET_SCHEMA_PREFIX,
        "target_schema_suffix": TARGET_SCHEMA_SUFFIX,
        "canonical_alias_policy": CANONICAL_ALIAS_POLICY,
        "score_mask_policy": SCORE_MASK_POLICY,
        "score_answer_body_tokens_only": True,
        "info_gain_type": INFO_GAIN_TYPE,
        "fast_path_structure": FAST_PATH_STRUCTURE,
        "target_tokenization_policy": TARGET_TOKENIZATION_POLICY,
        "answer_span_resolution": ANSWER_SPAN_RESOLUTION_POLICY,
        "mask_builder_version": MASK_BUILDER_VERSION,
        "position_builder_version": POSITION_BUILDER_VERSION,
        "production_precision_mode": PRODUCTION_PRECISION_MODE,
        "parameter_dtype": "float32",
        "activation_dtype": "float32",
        "logits_dtype": "float32",
        "log_probs_dtype": "float32",
        "autocast_enabled": False,
        "autocast_dtype": None,
        "allow_tf32": False,
        "allow_bf16_reduced_precision_reduction": False,
        "allow_fp16_reduced_precision_reduction": False,
    }
    for key, expected in required.items():
        checkpoint_value = checkpoint_exact.get(key)
        current_value = current_exact.get(key)
        if current_value != expected:
            raise RuntimeError(
                f"Current Exact-IG contract is invalid at {key}: "
                f"{current_value!r} != {expected!r}"
            )
        if checkpoint_value != expected:
            raise RuntimeError(
                "Checkpoint Exact-IG schema is incompatible with corrected training: "
                f"{key}={checkpoint_value!r}, required={expected!r}"
            )
