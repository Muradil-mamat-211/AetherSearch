from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

import numpy as np

from .masks import build_structural_attention_mask
from .position_ids import build_logical_position_ids
from .target_schema import (
    DEFAULT_TARGET_TEMPLATE,
    FAST_PATH_STRUCTURE,
    MASK_BUILDER_VERSION,
    POSITION_BUILDER_VERSION,
    EncodedExactIGTarget,
    encode_exact_ig_target,
)


@dataclass(frozen=True)
class PrefixScoreSpan:
    prefix_index: int
    prefix_end_position: int
    segment_start: int
    segment_end: int
    answer_token_positions: tuple[int, ...]
    logit_positions: tuple[int, ...]
    answer_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class VectorizedExactIGTask:
    prompt_global_id: str
    trajectory_id: str
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    answer_score_mask: np.ndarray
    canonical_target: EncodedExactIGTarget
    score_spans: tuple[PrefixScoreSpan, ...]
    prefix_count: int
    original_token_count: int
    original_attention_mask: np.ndarray
    original_position_ids: np.ndarray
    prefix_end_positions: tuple[int, ...]
    segment_starts: tuple[int, ...]
    segment_lengths: tuple[int, ...]
    maximum_extended_sequence_length: int
    maximum_position_id_exclusive: int
    fast_path_structure: str = FAST_PATH_STRUCTURE
    mask_builder_version: str = MASK_BUILDER_VERSION
    position_builder_version: str = POSITION_BUILDER_VERSION

    @property
    def canonical_answer(self) -> str:
        return self.canonical_target.canonical_answer

    @property
    def canonical_answer_hash(self) -> str:
        return self.canonical_target.canonical_answer_sha256

    @property
    def target_token_ids_hash(self) -> str:
        return self.canonical_target.token_ids_hash

    @property
    def score_span_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.canonical_target.score_span_hash.encode("ascii"))
        for span in self.score_spans:
            digest.update(str(span.prefix_index).encode("ascii"))
            digest.update(b":")
            digest.update(
                ",".join(str(value) for value in span.answer_token_positions).encode(
                    "ascii"
                )
            )
            digest.update(b"\n")
        return digest.hexdigest()

    @property
    def target_bundle_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.canonical_answer_hash,
            self.target_token_ids_hash,
            self.canonical_target.score_span_hash,
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def validate(self) -> None:
        if self.input_ids.ndim != 1:
            raise ValueError("input_ids must be rank 1")
        if self.attention_mask.shape != (self.input_ids.size, self.input_ids.size):
            raise ValueError("attention_mask shape mismatch")
        if self.position_ids.shape != self.input_ids.shape:
            raise ValueError("position_ids shape mismatch")
        if self.answer_score_mask.shape != self.input_ids.shape:
            raise ValueError("answer_score_mask shape mismatch")
        if self.answer_score_mask.dtype != np.bool_:
            raise ValueError("answer_score_mask must be boolean")
        if self.original_attention_mask.shape != (self.original_token_count,):
            raise ValueError("original_attention_mask shape mismatch")
        if self.original_position_ids.shape != (self.original_token_count,):
            raise ValueError("original_position_ids shape mismatch")
        if len(self.prefix_end_positions) != self.prefix_count:
            raise ValueError("prefix_end_positions count mismatch")
        if len(self.score_spans) != self.prefix_count:
            raise ValueError("Every prefix must have exactly one canonical GT copy")
        if len(self.segment_starts) != self.prefix_count:
            raise ValueError("segment_starts count mismatch")
        if len(self.segment_lengths) != self.prefix_count:
            raise ValueError("segment_lengths count mismatch")
        if np.any(self.answer_score_mask[: self.original_token_count]):
            raise ValueError("Original trajectory tokens cannot enter the answer score mask")

        scored_positions: set[int] = set()
        target_length = len(self.canonical_target.token_ids)
        answer_count = self.canonical_target.answer_token_count
        for span, segment_start, segment_length in zip(
            self.score_spans,
            self.segment_starts,
            self.segment_lengths,
            strict=True,
        ):
            if span.segment_start != segment_start:
                raise ValueError("GT segment metadata does not align")
            if span.segment_end != segment_start + segment_length:
                raise ValueError("GT segment end does not align")
            if segment_length != target_length:
                raise ValueError("Every prefix must append an identical GT copy")
            if len(span.answer_token_positions) != answer_count:
                raise ValueError("Answer score span has the wrong token count")
            if len(span.logit_positions) != answer_count:
                raise ValueError("Every answer token must have one predicting logit")
            if span.logit_positions != tuple(
                position - 1 for position in span.answer_token_positions
            ):
                raise ValueError("Exact-IG causal shift must use logits[p-1]")
            if span.logit_positions[0] < span.segment_start:
                raise ValueError(
                    "The first answer token must be predicted by its GT scaffold"
                )
            expected_answer_positions = tuple(
                segment_start + index
                for index, include in enumerate(self.canonical_target.score_mask)
                if include
            )
            if span.answer_token_positions != expected_answer_positions:
                raise ValueError("Answer score mask differs from target character span")
            observed = tuple(
                int(self.input_ids[position])
                for position in span.answer_token_positions
            )
            if observed != span.answer_token_ids:
                raise ValueError("Answer score positions do not align with answer IDs")
            if observed != self.canonical_target.answer_token_ids:
                raise ValueError("A GT copy changed the canonical answer tokens")
            if scored_positions.intersection(span.answer_token_positions):
                raise ValueError("A physical answer token is scored more than once")
            scored_positions.update(span.answer_token_positions)
            if not np.all(self.answer_score_mask[list(span.answer_token_positions)]):
                raise ValueError("A scored answer token is missing from score mask")
            scaffold_positions = range(
                segment_start,
                segment_start + self.canonical_target.answer_token_start,
            )
            suffix_positions = range(
                segment_start + self.canonical_target.answer_token_end,
                span.segment_end,
            )
            if any(self.answer_score_mask[position] for position in scaffold_positions):
                raise ValueError("Scaffold token entered the Exact-IG score mask")
            if any(self.answer_score_mask[position] for position in suffix_positions):
                raise ValueError("Closing tag token entered the Exact-IG score mask")

        expected_score_count = self.prefix_count * answer_count
        if int(self.answer_score_mask.sum()) != expected_score_count:
            raise ValueError("Answer-only score mask has an unexpected cardinality")
        if len(scored_positions) != expected_score_count:
            raise ValueError("Every answer token must be scored exactly once")
        if self.input_ids.size > self.maximum_extended_sequence_length:
            raise ValueError("Extended Exact-IG task exceeds its physical limit")
        if np.any(self.position_ids >= self.maximum_position_id_exclusive):
            raise ValueError("Exact-IG task exceeds its logical position limit")
        if self.fast_path_structure != FAST_PATH_STRUCTURE:
            raise ValueError("Only the official no-anchor Fast Path is permitted")


@dataclass(frozen=True)
class SequentialExactIGTask:
    """A fail-closed task whose packed Fast representation exceeds a hard limit."""

    prompt_global_id: str
    trajectory_id: str
    input_ids: np.ndarray
    canonical_target: EncodedExactIGTarget
    prefix_count: int
    original_token_count: int
    original_attention_mask: np.ndarray
    original_position_ids: np.ndarray
    prefix_end_positions: tuple[int, ...]
    projected_fast_packed_length: int
    maximum_extended_sequence_length: int
    maximum_position_id_exclusive: int
    fallback_reason: str = "single_fast_task_budget_exceeded"
    fast_path_structure: str = FAST_PATH_STRUCTURE
    mask_builder_version: str = MASK_BUILDER_VERSION
    position_builder_version: str = POSITION_BUILDER_VERSION

    @property
    def canonical_answer(self) -> str:
        return self.canonical_target.canonical_answer

    @property
    def canonical_answer_hash(self) -> str:
        return self.canonical_target.canonical_answer_sha256

    @property
    def target_token_ids_hash(self) -> str:
        return self.canonical_target.token_ids_hash

    @property
    def score_span_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.canonical_target.score_span_hash.encode("ascii"))
        for prefix_index in range(self.prefix_count):
            digest.update(f"{prefix_index}:sequential\n".encode("ascii"))
        return digest.hexdigest()

    @property
    def target_bundle_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.canonical_answer_hash,
            self.target_token_ids_hash,
            self.canonical_target.score_span_hash,
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def validate(self) -> None:
        if self.input_ids.ndim != 1 or self.input_ids.size == 0:
            raise ValueError("Sequential original input_ids must be non-empty rank 1")
        if self.input_ids.size != self.original_token_count:
            raise ValueError("Sequential task must store the original trajectory once")
        if self.original_attention_mask.shape != self.input_ids.shape:
            raise ValueError("Sequential original attention mask shape mismatch")
        if self.original_position_ids.shape != self.input_ids.shape:
            raise ValueError("Sequential original position IDs shape mismatch")
        if len(self.prefix_end_positions) != self.prefix_count:
            raise ValueError("Sequential prefix count mismatch")
        target_length = len(self.canonical_target.token_ids)
        for prefix_end in self.prefix_end_positions:
            if prefix_end <= 0 or prefix_end > self.original_token_count:
                raise ValueError("Sequential prefix endpoint is outside the trajectory")
            sequential_length = int(prefix_end) + target_length
            if sequential_length > self.maximum_extended_sequence_length:
                raise ValueError(
                    "A Sequential Exact-IG prefix exceeds the context limit"
                )
            maximum_logical_position = (
                int(self.original_position_ids[prefix_end - 1]) + target_length
            )
            if maximum_logical_position >= self.maximum_position_id_exclusive:
                raise ValueError(
                    "A Sequential Exact-IG prefix exceeds the position-ID limit"
                )
        if self.projected_fast_packed_length <= self.maximum_extended_sequence_length:
            raise ValueError(
                "Sequential-only task is invalid because its Fast representation fits"
            )
        if self.fallback_reason != "single_fast_task_budget_exceeded":
            raise ValueError("Unexpected Sequential fallback reason")


ExactIGTask: TypeAlias = VectorizedExactIGTask | SequentialExactIGTask


def assert_same_prompt_target_consistency(
    tasks: Sequence[ExactIGTask],
) -> str:
    if not tasks:
        raise ValueError("At least one Exact-IG task is required")
    prompt_ids = {task.prompt_global_id for task in tasks}
    if len(prompt_ids) != 1:
        raise ValueError("Target consistency can only be checked within one prompt")
    canonical_answers = {task.canonical_answer for task in tasks}
    hashes = {task.target_bundle_hash for task in tasks}
    if len(canonical_answers) != 1 or len(hashes) != 1:
        raise RuntimeError(
            "Canonical answer or answer-only target span changed within one prompt"
        )
    return next(iter(hashes))


class ExactIGTaskBuilder:
    def __init__(
        self,
        tokenizer: Any,
        *,
        target_template: str = DEFAULT_TARGET_TEMPLATE,
        maximum_extended_sequence_length: int,
        maximum_position_id_exclusive: int | None = None,
    ) -> None:
        if target_template != DEFAULT_TARGET_TEMPLATE:
            raise ValueError("Corrected Exact-IG locks one target scaffold")
        if maximum_extended_sequence_length <= 0:
            raise ValueError("maximum_extended_sequence_length must be positive")
        self.tokenizer = tokenizer
        self.target_template = target_template
        self.maximum_extended_sequence_length = int(
            maximum_extended_sequence_length
        )
        self.maximum_position_id_exclusive = int(
            maximum_position_id_exclusive
            if maximum_position_id_exclusive is not None
            else maximum_extended_sequence_length
        )

    def tokenize_canonical_answer(
        self,
        canonical_answer: Any,
    ) -> EncodedExactIGTarget:
        return encode_exact_ig_target(
            self.tokenizer,
            canonical_answer,
            target_template=self.target_template,
        )

    def build(
        self,
        *,
        prompt_global_id: str,
        trajectory_id: str,
        full_trajectory_input_ids: Sequence[int],
        prefix_end_positions: Sequence[int],
        canonical_answer: Any,
        original_attention_mask: Sequence[int],
        original_position_ids: Sequence[int] | None = None,
    ) -> ExactIGTask:
        original = np.asarray(full_trajectory_input_ids, dtype=np.int64)
        if original.ndim != 1 or original.size == 0:
            raise ValueError("full_trajectory_input_ids must be non-empty and rank 1")
        endpoints = tuple(int(position) for position in prefix_end_positions)
        if not endpoints:
            raise ValueError("At least the no-search prefix is required")
        if any(position <= 0 or position > original.size for position in endpoints):
            raise ValueError("Prefix endpoints must lie inside the original trajectory")
        if tuple(sorted(set(endpoints))) != endpoints:
            raise ValueError("Prefix endpoints must be strictly increasing")
        original_mask = np.asarray(original_attention_mask, dtype=np.int64)
        if original_mask.shape != original.shape:
            raise ValueError("original_attention_mask must align with the trajectory")
        if not np.all((original_mask == 0) | (original_mask == 1)):
            raise ValueError("original_attention_mask must be binary")
        if any(original_mask[position - 1] != 1 for position in endpoints):
            raise ValueError("Every prefix must end on a non-padding token")

        if original_position_ids is None:
            original_position_ids = (
                np.cumsum(original_mask, dtype=np.int64) - 1
            ).clip(min=0)
        original_positions = np.asarray(original_position_ids, dtype=np.int64)
        if original_positions.shape != original.shape:
            raise ValueError("original_position_ids must align with the trajectory")

        target = self.tokenize_canonical_answer(canonical_answer)
        target_ids = np.asarray(target.token_ids, dtype=np.int64)
        target_length = int(target_ids.size)
        extended_length = int(original.size) + len(endpoints) * target_length
        for prefix_end in endpoints:
            sequential_length = int(prefix_end) + target_length
            if sequential_length > self.maximum_extended_sequence_length:
                raise ValueError(
                    "Exact-IG Sequential prefix exceeds the configured context "
                    "limit; truncation and zero-reward substitution are forbidden"
                )
            maximum_logical_position = (
                int(original_positions[prefix_end - 1]) + target_length
            )
            if maximum_logical_position >= self.maximum_position_id_exclusive:
                raise ValueError(
                    "Exact-IG Sequential prefix exceeds the configured logical "
                    "position limit"
                )
        if extended_length > self.maximum_extended_sequence_length:
            deferred = SequentialExactIGTask(
                prompt_global_id=str(prompt_global_id),
                trajectory_id=str(trajectory_id),
                input_ids=original.copy(),
                canonical_target=target,
                prefix_count=len(endpoints),
                original_token_count=int(original.size),
                original_attention_mask=original_mask.copy(),
                original_position_ids=original_positions.copy(),
                prefix_end_positions=endpoints,
                projected_fast_packed_length=extended_length,
                maximum_extended_sequence_length=(
                    self.maximum_extended_sequence_length
                ),
                maximum_position_id_exclusive=(
                    self.maximum_position_id_exclusive
                ),
            )
            deferred.validate()
            return deferred

        extended_parts: list[np.ndarray] = [original]
        segment_starts: list[int] = []
        segment_lengths: list[int] = []
        spans: list[PrefixScoreSpan] = []
        score_mask = np.zeros(extended_length, dtype=np.bool_)
        cursor = int(original.size)
        for prefix_index, prefix_end in enumerate(endpoints):
            extended_parts.append(target_ids.copy())
            segment_starts.append(cursor)
            segment_lengths.append(target_length)
            answer_positions = tuple(
                cursor + index
                for index, include in enumerate(target.score_mask)
                if include
            )
            if not answer_positions or answer_positions[0] <= cursor:
                raise ValueError(
                    "Canonical answer must follow a non-empty scaffold in each GT copy"
                )
            score_mask[list(answer_positions)] = True
            spans.append(
                PrefixScoreSpan(
                    prefix_index=prefix_index,
                    prefix_end_position=prefix_end,
                    segment_start=cursor,
                    segment_end=cursor + target_length,
                    answer_token_positions=answer_positions,
                    logit_positions=tuple(position - 1 for position in answer_positions),
                    answer_token_ids=target.answer_token_ids,
                )
            )
            cursor += target_length

        task = VectorizedExactIGTask(
            prompt_global_id=str(prompt_global_id),
            trajectory_id=str(trajectory_id),
            input_ids=np.concatenate(extended_parts),
            attention_mask=build_structural_attention_mask(
                int(original.size),
                endpoints,
                segment_lengths,
                original_attention_mask=original_mask,
            ),
            position_ids=build_logical_position_ids(
                original_positions,
                endpoints,
                segment_lengths,
                maximum_position_id_exclusive=self.maximum_position_id_exclusive,
            ),
            answer_score_mask=score_mask,
            canonical_target=target,
            score_spans=tuple(spans),
            prefix_count=len(endpoints),
            original_token_count=int(original.size),
            original_attention_mask=original_mask.copy(),
            original_position_ids=original_positions.copy(),
            prefix_end_positions=endpoints,
            segment_starts=tuple(segment_starts),
            segment_lengths=tuple(segment_lengths),
            maximum_extended_sequence_length=self.maximum_extended_sequence_length,
            maximum_position_id_exclusive=self.maximum_position_id_exclusive,
        )
        task.validate()
        return task
