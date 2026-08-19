from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agentic_rl.exact_ig.alias_reduce import (
    immediate_ig_from_prefix_scores,
    telescoping_error,
)
from agentic_rl.exact_ig.target_schema import (
    ANSWER_SCAFFOLD_TEXT,
    DEFAULT_TARGET_TEMPLATE,
    TARGET_SCHEMA_SUFFIX,
    assert_exact_ig_checkpoint_compatible,
    encode_exact_ig_target,
    render_exact_ig_target,
    select_canonical_answer,
)
from agentic_rl.exact_ig.task_builder import (
    ExactIGTaskBuilder,
    SequentialExactIGTask,
    assert_same_prompt_target_consistency,
)
from agentic_rl.exact_ig.vectorized_scorer import VectorizedExactIGScorer


class CharacterTokenizer:
    name_or_path = "character-test-tokenizer"
    init_kwargs = {"revision": "test"}
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        result = {"input_ids": [ord(character) for character in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class BoundaryMergingTokenizer:
    """Tokenizer that merges the scaffold/answer boundary in whole-string mode."""

    name_or_path = "boundary-merging-test-tokenizer"
    init_kwargs = {}
    pad_token_id = 0

    def __init__(self) -> None:
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self.calls = 0

    def _identifier(self, piece: str) -> int:
        if piece not in self._piece_to_id:
            identifier = len(self._piece_to_id) + 1
            self._piece_to_id[piece] = identifier
            self._id_to_piece[identifier] = piece
        return self._piece_to_id[piece]

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        self.calls += 1
        pieces: list[str] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            if text[index : index + 2] == ">N":
                pieces.append(">N")
                offsets.append((index, index + 2))
                index += 2
            else:
                pieces.append(text[index])
                offsets.append((index, index + 1))
                index += 1
        result = {"input_ids": [self._identifier(piece) for piece in pieces]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join(self._id_to_piece[int(token_id)] for token_id in token_ids)


class NoOffsetCharacterTokenizer(CharacterTokenizer):
    name_or_path = "no-offset-character-test-tokenizer"

    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise NotImplementedError("offset_mapping is unavailable")
        return super().__call__(text, **kwargs)


class MaskAwareToyModel(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 256) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, position_ids, **_kwargs):
        batch, length = input_ids.shape
        if attention_mask.ndim == 4:
            allowed = (
                attention_mask[:, 0]
                if attention_mask.dtype == torch.bool
                else attention_mask[:, 0].eq(0)
            )
        elif attention_mask.ndim == 2:
            causal = torch.tril(
                torch.ones(
                    (length, length),
                    dtype=torch.bool,
                    device=input_ids.device,
                )
            )
            allowed = (
                causal.unsqueeze(0)
                & attention_mask.bool().unsqueeze(1)
            )
        else:
            raise ValueError("Unexpected attention mask rank")
        visible_ids = (
            allowed.to(torch.float32)
            * input_ids.to(torch.float32).unsqueeze(1)
        ).sum(dim=-1)
        centers = torch.remainder(
            visible_ids + position_ids.to(torch.float32),
            self.vocabulary_size,
        )
        vocabulary = torch.arange(
            self.vocabulary_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits = -(
            vocabulary.view(1, 1, -1) - centers.unsqueeze(-1)
        ).abs() / 17.0
        return SimpleNamespace(logits=logits + self.scale * 0.0)


def _builder(tokenizer=None, *, limit=10000):
    return ExactIGTaskBuilder(
        tokenizer or CharacterTokenizer(),
        maximum_extended_sequence_length=limit,
        maximum_position_id_exclusive=4096,
    )


def _task(
    *,
    trajectory_id="t",
    canonical_answer="A",
    original=(10, 11, 12, 13, 14),
    endpoints=(2, 5),
    mask=None,
):
    if mask is None:
        mask = (1,) * len(original)
    return _builder().build(
        prompt_global_id="p",
        trajectory_id=trajectory_id,
        full_trajectory_input_ids=original,
        original_attention_mask=mask,
        prefix_end_positions=endpoints,
        canonical_answer=canonical_answer,
    )


@pytest.mark.parametrize(
    "answer",
    (
        "A",
        "New York",
        "北京",
        "U.S.A.",
        " leading and trailing ",
        "answer, with punctuation!",
    ),
)
def test_target_and_score_mask_cover_only_canonical_answer(answer: str) -> None:
    tokenizer = CharacterTokenizer()
    target = encode_exact_ig_target(tokenizer, answer)
    assert DEFAULT_TARGET_TEMPLATE == (
        "<think>The retrieved evidence now supports the answer.</think>"
        "<answer>{answer}</answer>"
    )
    assert target.rendered_text == (
        ANSWER_SCAFFOLD_TEXT + answer + TARGET_SCHEMA_SUFFIX
    )
    scored = tuple(
        token_id
        for token_id, include in zip(
            target.token_ids,
            target.score_mask,
            strict=True,
        )
        if include
    )
    assert scored == target.answer_token_ids
    assert tokenizer.decode(scored) == answer
    assert sum(target.score_mask) == target.answer_token_count == len(answer)
    assert not any(target.score_mask[: target.answer_token_start])
    assert not any(target.score_mask[target.answer_token_end :])


def test_boundary_merge_uses_one_official_full_string_tokenization() -> None:
    tokenizer = BoundaryMergingTokenizer()
    target = encode_exact_ig_target(tokenizer, "New York")
    assert tokenizer.calls == 1
    assert target.span_resolution == "igpo_official_offset_covering"
    assert tokenizer.decode(target.token_ids) == render_exact_ig_target("New York")
    assert tokenizer.decode(target.answer_token_ids) == ">New York"
    assert target.left_boundary_crossing is True
    assert target.boundary_crossing_any is True
    assert sum(target.score_mask) == len(target.answer_token_ids)


def test_tokenizer_without_offsets_fails_closed() -> None:
    tokenizer = NoOffsetCharacterTokenizer()
    with pytest.raises(ValueError, match="offset_mapping"):
        encode_exact_ig_target(tokenizer, "New York")


def test_canonical_answer_is_scalar_or_first_alias_without_fallback() -> None:
    assert select_canonical_answer("primary") == "primary"
    assert select_canonical_answer(np.asarray("scalar array")) == "scalar array"
    assert select_canonical_answer(["first", "second"]) == "first"
    with pytest.raises(ValueError, match="empty"):
        select_canonical_answer([])
    with pytest.raises(ValueError, match="empty"):
        select_canonical_answer(["   ", "fallback forbidden"])
    with pytest.raises(ValueError, match=r"aliases\[0\]"):
        select_canonical_answer([1, "second"])
    task = _builder().build(
        prompt_global_id="p",
        trajectory_id="ordered-aliases",
        full_trajectory_input_ids=[10, 11],
        original_attention_mask=[1, 1],
        prefix_end_positions=[2],
        canonical_answer=["first", "second"],
    )
    assert task.canonical_answer == "first"


def test_no_anchor_extended_sequence_has_one_identical_copy_per_prefix() -> None:
    task = _task(canonical_answer="Paris")
    target = task.canonical_target.token_ids
    assert task.fast_path_structure == "official_no_anchor"
    assert task.input_ids.size == task.original_token_count + 2 * len(target)
    for start in task.segment_starts:
        assert tuple(task.input_ids[start : start + len(target)]) == target
    for span in task.score_spans:
        assert span.logit_positions == tuple(
            position - 1 for position in span.answer_token_positions
        )
        assert span.logit_positions[0] >= span.segment_start
        assert span.answer_token_positions[0] > span.segment_start


def test_structural_mask_visibility_is_prefix_specific_and_copy_isolated() -> None:
    task = _builder().build(
        prompt_global_id="p",
        trajectory_id="mask",
        full_trajectory_input_ids=[0, 10, 11, 12, 13, 14, 15, 16],
        original_attention_mask=[0, 1, 1, 1, 1, 1, 1, 1],
        original_position_ids=[0, 0, 1, 2, 3, 4, 5, 6],
        prefix_end_positions=[2, 5, 7],
        canonical_answer="x",
    )
    first_queries = [
        task.segment_starts[index]
        for index in range(task.prefix_count)
    ]
    gt0, gt1, gt2 = first_queries
    assert not task.attention_mask[gt0, 0]
    assert task.attention_mask[gt0, 1]
    assert not task.attention_mask[gt0, 2]
    assert task.attention_mask[gt1, 4]
    assert not task.attention_mask[gt1, 5]
    assert task.attention_mask[gt2, 6]
    assert not task.attention_mask[gt2, 7]
    assert not task.attention_mask[gt0, gt1]
    assert not task.attention_mask[gt1, gt0]
    own_second = gt1 + 1
    assert task.attention_mask[own_second, gt1]
    assert task.attention_mask[own_second, own_second]
    assert not task.attention_mask[gt1, own_second]


def test_logical_position_ids_match_independent_prefix_teacher_forcing() -> None:
    task = _builder().build(
        prompt_global_id="p",
        trajectory_id="positions",
        full_trajectory_input_ids=[0, 0, 10, 11, 12],
        original_attention_mask=[0, 0, 1, 1, 1],
        original_position_ids=[0, 0, 0, 1, 2],
        prefix_end_positions=[4, 5],
        canonical_answer="AB",
    )
    target_length = len(task.canonical_target.token_ids)
    for prefix_end, segment_start in zip(
        task.prefix_end_positions,
        task.segment_starts,
        strict=True,
    ):
        expected = np.arange(
            task.original_position_ids[prefix_end - 1] + 1,
            task.original_position_ids[prefix_end - 1] + 1 + target_length,
        )
        np.testing.assert_array_equal(
            task.position_ids[segment_start : segment_start + target_length],
            expected,
        )


def test_future_original_tokens_cannot_change_fast_prefix_phi() -> None:
    first = _task(
        trajectory_id="future-a",
        canonical_answer="A",
        original=(10, 11, 12, 13, 14),
        endpoints=(2,),
    )
    second = _task(
        trajectory_id="future-b",
        canonical_answer="A",
        original=(10, 11, 99, 98, 97),
        endpoints=(2,),
    )
    model = MaskAwareToyModel()
    scorer = VectorizedExactIGScorer(padding_token_id=0)
    left = scorer.score(model, first, torch.device("cpu"))
    right = scorer.score(model, second, torch.device("cpu"))
    assert left.score_by_prefix == right.score_by_prefix


def test_single_and_multi_trajectory_fast_batch_are_identical() -> None:
    tasks = (
        _task(trajectory_id="short", canonical_answer="A", endpoints=(2,)),
        _task(
            trajectory_id="long",
            canonical_answer="New York",
            original=(20, 21, 22, 23, 24, 25),
            endpoints=(2, 4, 6),
        ),
    )
    model = MaskAwareToyModel()
    scorer = VectorizedExactIGScorer(padding_token_id=0)
    singles = {
        task.trajectory_id: scorer.score(model, task, torch.device("cpu"))
        for task in tasks
    }
    batched = scorer.score_many(
        model,
        tasks,
        torch.device("cpu"),
        max_records_per_forward=2,
        max_attention_cost_per_batch=None,
        max_extended_tokens_per_batch=None,
    )
    for task in tasks:
        assert batched[task.trajectory_id].score_by_prefix == pytest.approx(
            singles[task.trajectory_id].score_by_prefix,
            abs=1.0e-12,
        )
        assert batched[task.trajectory_id].immediate_ig == pytest.approx(
            singles[task.trajectory_id].immediate_ig,
            abs=1.0e-12,
        )
    assert scorer.last_microbatch_profiles[0].batch_size == 2
    assert scorer.last_microbatch_profiles[0].padding_ratio > 0
    assert model.scale.grad is None


def test_same_prompt_target_is_immutable_across_rollouts_and_prefixes() -> None:
    first = _task(trajectory_id="one", canonical_answer="Paris")
    second = _task(
        trajectory_id="two",
        canonical_answer="Paris",
        original=(20, 21, 22),
        endpoints=(1, 3),
    )
    assert assert_same_prompt_target_consistency([first, second])
    changed = _task(trajectory_id="three", canonical_answer="City of Paris")
    with pytest.raises(RuntimeError, match="changed"):
        assert_same_prompt_target_consistency([first, changed])


def test_fast_context_overflow_builds_explicit_sequential_fallback() -> None:
    target_length = len(encode_exact_ig_target(CharacterTokenizer(), "A").token_ids)
    original = [10, 11, 12, 13, 14]
    limit = len(original) + target_length
    task = _builder(limit=limit).build(
        prompt_global_id="p",
        trajectory_id="fast-overflow",
        full_trajectory_input_ids=original,
        original_attention_mask=[1] * len(original),
        prefix_end_positions=[2, 5],
        canonical_answer="A",
    )
    assert isinstance(task, SequentialExactIGTask)
    assert task.projected_fast_packed_length > limit
    assert task.input_ids.tolist() == original
    assert task.fallback_reason == "single_fast_task_budget_exceeded"


def test_sequential_context_overflow_fails_closed_without_truncation() -> None:
    with pytest.raises(ValueError, match="Sequential prefix exceeds"):
        _builder(limit=12).build(
            prompt_global_id="p",
            trajectory_id="overflow",
            full_trajectory_input_ids=[10, 11],
            original_attention_mask=[1, 1],
            prefix_end_positions=[2],
            canonical_answer="Paris",
        )


def test_log_prob_diff_telescopes_without_probability_transform() -> None:
    prefix_scores = (-4.0, -3.5, -2.0)
    rewards = immediate_ig_from_prefix_scores(prefix_scores)
    assert rewards == (0.5, 1.5)
    assert math.isclose(telescoping_error(prefix_scores, rewards), 0.0)
    reducer_source = inspect.getsource(immediate_ig_from_prefix_scores)
    assert "exp(" not in reducer_source
    assert "sigmoid" not in reducer_source.lower()


def test_old_exact_ig_checkpoint_schema_is_rejected() -> None:
    from agentic_rl.config import load_config
    from config_support import TEST_CONFIG

    current = load_config(TEST_CONFIG)
    old = {"exact_ig": dict(current["exact_ig"])}
    old["exact_ig"]["exact_ig_version"] = "legacy_anchor_multi_alias"
    with pytest.raises(RuntimeError, match="incompatible"):
        assert_exact_ig_checkpoint_compatible(old, current)
    missing_policy = {"exact_ig": dict(current["exact_ig"])}
    missing_policy["exact_ig"].pop("target_tokenization_policy")
    with pytest.raises(RuntimeError, match="incompatible"):
        assert_exact_ig_checkpoint_compatible(missing_policy, current)
    wrong_commit = {"exact_ig": dict(current["exact_ig"])}
    wrong_commit["exact_ig"]["official_igpo_commit_sha"] = "legacy"
    with pytest.raises(RuntimeError, match="incompatible"):
        assert_exact_ig_checkpoint_compatible(wrong_commit, current)
    wrong_suffix = {"exact_ig": dict(current["exact_ig"])}
    wrong_suffix["exact_ig"]["target_schema_suffix"] = "</legacy>"
    with pytest.raises(RuntimeError, match="incompatible"):
        assert_exact_ig_checkpoint_compatible(wrong_suffix, current)
    assert_exact_ig_checkpoint_compatible(current, current)


def test_active_exact_ig_source_has_no_alias_max_or_anchor_path() -> None:
    from agentic_rl.exact_ig import task_builder, vectorized_scorer

    source = inspect.getsource(task_builder) + inspect.getsource(vectorized_scorer)
    assert "max_alias" not in source
    assert "alias_index" not in source
    assert "anchor_position" not in source
    assert "segmented_boundary_safe" not in source
    assert "segmented_no_offset_mapping" not in source
    assert "torch.exp(" not in source
    assert "sigmoid" not in source.lower()
