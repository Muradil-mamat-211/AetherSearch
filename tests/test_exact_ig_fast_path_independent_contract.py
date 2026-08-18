from __future__ import annotations

import inspect
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from agentic_rl.exact_ig.target_schema import (
    ANSWER_SCAFFOLD_TEXT,
    TARGET_SCHEMA_SUFFIX,
    select_canonical_answer,
)
from agentic_rl.exact_ig.task_builder import ExactIGTaskBuilder
from agentic_rl.runtime.fsdp_worker import _classify_exact_ig_canary

_AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_exact_ig_fast_path_production.py"
)
_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "exact_ig_independent_auditor",
    _AUDIT_PATH,
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT)
audit_task_contract = _AUDIT.audit_task_contract
independent_answer_range = _AUDIT.independent_answer_range
independent_expected_mask = _AUDIT.independent_expected_mask
independent_expected_positions = _AUDIT.independent_expected_positions
independent_score_positions = _AUDIT.independent_score_positions


class CharacterTokenizer:
    name_or_path = "independent-contract-character-tokenizer"
    init_kwargs = {"revision": "test"}
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, **kwargs):
        self.calls += 1
        assert kwargs["add_special_tokens"] is False
        result = {"input_ids": [ord(character) for character in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _task():
    tokenizer = CharacterTokenizer()
    task = ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=4096,
        maximum_position_id_exclusive=4096,
    ).build(
        prompt_global_id="prompt-1",
        trajectory_id="trajectory-1",
        full_trajectory_input_ids=(101, 102, 103, 104, 105, 106),
        original_attention_mask=(1, 1, 1, 1, 1, 1),
        original_position_ids=(7, 8, 9, 10, 11, 12),
        prefix_end_positions=(2, 4, 6),
        canonical_answer="Paris",
    )
    return tokenizer, task


def test_independent_auditor_does_not_call_production_builders() -> None:
    sources = "\n".join(
        inspect.getsource(function)
        for function in (
            independent_answer_range,
            independent_expected_mask,
            independent_expected_positions,
            independent_score_positions,
            audit_task_contract,
        )
    )
    forbidden = (
        "build_structural_attention_mask",
        "build_logical_position_ids",
        "ExactIGTaskBuilder.build",
        "VectorizedExactIGTask.validate",
    )
    assert all(name not in sources for name in forbidden)


def test_independent_contract_matches_production_task_exhaustively() -> None:
    tokenizer, task = _task()
    result = audit_task_contract(task)
    assert tokenizer.calls == 1
    assert result["packed_structure_pass"] is True
    assert result["attention_mask_exhaustive_pass"] is True
    assert result["position_ids_pass"] is True
    assert result["p_minus_one_shift_pass"] is True
    assert result["no_anchor_pass"] is True

    target = task.canonical_target
    assert target.rendered_text == (
        ANSWER_SCAFFOLD_TEXT + "Paris" + TARGET_SCHEMA_SUFFIX
    )
    expected_range = independent_answer_range(
        target.offset_mapping,
        len(ANSWER_SCAFFOLD_TEXT),
        len(ANSWER_SCAFFOLD_TEXT) + len("Paris"),
    )
    assert expected_range == (
        target.answer_token_start,
        target.answer_token_end,
    )


def test_independent_mask_has_exact_prefix_and_copy_visibility() -> None:
    _, task = _task()
    expected = independent_expected_mask(
        original_token_count=task.original_token_count,
        original_attention_mask=task.original_attention_mask,
        prefix_end_positions=task.prefix_end_positions,
        segment_starts=task.segment_starts,
        segment_lengths=task.segment_lengths,
    )
    assert np.array_equal(expected, task.attention_mask)
    for prefix_index, (prefix_end, start, length) in enumerate(
        zip(
            task.prefix_end_positions,
            task.segment_starts,
            task.segment_lengths,
            strict=True,
        )
    ):
        for query in range(start, start + length):
            assert expected[query, :prefix_end].all()
            assert not expected[
                query, prefix_end : task.original_token_count
            ].any()
            for other_index, (other_start, other_length) in enumerate(
                zip(task.segment_starts, task.segment_lengths, strict=True)
            ):
                if other_index == prefix_index:
                    assert expected[query, other_start : query + 1].all()
                    assert not expected[
                        query, query + 1 : other_start + other_length
                    ].any()
                else:
                    assert not expected[
                        query, other_start : other_start + other_length
                    ].any()
    assert not expected[: task.original_token_count, task.original_token_count :].any()


def test_independent_positions_match_each_sequential_prefix() -> None:
    _, task = _task()
    expected = independent_expected_positions(
        original_position_ids=task.original_position_ids,
        prefix_end_positions=task.prefix_end_positions,
        segment_starts=task.segment_starts,
        segment_lengths=task.segment_lengths,
    )
    assert np.array_equal(expected, task.position_ids)
    for prefix_end, start, length in zip(
        task.prefix_end_positions,
        task.segment_starts,
        task.segment_lengths,
        strict=True,
    ):
        first = int(task.original_position_ids[prefix_end - 1]) + 1
        sequential_target_positions = np.arange(first, first + length)
        assert np.array_equal(
            task.position_ids[start : start + length],
            sequential_target_positions,
        )
        assert int(task.position_ids[start]) != start


def test_independent_score_rows_use_p_minus_one_and_answer_only() -> None:
    _, task = _task()
    answer_positions, logit_positions = independent_score_positions(
        segment_starts=task.segment_starts,
        answer_token_start=task.canonical_target.answer_token_start,
        answer_token_end=task.canonical_target.answer_token_end,
    )
    actual_answer_positions = tuple(
        position
        for span in task.score_spans
        for position in span.answer_token_positions
    )
    actual_logit_positions = tuple(
        position
        for span in task.score_spans
        for position in span.logit_positions
    )
    assert actual_answer_positions == answer_positions
    assert actual_logit_positions == logit_positions
    assert all(logit == token - 1 for token, logit in zip(
        answer_positions, logit_positions, strict=True
    ))
    assert int(task.answer_score_mask.sum()) == (
        task.prefix_count * task.canonical_target.answer_token_count
    )


def test_canonical_answer_is_strictly_first_alias() -> None:
    assert select_canonical_answer("Paris") == "Paris"
    assert select_canonical_answer(["Paris", "Paris, France"]) == "Paris"
    with pytest.raises(ValueError):
        select_canonical_answer([])
    with pytest.raises(ValueError):
        select_canonical_answer([3, "Paris"])
    with pytest.raises(ValueError):
        select_canonical_answer([" ", "Paris"])


def _canary(**overrides):
    values = {
        "token_allclose": False,
        "phi_allclose": False,
        "ig_allclose": False,
        "finite": True,
        "target_coverage": True,
        "canonical_answer_agreement": True,
        "non_ambiguous_sign_agreement": True,
        "turn_ranking_agreement": True,
        "token_error": 2.0e-4,
        "phi_error": 2.0e-4,
        "ig_error": 2.0e-4,
        "telescoping_error": 0.0,
        "telemetry_token_error": 2.0e-5,
        "telemetry_phi_error": 2.0e-5,
        "telemetry_ig_error": 2.0e-5,
        "maximum_phi_safety_error": 1.0e-3,
        "maximum_ig_safety_error": 1.0e-3,
        "maximum_telescoping_error": 1.0e-10,
    }
    values.update(overrides)
    return _classify_exact_ig_canary(**values)


def test_shape_dependent_numeric_drift_is_telemetry_not_hard_failure() -> None:
    telemetry_warning, hard_failure = _canary()
    assert telemetry_warning is True
    assert hard_failure is False


@pytest.mark.parametrize(
    "override",
    (
        {"phi_error": math.nextafter(1.0e-3, math.inf)},
        {"ig_error": math.nextafter(1.0e-3, math.inf)},
        {"finite": False},
        {"target_coverage": False},
        {"canonical_answer_agreement": False},
        {"non_ambiguous_sign_agreement": False},
        {"turn_ranking_agreement": False},
    ),
)
def test_structural_semantic_and_safety_failures_remain_hard(
    override,
) -> None:
    _, hard_failure = _canary(**override)
    assert hard_failure is True


def test_p99_drift_is_hard_only_after_calibrated_enforcement() -> None:
    _, hard_failure = _canary(
        observed_p99_ig_error=2.1e-4,
        calibration_p99_ig_error=1.0e-4,
        enforce_p99_drift=True,
    )
    assert hard_failure is True
    _, hard_failure = _canary(
        observed_p99_ig_error=2.1e-4,
        calibration_p99_ig_error=1.0e-4,
        enforce_p99_drift=False,
    )
    assert hard_failure is False
