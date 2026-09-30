from __future__ import annotations

import copy
import ast
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from agentic_rl.exact_ig.precision_policy import production_precision_policy
from agentic_rl.exact_ig.sequential_oracle import sequential_teacher_forced_oracle
from agentic_rl.exact_ig.target_schema import (
    DEFAULT_TARGET_TEMPLATE,
    EXACT_IG_VERSION,
    PRIOR_TARGET_TEMPLATE,
    TARGET_STATE_POLICY,
    assert_exact_ig_checkpoint_compatible,
    encode_exact_ig_target,
)
from agentic_rl.exact_ig.task_builder import ExactIGTaskBuilder, SequentialExactIGTask
from agentic_rl.exact_ig.vectorized_scorer import VectorizedExactIGScorer
from support.exact_ig_fast_path_audit import audit_task_contract
from test_exact_ig_structure import CharacterTokenizer, MaskAwareToyModel


class StateMergingTokenizer(CharacterTokenizer):
    """Exercise state-dependent answer token counts, not just scaffold lengths."""

    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        if text.startswith("<think>Reliable"):
            start = text.index("Paris")
            result["input_ids"][start:start + 2] = [255]
            result["offset_mapping"][start:start + 2] = [(start, start + 2)]
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join("Pa" if int(value) == 255 else chr(int(value)) for value in token_ids)


def build(tokenizer=None, *, endpoints=(2, 4, 6), limit=2048):
    return ExactIGTaskBuilder(
        tokenizer or CharacterTokenizer(),
        maximum_extended_sequence_length=limit,
        maximum_position_id_exclusive=4096,
    ).build(
        prompt_global_id="p", trajectory_id="t",
        full_trajectory_input_ids=[10, 11, 12, 13, 14, 15],
        original_attention_mask=[1] * 6,
        prefix_end_positions=endpoints, canonical_answer="Paris",
    )


@pytest.mark.parametrize("endpoints", [(2,), (2, 4), (2, 4, 6)])
@pytest.mark.parametrize("tokenizer_type", [CharacterTokenizer, StateMergingTokenizer])
def test_packed_and_sequential_state_targets_agree(endpoints, tokenizer_type):
    tokenizer = tokenizer_type()
    task = build(tokenizer, endpoints=endpoints)
    expected = [PRIOR_TARGET_TEMPLATE] + [DEFAULT_TARGET_TEMPLATE] * (len(endpoints) - 1)
    for start, target, template in zip(task.segment_starts, task.targets_by_prefix, expected, strict=True):
        assert target.rendered_text == template.format(answer="Paris")
        assert tuple(task.input_ids[start:start + len(target.token_ids)]) == target.token_ids
        assert tokenizer.decode(target.answer_token_ids) == "Paris"
    audit = audit_task_contract(task)
    assert all(audit[key] for key in (
        "packed_structure_pass", "attention_mask_exhaustive_pass",
        "position_ids_pass", "p_minus_one_shift_pass", "no_anchor_pass",
    ))
    policy = production_precision_policy("fp32_exact_ig")
    model = MaskAwareToyModel()
    scorer = VectorizedExactIGScorer(precision_policy=policy, tokenizer=tokenizer, padding_token_id=0)
    fast = scorer.score(model, task, torch.device("cpu"))
    oracle = sequential_teacher_forced_oracle(
        model=model, tokenizer=tokenizer,
        full_trajectory_input_ids=task.input_ids[:task.original_token_count],
        original_attention_mask=task.original_attention_mask,
        prefix_end_positions=endpoints, canonical_answer="Paris",
        device=torch.device("cpu"), precision_policy=policy,
    )
    assert fast.score_by_prefix == pytest.approx(oracle.score_by_prefix, abs=1e-6)
    assert fast.immediate_ig == pytest.approx(oracle.immediate_ig, abs=1e-6)
    assert len(fast.immediate_ig) == len(endpoints) - 1
    assert abs(fast.telescoping_error) < 1e-6
    assert fast.runtime_metadata["target_schema_by_prefix"] == ["prior_knowledge"] + ["retrieved_evidence"] * (len(endpoints) - 1)
    assert oracle.scored_answer_token_count == sum(target.answer_token_count for target in task.targets_by_prefix)
    assert fast.score_token_ids_by_prefix == tuple(target.answer_token_ids for target in task.targets_by_prefix)


@pytest.mark.parametrize("tokenizer_type", [CharacterTokenizer, StateMergingTokenizer])
def test_sequential_fallback_has_identical_state_semantics(tokenizer_type):
    tokenizer = tokenizer_type()
    fast_task = build(tokenizer)
    limit = max(end + len(target.token_ids) for end, target in zip(fast_task.prefix_end_positions, fast_task.targets_by_prefix, strict=True))
    fallback = build(tokenizer, limit=limit)
    assert isinstance(fallback, SequentialExactIGTask)
    assert fallback.projected_fast_packed_length == fast_task.input_ids.size
    policy = production_precision_policy("fp32_exact_ig")
    scorer = VectorizedExactIGScorer(precision_policy=policy, tokenizer=tokenizer, padding_token_id=0)
    model = MaskAwareToyModel()
    fast = scorer.score(model, fast_task, torch.device("cpu"))
    slow = scorer._score_sequential_fallback(model, fallback, torch.device("cpu"))
    assert fast.score_by_prefix == pytest.approx(slow.score_by_prefix, abs=1e-6)
    assert fast.target_score_span_hash == slow.target_score_span_hash
    assert fast.runtime_metadata["target_token_ids_hash_by_prefix"] == slow.runtime_metadata["target_token_ids_hash_by_prefix"]


def test_prior_length_is_used_for_context_and_logical_limits():
    task = build(endpoints=(2,))
    short_limit = 2 + len(task.canonical_target.token_ids)
    assert short_limit < 2 + len(task.prior_target.token_ids)
    with pytest.raises(ValueError, match="Sequential prefix exceeds"):
        build(endpoints=(2,), limit=short_limit)
    invalid = replace(task, maximum_position_id_exclusive=2 + len(task.prior_target.token_ids) - 1)
    with pytest.raises(ValueError, match="logical position limit"):
        invalid.validate()


def test_bundle_tracks_both_states_but_not_number_of_searches():
    direct, multi = build(endpoints=(2,)), build()
    assert direct.target_bundle_hash == multi.target_bundle_hash
    assert direct.target_bundle_hash != replace(direct, prior_target=direct.canonical_target).target_bundle_hash
    assert direct.target_bundle_hash != replace(direct, canonical_target=direct.prior_target).target_bundle_hash


def test_invalid_state_or_prefix_metadata_is_rejected():
    task = build()
    with pytest.raises(ValueError, match="retrieval-state schema"):
        replace(task, prior_target=task.canonical_target).validate()
    spans = (replace(task.score_spans[0], prefix_index=1), *task.score_spans[1:])
    with pytest.raises(ValueError, match="prefix state"):
        replace(task, score_spans=spans).validate()


def test_rl_scaffolds_match_sft_export_constants():
    root = Path(__file__).resolve().parents[1]
    teacher = root / "sft/data_generation/search_sft_teacher/controlled_rollout.py"
    constants = {}
    for node in ast.parse(teacher.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"RETRIEVED_FINAL_THINK", "DIRECT_FINAL_THINK"}:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    for key, template in (
        ("RETRIEVED_FINAL_THINK", DEFAULT_TARGET_TEMPLATE),
        ("DIRECT_FINAL_THINK", PRIOR_TARGET_TEMPLATE),
    ):
        assert template == "<think>" + constants[key] + "</think><answer>{answer}</answer>"


def test_v4_and_mismatched_state_policy_checkpoints_are_rejected():
    from config_support import load_exact_ig_contract

    current = load_exact_ig_contract()
    assert current["exact_ig"]["exact_ig_version"] == EXACT_IG_VERSION
    assert_exact_ig_checkpoint_compatible(current, current)
    for key, value in (
        ("exact_ig_version", "exact_ig_official_offset_fp32_no_anchor_v4"),
        ("target_state_policy", "always_retrieved"),
        ("prior_target_template", DEFAULT_TARGET_TEMPLATE),
    ):
        old = copy.deepcopy(current)
        old["exact_ig"][key] = value
        with pytest.raises(RuntimeError, match="incompatible"):
            assert_exact_ig_checkpoint_compatible(old, current)
    assert current["exact_ig"]["target_state_policy"] == TARGET_STATE_POLICY


def test_actual_student_tokenizer_encodes_each_state_once():
    model_path = os.environ.get("AETHERSEARCH_ACTOR_MODEL")
    if not model_path:
        pytest.skip("Set AETHERSEARCH_ACTOR_MODEL to the local student tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    for answer in ("Paris", "New York", " 1972", "<answer>"):
        for template in (PRIOR_TARGET_TEMPLATE, DEFAULT_TARGET_TEMPLATE):
            target = encode_exact_ig_target(tokenizer, answer, target_template=template)
            assert tokenizer.decode(target.token_ids) == template.format(answer=answer)
            assert target.answer_char_start == len(template.split("{answer}")[0])
            assert target.answer_char_end == target.answer_char_start + len(answer)
            assert np.count_nonzero(target.score_mask) == target.answer_token_count
