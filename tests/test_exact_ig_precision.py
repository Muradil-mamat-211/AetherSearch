from __future__ import annotations

import inspect
import math
import os
from types import SimpleNamespace

import torch

from agentic_rl.config import DEFAULT_CONFIG, load_config
from agentic_rl.exact_ig.precision_policy import (
    ExactIGPrecisionPolicy,
    exact_ig_precision_context,
    production_precision_policy,
)
from agentic_rl.exact_ig.sequential_oracle import (
    sequential_teacher_forced_oracle,
)
from agentic_rl.exact_ig.target_schema import (
    ANSWER_SCAFFOLD_TEXT,
    TARGET_SCHEMA_SUFFIX,
    encode_exact_ig_target,
    render_exact_ig_target,
)


class CharacterTokenizer:
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


class ZeroLogitModel(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 256) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(
            logits=self.scale
            * torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.vocabulary_size),
                dtype=torch.float32,
                device=input_ids.device,
            )
        )


def _cpu_policy() -> ExactIGPrecisionPolicy:
    return ExactIGPrecisionPolicy(
        mode="cpu_oracle_test",
        autocast_enabled=False,
        autocast_dtype=None,
        temperature=1.0,
        attention_implementation="eager",
        sdpa_backend=None,
    )


def test_sequential_oracle_is_logically_independent_from_fast_path() -> None:
    source = inspect.getsource(sequential_teacher_forced_oracle)
    assert "task_builder" not in source
    assert "VectorizedExactIGTask" not in source
    assert "build_structural_attention_mask" not in source
    assert "score_spans" not in source
    assert "target_start - 1 : target_end - 1" in source
    assert "expected_full" in source
    assert "torch.nn.functional.log_softmax" in source
    assert "build_structural_attention_mask" not in source


def test_sequential_oracle_scores_answer_body_and_uses_p_minus_one() -> None:
    tokenizer = CharacterTokenizer()
    model = ZeroLogitModel()
    result = sequential_teacher_forced_oracle(
        model=model,
        tokenizer=tokenizer,
        full_trajectory_input_ids=[11, 12, 13],
        original_attention_mask=[1, 1, 1],
        original_position_ids=[0, 1, 2],
        prefix_end_positions=[2, 3],
        canonical_answer="AB",
        device=torch.device("cpu"),
        precision_policy=_cpu_policy(),
    )
    assert result.answer_token_count == 2
    assert result.scored_answer_token_count == 4
    assert result.score_token_ids_by_prefix == (
        (ord("A"), ord("B")),
        (ord("A"), ord("B")),
    )
    assert all(
        math.isclose(value, -math.log(256), abs_tol=1.0e-6)
        for value in result.score_by_prefix
    )
    first = result.token_scores[0]
    assert first.physical_token_index == (
        2 + len(ANSWER_SCAFFOLD_TEXT)
    )
    assert first.predicting_logit_index == first.physical_token_index - 1
    assert first.token_id == ord("A")
    assert first.score_mask is True
    assert model.scale.grad is None


def test_target_schema_is_exactly_locked() -> None:
    assert render_exact_ig_target("Paris") == (
        ANSWER_SCAFFOLD_TEXT + "Paris" + TARGET_SCHEMA_SUFFIX
    )
    assert ANSWER_SCAFFOLD_TEXT == (
        "<think>The retrieved evidence now supports the answer.</think><answer>"
    )
    assert TARGET_SCHEMA_SUFFIX == "</answer>"


def test_real_dpo_tokenizer_uses_official_answer_covering_offsets() -> None:
    from transformers import AutoTokenizer

    model_path = os.environ.get("AETHERSEARCH_ACTOR_MODEL", "")
    if not model_path:
        import pytest

        pytest.skip("AETHERSEARCH_ACTOR_MODEL is required for asset-backed tests")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    for answer in ("New York", "北京", "U.S.A.", " leading space"):
        target = encode_exact_ig_target(tokenizer, answer)
        assert tokenizer.decode(
            list(target.token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) == target.rendered_text
        assert sum(target.score_mask) == target.answer_token_count
        assert target.answer_token_ids == target.token_ids[
            target.answer_token_start : target.answer_token_end
        ]


def test_fp32_policy_disables_reduced_precision_and_preserves_model_dtype() -> None:
    policy = production_precision_policy("fp32_exact_ig")
    assert policy.autocast_enabled is False
    assert policy.autocast_dtype is None
    assert policy.temperature == 1.0
    assert policy.attention_implementation == "sdpa"
    assert policy.sdpa_backend == "math"
    assert policy.allow_tf32 is False
    assert policy.allow_bf16_reduced_precision_reduction is False
    assert policy.allow_fp16_reduced_precision_reduction is False
    model = ZeroLogitModel()
    dtype_before = next(model.parameters()).dtype
    with exact_ig_precision_context(model, policy):
        assert next(model.parameters()).dtype is dtype_before
    assert next(model.parameters()).dtype is dtype_before


def test_corrected_exact_ig_config_contract() -> None:
    config = load_config(DEFAULT_CONFIG)
    exact = config["exact_ig"]
    assert exact["production_precision_mode"] == "fp32_exact_ig"
    assert exact["exact_ig_version"] == (
        "exact_ig_official_offset_fp32_no_anchor_v4"
    )
    assert exact["canonical_alias_policy"] == "first"
    assert exact["score_mask_policy"] == "igpo_official_answer_covering_span"
    assert exact["info_gain_type"] == "log_prob_diff"
    assert exact["fast_path_structure"] == "official_no_anchor"
    assert exact["target_template"] == (
        "<think>The retrieved evidence now supports the answer.</think>"
        "<answer>{answer}</answer>"
    )
    assert exact["parameter_dtype"] == "float32"
    assert exact["activation_dtype"] == "float32"
    assert exact["logits_dtype"] == "float32"
    assert exact["log_probs_dtype"] == "float32"
    assert exact["autocast_enabled"] is False
    assert exact["autocast_dtype"] is None
    assert exact["allow_tf32"] is False
    assert exact["temperature"] == 1.0
    assert exact["scoring_logits_mode"] == "official_full_logits"
    assert exact["selected_positions_enabled"] is False
    assert exact["encode_complete_target_once_per_prompt"] is True
    assert (
        exact["target_tokenization_policy"]
        == "official_full_string_single_tokenization"
    )
    assert exact["answer_span_resolution"] == "igpo_official_offset_covering"
    assert exact["parity_rtol"] == 1.0e-5
    assert exact["parity_atol"] == 2.0e-5
    assert exact["maximum_token_log_prob_abs_diff"] == 2.0e-5
    assert exact["maximum_phi_abs_diff"] == 2.0e-5
    assert exact["maximum_ig_abs_diff"] == 2.0e-5
    assert exact["maximum_telescoping_error"] == 1.0e-10
    assert exact["logits_element_size"] == 4
    assert exact["numerical_gate_status"] == "PASS"
    assert exact["structural_audit_status"] == "PASS"
    assert exact["maximum_phi_safety_abs_diff"] == 1.0e-3
    assert exact["maximum_ig_safety_abs_diff"] == 1.0e-3
