from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agentic_rl.exact_ig.fsdp_scoring_window import (
    FSDPReshardRestoreError,
    FSDPReshardStateRegistry,
    exact_ig_scoring_window,
)
from agentic_rl.exact_ig.precision_policy import ExactIGPrecisionPolicy
from agentic_rl.exact_ig.task_builder import ExactIGTaskBuilder
from agentic_rl.exact_ig.vectorized_scorer import (
    OFFICIAL_ADDITIVE_MASK,
    BOOLEAN_4D_MASK,
    SingleFastTaskBudgetExceeded,
    VectorizedExactIGScorer,
    estimate_exact_ig_batch,
    pack_exact_ig_microbatches,
)


class CharacterTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        if not kwargs.get("return_offsets_mapping"):
            raise AssertionError("Official target encoding requires offsets")
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class TinyCausalModel(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 256) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocabulary_size)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        use_cache=False,
    ):
        del attention_mask, position_ids, use_cache
        logits = torch.zeros(
            (*input_ids.shape, self.config.vocab_size),
            dtype=self.scale.dtype,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits + self.scale * 0.0)


def _policy() -> ExactIGPrecisionPolicy:
    return ExactIGPrecisionPolicy(
        mode="cpu_official_semantics",
        autocast_enabled=False,
        autocast_dtype=None,
        temperature=1.0,
        attention_implementation="eager",
        sdpa_backend=None,
    )


def _task():
    tokenizer = CharacterTokenizer()
    task = ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=4096,
        maximum_position_id_exclusive=4096,
    ).build(
        prompt_global_id="p",
        trajectory_id="t",
        full_trajectory_input_ids=[11, 12, 13, 14],
        original_attention_mask=[1, 1, 1, 1],
        prefix_end_positions=[2, 4],
        canonical_answer="A",
    )
    return tokenizer, task


def _estimate(task):
    return estimate_exact_ig_batch(
        (task,),
        vocabulary_size=256,
        logits_element_size=4,
        structural_mask_element_size=4,
    )


def _pack(task, **overrides):
    estimate = _estimate(task)
    arguments = {
        "max_records_per_forward": 1,
        "max_attention_cost_per_batch": estimate.padded_attention_cost,
        "max_extended_tokens_per_batch": estimate.padded_token_count,
        "max_full_logits_bytes": estimate.full_logits_estimated_bytes,
        "max_selected_logits_bytes": estimate.selected_logits_estimated_bytes,
        "vocabulary_size": 256,
        "logits_element_size": 4,
        "structural_mask_element_size": 4,
    }
    arguments.update(overrides)
    return pack_exact_ig_microbatches((task,), **arguments)


def test_single_task_equal_to_every_budget_is_accepted() -> None:
    _, task = _task()
    assert _pack(task) == ((task,),)


def test_single_task_just_below_every_budget_is_accepted() -> None:
    _, task = _task()
    estimate = _estimate(task)
    assert _pack(
        task,
        max_attention_cost_per_batch=estimate.padded_attention_cost + 1,
        max_extended_tokens_per_batch=estimate.padded_token_count + 1,
        max_full_logits_bytes=estimate.full_logits_estimated_bytes + 1,
        max_selected_logits_bytes=estimate.selected_logits_estimated_bytes + 1,
    ) == ((task,),)


@pytest.mark.parametrize(
    ("field", "estimate_field"),
    (
        ("max_attention_cost_per_batch", "padded_attention_cost"),
        ("max_extended_tokens_per_batch", "padded_token_count"),
        ("max_full_logits_bytes", "full_logits_estimated_bytes"),
        ("max_selected_logits_bytes", "selected_logits_estimated_bytes"),
    ),
)
def test_single_task_over_each_hard_budget_is_rejected_before_batching(
    field: str,
    estimate_field: str,
) -> None:
    _, task = _task()
    estimate = _estimate(task)
    with pytest.raises(SingleFastTaskBudgetExceeded) as raised:
        _pack(task, **{field: getattr(estimate, estimate_field) - 1})
    assert field in raised.value.reasons


def test_fast_budget_overflow_uses_explicit_official_sequential_fallback() -> None:
    tokenizer, task = _task()
    model = TinyCausalModel()
    target_length = len(task.canonical_target.token_ids)
    largest_sequential = max(task.prefix_end_positions) + target_length
    fast_length = int(task.input_ids.size)
    assert largest_sequential < fast_length
    budget = largest_sequential * largest_sequential
    scorer = VectorizedExactIGScorer(
        precision_policy=_policy(),
        padding_token_id=0,
        tokenizer=tokenizer,
    )
    result = scorer.score_many(
        model,
        (task,),
        torch.device("cpu"),
        max_records_per_forward=1,
        max_attention_cost_per_batch=budget,
        max_extended_tokens_per_batch=fast_length,
        max_full_logits_bytes=None,
        max_selected_logits_bytes=None,
    )["t"]
    assert result.execution_path == "official_sequential_fallback"
    assert result.runtime_metadata["fallback_reason"] == (
        "single_fast_task_budget_exceeded"
    )


def test_sequential_prefix_over_budget_fails_closed() -> None:
    tokenizer, task = _task()
    model = TinyCausalModel()
    scorer = VectorizedExactIGScorer(
        precision_policy=_policy(),
        padding_token_id=0,
        tokenizer=tokenizer,
    )
    with pytest.raises(RuntimeError, match="both Fast and Sequential"):
        scorer.score_many(
            model,
            (task,),
            torch.device("cpu"),
            max_records_per_forward=1,
            max_attention_cost_per_batch=1,
            max_extended_tokens_per_batch=None,
            max_full_logits_bytes=None,
            max_selected_logits_bytes=None,
        )


def test_sequential_logical_position_overflow_fails_closed() -> None:
    tokenizer = CharacterTokenizer()
    builder = ExactIGTaskBuilder(
        tokenizer,
        maximum_extended_sequence_length=4096,
        maximum_position_id_exclusive=80,
    )
    with pytest.raises(ValueError, match="logical position limit"):
        builder.build(
            prompt_global_id="p",
            trajectory_id="position-overflow",
            full_trajectory_input_ids=[11, 12],
            original_attention_mask=[1, 1],
            original_position_ids=[70, 71],
            prefix_end_positions=[2],
            canonical_answer="A",
        )


def test_boolean_and_official_additive_masks_have_cpu_score_parity() -> None:
    _, task = _task()
    model = TinyCausalModel()
    additive = VectorizedExactIGScorer(
        precision_policy=_policy(),
        attention_mask_mode=OFFICIAL_ADDITIVE_MASK,
    ).score(model, task, torch.device("cpu"))
    boolean = VectorizedExactIGScorer(
        precision_policy=_policy(),
        attention_mask_mode=BOOLEAN_4D_MASK,
    ).score(model, task, torch.device("cpu"))
    assert boolean.score_by_prefix == pytest.approx(
        additive.score_by_prefix,
        abs=0.0,
    )


class FakeFSDPModule:
    def __init__(self, initial: bool, fail_on_call: int | None = None) -> None:
        self.value = bool(initial)
        self.calls = 0
        self.fail_on_call = fail_on_call

    def set_reshard_after_forward(self, value: bool) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected setter failure")
        self.value = bool(value)


class FakeFSDPModel:
    def __init__(self, modules):
        self._modules = tuple(modules)

    def modules(self):
        return iter(self._modules)


@pytest.mark.parametrize("initial", (True, False))
def test_fsdp_window_restores_uniform_original_state(initial: bool) -> None:
    modules = [FakeFSDPModule(initial), FakeFSDPModule(initial)]
    model = FakeFSDPModel(modules)
    registry = FSDPReshardStateRegistry()
    registry.register_model(model, initial)
    with exact_ig_scoring_window(model, registry=registry) as report:
        assert [module.value for module in modules] == [False, False]
    assert [module.value for module in modules] == [initial, initial]
    assert report.restore_succeeded is True


def test_fsdp_window_restores_mixed_state_and_is_repeatable() -> None:
    modules = [FakeFSDPModule(True), FakeFSDPModule(False)]
    model = FakeFSDPModel(modules)
    registry = FSDPReshardStateRegistry()
    registry.register_model(model, {modules[0]: True, modules[1]: False})
    for _ in range(2):
        with exact_ig_scoring_window(model, registry=registry):
            assert [module.value for module in modules] == [False, False]
        assert [module.value for module in modules] == [True, False]


def test_fsdp_window_restores_after_body_exception() -> None:
    module = FakeFSDPModule(True)
    model = FakeFSDPModel([module])
    registry = FSDPReshardStateRegistry()
    registry.register_model(model, True)
    with pytest.raises(ValueError, match="body"):
        with exact_ig_scoring_window(model, registry=registry):
            raise ValueError("body")
    assert module.value is True


def test_fsdp_window_restore_failure_is_fail_closed() -> None:
    module = FakeFSDPModule(True, fail_on_call=2)
    model = FakeFSDPModel([module])
    registry = FSDPReshardStateRegistry()
    registry.register_model(model, True)
    with pytest.raises(FSDPReshardRestoreError, match="restore"):
        with exact_ig_scoring_window(model, registry=registry):
            pass
