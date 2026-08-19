from __future__ import annotations

from gpu_test_guard import skip_if_no_gpu

skip_if_no_gpu()

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rl.config import load_config
from agentic_rl.controller.attempt_state import TrainingState
from agentic_rl.controller.transaction import StrictUpdateTransaction
from agentic_rl.runtime.verl_config import (
    build_verl_config,
    effective_rollout_topology,
    unresolved_formal_fields,
)
from agentic_rl.runtime.verl_runtime_adapter import (
    RuntimeGateError,
    VerlAttemptRuntimeAdapter,
    _debug_shape,
    _resolve_pad_token_id,
    _with_runtime_smoke_schedule,
    assert_exact_ig_parity_gate,
)
from agentic_rl.runtime.fsdp_worker import (
    StrictOnPolicyFSDP2Worker,
    _classify_exact_ig_canary,
    _to_builtin_optimizer_metadata,
    _uniform_sample_indices,
)

from config_support import MICA_CONFIG, TEST_CONFIG


def test_exact_ig_structural_audit_replaces_old_numeric_hard_gate() -> None:
    config = load_config(TEST_CONFIG)
    summary = config["exact_ig"]["structural_audit_path"]
    payload = json.loads(open(summary, encoding="utf-8").read())
    assert payload["allow_fast_path_training"] is True
    assert all(
        value is True
        for key, value in payload["gates"].items()
        if key
        not in {"OPTIMIZER_STEPS", "SCHEDULER_STEPS", "CHECKPOINT_WRITES"}
    )
    approved = assert_exact_ig_parity_gate(config)
    assert approved["gate_pass"] is True
    assert approved["runtime_approval"] is None
    assert approved["numeric_difference_policy"] == (
        "telemetry_only_unless_semantic_or_safety_drift"
    )
    assert approved["ragen"]["selected_ids_equal"] is True

    config["exact_ig"]["structural_audit_status"] = "FAIL"
    with pytest.raises(RuntimeGateError, match="structural audit PASS"):
        assert_exact_ig_parity_gate(config)


def test_production_exact_ig_worker_uses_canonical_batched_no_anchor_contract() -> None:
    source = inspect.getsource(StrictOnPolicyFSDP2Worker.score_exact_ig_tasks)
    assert "scorer.score_many(" in source
    assert "canonical_answer=task.canonical_answer" in source
    assert "score_by_prefix_alias" not in source
    assert "task.aliases" not in source
    assert "alias_argmax" not in source
    assert "exact_ig_max_records_per_forward" in source
    assert "micro_batches" in source
    for metadata_key in (
        "target_tokenization_policy",
        "official_igpo_commit_sha",
        "mask_builder_version",
        "position_builder_version",
        "scaffold_text",
        "tokenizer_name_or_path",
        "tokenizer_revision",
    ):
        assert metadata_key in source


def test_global_exact_ig_dispatch_asserts_same_prompt_target_consistency() -> None:
    source = inspect.getsource(VerlAttemptRuntimeAdapter._score_exact_ig_tasks)
    assert "tasks_by_prompt" in source
    assert "assert_same_prompt_target_consistency(prompt_tasks)" in source


def test_verl_mapping_resolves_four_independent_tp1_replicas() -> None:
    config = load_config(TEST_CONFIG)
    resolved = build_verl_config(config, require_optimizer=False)
    assert effective_rollout_topology(resolved) == {
        "worker_world_size": 4,
        "per_replica_world_size": 1,
        "replica_count": 4,
        "aggregate_data_parallel_size": 4,
        "tensor_parallel_size": 1,
    }
    assert resolved.actor_rollout_ref.actor.strategy == "fsdp2"
    assert resolved.actor_rollout_ref.actor.ppo_epochs == 1
    assert resolved.actor_rollout_ref.rollout.name == "vllm"
    assert resolved.actor_rollout_ref.rollout.mode == "async"
    assert resolved.actor_rollout_ref.rollout.data_parallel_size == 1
    assert resolved.actor_rollout_ref.exact_ig_structural_audit_path == (
        config["exact_ig"]["structural_audit_path"]
    )
    assert (
        resolved.actor_rollout_ref.exact_ig_maximum_phi_safety_abs_diff
        == 1.0e-3
    )
    assert (
        resolved.actor_rollout_ref.exact_ig_maximum_ig_safety_abs_diff
        == 1.0e-3
    )
    assert (
        resolved.actor_rollout_ref.exact_ig_numeric_ambiguity_epsilon
        == config["exact_ig"]["numeric_ambiguity_epsilon"]
    )
    assert (
        resolved.actor_rollout_ref.exact_ig_calibration_p99_ig_abs_diff
        == config["exact_ig"]["calibration_p99_ig_abs_diff"]
    )


def test_online_canary_records_numeric_warning_without_using_it_as_hard_gate() -> None:
    common = {
        "token_allclose": False,
        "phi_allclose": True,
        "ig_allclose": True,
        "finite": True,
        "target_coverage": True,
        "canonical_answer_agreement": True,
        "non_ambiguous_sign_agreement": True,
        "turn_ranking_agreement": True,
        "token_error": 2.47955322265625e-5,
        "phi_error": 8.58306884765625e-6,
        "ig_error": 8.344650268554688e-6,
        "telescoping_error": 0.0,
        "telemetry_token_error": 2.0e-5,
        "telemetry_phi_error": 2.0e-5,
        "telemetry_ig_error": 2.0e-5,
        "maximum_phi_safety_error": 1.0e-3,
        "maximum_ig_safety_error": 1.0e-3,
        "maximum_telescoping_error": 1.0e-10,
    }
    threshold_exceeded, hard_failure = _classify_exact_ig_canary(**common)
    assert threshold_exceeded is True
    assert hard_failure is False

    _, hard_failure_above_safety_limit = _classify_exact_ig_canary(
        **{**common, "ig_error": 1.00001e-3}
    )
    assert hard_failure_above_safety_limit is True

    threshold_exceeded, hard_failure_ig_soft_warning = (
        _classify_exact_ig_canary(
            **{
                **common,
                "token_error": 4.172325134277344e-5,
                "phi_error": 1.7404556274414062e-5,
                "ig_error": 2.0265579223632812e-5,
            },
        )
    )
    assert threshold_exceeded is True
    assert hard_failure_ig_soft_warning is False


def test_runtime_imports_installed_verl_061_not_legacy_search_r1_copy() -> None:
    import verl

    path = Path(inspect.getfile(verl)).resolve()
    assert "site-packages/verl" in path.as_posix()
    assert getattr(verl, "__version__", None) == "0.6.1"
    assert "/code/Search-R1/verl/" not in path.as_posix()


def test_formal_runtime_fails_closed_on_unapproved_hyperparameters() -> None:
    config = load_config(TEST_CONFIG)
    unresolved = unresolved_formal_fields(config)
    assert "learning_rate" in unresolved
    assert "maximum_prompt_length" in unresolved
    assert "maximum_response_length" in unresolved
    assert "maximum_model_length" in unresolved
    smoke = _debug_shape(config, prompt_count=4, group_size=4)
    assert unresolved_formal_fields(smoke) == ()
    assert smoke["formal_schedule"]["learning_rate"] == 2.0e-7
    assert smoke["formal_schedule"]["learner_micro_batch_size"] == 6
    assert config["formal_schedule"]["learning_rate"] is None
    assert config["formal_schedule"]["learner_micro_batch_size"] is None


def test_optimizer_debug_shape_uses_largest_compatible_micro_batch() -> None:
    mica_config_path = MICA_CONFIG
    for config_path, expected_world_size, expected_micro_batch in (
        (TEST_CONFIG, 4, 4),
        (mica_config_path, 3, 3),
    ):
        config = load_config(config_path)
        original_micro_batch = config["formal_schedule"][
            "learner_micro_batch_size"
        ]
        smoke = _debug_shape(
            config,
            prompt_count=4,
            group_size=4,
            require_optimizer_compatible=True,
            preserve_formal_schedule=(config_path == mica_config_path),
        )

        assert smoke["learner"]["world_size"] == expected_world_size
        assert (
            smoke["formal_schedule"]["learner_micro_batch_size"]
            == expected_micro_batch
        )
        normalized_mini_batch = (4 * 4 * 4) // expected_world_size
        assert normalized_mini_batch % expected_micro_batch == 0
        assert (
            config["formal_schedule"]["learner_micro_batch_size"]
            == original_micro_batch
        )
        if config_path == mica_config_path:
            assert smoke["formal_schedule"]["warmup"] == 10
        else:
            assert smoke["formal_schedule"]["warmup"] == 0

    mica_config = load_config(mica_config_path)
    mica_smoke = _debug_shape(
        mica_config,
        prompt_count=8,
        group_size=4,
        require_optimizer_compatible=True,
        preserve_formal_schedule=True,
    )
    assert mica_smoke["formal_schedule"]["learner_micro_batch_size"] == 6
    assert ((8 * 4 * 4) // 3) % 6 == 0


def test_stage_d_uses_runtime_schedule_without_changing_production_shape() -> None:
    config = load_config(TEST_CONFIG)
    stage_d = _with_runtime_smoke_schedule(config)

    assert stage_d["rollout"]["candidate_prompts_initial"] == 64
    assert stage_d["rollout"]["candidate_prompts_max"] == 128
    assert stage_d["rollout"]["group_size"] == 16
    assert stage_d["formal_schedule"]["learning_rate"] == 2.0e-7
    assert stage_d["formal_schedule"]["learner_micro_batch_size"] == 6
    assert stage_d["exact_ig"]["oracle_canary_fail_closed"] is False
    assert unresolved_formal_fields(stage_d) == ()

    # The persisted formal input remains fail-closed.
    assert config["formal_schedule"]["learning_rate"] is None
    assert config["formal_schedule"]["learner_micro_batch_size"] is None
    assert config["exact_ig"]["oracle_canary_fail_closed"] is False


def test_transaction_bounds_can_be_narrowed_only_for_isolated_debug_shape() -> None:
    transaction = StrictUpdateTransaction(
        TrainingState(),
        allowed_candidate_counts=(4, 8),
        minimum_selected_prompts=1,
        maximum_selected_prompts=4,
    )
    assert transaction.allowed_candidate_counts == (4, 8)
    assert transaction.minimum_selected_prompts == 1
    assert transaction.maximum_selected_prompts == 4


def test_checksum_indices_remain_in_bounds_for_large_embedding() -> None:
    import torch

    numel = 151_936 * 2_048
    indices = _uniform_sample_indices(
        numel,
        maximum_samples=32,
        device=torch.device("cpu"),
    )
    assert indices.dtype == torch.int64
    assert indices.shape == (32,)
    assert int(indices[0]) == 0
    assert int(indices[-1]) == numel - 1
    assert bool(torch.all(indices[1:] > indices[:-1]))


def test_actor_checksum_canonicalization_reshards_every_fsdp_module_child_first() -> None:
    import torch

    calls: list[str] = []

    class FakeFSDPModule(torch.nn.Module):
        def __init__(self, name: str, child: torch.nn.Module | None = None) -> None:
            super().__init__()
            self.name = name
            if child is not None:
                self.child = child

        def reshard(self) -> None:
            calls.append(self.name)

    root = FakeFSDPModule(
        "root",
        FakeFSDPModule("block", FakeFSDPModule("leaf")),
    )
    worker = SimpleNamespace(actor_module_fsdp=root)
    count = StrictOnPolicyFSDP2Worker._reshard_all_actor_modules(worker)
    assert count == 3
    assert calls == ["leaf", "block", "root"]


def test_optimizer_checkpoint_metadata_contains_no_omegaconf_containers() -> None:
    from omegaconf import OmegaConf

    value = OmegaConf.create(
        {
            "betas": [0.9, 0.999],
            "nested": {"values": [1, 2]},
        }
    )
    converted = _to_builtin_optimizer_metadata(value)
    assert converted == {
        "betas": (0.9, 0.999),
        "nested": {"values": (1, 2)},
    }
    assert type(converted) is dict
    assert type(converted["betas"]) is tuple
    assert type(converted["nested"]) is dict
    assert type(converted["nested"]["values"]) is tuple


def test_stage_a_provenance_canary_masks_non_model_tokens() -> None:
    result = VerlAttemptRuntimeAdapter._stage_a_token_provenance_canary()
    assert result["status"] == "PASS"
    assert result["token_sources"] == [
        "prompt",
        "model",
        "environment",
        "code_inserted",
    ]
    assert result["action_mask"] == (0, 1, 0, 0)
    assert result["policy_mask"] == (0, 1, 0, 0)
    assert result["kl_mask"] == (0, 1, 0, 0)
    assert result["terminal_policy_credit_turn_index"] is None


def test_runtime_smoke_stages_never_write_model_checkpoints() -> None:
    config = load_config(TEST_CONFIG)
    for stage in ("A", "B", "C", "D"):
        adapter = VerlAttemptRuntimeAdapter(config)
        adapter.stage = stage
        assert adapter._should_checkpoint(1) is False
        assert adapter._should_checkpoint(5) is False
        assert adapter._should_checkpoint(50) is False
        with pytest.raises(
            RuntimeError,
            match="must not write model checkpoints",
        ):
            adapter._save_checkpoint(TrainingState())


def test_pilot_and_formal_checkpoint_policies_remain_separate_from_smoke() -> None:
    config = load_config(TEST_CONFIG)
    adapter = VerlAttemptRuntimeAdapter(config)
    adapter.stage = "PILOT50"
    assert adapter._should_checkpoint(1) is True
    assert adapter._should_checkpoint(2) is False
    assert adapter._should_checkpoint(50) is True

    approved = _debug_shape(config, prompt_count=4, group_size=4)
    formal = VerlAttemptRuntimeAdapter(approved)
    formal.stage = "FORMAL"
    assert formal._should_checkpoint(19) is False
    assert formal._should_checkpoint(20) is True


def test_action_hidden_forward_uses_causal_lm_root_without_full_vocab_logits() -> None:
    import torch

    class FakeCausalLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=3)
            self.embedding = torch.nn.Embedding(16, 3)
            self.lm_head = torch.nn.Linear(3, 16, bias=False)
            self.root_forward_calls = 0

        def forward(self, input_ids, **kwargs):
            del kwargs
            self.root_forward_calls += 1
            return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))

    model = FakeCausalLM()
    worker = object.__new__(VerlAttemptRuntimeAdapter)
    hidden = StrictOnPolicyFSDP2Worker._forward_action_hidden(
        worker,
        model,
        {"input_ids": torch.tensor([[1, 2]])},
        inference=False,
    )
    assert model.root_forward_calls == 1
    assert hidden.shape == (1, 2, 3)
    assert hidden.requires_grad
    assert model.lm_head(hidden).shape == (1, 2, 16)
    inference_hidden = StrictOnPolicyFSDP2Worker._forward_action_hidden(
        worker,
        model,
        {"input_ids": torch.tensor([[1, 2]])},
        inference=True,
    )
    assert inference_hidden.requires_grad is False
    assert inference_hidden.is_inference() is False


def test_pad_token_id_uses_generation_config_when_tokenizer_id_is_null(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<|endoftext|>", "pad_token_id": None}),
        encoding="utf-8",
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"pad_token_id": 151643, "eos_token_id": [151645, 151643]}),
        encoding="utf-8",
    )
    assert _resolve_pad_token_id(tmp_path) == 151643


def test_pad_token_id_falls_back_to_first_numeric_eos_id(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [None, 151645, 151643]}),
        encoding="utf-8",
    )
    assert _resolve_pad_token_id(tmp_path) == 151645


def test_pad_token_id_resolution_fails_closed_without_numeric_id(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<|endoftext|>"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeGateError, match="no numeric pad/eos token id"):
        _resolve_pad_token_id(tmp_path)
