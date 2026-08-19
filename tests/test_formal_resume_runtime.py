from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rl.config import load_config

from config_support import FORMAL_RESUME_CONFIG, MICA_CONFIG, PILOT_CONFIG
from agentic_rl.runtime.formal_state import (
    claim_next_eval,
    complete_eval,
    enqueue_eval,
    eval_queue_snapshot,
    seed_completed_eval,
)
from agentic_rl.runtime.async_eval_worker import _target_successful_update
from agentic_rl.runtime.fsdp_worker import (
    StrictOnPolicyFSDP2Worker,
    _load_safetensors_state_dict_with_tied_key_validation,
    _update_sampled_state_digest_entry,
)
from agentic_rl.runtime.verl_runtime_adapter import VerlAttemptRuntimeAdapter
from agentic_rl.rollout.trajectory_schema import TurnType


ROOT = Path(__file__).resolve().parents[1]


def test_restored_reward_loader_accepts_explicit_equal_tied_keys(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4, bias=False)
            self.head.weight = self.embedding.weight

    model = TiedModel()
    expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    artifact = tmp_path / "model.safetensors"
    save_file(
        {
            "embedding.weight": expected.clone(),
            "head.weight": expected.clone(),
        },
        artifact,
    )

    tied_groups = _load_safetensors_state_dict_with_tied_key_validation(
        model,
        artifact,
    )

    assert tied_groups == 1
    assert torch.equal(model.embedding.weight, expected)
    assert model.embedding.weight.data_ptr() == model.head.weight.data_ptr()


def test_restored_reward_loader_rejects_divergent_tied_keys(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4, bias=False)
            self.head.weight = self.embedding.weight

    artifact = tmp_path / "model.safetensors"
    save_file(
        {
            "embedding.weight": torch.zeros(4, 3),
            "head.weight": torch.ones(4, 3),
        },
        artifact,
    )

    with pytest.raises(RuntimeError, match="divergent tied weights"):
        _load_safetensors_state_dict_with_tied_key_validation(
            TiedModel(),
            artifact,
        )


def test_reward_snapshot_digest_supports_streaming_entries() -> None:
    import hashlib

    import torch

    state = {
        "a.weight": torch.arange(12, dtype=torch.bfloat16).reshape(4, 3),
        "b.buffer": torch.tensor([3, 5, 8], dtype=torch.int64),
    }
    streamed = hashlib.sha256()
    for name in sorted(state):
        _update_sampled_state_digest_entry(streamed, name, state[name])

    assert streamed.hexdigest() == (
        StrictOnPolicyFSDP2Worker._sampled_state_digest(state)
    )


def test_snapshot_sync_and_hf_export_use_bounded_streaming_state() -> None:
    snapshot_source = inspect.getsource(
        StrictOnPolicyFSDP2Worker._sync_reward_snapshot
    )
    export_source = inspect.getsource(
        StrictOnPolicyFSDP2Worker.export_hf_model_checkpoint
    )
    lifecycle_source = inspect.getsource(
        StrictOnPolicyFSDP2Worker.validate_reward_snapshot_sync_cycles
    )

    assert "get_fsdp_full_state_dict" not in snapshot_source
    assert "get_fsdp_full_state_dict" not in export_source
    assert "_streaming_actor_state" in snapshot_source
    assert "_materialize_streamed_actor_tensor" in snapshot_source
    assert "_streaming_actor_state" in export_source
    assert "_materialize_streamed_actor_tensor" in export_source
    assert "_sync_reward_snapshot" in lifecycle_source
    assert "optimizer.step" not in lifecycle_source
    assert "scheduler.step" not in lifecycle_source

    validation_source = inspect.getsource(
        VerlAttemptRuntimeAdapter._run_resume_world_size_validation
    )
    assert "validate_reward_snapshot_sync_cycles" in validation_source
    assert "export_hf_model_checkpoint" in validation_source
    assert "tied_embedding_lm_head_equal" in validation_source
    assert "shutil.rmtree(export_root)" in validation_source


def test_formal_resume_config_preserves_pilot_state_contract() -> None:
    config = load_config(FORMAL_RESUME_CONFIG)
    assert config["formal"]["resume_from_successful_update"] == 20
    assert config["formal_schedule"]["total_successful_updates"] == 500
    assert config["formal_schedule"]["warmup"] == 2
    assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6
    assert config["rollout"]["candidate_prompts_max"] == 128
    assert config["checkpoint"]["formal_limit"] == 3
    assert config["evaluation"]["asynchronous"] is True
    assert config["rollout"]["group_size"] == 16
    assert config["policy"]["optimizer_steps_per_successful_update"] == 1


def test_mica_formal_checkpoint_retention_is_unlimited() -> None:
    config = load_config(MICA_CONFIG)
    assert config["checkpoint"]["formal_limit"] is None


def test_checkpoint_limit_none_does_not_delete_complete_checkpoints(
    tmp_path: Path,
) -> None:
    import inspect

    source = inspect.getsource(VerlAttemptRuntimeAdapter._enforce_checkpoint_limit)
    assert "raw_limit is None" in source
    assert "shutil.rmtree" in source


def test_formal_checkpoint_cadence_is_every_twenty(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RL_RUNTIME_STAGE", "FORMAL")
    config = load_config(FORMAL_RESUME_CONFIG)
    adapter = VerlAttemptRuntimeAdapter(config)
    assert adapter._should_checkpoint(20)
    assert adapter._should_checkpoint(40)
    assert not adapter._should_checkpoint(41)
    assert adapter._should_checkpoint(500)


def test_pilot_checkpoint_and_async_eval_target_are_update_twenty(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RL_RUNTIME_STAGE", "PILOT20")
    config = load_config(PILOT_CONFIG)
    adapter = VerlAttemptRuntimeAdapter(config)
    assert not adapter._should_checkpoint(10)
    assert adapter._should_checkpoint(20)
    assert _target_successful_update(config) == 20


def test_runtime_persists_complete_search_advantage_component_summary() -> None:
    config = load_config(PILOT_CONFIG)
    config["advantage"]["search_task_mode"] = "sufficiency_novelty_local_ig"
    adapter = VerlAttemptRuntimeAdapter(config)
    advantage = SimpleNamespace(
        future_ig_sum={},
        accumulated_ig_count={},
        future_ig_rescaled={},
        search_task_advantage={},
        search_advantage={0: -1.0, 1: -1.0},
        stop_continue_by_search_index={},
        normalized_ig={0: 2.0, 1: -1.0},
        sufficient_before_search={0: True, 1: False},
        no_new_observation={0: False, 1: False},
        search_branch_by_search_index={
            0: "sufficient_before_search",
            1: "normalized_local_ig",
        },
        normalized_outcome=0.25,
        centered_format_indicator=-0.25,
        answer_advantage=0.0,
    )
    turns = (
        SimpleNamespace(
            turn_type=TurnType.SEARCH,
            search_index=0,
            turn_index=10,
            policy_credit_eligible=True,
            exact_query_repeat=False,
            different_query_no_new_passage=False,
        ),
        SimpleNamespace(
            turn_type=TurnType.SEARCH,
            search_index=1,
            turn_index=11,
            policy_credit_eligible=True,
            exact_query_repeat=False,
            different_query_no_new_passage=False,
        ),
    )
    item = SimpleNamespace(
        record=SimpleNamespace(
            trajectory_id="trajectory-0",
            turns=turns,
            immediate_ig={0: 0.2, 1: -0.1},
        ),
        advantage=advantage,
        advantage_by_turn={10: -1.0, 11: -1.0},
    )
    adapter._prepared_groups = ((item,),)
    adapter._attempt_context = {}
    adapter._validate_and_record_search_advantage_components()
    metrics = adapter._attempt_context["advantage_component_metrics"]
    assert metrics["advantage_component_coverage_pass"] is True
    assert metrics["local_ig_hat_count"] == 2
    assert metrics["A_search_count"] == 2
    assert metrics["search_advantage_formula_assertion_pass"] is True
    assert metrics["answer_advantage_formula_assertion_pass"] is True
    assert metrics["search_z_o_entry_count"] == 0
    assert metrics["search_a_sc_entry_count"] == 0
    assert metrics["future_ig_contribution_count"] == 0
    assert metrics["sqrt_n_rescale_call_count"] == 0
    assert metrics["external_ig_multiplier_call_count"] == 0
    assert metrics["S_count"] == 1
    assert metrics["N_count"] == 0
    assert metrics["normal_local_ig_branch_count"] == 1


def test_runtime_accepts_answer_only_selected_payload_without_fake_search() -> None:
    config = load_config(PILOT_CONFIG)
    config["advantage"]["search_task_mode"] = "sufficiency_novelty_local_ig"
    adapter = VerlAttemptRuntimeAdapter(config)
    advantage = SimpleNamespace(
        future_ig_sum={},
        accumulated_ig_count={},
        future_ig_rescaled={},
        search_task_advantage={},
        search_advantage={},
        stop_continue_by_search_index={},
        normalized_ig={},
        sufficient_before_search={},
        no_new_observation={},
        search_branch_by_search_index={},
        normalized_outcome=0.5,
        centered_format_indicator=-0.5,
        answer_advantage=0.0,
    )
    item = SimpleNamespace(
        record=SimpleNamespace(
            trajectory_id="direct-answer",
            turns=(),
            immediate_ig={},
        ),
        advantage=advantage,
        advantage_by_turn={},
    )
    adapter._prepared_groups = ((item,),)
    adapter._attempt_context = {}

    adapter._validate_and_record_search_advantage_components()

    metrics = adapter._attempt_context["advantage_component_metrics"]
    assert metrics["advantage_component_coverage_pass"] is True
    assert metrics["local_ig_hat_count"] == 0
    assert metrics["A_search_count"] == 0
    assert metrics["searched_trajectory_count"] == 0
    assert metrics["no_search_trajectory_count"] == 1
    assert metrics["no_search_trajectory_rate"] == 1.0


def test_eval_queue_is_ordered_idempotent_and_recoverable(tmp_path: Path) -> None:
    model40 = tmp_path / "model40"
    model60 = tmp_path / "model60"
    enqueue_eval(tmp_path, update=60, model_path=model60, actor_checksum="b")
    enqueue_eval(tmp_path, update=40, model_path=model40, actor_checksum="a")
    enqueue_eval(tmp_path, update=40, model_path=model40, actor_checksum="a")
    first = claim_next_eval(tmp_path, worker_pid=123)
    assert first is not None and first["update"] == 40
    complete_eval(tmp_path, update=40, error="temporary")
    retry = claim_next_eval(tmp_path, worker_pid=124)
    assert retry is not None and retry["update"] == 40
    complete_eval(tmp_path, update=40, error=None)
    second = claim_next_eval(tmp_path, worker_pid=125)
    assert second is not None and second["update"] == 60
    complete_eval(tmp_path, update=60, error=None)
    snapshot = eval_queue_snapshot(tmp_path)
    assert [row["update"] for row in snapshot["tasks"]] == [40, 60]
    assert all(row["status"] == "completed" for row in snapshot["tasks"])


def test_seeded_pilot_eval_is_completed_without_loading_model(tmp_path: Path) -> None:
    seed_completed_eval(
        tmp_path,
        update=20,
        model_path=tmp_path / "pilot-resume",
        actor_checksum="pilot",
    )
    snapshot = eval_queue_snapshot(tmp_path)
    assert snapshot["tasks"][0]["update"] == 20
    assert snapshot["tasks"][0]["status"] == "completed"


def test_reward_scorer_and_export_are_explicit_and_nonmutating() -> None:
    initialization = inspect.getsource(StrictOnPolicyFSDP2Worker.init_model)
    export = inspect.getsource(
        StrictOnPolicyFSDP2Worker.export_hf_model_checkpoint
    )
    assert "actor_parameter_dtypes" in initialization
    assert "self._reward_parameter_dtype = torch.float32" in initialization
    assert "dtype=torch.float32" in initialization
    assert "requires_grad_(False)" in initialization
    assert "self._reward_model.eval()" in initialization
    assert 'attn_implementation="sdpa"' in initialization
    assert ".bfloat16()" not in initialization
    assert "before_checksum" in export
    assert "after_checksum" in export
    assert "before_dtype" in export
    assert "after_dtype" in export
    assert "actor_optimizer.step" not in export


def test_formal_launcher_has_independent_lifecycle_processes() -> None:
    launcher = (ROOT / "scripts/train_rl.sh").read_text()
    supervisor = (ROOT / "scripts/_run_runtime_job.sh").read_text()
    assert "recipes/rl/train_4x48gb.yaml" in launcher
    assert "preflight_mica_formal.py" in launcher
    assert "launch_retriever.sh" in supervisor
    assert "async_eval_worker.sh" in supervisor
