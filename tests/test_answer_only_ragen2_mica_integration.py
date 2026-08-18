from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import yaml

from agentic_rl.advantage.mica_ig import (
    ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE,
)
from agentic_rl.config import load_config
from agentic_rl.controller.attempt_state import TrainingState
from agentic_rl.runtime.formal_state import eval_queue_snapshot
from agentic_rl.runtime import verl_runtime_adapter as runtime_adapter_module
from agentic_rl.runtime.learner_batch import (
    build_synchronized_microbatch_rounds,
    prepare_selected_trajectories,
)
from agentic_rl.rollout.trajectory_schema import (
    PromptTrajectoryGroup,
    TokenSource,
    TrajectoryRecord,
    TurnRecord,
    TurnType,
)
from agentic_rl.selection.candidate_pool import (
    ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
    CandidatePool,
    PromptGroup,
    prompt_group_from_outcomes,
)
from agentic_rl.selection.channel_scale import ChannelScaleState

CONFIG = "configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml"
ROOT = Path(__file__).resolve().parents[1]


def test_verified_resume_recreates_only_missing_cadence_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENTIC_RL_RUNTIME_STAGE", "FORMAL")
    resume_checkpoint = tmp_path / "source" / "update_020"
    resume_checkpoint.mkdir(parents=True)
    monkeypatch.setenv(
        "AGENTIC_RL_RESUME_CHECKPOINT",
        str(resume_checkpoint),
    )
    config = load_config(
        ROOT / "configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml"
    )
    config["paths"]["runtime_root"] = str(tmp_path / "run")
    config["checkpoint"][
        "materialize_missing_cadence_artifacts_on_resume"
    ] = True
    adapter = runtime_adapter_module.VerlAttemptRuntimeAdapter(config)
    checksum = "actor-checksum-u020"
    monkeypatch.setattr(adapter, "actor_parameter_checksum", lambda: checksum)

    class FakeWorkerGroup:
        def execute_all_sync(self, method: str, model: str, step: int, actor: str):
            assert method == "load_restored_reward_snapshot_from_hf"
            assert Path(model).name == "update_020"
            assert step == 20
            assert actor == checksum
            return [
                {
                    "rank": rank,
                    "successful_update_step": 20,
                    "actor_checksum_before": checksum,
                    "actor_checksum_after": checksum,
                    "reward_snapshot_checksum": "reward-u020",
                    "reward_parameter_dtype": "float32",
                }
                for rank in range(3)
            ]

    adapter.worker_group = FakeWorkerGroup()

    def fake_export(
        state: TrainingState,
        *,
        actor_checksum: str,
        allow_restored_checkpoint_boundary: bool = False,
    ) -> Path:
        assert state.successful_update_step == 20
        assert actor_checksum == checksum
        assert allow_restored_checkpoint_boundary is True
        destination = (
            Path(config["paths"]["runtime_root"])
            / "checkpoints"
            / "models"
            / "update_020"
        )
        destination.mkdir(parents=True)
        model_file = destination / "model.safetensors"
        model_file.write_bytes(b"verified-resume-model")
        import hashlib
        import json

        model_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()
        (destination / "training_metadata.json").write_text(
            json.dumps(
                {
                    "successful_update_step": 20,
                    "actor_checksum": checksum,
                    "manifest": {"model.safetensors": model_sha256},
                }
            ),
            encoding="utf-8",
        )
        (destination / "COMPLETED").write_text("20\n", encoding="utf-8")
        adapter._last_model_checkpoint = destination
        return destination

    monkeypatch.setattr(adapter, "_export_model_checkpoint", fake_export)
    state = TrainingState(
        attempt_id=20,
        successful_update_step=20,
        data_cursor=1984,
    )
    first = adapter._materialize_missing_resume_cadence_artifacts(state)
    second = adapter._materialize_missing_resume_cadence_artifacts(state)
    assert first is not None and first["model_export_created"] is True
    assert second is not None and second["model_export_created"] is False
    queue = eval_queue_snapshot(config["paths"]["runtime_root"])
    assert len(queue["tasks"]) == 1
    assert queue["tasks"][0]["update"] == 20
    assert queue["tasks"][0]["status"] == "pending"
    assert first["optimizer_steps_during_recovery"] == 0
    assert first["scheduler_steps_during_recovery"] == 0
    assert first["resume_checkpoint_writes_during_recovery"] == 0
    assert first["reward_snapshot_preloaded"] is True


def test_verified_resume_reuses_external_cadence_artifact_without_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import hashlib
    import json

    monkeypatch.setenv("AGENTIC_RL_RUNTIME_STAGE", "FORMAL")
    resume_checkpoint = tmp_path / "source" / "update_020"
    resume_checkpoint.mkdir(parents=True)
    monkeypatch.setenv("AGENTIC_RL_RESUME_CHECKPOINT", str(resume_checkpoint))
    checksum = "actor-checksum-u020"
    artifact = tmp_path / "derived" / "update_020"
    artifact.mkdir(parents=True)
    model = artifact / "model.safetensors"
    model.write_bytes(b"immutable-derived-model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    (artifact / "training_metadata.json").write_text(
        json.dumps(
            {
                "successful_update_step": 20,
                "actor_checksum": checksum,
                "manifest": {"model.safetensors": model_sha},
            }
        ),
        encoding="utf-8",
    )
    (artifact / "COMPLETED").write_text("20\n", encoding="utf-8")
    config = load_config(ROOT / CONFIG)
    config["paths"]["runtime_root"] = str(tmp_path / "run")
    config["checkpoint"]["materialize_missing_cadence_artifacts_on_resume"] = True
    config["checkpoint"]["resume_cadence_model_artifact_source"] = str(artifact)
    adapter = runtime_adapter_module.VerlAttemptRuntimeAdapter(config)
    monkeypatch.setattr(adapter, "actor_parameter_checksum", lambda: checksum)
    monkeypatch.setattr(
        adapter,
        "_export_model_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("external artifact must avoid live FSDP export")
        ),
    )

    class FakeWorkerGroup:
        def execute_all_sync(self, method: str, model_root: str, step: int, actor: str):
            assert method == "load_restored_reward_snapshot_from_hf"
            assert Path(model_root).resolve() == artifact.resolve()
            return [
                {
                    "rank": rank,
                    "successful_update_step": step,
                    "actor_checksum_before": actor,
                    "actor_checksum_after": actor,
                    "reward_snapshot_checksum": "reward-u020",
                    "reward_parameter_dtype": "float32",
                }
                for rank in range(3)
            ]

    adapter.worker_group = FakeWorkerGroup()
    report = adapter._materialize_missing_resume_cadence_artifacts(
        TrainingState(attempt_id=20, successful_update_step=20, data_cursor=1984)
    )
    assert report is not None
    assert report["model_export_created"] is False
    assert report["external_model_artifact_reused"] is True
    assert report["reward_snapshot_preloaded"] is True
    assert Path(report["model_path"]).resolve() == artifact.resolve()


def _record(trajectory_id: str, raw_ig: float, outcome: float, fmt: int):
    record = TrajectoryRecord(
        prompt_global_id="p",
        trajectory_id=trajectory_id,
        input_ids=[10, 20, 21, 30, 40],
        token_sources=[
            TokenSource.PROMPT,
            TokenSource.MODEL,
            TokenSource.MODEL,
            TokenSource.ENVIRONMENT,
            TokenSource.MODEL,
        ],
        turn_ids=[-1, 0, 0, -1, 1],
        turns=[
            TurnRecord(
                turn_index=0,
                turn_type=TurnType.SEARCH,
                search_index=0,
                model_text="<think>x</think><search>q</search>",
                search_action_span_valid=True,
                search_prefix_valid=True,
                ig_reward_eligible=True,
            ),
            TurnRecord(
                turn_index=1,
                turn_type=TurnType.ANSWER,
                model_text="<answer>a</answer>",
            ),
        ],
        search_prefix_end_positions=[1, 4],
        search_prefix_before_search_end_positions={0: 1},
        immediate_ig={0: raw_ig},
        task_outcome=outcome,
        answer_format_indicator=fmt,
        terminal_answer_valid=True,
        trajectory_protocol_valid=True,
    )
    record.validate()
    return record


def _select(groups):
    pool = CandidatePool(group_size=1, maximum_prompts=4)
    pool.add(groups)
    return pool.select(
        ig_state=ChannelScaleState(),
        outcome_state=ChannelScaleState(),
        top_p_mass=0.9,
        alpha_ig=0.5,
        alpha_outcome=0.5,
        noise_floor_ig=1.0e-12,
        noise_floor_outcome=1.0e-12,
        minimum_positive_prompts=1,
        health_threshold_ratio=0.1,
        minimum_selected_prompts=1,
        maximum_selected_prompts=4,
        allow_provisional_scale=True,
        signal_mode=ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL,
    )


def test_answer_only_ragen_selection_is_independent_of_exact_ig() -> None:
    first = _select(
        [
            PromptGroup(f"p{index}", (index,), float(index), float(4 - index))
            for index in range(4)
        ]
    )
    changed_ig = _select(
        [
            PromptGroup(
                f"p{index}",
                (index,),
                float(10_000 * (4 - index)),
                float(4 - index),
            )
            for index in range(4)
        ]
    )
    assert first.selected_ids == changed_ig.selected_ids
    assert first.score_by_prompt == changed_ig.score_by_prompt
    assert first.signal_mode == ANSWER_OUTCOME_ONLY_SELECTION_SIGNAL


def test_candidate_group_can_be_built_before_exact_ig_gpu_scoring() -> None:
    records = [_record("t0", -1.0, 0.0, 0), _record("t1", 1.0, 1.0, 1)]
    for record in records:
        record.immediate_ig.clear()
    group = prompt_group_from_outcomes(
        PromptTrajectoryGroup("p", tuple(records), ("a",)),
        expected_group_size=2,
    )
    assert group.ig_variance == 0.0
    assert group.outcome_variance == 0.5
    assert group.metadata["exact_ig_deferred"] is True


def test_new_config_locks_answer_only_ragen_and_mica_v1() -> None:
    config = load_config(CONFIG)
    assert config["algorithm_mode"] == (
        ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE
    )
    assert config["selection"]["signal"] == "answer_outcome_only"
    assert config["mica"]["gamma"] == 1.0
    assert config["mica"]["alpha"] == 0.5
    assert config["mica"]["normalization_scope"] == "prompt_search_depth"
    assert config["mica"]["singleton_fallback"] == (
        "normalized_terminal_outcome"
    )
    assert config["advantage"]["sufficiency_probe"]["enabled"] is False
    assert config["checkpoint"]["live_distributed_reload_verification"] is False


def test_formal_checkpoint_validation_does_not_reload_live_workers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    checkpoint = tmp_path / "update_020"
    checkpoint.mkdir()
    metadata = SimpleNamespace(
        algorithm_config=config,
        attempt_id=20,
        successful_update_step=20,
        data_cursor=1984,
    )

    class FakeCommitter:
        def __init__(self, _root: Path) -> None:
            pass

        def validate(self, _checkpoint: Path):
            return metadata

    class FakeWorkerGroup:
        def execute_all_sync(self, method: str):
            assert method == "local_optimizer_scheduler_digest"
            return [{"rank": rank, "digest": "unchanged"} for rank in range(3)]

    monkeypatch.setattr(
        runtime_adapter_module,
        "AtomicCheckpointCommitter",
        FakeCommitter,
    )
    monkeypatch.setattr(
        runtime_adapter_module,
        "assert_exact_ig_checkpoint_compatible",
        lambda _saved, _active: None,
    )
    adapter = object.__new__(runtime_adapter_module.VerlAttemptRuntimeAdapter)
    adapter.config = config
    adapter.worker_group = FakeWorkerGroup()
    adapter.actor_parameter_checksum = lambda: "actor-checksum"
    restored_called = False

    def forbidden_restore(_checkpoint: Path):
        nonlocal restored_called
        restored_called = True
        raise AssertionError("live checkpoint restore must not be called")

    adapter._restore_checkpoint = forbidden_restore
    result = adapter._verify_checkpoint_without_live_reload(
        checkpoint,
        SimpleNamespace(
            attempt_id=20,
            successful_update_step=20,
            data_cursor=1984,
        ),
    )
    assert result["status"] == "PASS"
    assert result["fresh_runtime_restore_required"] is True
    assert result["actor_checksum_before"] == result["actor_checksum_after"]
    assert restored_called is False


def test_preformal_runtime_stages_call_the_production_mica_path() -> None:
    adapter = (ROOT / "src/agentic_rl/runtime/verl_runtime_adapter.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "scripts/run_mica_preformal_gate.sh").read_text(
        encoding="utf-8"
    )
    for stage in (
        "MICA_E2E_NOUPDATE",
        "MICA_ONE_UPDATE",
        "MICA_FORMAL_SHAPE",
    ):
        assert stage in adapter
        assert stage in runner
    assert "self.finalize_selected_exact_ig(selected)" in adapter
    assert "self.selected_microbatches(selected)" in adapter
    assert "self._validate_mica_search_advantages()" in adapter
    assert "AGENTIC_RL_SMOKE_MODEL_CHECKPOINTS=0" in runner
    assert "resolve_mica_formal_config.py" in runner


def test_formal_operational_wrappers_resolve_the_isolated_project() -> None:
    historical_root = "igpo_ragen2_a2tgpo_strict_onpolicy_v1"
    for name in (
        "async_eval_gpu0_worker.sh",
        "monitor_formal_training_10min.sh",
        "formal_training_watchdog.sh",
        "status_formal_training.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'dirname "${BASH_SOURCE[0]}"' in source
        assert historical_root not in source


def test_preformal_resolver_preserves_formal_schedule(tmp_path: Path) -> None:
    output = tmp_path / "resolved.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/resolve_mica_formal_config.py"),
            "--input",
            str(ROOT / CONFIG),
            "--output",
            str(output),
            "--total-successful-updates",
            "500",
        ],
        check=True,
    )
    resolved = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert resolved["formal"]["total_successful_updates"] == 500
    assert resolved["formal_schedule"]["total_successful_updates"] == 500
    assert resolved["scheduler"]["total_successful_updates"] == 500
    assert resolved["formal_schedule"]["warmup"] == 10
    assert resolved["formal_schedule"]["learner_micro_batch_size"] == 6


def test_mica_transport_has_no_role_gate_loss_and_masks_observation() -> None:
    records = (_record("t0", -1.0, 0.0, 0), _record("t1", 1.0, 1.0, 1))
    group = PromptGroup("p", records, ig_variance=1.0, outcome_variance=0.5)
    prepared = prepare_selected_trajectories(
        (group,),
        expected_group_size=2,
        advantage_config={
            "search_task_mode": (
                ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE
            ),
            "lambda_outcome": 1.0,
            "lambda_format": 1.0,
            "normalization_epsilon": 1.0e-6,
            "zero_variance_tolerance": 1.0e-12,
            "mica": {"gamma": 1.0, "alpha": 0.5},
        },
    )
    for item in prepared[0]:
        assert item.decision_advantage_by_turn == {}
        assert item.query_advantage_by_turn == {}
        assert item.decision_token_mask == ()
        assert item.query_token_mask == ()
        assert item.record.policy_mask[3] == 0
    rounds = build_synchronized_microbatch_rounds(
        (prepared[0],),
        micro_batch_size_per_rank=2,
        pad_token_id=0,
        snapshot_step=0,
        global_prompt_count=1,
        group_size=2,
        action_state_chunk_size=1,
        vocabulary_chunk_size=8,
        kl_coefficient=0.01,
        lambda_decision=0.0,
        lambda_query=0.0,
    )
    payload = rounds[0][0]
    assert payload["search_task_mode"] == (
        ANSWER_ONLY_RAGEN2_MICA_IG_V1_SINGLETON_OUTCOME_MODE
    )
    assert not payload["decision_token_mask"].any().item()
    assert not payload["query_token_mask"].any().item()
    assert not payload["policy_mask"][:, 3].any().item()
