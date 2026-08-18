#!/usr/bin/env python3
"""Prepare a new formal run from a fresh-runtime-verified resume checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from agentic_rl.checkpoint.atomic_commit import AtomicCheckpointCommitter
from agentic_rl.config import load_config, validate_config
from agentic_rl.runtime.formal_state import atomic_write_json
from agentic_rl.runtime.verl_config import assert_formal_hyperparameters_approved


MICA_MODE = "answer_only_ragen2_mica_ig_v1_singleton_outcome"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_successful_update(source_run: Path) -> int | None:
    path = source_run / "state" / "latest_successful_update"
    return int(path.read_text(encoding="utf-8").strip()) if path.is_file() else None


def _validate_cadence_model_artifact(
    root: Path,
    *,
    successful_update_step: int,
    actor_checksum: str,
) -> dict[str, Any]:
    root = root.resolve()
    metadata_path = root / "training_metadata.json"
    completed_path = root / "COMPLETED"
    model_path = root / "model.safetensors"
    for required in (metadata_path, completed_path, model_path):
        if not required.is_file():
            raise SystemExit(f"incomplete cadence model artifact: {required}")
    metadata = _read_json(metadata_path)
    if int(metadata.get("successful_update_step", -1)) != int(
        successful_update_step
    ):
        raise SystemExit("cadence model artifact update differs from checkpoint")
    if str(metadata.get("actor_checksum")) != str(actor_checksum):
        raise SystemExit("cadence model artifact Actor checksum differs")
    expected_sha256 = metadata.get("manifest", {}).get("model.safetensors")
    observed_sha256 = _sha256_file(model_path)
    if not expected_sha256 or observed_sha256 != expected_sha256:
        raise SystemExit("cadence model artifact checksum failed")
    return {
        "path": str(root),
        "model_sha256": observed_sha256,
        "metadata_sha256": _sha256_file(metadata_path),
        "successful_update_step": int(successful_update_step),
        "actor_checksum": str(actor_checksum),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--restore-validation", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--cadence-model-artifact", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    restore_validation_path = args.restore_validation.resolve()
    source_run = args.source_run.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    for required in (
        config_path,
        checkpoint / "metadata.json",
        checkpoint / "integrity.sha256.json",
        checkpoint / "controller" / "state.json",
        restore_validation_path,
    ):
        if not required.exists():
            raise SystemExit(f"missing verified resume input: {required}")

    metadata = AtomicCheckpointCommitter(checkpoint.parent).validate(checkpoint)
    step = int(metadata.successful_update_step)
    validation = _read_json(restore_validation_path)
    if validation.get("status") != "PASS":
        raise SystemExit("fresh-runtime restore validation did not PASS")
    if Path(str(validation.get("checkpoint", ""))).resolve() != checkpoint:
        raise SystemExit("restore validation refers to another checkpoint")
    if int(validation.get("successful_update_step", -1)) != step:
        raise SystemExit("restore validation successful-update counter differs")
    actor_checksums = {str(value) for value in validation.get("actor_checksums", [])}
    if len(actor_checksums) != 1:
        raise SystemExit("restore validation rank Actor checksums disagree")
    if int(validation.get("optimizer_steps", -1)) != 0:
        raise SystemExit("restore validation unexpectedly stepped the optimizer")
    if int(validation.get("scheduler_steps", -1)) != 0:
        raise SystemExit("restore validation unexpectedly stepped the scheduler")
    if int(validation.get("checkpoint_writes", -1)) != 0:
        raise SystemExit("restore validation unexpectedly wrote a checkpoint")
    actor_checksum = next(iter(actor_checksums))
    cadence_model_artifact = (
        _validate_cadence_model_artifact(
            args.cadence_model_artifact,
            successful_update_step=step,
            actor_checksum=actor_checksum,
        )
        if args.cadence_model_artifact is not None
        else None
    )

    controller = _read_json(checkpoint / "controller" / "state.json")
    restored_state = controller["training_state"]
    restored_counters = (
        int(restored_state["attempt_id"]),
        int(restored_state["successful_update_step"]),
        int(restored_state["data_cursor"]),
    )
    metadata_counters = (
        int(metadata.attempt_id),
        step,
        int(metadata.data_cursor),
    )
    if restored_counters != metadata_counters:
        raise SystemExit(
            f"checkpoint/controller counters differ: {restored_counters} != "
            f"{metadata_counters}"
        )

    config = load_config(config_path)
    if config["advantage"]["search_task_mode"] != MICA_MODE:
        raise SystemExit("verified checkpoint config is not the MICA V1 mode")
    if metadata.algorithm_config["advantage"]["search_task_mode"] != MICA_MODE:
        raise SystemExit("checkpoint algorithm mode is not MICA V1")
    if int(config["learner"]["world_size"]) != int(
        validation["target_fsdp_world_size"]
    ):
        raise SystemExit("restore validation topology differs from formal config")

    config["paths"]["runtime_root"] = str(run_dir)
    config["formal"]["fresh_start_required"] = False
    config["formal"]["resume_from_successful_update"] = step
    config.setdefault("checkpoint", {})[
        "live_distributed_reload_verification"
    ] = False
    config["checkpoint"][
        "materialize_missing_cadence_artifacts_on_resume"
    ] = True
    # Preserve all complete checkpoints in the recovered run. Retention is
    # separate from cadence and must not silently delete earlier evidence.
    config["checkpoint"]["formal_limit"] = None
    if cadence_model_artifact is not None:
        config["checkpoint"]["resume_cadence_model_artifact_source"] = (
            cadence_model_artifact["path"]
        )
    validate_config(config)
    assert_formal_hyperparameters_approved(config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "checkpoint": str(checkpoint),
                    "successful_update": step,
                    "actor_checksum": actor_checksum,
                    "restore_validation": str(restore_validation_path),
                    "cadence_model_artifact": cadence_model_artifact,
                },
                sort_keys=True,
            )
        )
        return 0

    for relative in (
        "configs",
        "logs",
        "metrics",
        "checkpoints/models",
        "checkpoints/resume",
        "eval",
        "state/pids",
        "artifacts/pids",
        "monitor/snapshots",
        "reports",
        "events",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "logs/console.log",
        "logs/train_rank0.log",
        "logs/ray_driver.log",
        "logs/retriever.log",
        "logs/eval_worker.log",
        "logs/watchdog.log",
        "logs/errors.log",
        "monitor/monitor_10min.log",
        "monitor/alerts.log",
    ):
        (run_dir / relative).touch()

    shutil.copy2(config_path, run_dir / "configs" / "source_resolved_config.yaml")
    resolved_path = run_dir / "configs" / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    if (project_root / "MANIFEST.sha256").is_file():
        shutil.copy2(
            project_root / "MANIFEST.sha256",
            run_dir / "configs" / "project_manifest.sha256",
        )
    shutil.copy2(
        restore_validation_path,
        run_dir / "configs" / "fresh_runtime_restore_validation.json",
    )

    now = time.time()
    source_committed_step = _source_successful_update(source_run)
    manifest = {
        "run_id": run_dir.name,
        "created_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)
        ),
        "algorithm_mode": MICA_MODE,
        "recovery_kind": "verified_post_step_pre_artifact_checkpoint",
        "source_run": str(source_run),
        "source_run_last_fully_logged_successful_update": source_committed_step,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_metadata_sha256": _sha256_file(
            checkpoint / "metadata.json"
        ),
        "source_checkpoint_integrity_sha256": _sha256_file(
            checkpoint / "integrity.sha256.json"
        ),
        "fresh_runtime_restore_validation": str(restore_validation_path),
        "fresh_runtime_restore_validation_sha256": _sha256_file(
            restore_validation_path
        ),
        "restored_actor_checksum": actor_checksum,
        "restored_successful_update": step,
        "restored_attempt_id": int(metadata.attempt_id),
        "restored_data_cursor": int(metadata.data_cursor),
        "optimizer_state": "restored",
        "scheduler_state": "restored",
        "u20_model_export_recovery": True,
        "u20_async_eval_recovery": True,
        "reused_verified_cadence_model_artifact": cadence_model_artifact,
        "config_path": str(resolved_path),
        "config_sha256": _sha256_file(resolved_path),
        "target_successful_update": int(
            config["formal_schedule"]["total_successful_updates"]
        ),
        "checkpoint_every_successful_updates": int(
            config["formal_schedule"]["checkpoint_every_successful_updates"]
        ),
        "fixed_eval_every_successful_updates": int(
            config["formal_schedule"]["fixed_eval_every_successful_updates"]
        ),
        "async_eval": True,
        "tmux_session": args.session,
        "rl_physical_gpus": list(config["hardware"]["rl_physical_gpus"]),
        "retriever_physical_gpu": int(
            config["hardware"]["retriever_physical_gpu"]
        ),
        "fsdp_world_size": int(config["learner"]["world_size"]),
        "vllm_replicas": int(config["rollout"]["replicas"]),
    }
    atomic_write_json(run_dir / "state" / "run_manifest.json", manifest)
    atomic_write_json(
        run_dir / "state" / "training_progress.json",
        {
            "status": "starting_from_verified_checkpoint",
            "attempt_id": int(metadata.attempt_id),
            "successful_update_step": step,
            "successful_updates_since_resume": 0,
            "data_cursor": int(metadata.data_cursor),
            "optimizer_steps_total": step,
            "scheduler_steps_total": step,
            "latest_resume_checkpoint": str(checkpoint),
            "latest_model_checkpoint": None,
            "actor_checksum": actor_checksum,
            "updated_at": now,
        },
    )
    atomic_write_json(
        run_dir / "state" / "latest_resume_checkpoint.json",
        {
            "successful_update_step": step,
            "path": str(checkpoint),
            "actor_checksum": actor_checksum,
        },
    )
    atomic_write_json(
        run_dir / "state" / "fatal_status.json",
        {"fatal": False, "timestamp": now},
    )
    (run_dir / "state" / "latest_successful_update").write_text(
        f"{step}\n", encoding="utf-8"
    )
    (run_dir / "state" / "latest_checkpoint").write_text(
        f"{checkpoint}\n", encoding="utf-8"
    )
    (run_dir / "state" / "tmux_session").write_text(
        f"{args.session}\n", encoding="utf-8"
    )
    launch_command = (
        f"bash {project_root / 'scripts' / '_run_runtime_job.sh'} FORMAL "
        f"{resolved_path} {run_dir} {checkpoint}\n"
    )
    (run_dir / "configs" / "launch_command.sh").write_text(
        launch_command,
        encoding="utf-8",
    )
    atomic_write_json(
        run_dir / "state" / "processes.json",
        {
            "run_id": run_dir.name,
            "tmux_session": args.session,
            "started_at": now,
            "source_checkpoint": str(checkpoint),
            "controller_pid": None,
            "eval_worker_pid": None,
            "retriever_pid": None,
            "monitor_pid": None,
            "watchdog_pid": None,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "run_dir": str(run_dir),
                "config": str(resolved_path),
                "checkpoint": str(checkpoint),
                "successful_update": step,
                "actor_checksum": actor_checksum,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
