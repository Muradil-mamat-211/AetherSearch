#!/usr/bin/env python3
"""Prepare durable metadata for a new formal run from a committed checkpoint."""

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
from agentic_rl.runtime.formal_state import append_jsonl, atomic_write_json, seed_completed_eval


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--retriever-pid", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    source_model = args.source_model.resolve()
    source_run = args.source_run.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")

    for required in (
        checkpoint / "metadata.json",
        checkpoint / "integrity.sha256.json",
        checkpoint / "controller/state.json",
        source_model / "training_metadata.json",
    ):
        if not required.exists():
            raise SystemExit(f"missing required resume input: {required}")

    metadata = AtomicCheckpointCommitter(checkpoint.parent).validate(checkpoint)
    step = int(metadata.successful_update_step)
    eval_dir = source_run / "eval" / f"update_{step:03d}"
    for required in (eval_dir / "metrics.json", eval_dir / "COMPLETED"):
        if not required.exists():
            raise SystemExit(
                f"checkpoint U{step} requires a completed matching Eval: {required}"
            )
    controller = read_json(checkpoint / "controller/state.json")
    state = controller["training_state"]
    actual = (int(state["successful_update_step"]), int(state["attempt_id"]), int(state["data_cursor"]))
    expected = (step, int(metadata.attempt_id), int(metadata.data_cursor))
    if actual != expected:
        raise SystemExit(f"checkpoint/controller mismatch: {actual} != {expected}")

    config = load_config(config_path)
    config["paths"]["runtime_root"] = str(run_dir)
    config["formal"]["resume_from_successful_update"] = step
    validate_config(config)

    model_metadata = read_json(source_model / "training_metadata.json")
    actor_checksum = str(model_metadata["actor_checksum"])
    eval_metrics = read_json(eval_dir / "metrics.json")
    eval_checksums = {str(row["actor_checksum"]) for row in eval_metrics["metrics"]}
    if eval_checksums != {actor_checksum}:
        raise SystemExit(
            f"U{step} actor checksum differs between model and completed Eval"
        )

    run_dir.mkdir(parents=True)
    for relative in ("configs", "logs", "metrics", "checkpoints/models", "checkpoints/resume", "eval", "state/pids", "final"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "logs/console.log", "logs/train_rank0.log", "logs/ray_driver.log",
        "logs/fsdp_rank0.log", "logs/fsdp_rank1.log", "logs/fsdp_rank2.log",
        "logs/eval_worker.log", "logs/errors.log",
    ):
        (run_dir / relative).touch()

    shutil.copy2(config_path, run_dir / "configs/template_config.yaml")
    resolved_path = run_dir / "configs/resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    if (project_root / "MANIFEST.sha256").exists():
        shutil.copy2(project_root / "MANIFEST.sha256", run_dir / "configs/project_manifest.sha256")
    shutil.copytree(eval_dir, run_dir / "eval" / eval_dir.name)
    for row in eval_metrics["metrics"]:
        append_jsonl(run_dir / "metrics/eval_metrics.jsonl", row)
    seed_completed_eval(
        run_dir,
        update=step,
        model_path=source_model,
        actor_checksum=actor_checksum,
    )

    now = time.time()
    run_manifest = {
        "run_id": run_dir.name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(resolved_path),
        "config_sha256": sha256_file(resolved_path),
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_metadata_sha256": sha256_file(checkpoint / "metadata.json"),
        "source_checkpoint_integrity_sha256": sha256_file(checkpoint / "integrity.sha256.json"),
        "source_model_checkpoint": str(source_model),
        "source_actor_checksum": actor_checksum,
        "source_successful_update": step,
        "target_successful_update": int(config["formal_schedule"]["total_successful_updates"]),
        "checkpoint_every_successful_updates": int(config["formal_schedule"]["checkpoint_every_successful_updates"]),
        "fixed_eval_every_successful_updates": int(config["formal_schedule"]["fixed_eval_every_successful_updates"]),
        "async_eval": True,
        "eval_tmux_window": "eval",
        "monitor_enabled": False,
        "watchdog_enabled": False,
        "retriever_reused": True,
        "retriever_pid": args.retriever_pid,
        "retriever_endpoint": str(config["retriever"]["service_url"]),
        "rl_cuda_visible_devices": "1,2,3",
        "retriever_physical_gpu": 0,
        "fsdp_world_size": 3,
        "vllm_replicas": 3,
    }
    atomic_write_json(run_dir / "state/run_manifest.json", run_manifest)
    atomic_write_json(run_dir / "state/processes.json", {
        "run_id": run_dir.name, "tmux_session": args.session, "started_at": now,
        "source_run": str(source_run), "source_checkpoint": str(checkpoint),
        "source_model_checkpoint": str(source_model), "rl_cuda_visible_devices": "1,2,3",
        "retriever_physical_gpu": 0, "retriever_reused": True, "retriever_pid": args.retriever_pid,
        "fsdp_world_size": 3, "vllm_replica_count": 3,
        "monitor_enabled": False, "watchdog_enabled": False,
    })
    atomic_write_json(run_dir / "state/fatal_status.json", {"fatal": False, "timestamp": now})
    atomic_write_json(run_dir / "state/recovery_state.json", {"attempts": 0, "timestamp": now})
    atomic_write_json(run_dir / "state/training_progress.json", {
        "status": "starting", "attempt_id": int(metadata.attempt_id),
        "successful_update_step": step, "successful_updates_since_resume": 0,
        "data_cursor": int(metadata.data_cursor), "optimizer_steps_total": step,
        "scheduler_steps_total": step, "latest_resume_checkpoint": str(checkpoint),
        "latest_model_checkpoint": str(source_model), "actor_checksum": actor_checksum,
        "updated_at": now,
    })
    atomic_write_json(run_dir / "state/latest_resume_checkpoint.json", {
        "successful_update_step": step, "path": str(checkpoint), "actor_checksum": actor_checksum,
    })
    atomic_write_json(run_dir / "state/latest_model_checkpoint.json", {
        "successful_update_step": step, "path": str(source_model), "actor_checksum": actor_checksum,
    })
    (run_dir / "state/latest_successful_update").write_text(f"{step}\n", encoding="utf-8")
    (run_dir / "state/latest_checkpoint").write_text(f"{checkpoint}\n", encoding="utf-8")
    (run_dir / "configs/manifest.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path}" for path in (resolved_path, run_dir / "state/run_manifest.json")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "config": str(resolved_path), "checkpoint": str(checkpoint), "successful_update": step, "actor_checksum": actor_checksum}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
