#!/usr/bin/env python3
"""Validate and initialize a new U20 -> U500 formal run directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from agentic_rl.config import load_config, validate_config
from agentic_rl.runtime.formal_state import (
    append_jsonl,
    atomic_write_json,
    seed_completed_eval,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    profile_path: Path,
    checkpoint: Path,
    source_model: Path,
    source_run: Path,
    validation_result: Path,
) -> tuple[dict, dict, dict, dict, dict]:
    config = load_config(profile_path)
    checkpoint_meta = read_json(checkpoint / "metadata.json")
    model_meta = read_json(source_model / "training_metadata.json")
    validation = read_json(validation_result)
    eval_metrics = read_json(source_run / "eval/update_020/metrics.json")

    assert validation["status"] == "PASS"
    assert validation["successful_update_step"] == 20
    assert validation["target_fsdp_world_size"] == 3
    assert validation["worker_count"] == 3
    assert validation["optimizer_steps"] == 0
    assert validation["scheduler_steps"] == 0
    assert validation["checkpoint_writes"] == 0
    assert checkpoint_meta["successful_update_step"] == 20
    assert checkpoint_meta["actor_state_present"] is True
    assert checkpoint_meta["optimizer_state_present"] is True
    assert checkpoint_meta["scheduler_state_present"] is True
    assert model_meta["attempt_id"] == checkpoint_meta["attempt_id"]
    assert model_meta["data_cursor"] == checkpoint_meta["data_cursor"]
    checksums = {str(row["actor_checksum"]) for row in eval_metrics["metrics"]}
    assert checksums == {str(model_meta["actor_checksum"])}

    assert config["formal"]["fresh_start_required"] is False
    assert config["formal"]["resume_from_successful_update"] == 20
    assert config["formal"]["total_successful_updates"] == 500
    assert config["hardware"]["expected_cpu_cores"] == 48
    assert config["hardware"]["total_physical_gpus"] == 4
    assert config["hardware"]["rl_physical_gpus"] == [1, 2, 3]
    assert config["hardware"]["rl_world_size"] == 3
    assert config["rollout"]["data_parallel_size"] == 3
    assert config["rollout"]["replicas"] == 3
    assert config["rollout"]["gpu_memory_utilization"] == 0.48
    assert config["rollout"]["max_num_seqs"] == 64
    assert config["formal_schedule"]["learner_micro_batch_size"] == 6
    assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
    assert config["formal_schedule"]["fixed_eval_every_successful_updates"] == 20
    assert config["evaluation"]["asynchronous"] is True
    assert config["checkpoint"]["formal_limit"] == 3
    assert config["checkpoint"]["model_checkpoints_retain_all"] is True
    assert config["monitoring"]["minimum_disk_free_gib"] == 80

    algorithm = checkpoint_meta["algorithm_config"]
    assert algorithm["advantage"]["search_task_mode"] == config["advantage"]["search_task_mode"]
    assert algorithm["advantage"]["role_localized_gate"]["lambda_decision"] == config["advantage"]["role_localized_gate"]["lambda_decision"]
    assert algorithm["advantage"]["role_localized_gate"]["lambda_query"] == config["advantage"]["role_localized_gate"]["lambda_query"]
    assert algorithm["exact_ig"]["exact_ig_version"] == config["exact_ig"]["exact_ig_version"]
    assert algorithm["data"]["ordered_view_identity_sha256"] == config["data"]["ordered_view_identity_sha256"]
    assert algorithm["data"]["shuffle_seed"] == config["data"]["shuffle_seed"]
    assert algorithm["optimizer"]["learning_rate"] == config["optimizer"]["learning_rate"]
    assert algorithm["formal_schedule"]["warmup"] == config["formal_schedule"]["warmup"]
    return config, checkpoint_meta, model_meta, validation, eval_metrics


def write_environment(path: Path) -> None:
    rl_python = Path(
        os.environ.get(
            "RL_ENV",
            str(Path(__file__).resolve().parents[2] / "envs/igpo-ragen2-fsdp2-vllm011"),
        )
    ) / "bin/python"
    commands = [
        ["nvidia-smi"],
        [str(rl_python), "--version"],
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(time.strftime("timestamp_utc=%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
        for command in commands:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            handle.write("$ " + " ".join(command) + "\n")
            handle.write(result.stdout)
            handle.write(result.stderr)


def initialize_run(
    profile_path: Path,
    checkpoint: Path,
    source_model: Path,
    source_run: Path,
    run_dir: Path,
    session: str,
    validation_result: Path,
) -> None:
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    config, checkpoint_meta, model_meta, _validation, _eval_metrics = validate_inputs(
        profile_path, checkpoint, source_model, source_run, validation_result
    )
    run_dir.mkdir(parents=True)
    for relative in (
        "configs",
        "logs",
        "metrics",
        "checkpoints/models",
        "checkpoints/resume",
        "eval",
        "monitor",
        "state/pids",
        "final",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "logs/console.log",
        "logs/train_rank0.log",
        "logs/retriever.log",
        "logs/ray_driver.log",
        "logs/eval_worker.log",
        "logs/watchdog.log",
        "logs/errors.log",
        "monitor/monitor_10min.log",
        "monitor/monitor_10min.jsonl",
        "monitor/alerts.log",
    ):
        (run_dir / relative).touch()

    shutil.copy2(profile_path, run_dir / "configs/profile_config.yaml")
    project_root = Path(__file__).resolve().parents[1]
    shutil.copy2(project_root / "MANIFEST.sha256", run_dir / "configs/project_manifest.sha256")

    config["paths"]["runtime_root"] = str(run_dir.resolve())
    validate_config(config)
    resolved_path = run_dir / "configs/resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copytree(
        source_run / "eval/update_020",
        run_dir / "eval/update_020",
    )
    write_environment(run_dir / "configs/environment.txt")

    model_path = source_model.resolve()
    actor_checksum = str(model_meta["actor_checksum"])
    for row in _eval_metrics["metrics"]:
        append_jsonl(run_dir / "metrics/eval_metrics.jsonl", row)
    seed_completed_eval(
        run_dir,
        update=20,
        model_path=model_path,
        actor_checksum=actor_checksum,
    )
    now = time.time()
    atomic_write_json(
        run_dir / "state/processes.json",
        {
            "run_id": run_dir.name,
            "tmux_session": session,
            "started_at": now,
            "source_run": str(source_run.resolve()),
            "source_checkpoint": str(checkpoint.resolve()),
            "source_model_checkpoint": str(model_path),
            "rl_cuda_visible_devices": "1,2,3",
            "retriever_physical_gpu": 0,
            "fsdp_world_size": 3,
            "vllm_replica_count": 3,
        },
    )
    atomic_write_json(run_dir / "state/fatal_status.json", {"fatal": False, "timestamp": now})
    atomic_write_json(run_dir / "state/recovery_state.json", {"attempts": 0, "timestamp": now})
    atomic_write_json(
        run_dir / "state/training_progress.json",
        {
            "status": "starting",
            "attempt_id": int(checkpoint_meta["attempt_id"]),
            "successful_update_step": 20,
            "successful_updates_since_resume": 0,
            "data_cursor": int(checkpoint_meta["data_cursor"]),
            "optimizer_steps_total": 20,
            "scheduler_steps_total": 20,
            "latest_resume_checkpoint": str(checkpoint.resolve()),
            "latest_model_checkpoint": str(model_path),
            "actor_checksum": actor_checksum,
            "updated_at": now,
        },
    )
    atomic_write_json(
        run_dir / "state/latest_resume_checkpoint.json",
        {
            "successful_update_step": 20,
            "path": str(checkpoint.resolve()),
            "actor_checksum": actor_checksum,
        },
    )
    (run_dir / "state/latest_successful_update").write_text("20\n", encoding="utf-8")
    (run_dir / "state/latest_checkpoint").write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_dir.name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(resolved_path),
        "config_sha256": sha256_file(resolved_path),
        "profile_config_sha256": sha256_file(run_dir / "configs/profile_config.yaml"),
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_metadata_sha256": sha256_file(checkpoint / "metadata.json"),
        "source_checkpoint_integrity_sha256": sha256_file(checkpoint / "integrity.sha256.json"),
        "source_model_checkpoint": str(model_path),
        "source_actor_checksum": actor_checksum,
        "source_successful_update": 20,
        "target_successful_update": 500,
        "checkpoint_every_successful_updates": 20,
        "fixed_eval_every_successful_updates": 20,
        "async_eval": True,
        "eval_tmux_window": "eval",
        "retriever_physical_gpu": 0,
        "rl_physical_gpus": [1, 2, 3],
        "fsdp_world_size": 3,
        "vllm_replicas": 3,
    }
    atomic_write_json(run_dir / "state/run_manifest.json", manifest)
    launch = run_dir / "configs/launch_command.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"cd {project_root}\n"
        "bash scripts/resume_formal_u20_3rank_48cpu.sh\n",
        encoding="utf-8",
    )
    launch.chmod(0o755)
    with (run_dir / "configs/manifest.sha256").open("w", encoding="utf-8") as handle:
        for path in (
            run_dir / "configs/profile_config.yaml",
            resolved_path,
            run_dir / "configs/project_manifest.sha256",
            run_dir / "state/run_manifest.json",
        ):
            handle.write(f"{sha256_file(path)}  {path}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--validation-result", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    validate_inputs(
        args.profile,
        args.checkpoint,
        args.source_model,
        args.source_run,
        args.validation_result,
    )
    if args.check_only:
        print("FORMAL_RESUME_INPUTS=PASS")
        return 0
    if args.run_dir is None or not args.session:
        parser.error("--run-dir and --session are required unless --check-only is used")
    initialize_run(
        args.profile,
        args.checkpoint,
        args.source_model,
        args.source_run,
        args.run_dir,
        args.session,
        args.validation_result,
    )
    print(f"FORMAL_RESUME_RUN_PREPARED=PASS\nrun_dir={args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
