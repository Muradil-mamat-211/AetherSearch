from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from agentic_rl.config import load_config
from agentic_rl.retriever.health import query_health

from .formal_state import append_jsonl, atomic_write_json, read_json


FORBIDDEN_RECOVERY_MARKERS = (
    "out of memory",
    "cuda oom",
    "nan",
    "inf",
    "checksum",
    "cursor",
    "version mismatch",
    "optimizer_steps",
    "scheduler_steps",
    "nonfinite",
    "both_selection_channels_inactive",
)


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def _process_payload(run_dir: Path) -> dict[str, Any]:
    payload = read_json(run_dir / "state" / "processes.json", {})
    for name in ("trainer", "retriever", "eval_worker", "monitor", "watchdog"):
        path = run_dir / "state" / "pids" / f"{name}.pid"
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            payload[f"{name}_pid"] = int(value) if value.isdigit() else None
    return payload


def _tmux_alive(name: Any) -> bool:
    if not name:
        return False
    return subprocess.run(
        ["tmux", "has-session", "-t", str(name)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _rl_gpu_process_count() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    # The trainer records the physical UUID map in system metrics. At this
    # liveness layer, count all GPU compute processes and subtract the known
    # GPU0 retriever/evaluator PIDs instead of guessing Ray process names.
    pids = {
        line.split(",", maxsplit=1)[1].strip()
        for line in completed.stdout.splitlines()
        if "," in line
    }
    return len(pids)


def _latest_resume(run_dir: Path, initial: str) -> str:
    payload = read_json(run_dir / "state" / "latest_resume_checkpoint.json", {})
    path = payload.get("path") or initial
    return str(Path(path).resolve())


def _attempt_recovery(
    *,
    run_dir: Path,
    config_path: Path,
    initial_checkpoint: str,
) -> bool:
    recovery = read_json(
        run_dir / "state" / "recovery_state.json",
        {"attempts": 0},
    )
    if int(recovery.get("attempts", 0)) >= 1:
        return False
    exit_state = read_json(run_dir / "state" / "trainer_exit.json", {})
    error_text = json.dumps(exit_state, sort_keys=True).lower()
    if any(marker in error_text for marker in FORBIDDEN_RECOVERY_MARKERS):
        return False
    checkpoint = _latest_resume(run_dir, initial_checkpoint)
    if not (Path(checkpoint) / "metadata.json").is_file():
        return False
    command = [
        str(run_dir.parent.parent.parent / "scripts" / "_formal_trainer_process.sh")
    ]
    # Resolve from the immutable project path, not from output-directory depth.
    project_root = Path(__file__).resolve().parents[3]
    command = [
        str(project_root / "scripts" / "_formal_trainer_process.sh"),
        str(config_path),
        str(run_dir),
        checkpoint,
    ]
    log = (run_dir / "logs" / "watchdog_recovery.log").open(
        "a", encoding="utf-8", buffering=1
    )
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    atomic_write_json(
        run_dir / "state" / "recovery_state.json",
        {
            "attempts": 1,
            "checkpoint": checkpoint,
            "launcher_pid": process.pid,
            "timestamp": time.time(),
        },
    )
    append_jsonl(
        run_dir / "monitor" / "alerts.log.jsonl",
        {
            "level": "YELLOW",
            "event": "automatic_transient_recovery",
            "checkpoint": checkpoint,
            "launcher_pid": process.pid,
            "timestamp": time.time(),
        },
    )
    return True


def run_watchdog(
    config_path: Path,
    run_dir: Path,
    initial_checkpoint: str,
) -> int:
    config = load_config(config_path)
    interval = int(config["monitoring"]["watchdog_interval_seconds"])
    stale_seconds = int(config["monitoring"]["stale_metrics_seconds"])
    minimum_disk = int(config["monitoring"]["minimum_disk_free_gib"]) * 1024**3
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stop:
        now = time.time()
        processes = _process_payload(run_dir)
        trainer_result = read_json(run_dir / "state" / "trainer_result.json", {})
        eval_result = read_json(run_dir / "state" / "eval_worker_result.json", {})
        monitor_result = read_json(run_dir / "state" / "monitor_result.json", {})
        fatal = read_json(run_dir / "state" / "fatal_status.json", {})
        retriever_error = None
        try:
            query_health(str(config["retriever"]["service_url"]))
        except BaseException as exc:
            retriever_error = f"{type(exc).__name__}: {exc}"
        update_metrics = run_dir / "metrics" / "update_metrics.jsonl"
        metrics_age = (
            now - update_metrics.stat().st_mtime
            if update_metrics.is_file()
            else None
        )
        stat = os.statvfs(run_dir)
        disk_free = stat.f_bavail * stat.f_frsize
        record = {
            "timestamp": now,
            "trainer_alive": _alive(processes.get("trainer_pid")),
            "retriever_alive": _alive(processes.get("retriever_pid")),
            "eval_worker_alive": _alive(processes.get("eval_worker_pid")),
            "monitor_alive": _alive(processes.get("monitor_pid")),
            "watchdog_pid": os.getpid(),
            "tmux_alive": _tmux_alive(processes.get("tmux_session")),
            "retriever_health_error": retriever_error,
            "metrics_age_seconds": metrics_age,
            "disk_free_bytes": disk_free,
            "gpu_compute_process_count": _rl_gpu_process_count(),
            "fatal": fatal,
        }
        append_jsonl(run_dir / "logs" / "watchdog_events.jsonl", record)

        if fatal.get("fatal"):
            return 2
        if disk_free < minimum_disk:
            atomic_write_json(
                run_dir / "state" / "fatal_status.json",
                {"fatal": True, "source": "watchdog", "error": "disk_space_below_safety_floor", "timestamp": now},
            )
            trainer_pid = processes.get("trainer_pid")
            if _alive(trainer_pid):
                os.kill(int(trainer_pid), signal.SIGTERM)
            return 2
        if metrics_age is not None and metrics_age > stale_seconds and record["trainer_alive"]:
            append_jsonl(
                run_dir / "monitor" / "alerts.log.jsonl",
                {"level": "YELLOW", "event": "metrics_stale", "age_seconds": metrics_age, "timestamp": now},
            )

        if trainer_result.get("status") == "PASS" and eval_result.get("status") == "PASS":
            if monitor_result.get("status") != "PASS":
                time.sleep(interval)
                continue
            retriever_pid = processes.get("retriever_pid")
            if _alive(retriever_pid):
                os.kill(int(retriever_pid), signal.SIGTERM)
            atomic_write_json(
                run_dir / "state" / "watchdog_result.json",
                {"status": "PASS", "timestamp": time.time()},
            )
            return 0

        if not record["trainer_alive"] and not trainer_result:
            exit_state = read_json(run_dir / "state" / "trainer_exit.json", {})
            if exit_state and int(exit_state.get("exit_code", 1)) != 0:
                if not _attempt_recovery(
                    run_dir=run_dir,
                    config_path=config_path,
                    initial_checkpoint=initial_checkpoint,
                ):
                    atomic_write_json(
                        run_dir / "state" / "fatal_status.json",
                        {"fatal": True, "source": "watchdog", "error": "trainer_exit_not_recoverable", "details": exit_state, "timestamp": now},
                    )
                    return 2
        time.sleep(interval)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal runtime watchdog")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    arguments = parser.parse_args()
    raise SystemExit(
        run_watchdog(
            Path(arguments.config).resolve(),
            Path(arguments.run_dir).resolve(),
            str(Path(arguments.initial_checkpoint).resolve()),
        )
    )


if __name__ == "__main__":
    main()
