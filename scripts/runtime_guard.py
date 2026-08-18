#!/usr/bin/env python3
"""Persistent liveness and metric snapshots for a formal runtime job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_pid(run_dir: Path, name: str) -> int | None:
    for root in (run_dir / "artifacts" / "pids", run_dir / "state" / "pids"):
        path = root / f"{name}.pid"
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if value.isdigit():
            return int(value)
    return None


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _last_jsonl(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(0, end - 512 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return {}
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 4:
            rows.append(
                {
                    "gpu": int(fields[0]),
                    "utilization_percent": int(fields[1]),
                    "memory_used_mib": int(fields[2]),
                    "memory_total_mib": int(fields[3]),
                }
            )
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, row: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--kind", choices=("watchdog", "monitor"), required=True)
    parser.add_argument("--interval", required=True, type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    pid_root = run_dir / "state" / "pids"
    pid_root.mkdir(parents=True, exist_ok=True)
    (pid_root / f"{args.kind}.pid").write_text(
        f"{os.getpid()}\n", encoding="utf-8"
    )
    output = run_dir / "monitor" / f"{args.kind}_events.jsonl"
    latest = run_dir / "monitor" / f"latest_{args.kind}.json"
    started = time.time()
    while True:
        pids = {
            name: _read_pid(run_dir, name)
            for name in ("driver", "retriever", "eval_worker")
        }
        update = _last_jsonl(run_dir / "metrics" / "update_metrics.jsonl")
        progress = _read_json(run_dir / "state" / "training_progress.json")
        fatal = _read_json(run_dir / "state" / "fatal_status.json")
        trainer_result = _read_json(run_dir / "state" / "trainer_result.json")
        eval_result = _read_json(run_dir / "state" / "eval_worker_result.json")
        queue = _read_json(run_dir / "state" / "eval_queue.json")
        successful_update = int(
            update.get(
                "successful_update_step",
                progress.get("successful_update_step", 0),
            )
        )
        row = {
            "timestamp_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "kind": args.kind,
            "pids": pids,
            "alive": {name: _alive(pid) for name, pid in pids.items()},
            "successful_update": successful_update,
            "attempt_id": update.get("attempt_id", progress.get("attempt_id")),
            "latest_update": update,
            "training_progress": progress,
            "fatal": fatal,
            "trainer_result": trainer_result,
            "eval_result": eval_result,
            "eval_queue": queue,
            "gpus": _gpu_snapshot(),
        }
        startup_grace = time.time() - started < 300
        row["alert"] = bool(fatal.get("fatal")) or (
            not startup_grace
            and any(not status for status in row["alive"].values())
            and trainer_result.get("status") not in {"PASS"}
        )
        _append_jsonl(output, row)
        _atomic_json(latest, row)
        if trainer_result.get("status") in {"PASS", "FAIL"}:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
