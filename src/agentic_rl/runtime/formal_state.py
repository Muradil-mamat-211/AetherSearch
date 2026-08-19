from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: str | Path, default: Any = None) -> Any:
    source = Path(path)
    if not source.is_file():
        return default
    return json.loads(source.read_text(encoding="utf-8"))


@contextmanager
def _locked_queue(run_dir: str | Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    state_root = Path(run_dir) / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / "eval_queue.lock"
    queue_path = state_root / "eval_queue.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload = read_json(
            queue_path,
            {"schema_version": 1, "tasks": [], "updated_at": None},
        )
        try:
            yield queue_path, payload
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def enqueue_eval(
    run_dir: str | Path,
    *,
    update: int,
    model_path: str | Path,
    actor_checksum: str,
) -> dict[str, Any]:
    with _locked_queue(run_dir) as (path, payload):
        existing = next(
            (item for item in payload["tasks"] if int(item["update"]) == int(update)),
            None,
        )
        if existing is not None:
            if Path(existing["model_path"]).resolve() != Path(model_path).resolve():
                raise RuntimeError(f"Eval Update {update} was queued with another model")
            return dict(existing)
        now = time.time()
        task = {
            "update": int(update),
            "model_path": str(Path(model_path).resolve()),
            "actor_checksum": str(actor_checksum),
            "status": "pending",
            "queued_at": now,
            "started_at": None,
            "completed_at": None,
            "worker_pid": None,
            "attempts": 0,
            "last_error": None,
            "wait_reason": None,
        }
        payload["tasks"].append(task)
        payload["tasks"].sort(key=lambda item: int(item["update"]))
        payload["updated_at"] = now
        atomic_write_json(path, payload)
        return dict(task)


def seed_completed_eval(
    run_dir: str | Path,
    *,
    update: int,
    model_path: str | Path,
    actor_checksum: str,
) -> None:
    enqueue_eval(
        run_dir,
        update=update,
        model_path=model_path,
        actor_checksum=actor_checksum,
    )
    complete_eval(run_dir, update=update, error=None)


def claim_next_eval(run_dir: str | Path, *, worker_pid: int) -> dict[str, Any] | None:
    with _locked_queue(run_dir) as (path, payload):
        pending = sorted(
            (item for item in payload["tasks"] if item["status"] == "pending"),
            key=lambda item: int(item["update"]),
        )
        if not pending:
            return None
        task = pending[0]
        now = time.time()
        task.update(
            {
                "status": "running",
                "started_at": now,
                "worker_pid": int(worker_pid),
                "attempts": int(task.get("attempts", 0)) + 1,
                "wait_reason": None,
            }
        )
        payload["updated_at"] = now
        atomic_write_json(path, payload)
        return dict(task)


def defer_eval(run_dir: str | Path, *, update: int, reason: str) -> None:
    with _locked_queue(run_dir) as (path, payload):
        task = next(item for item in payload["tasks"] if int(item["update"]) == int(update))
        task.update(
            {
                "status": "pending",
                "worker_pid": None,
                "wait_reason": str(reason),
                "last_error": None,
            }
        )
        payload["updated_at"] = time.time()
        atomic_write_json(path, payload)


def complete_eval(
    run_dir: str | Path,
    *,
    update: int,
    error: str | None,
) -> None:
    with _locked_queue(run_dir) as (path, payload):
        task = next(item for item in payload["tasks"] if int(item["update"]) == int(update))
        now = time.time()
        task.update(
            {
                "status": "completed" if error is None else "pending",
                "completed_at": now if error is None else None,
                "worker_pid": None,
                "last_error": error,
                "wait_reason": None if error is None else "evaluation_error",
            }
        )
        payload["updated_at"] = now
        atomic_write_json(path, payload)


def eval_queue_snapshot(run_dir: str | Path) -> dict[str, Any]:
    with _locked_queue(run_dir) as (_path, payload):
        return json.loads(json.dumps(payload))
