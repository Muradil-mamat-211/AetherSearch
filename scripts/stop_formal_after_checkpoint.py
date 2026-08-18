#!/usr/bin/env python3
"""Gracefully stop one known formal run after a complete checkpoint commit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def checkpoint_commit_is_complete(run: Path, update: int) -> tuple[bool, str]:
    model = run / "checkpoints" / "models" / f"update_{update}"
    resume = run / "checkpoints" / "resume" / f"update_{update}"
    required = [
        model / "COMPLETED",
        model / "training_metadata.json",
        resume / "metadata.json",
        resume / "integrity.sha256.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    latest = read_json(run / "state" / "latest_resume_checkpoint.json")
    if not latest or Path(str(latest.get("path", ""))).name != f"update_{update}":
        missing.append("state/latest_resume_checkpoint.json")
    if missing:
        return False, "missing=" + ",".join(missing)
    return True, "model_and_resume_artifacts_present"


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--tmux-target", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--post-sigint-seconds", type=int, default=180)
    args = parser.parse_args()

    log = args.run / "logs" / f"stop_after_u{args.update}.log"
    append_log(log, f"watcher_started update={args.update} tmux_target={args.tmux_target}")
    while True:
        ok, detail = checkpoint_commit_is_complete(args.run, args.update)
        if ok:
            append_log(log, f"checkpoint_commit_verified update={args.update} detail={detail}")
            session = args.tmux_target.split(":", 1)[0]
            probe = subprocess.run(
                ["tmux", "has-session", "-t", session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if probe.returncode != 0:
                append_log(log, "BLOCKED trainer_tmux_session_missing_before_stop")
                return 2
            append_log(log, f"sending_graceful_sigint target={args.tmux_target}")
            subprocess.run(
                ["tmux", "send-keys", "-t", args.tmux_target, "C-c"],
                check=False,
            )
            deadline = time.monotonic() + args.post_sigint_seconds
            while time.monotonic() < deadline:
                probe = subprocess.run(
                    ["tmux", "has-session", "-t", session],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if probe.returncode != 0:
                    append_log(log, "trainer_tmux_session_exited_after_sigint")
                    return 0
                time.sleep(5)
            append_log(log, "SIGINT_SENT_TRAINER_DID_NOT_EXIT_WITHIN_GRACE_WINDOW")
            return 3
        append_log(log, f"waiting update={args.update} {detail}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
