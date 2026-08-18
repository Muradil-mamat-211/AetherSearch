#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rl.checkpoint.atomic_commit import AtomicCheckpointCommitter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only integrity and controller-state checkpoint verifier"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-attempt", type=int, required=True)
    parser.add_argument("--expected-cursor", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    metadata = AtomicCheckpointCommitter(checkpoint.parent).validate(checkpoint)
    controller = json.loads(
        (checkpoint / "controller" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    actual = (
        int(metadata.successful_update_step),
        int(metadata.attempt_id),
        int(metadata.data_cursor),
    )
    expected = (
        int(args.expected_step),
        int(args.expected_attempt),
        int(args.expected_cursor),
    )
    if actual != expected:
        raise SystemExit(
            f"checkpoint counters differ: expected={expected}, actual={actual}"
        )
    training_state = controller["training_state"]
    controller_actual = (
        int(training_state["successful_update_step"]),
        int(training_state["attempt_id"]),
        int(training_state["data_cursor"]),
    )
    if controller_actual != expected:
        raise SystemExit(
            "controller state differs from metadata: "
            f"metadata={actual}, controller={controller_actual}"
        )
    payload = {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "successful_update_step": actual[0],
        "attempt_id": actual[1],
        "data_cursor": actual[2],
        "ig_channel": metadata.as_dict()["ig_channel"],
        "outcome_channel": metadata.as_dict()["outcome_channel"],
        "actor_present": (checkpoint / "actor").is_dir(),
        "optimizer_present": (checkpoint / "optimizer").is_dir(),
        "scheduler_present": (checkpoint / "scheduler").is_dir(),
        "integrity_verified": True,
        "model_loaded": False,
        "optimizer_step_executed": False,
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
