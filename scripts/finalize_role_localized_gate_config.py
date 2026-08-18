#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from agentic_rl.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite formal config: {args.output}")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "PASS":
        raise RuntimeError("Calibration manifest is not PASS")
    overlay = yaml.safe_load(args.calibration_config.read_text(encoding="utf-8"))
    advantage = dict(overlay["advantage"])
    role = dict(advantage["role_localized_gate"])
    role.update(
        {
            "lambda_decision": float(manifest["lambda_decision"]),
            "lambda_query": float(manifest["lambda_query"]),
            "calibration_manifest": str(args.manifest.resolve()),
            "calibration_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "calibration_pending": False,
        }
    )
    advantage["role_localized_gate"] = role
    formal_overlay = {
        "extends": "formal_train.yaml",
        "scheduler": {"total_successful_updates": 500},
        "formal": {
            "total_successful_updates": 500,
            "resume_from_successful_update": 0,
            "fresh_start_required": True,
        },
        "formal_schedule": {"total_successful_updates": 500},
        "advantage": advantage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(formal_overlay, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    resolved = load_config(args.output)
    if resolved["advantage"]["role_localized_gate"]["calibration_pending"]:
        raise RuntimeError("Final formal config remains calibration-pending")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "lambda_decision": role["lambda_decision"],
                "lambda_query": role["lambda_query"],
                "manifest_sha256": role["calibration_manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
