#!/usr/bin/env python3
"""Fail-closed compatibility entry point for the superseded V2 validator."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--audit-dir")
    parser.add_argument("--device")
    parser.add_argument("--cpu-threads")
    parser.add_argument("--maximum-validation-length")
    parser.add_argument("--model-mode")
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def run(_args: argparse.Namespace) -> int:
    print(
        "This Exact-IG V2 validator is superseded and cannot authorize "
        "runtime execution. Run artifacts/exact_ig_official_alignment_"
        "v3_20260730/run_exact_ig_official_alignment_validation.sh."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
