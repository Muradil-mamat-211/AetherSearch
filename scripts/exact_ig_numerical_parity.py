#!/usr/bin/env python3
"""Compatibility entry point for the corrected Exact-IG validation."""

from validate_exact_ig_fast_path_repair import parse_args, run


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
