#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  "Pre-repair RAGEN equivalence is invalid after the Exact-IG schema change." \
  "This restricted repair intentionally does not execute or alter RAGEN."
exit 2
