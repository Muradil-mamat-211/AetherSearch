#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
exec \
  "${PROJECT_ROOT}/artifacts/exact_ig_official_alignment_v3_20260730/run_exact_ig_official_alignment_validation.sh"
