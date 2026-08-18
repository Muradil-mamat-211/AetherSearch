#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
exec "${PROJECT_ROOT}/scripts/run_exact_ig_fast_path_repair_validation.sh" "${1:-}"
