#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
RUN_DIR=""
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: tail_formal_logs.sh [--run-dir /absolute/path/to/RUN_ID] [--dry-run]'
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-dir) [[ "$#" -ge 2 ]] || exit 2; RUN_DIR="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(readlink -f "${PROJECT_ROOT}/outputs/formal_training/latest")"
else
  RUN_DIR="$(readlink -f "${RUN_DIR}")"
fi
LOG="${RUN_DIR}/monitor/monitor_10min.log"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'FORMAL_TAIL_DRY_RUN=PASS\nlog=%s\n' "${LOG}"
  exit 0
fi
test -f "${LOG}"
exec tail -n 100 -F "${LOG}"
