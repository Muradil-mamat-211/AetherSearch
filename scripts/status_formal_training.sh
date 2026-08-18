#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR=""
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: status_formal_training.sh [--run-dir /absolute/path/to/RUN_ID] [--dry-run]'
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
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'FORMAL_STATUS_DRY_RUN=PASS\nrun_dir=%s\n' "${RUN_DIR}"
  exit 0
fi
printf 'RUN_DIR=%s\n' "${RUN_DIR}"
for name in trainer retriever; do
  pid_file="${RUN_DIR}/state/pids/${name}.pid"
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
    printf '%s_pid=%s status=RUNNING\n' "${name}" "${pid}"
  else
    printf '%s_pid=%s status=STOPPED\n' "${name}" "${pid:-NONE}"
  fi
done
for name in eval_worker monitor watchdog; do
  pid_file="${RUN_DIR}/state/pids/${name}.pid"
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
    printf '%s_pid=%s status=RUNNING\n' "${name}" "${pid}"
  else
    printf '%s_pid=%s status=STOPPED\n' "${name}" "${pid:-NONE}"
  fi
done
for state in training_state trainer_result fatal_status eval_queue; do
  if [[ -f "${RUN_DIR}/state/${state}.json" ]]; then
    printf '%s=%s\n' "${state}" "$(tr '\n' ' ' <"${RUN_DIR}/state/${state}.json")"
  fi
done
if [[ -f "${RUN_DIR}/metrics/update_metrics.jsonl" ]]; then
  printf 'successful_update_records=%s\n' \
    "$(wc -l <"${RUN_DIR}/metrics/update_metrics.jsonl")"
  tail -n 1 "${RUN_DIR}/metrics/update_metrics.jsonl"
fi
