#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  printf 'Usage: monitor_formal_training_10min.sh CONFIG RUN_DIR [--once]\n' >&2
  exit 2
fi
CONFIG="$(readlink -f "$1")"
RUN_DIR="$(readlink -f "$2")"
ONCE="${3:-}"
mkdir -p "${RUN_DIR}/state/pids" "${RUN_DIR}/monitor"
printf '%s\n' "$$" >"${RUN_DIR}/state/pids/monitor.pid"
exec > >(tee -a "${RUN_DIR}/monitor/monitor_10min.log") \
     2> >(tee -a "${RUN_DIR}/monitor/alerts.log" >&2)
ARGS=(--config "${CONFIG}" --run-dir "${RUN_DIR}")
if [[ "${ONCE}" == "--once" ]]; then
  ARGS+=(--once)
elif [[ -n "${ONCE}" ]]; then
  printf 'Unknown argument: %s\n' "${ONCE}" >&2
  exit 2
fi
exec "${RL_ENV}/bin/python" -u -m agentic_rl.runtime.formal_monitor "${ARGS[@]}"
