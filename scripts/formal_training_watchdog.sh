#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
if [[ "$#" -ne 3 ]]; then
  printf 'Usage: formal_training_watchdog.sh CONFIG RUN_DIR INITIAL_CHECKPOINT\n' >&2
  exit 2
fi
CONFIG="$(readlink -f "$1")"
RUN_DIR="$(readlink -f "$2")"
CHECKPOINT="$(readlink -f "$3")"
mkdir -p "${RUN_DIR}/state/pids" "${RUN_DIR}/logs"
printf '%s\n' "$$" >"${RUN_DIR}/state/pids/watchdog.pid"
exec > >(tee -a "${RUN_DIR}/logs/watchdog.log") \
     2> >(tee -a "${RUN_DIR}/logs/watchdog.log" "${RUN_DIR}/logs/errors.log" >&2)
exec "${RL_ENV}/bin/python" -u -m agentic_rl.runtime.formal_watchdog \
  --config "${CONFIG}" --run-dir "${RUN_DIR}" \
  --initial-checkpoint "${CHECKPOINT}"
