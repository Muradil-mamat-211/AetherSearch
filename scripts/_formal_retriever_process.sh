#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  printf 'Usage: _formal_retriever_process.sh [CONFIG] RUN_DIR\n' >&2
  exit 2
fi
if [[ "$#" -eq 1 ]]; then
  RUN_DIR="$(readlink -f "$1")"
  CONFIG="${RUN_DIR}/configs/resolved_config.yaml"
else
  CONFIG="$(readlink -f "$1")"
  RUN_DIR="$(readlink -f "$2")"
fi
test -f "${CONFIG}"
mkdir -p "${RUN_DIR}/state/pids" "${RUN_DIR}/logs"
printf '%s\n' "$$" >"${RUN_DIR}/state/pids/retriever.pid"
exec >>"${RUN_DIR}/logs/retriever.log" 2>&1
exec "${PROJECT_ROOT}/scripts/launch_retriever.sh" "${CONFIG}" "${RUN_DIR}"
