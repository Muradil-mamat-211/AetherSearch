#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -ne 1 ]]; then
  printf 'Usage: _formal_retriever_process.sh RUN_DIR\n' >&2
  exit 2
fi
RUN_DIR="$(readlink -f "$1")"
mkdir -p "${RUN_DIR}/state/pids" "${RUN_DIR}/logs"
printf '%s\n' "$$" >"${RUN_DIR}/state/pids/retriever.pid"
exec >>"${RUN_DIR}/logs/retriever.log" 2>&1
exec "${PROJECT_ROOT}/scripts/launch_retriever_gpu0.sh"
