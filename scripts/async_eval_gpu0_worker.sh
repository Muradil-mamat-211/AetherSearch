#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
if [[ "$#" -ne 2 ]]; then
  printf 'Usage: async_eval_gpu0_worker.sh CONFIG RUN_DIR\n' >&2
  exit 2
fi
CONFIG="$(readlink -f "$1")"
RUN_DIR="$(readlink -f "$2")"
RL_PYTHON="$("${RL_PYTHON}" -m agentic_rl.config --config "${CONFIG}" --get paths.rl_python)"
export CUDA_VISIBLE_DEVICES="${AETHERSEARCH_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
mkdir -p "${RUN_DIR}/state/pids" "${RUN_DIR}/logs"
printf '%s\n' "$$" >"${RUN_DIR}/state/pids/eval_worker.pid"
exec > >(tee -a "${RUN_DIR}/logs/eval_worker.log") \
     2> >(tee -a "${RUN_DIR}/logs/eval_worker.log" "${RUN_DIR}/logs/errors.log" >&2)
exec "${RL_PYTHON}" -u -m agentic_rl.runtime.async_eval_worker \
  --config "${CONFIG}" --run-dir "${RUN_DIR}"
