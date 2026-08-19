#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

usage() {
  cat <<'EOF'
Usage: _run_runtime_job.sh STAGE CONFIG RUN_DIR [RESUME_CHECKPOINT]

Internal process supervisor. Use train_rl.sh or resume_rl.sh for public
launches.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  usage >&2
  exit 2
fi

STAGE="${1^^}"
CONFIG="$(readlink -f "$2")"
RUN_DIR="$(readlink -m "$3")"
RESUME_CHECKPOINT="${4:-}"
test -f "${CONFIG}"

mapfile -d '' -t RUNTIME_VALUES < <(
  "${RL_PYTHON}" - "${CONFIG}" <<'PY'
import sys
from agentic_rl.config import load_config
from agentic_rl.topology import TopologyPlan

config = load_config(sys.argv[1])
plan = TopologyPlan.from_config(config)
values = (
    config["paths"]["rl_python"],
    "" if plan.retriever_physical_gpu is None else plan.retriever_cuda_visible_devices,
    plan.eval_cuda_visible_devices,
    plan.rl_cuda_visible_devices,
    config["retriever"]["service_url"],
    plan.learner_world_size,
    config["ray"].get("memory_monitor_refresh_ms", 1000),
    config["ray"].get("memory_usage_threshold", 0.80),
)
for value in values:
    print(value, end="\0")
PY
)
if [[ "${#RUNTIME_VALUES[@]}" -ne 8 ]]; then
  printf 'Failed to resolve the runtime configuration.\n' >&2
  exit 1
fi
RL_PYTHON="${RUNTIME_VALUES[0]}"
RETRIEVER_GPU="${RUNTIME_VALUES[1]}"
EVAL_GPU="${RUNTIME_VALUES[2]}"
RL_GPUS="${RUNTIME_VALUES[3]}"
RETRIEVER_URL="${RUNTIME_VALUES[4]}"
RL_WORLD_SIZE="${RUNTIME_VALUES[5]}"
RAY_MEMORY_REFRESH_MS="${RUNTIME_VALUES[6]}"
RAY_MEMORY_THRESHOLD="${RUNTIME_VALUES[7]}"
if [[ -z "${RETRIEVER_GPU}" || -z "${RL_GPUS}" ]]; then
  printf 'Resolved topology must assign Retriever and RL GPU roles.\n' >&2
  exit 1
fi
LOG_DIR="${RUN_DIR}/logs"
PID_DIR="${RUN_DIR}/artifacts/pids"

test -x "${RL_PYTHON}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"
touch \
  "${LOG_DIR}/console.log" \
  "${LOG_DIR}/train_rank0.log" \
  "${LOG_DIR}/retriever.log" \
  "${LOG_DIR}/ray_driver.log" \
  "${LOG_DIR}/eval_worker.log" \
  "${LOG_DIR}/errors.log"
for rank in $(seq 0 $((RL_WORLD_SIZE - 1))); do
  touch "${LOG_DIR}/fsdp_rank${rank}.log"
done

exec > >(tee -a "${LOG_DIR}/console.log") \
     2> >(tee -a "${LOG_DIR}/errors.log" >&2)

DRIVER_PID=""
RETRIEVER_PID=""
EVAL_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${DRIVER_PID}" ]] && kill -0 "${DRIVER_PID}" 2>/dev/null; then
    kill -TERM "${DRIVER_PID}" 2>/dev/null || true
    wait "${DRIVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${EVAL_PID}" ]] && kill -0 "${EVAL_PID}" 2>/dev/null; then
    kill -TERM "${EVAL_PID}" 2>/dev/null || true
    wait "${EVAL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RETRIEVER_PID}" ]] && kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    kill -TERM "${RETRIEVER_PID}" 2>/dev/null || true
    wait "${RETRIEVER_PID}" 2>/dev/null || true
  fi
  rm -f \
    "${PID_DIR}/driver.pid" \
    "${PID_DIR}/retriever.pid" \
    "${PID_DIR}/eval_worker.pid"
  exit "${status}"
}
trap cleanup EXIT INT TERM

printf 'stage=%s\nconfig=%s\nrun_dir=%s\n' \
  "${STAGE}" "${CONFIG}" "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}" \
  "${PROJECT_ROOT}/scripts/launch_retriever.sh" "${CONFIG}" "${RUN_DIR}" \
  >>"${LOG_DIR}/retriever.log" 2>&1 &
RETRIEVER_PID=$!
printf '%s\n' "${RETRIEVER_PID}" >"${PID_DIR}/retriever.pid"

for _ in $(seq 1 240); do
  if "${RL_PYTHON}" -c \
    'import sys; from agentic_rl.retriever.health import query_health; query_health(sys.argv[1])' \
    "${RETRIEVER_URL}" \
    >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    printf 'Retriever exited before health became ready.\n' >&2
    exit 1
  fi
  sleep 2
done
"${RL_PYTHON}" -c \
  'import sys; from agentic_rl.retriever.health import query_health; query_health(sys.argv[1])' \
  "${RETRIEVER_URL}" \
  >/dev/null

if [[ "${STAGE}" == "PILOT20" || "${STAGE}" == "FORMAL" ]]; then
  if [[ -n "${EVAL_GPU}" ]]; then
    AETHERSEARCH_EVAL_CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
      "${PROJECT_ROOT}/scripts/async_eval_worker.sh" \
      "${CONFIG}" "${RUN_DIR}" &
    EVAL_PID=$!
    printf '%s\n' "${EVAL_PID}" >"${PID_DIR}/eval_worker.pid"
  else
    printf 'No eval role is configured; continuing without async evaluation.\n'
    EVAL_PID=""
  fi
fi

export CUDA_VISIBLE_DEVICES="${RL_GPUS}"
export AGENTIC_RL_EXPECTED_RL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Keep Ray's memory monitor active using the resolved runtime profile.
export RAY_memory_monitor_refresh_ms="${RAY_MEMORY_REFRESH_MS}"
export RAY_memory_usage_threshold="${RAY_MEMORY_THRESHOLD}"
export AGENTIC_RL_RUNTIME_STAGE="${STAGE}"
export AGENTIC_RL_RUN_DIR="${RUN_DIR}"
if [[ "${STAGE}" == "FORMAL" ]]; then
  export AGENTIC_RL_FORMAL_RUN_ROOT="${RUN_DIR}"
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
# A fresh formal launch must not inherit a resume path from the invoking shell.
unset AGENTIC_RL_RESUME_CHECKPOINT
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  export AGENTIC_RL_RESUME_CHECKPOINT="$(readlink -f "${RESUME_CHECKPOINT}")"
fi

stdbuf -oL -eL "${RL_PYTHON}" -u \
  -m agentic_rl.runtime.entrypoint \
  --config "${CONFIG}" \
  > >(tee -a "${LOG_DIR}/ray_driver.log" "${LOG_DIR}/train_rank0.log") \
  2> >(tee -a "${LOG_DIR}/ray_driver.log" "${LOG_DIR}/train_rank0.log" >&2) &
DRIVER_PID=$!
printf '%s\n' "${DRIVER_PID}" >"${PID_DIR}/driver.pid"
wait "${DRIVER_PID}"
DRIVER_PID=""
if [[ -n "${EVAL_PID}" ]]; then
  wait "${EVAL_PID}"
  EVAL_PID=""
fi
