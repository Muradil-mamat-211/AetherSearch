#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

GATE="${1:-}"
case "${GATE}" in
  A) STAGE=MICA_E2E_NOUPDATE; GATE_DIR=gate_a_e2e_no_update ;;
  B) STAGE=MICA_ONE_UPDATE; GATE_DIR=gate_b_one_update ;;
  C) STAGE=MICA_FORMAL_SHAPE; GATE_DIR=gate_c_formal_shape ;;
  *) printf 'Usage: %s A|B|C\n' "$0" >&2; exit 2 ;;
esac

BASE_CONFIG="${PROJECT_ROOT}/configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml"
REPORT_ROOT="${PROJECT_ROOT}/reports/preformal_runtime_qualification"
RUN_DIR="${REPORT_ROOT}/runtime/${GATE_DIR}"
RL_PYTHON="${RL_ENV}/bin/python"

test -f "${BASE_CONFIG}"
test "$(nvidia-smi -L | wc -l)" -eq 4
if [[ -n "${AGENTIC_RL_RESUME_CHECKPOINT:-}" ]]; then
  printf 'Preformal MICA qualification rejects resume checkpoints.\n' >&2
  exit 1
fi
unset AGENTIC_RL_RESUME_CHECKPOINT
if [[ -e "${RUN_DIR}" ]]; then
  printf 'Qualification run directory already exists: %s\n' "${RUN_DIR}" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf 'GPU compute processes are present before qualification.\n' >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/stage_results"
CONFIG="${RUN_DIR}/resolved_gate_config.yaml"
"${RL_PYTHON}" "${PROJECT_ROOT}/scripts/resolve_mica_formal_config.py" \
  --input "${BASE_CONFIG}" \
  --output "${CONFIG}" \
  --total-successful-updates 500
RETRIEVER_PID=""
MONITOR_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RETRIEVER_PID}" ]] && kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    kill -TERM "${RETRIEVER_PID}" 2>/dev/null || true
    wait "${RETRIEVER_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 "${PROJECT_ROOT}/scripts/launch_retriever_gpu0.sh" \
  >"${RUN_DIR}/logs/retriever.log" 2>&1 &
RETRIEVER_PID=$!
for _ in $(seq 1 240); do
  if "${RL_PYTHON}" -c \
    "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
    >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    printf 'Retriever exited before becoming healthy.\n' >&2
    exit 1
  fi
  sleep 2
done
"${RL_PYTHON}" -c \
  "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
  >/dev/null

(
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';'
    printf '\n'
    sleep 2
  done
) >"${RUN_DIR}/logs/gpu_poll.csv" &
MONITOR_PID=$!

export CUDA_VISIBLE_DEVICES=1,2,3
export AGENTIC_RL_EXPECTED_RL_CUDA_VISIBLE_DEVICES=1,2,3
export AGENTIC_RL_RUNTIME_STAGE="${STAGE}"
export AGENTIC_RL_RUN_DIR="${RUN_DIR}"
export AGENTIC_RL_SMOKE_MODEL_CHECKPOINTS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_memory_monitor_refresh_ms=1000
export RAY_memory_usage_threshold=0.80
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1

stdbuf -oL -eL "${RL_PYTHON}" -u \
  -m agentic_rl.runtime.entrypoint --config "${CONFIG}" \
  > >(tee "${RUN_DIR}/logs/runtime.log") \
  2> >(tee "${RUN_DIR}/logs/runtime.err" >&2)

if [[ -d "${RUN_DIR}/checkpoints" ]] && \
   find "${RUN_DIR}/checkpoints" -mindepth 1 -print -quit | grep -q .; then
  printf 'Qualification stage wrote a forbidden checkpoint.\n' >&2
  exit 1
fi

RESULT="${RUN_DIR}/stage_results/stage_${STAGE,,}.json"
"${RL_PYTHON}" - "${RESULT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing qualification result: {path}")
result = json.loads(path.read_text(encoding="utf-8"))
if result.get("status") != "PASS":
    raise SystemExit(f"Qualification failed: {result}")
print(path)
PY
