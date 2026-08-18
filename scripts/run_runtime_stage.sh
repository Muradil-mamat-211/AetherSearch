#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

STAGE="${1:?runtime stage A, B, C, or D is required}"
STAGE="${STAGE^^}"
case "${STAGE}" in
  A) PREVIOUS="" ;;
  SC) PREVIOUS="" ;;
  B) PREVIOUS="a" ;;
  C) PREVIOUS="b" ;;
  D) PREVIOUS="c" ;;
  *) printf 'Unsupported runtime stage: %s\n' "${STAGE}" >&2; exit 2 ;;
esac

RL_PYTHON="${RL_ENV}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/base.yaml"
RUNTIME="${PROJECT_ROOT}/runtime"
LOG="${RUNTIME}/logs/runtime_stage_${STAGE}.log"
mkdir -p "${RUNTIME}/logs" "${RUNTIME}/pids" "${RUNTIME}/stage_results"

if [[ -n "${PREVIOUS}" ]]; then
  "${RL_PYTHON}" - "${RUNTIME}/stage_results/stage_${PREVIOUS}.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text()).get("status") != "PASS":
    raise SystemExit(f"Previous runtime stage is not PASS: {path}")
PY
fi

RETRIEVER_STARTED=0
if ! "${RL_PYTHON}" -c \
  "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
  >/dev/null 2>&1; then
  "${PROJECT_ROOT}/scripts/launch_retriever_gpu0.sh" \
    >"${RUNTIME}/logs/retriever_stage_${STAGE}.launcher.log" 2>&1 &
  RETRIEVER_PID=$!
  RETRIEVER_STARTED=1
  printf '%s\n' "${RETRIEVER_PID}" >"${RUNTIME}/pids/retriever_stage_${STAGE}.pid"
fi

cleanup() {
  if [[ "${RETRIEVER_STARTED}" -eq 1 ]] && \
      kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    kill -TERM "${RETRIEVER_PID}"
    wait "${RETRIEVER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if "${RL_PYTHON}" -c \
    "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
    >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
"${RL_PYTHON}" -c \
  "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
  >/dev/null

if [[ "$(nvidia-smi -L | wc -l)" -ne 5 ]]; then
  printf '%s\n' "Runtime stages require five physical GPUs." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_memory_monitor_refresh_ms=1000
export RAY_memory_usage_threshold=0.80
export AGENTIC_RL_RUNTIME_STAGE="${STAGE}"
export AGENTIC_RL_SMOKE_MODEL_CHECKPOINTS=0
# CPU-only Ray actors are numerous (AgentLoop, Outcome, task builders). Keep
# each process single-threaded; the actor count supplies inter-op parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1

"${RL_PYTHON}" -m agentic_rl.runtime.entrypoint --config "${CONFIG}" \
  2>&1 | tee "${LOG}"

SMOKE_OUTPUT="${PROJECT_ROOT}/outputs/runtime_stage_${STAGE,,}"
if [[ -d "${SMOKE_OUTPUT}/checkpoints" ]] && \
    find "${SMOKE_OUTPUT}/checkpoints" -mindepth 1 -print -quit | \
      grep -q .; then
  printf 'Smoke Stage %s wrote a forbidden checkpoint under %s\n' \
    "${STAGE}" "${SMOKE_OUTPUT}" >&2
  exit 1
fi
