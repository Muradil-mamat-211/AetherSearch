#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

RL_PYTHON="${RL_ENV}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/base.yaml"
PARITY="${PROJECT_ROOT}/artifacts/exact_ig_official_alignment_v3_20260730/EXACT_IG_FAST_SEQUENTIAL_PARITY_V3.json"
RUNTIME="${PROJECT_ROOT}/runtime"

"${RL_PYTHON}" - "${PARITY}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Exact-IG precision gate is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("gate_pass") is not True:
    raise SystemExit(f"Exact-IG precision gate is not PASS: {path}")
if payload.get("oracle_validated") is not True:
    raise SystemExit(f"Sequential Oracle validation is not PASS: {path}")
if payload.get("allow_next_stage") is not True:
    raise SystemExit(f"Exact-IG V3 does not allow runtime stages: {path}")
if payload.get("selected_mode") != "OFFICIAL_BF16_FAST_FULL_LOGITS":
    raise SystemExit(f"Official BF16 full-logits Fast mode is absent: {path}")
equivalence = payload.get("ragen", {})
if equivalence.get("selected_ids_equal") is not True:
    raise SystemExit(f"Fast/Oracle selected IDs differ: {path}")
PY

if [[ "$(nvidia-smi -L | wc -l)" -ne 5 ]]; then
  printf '%s\n' "Exactly five GPUs are required." >&2
  exit 1
fi

mkdir -p "${RUNTIME}/logs" "${RUNTIME}/pids"
RETRIEVER_STARTED=0
if ! "${RL_PYTHON}" -c \
  "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
  >/dev/null 2>&1; then
  "${PROJECT_ROOT}/scripts/launch_retriever_gpu0.sh" \
    >"${RUNTIME}/logs/retriever_parallel_smoke.log" 2>&1 &
  RETRIEVER_PID=$!
  RETRIEVER_STARTED=1
  printf '%s\n' "${RETRIEVER_PID}" >"${RUNTIME}/pids/retriever_smoke.pid"
fi

cleanup() {
  if [[ "${RETRIEVER_STARTED}" -eq 1 ]] && kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
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

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_memory_monitor_refresh_ms=1000
export RAY_memory_usage_threshold=0.80

for stage in A B C D; do
  export AGENTIC_RL_RUNTIME_STAGE="${stage}"
  "${RL_PYTHON}" -m agentic_rl.runtime.entrypoint --config "${CONFIG}" \
    >"${RUNTIME}/logs/runtime_stage_${stage}.log" 2>&1
done
