#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

if [[ "$#" -ne 3 ]]; then
  printf 'Usage: _formal_trainer_process.sh CONFIG RUN_DIR RESUME_CHECKPOINT\n' >&2
  exit 2
fi
CONFIG="$(readlink -f "$1")"
RUN_DIR="$(readlink -f "$2")"
RESUME_CHECKPOINT="$(readlink -f "$3")"
RL_PYTHON="$("${RL_PYTHON}" -m agentic_rl.config --config "${CONFIG}" --get paths.rl_python)"
RL_GPUS="$("${RL_PYTHON}" - "${CONFIG}" <<'PY'
import sys
from agentic_rl.config import load_config
print(",".join(str(value) for value in load_config(sys.argv[1])["hardware"]["rl_physical_gpus"]))
PY
)"

export CUDA_VISIBLE_DEVICES="${AGENTIC_RL_RL_CUDA_VISIBLE_DEVICES:-${RL_GPUS}}"
export AGENTIC_RL_EXPECTED_RL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Keep Ray's cgroup-aware memory monitor active with a conservative threshold.
export RAY_memory_monitor_refresh_ms=1000
export RAY_memory_usage_threshold=0.80
export AGENTIC_RL_RUNTIME_STAGE=FORMAL
export AGENTIC_RL_RUN_DIR="${RUN_DIR}"
export AGENTIC_RL_FORMAL_RUN_ROOT="${RUN_DIR}"
export AGENTIC_RL_RESUME_CHECKPOINT="${RESUME_CHECKPOINT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1

mkdir -p "${RUN_DIR}/state/pids" "${RUN_DIR}/logs"
CHILD_PID=""
forward_signal() {
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
  fi
}
trap forward_signal INT TERM

set +e
stdbuf -oL -eL "${RL_PYTHON}" -u \
  -m agentic_rl.runtime.entrypoint \
  --config "${CONFIG}" \
  > >(tee -a "${RUN_DIR}/logs/console.log" "${RUN_DIR}/logs/train_rank0.log" "${RUN_DIR}/logs/ray_driver.log") \
  2> >(tee -a "${RUN_DIR}/logs/console.log" "${RUN_DIR}/logs/train_rank0.log" "${RUN_DIR}/logs/ray_driver.log" "${RUN_DIR}/logs/errors.log" >&2) &
CHILD_PID=$!
printf '%s\n' "${CHILD_PID}" >"${RUN_DIR}/state/pids/trainer.pid"
wait "${CHILD_PID}"
STATUS=$?
set -e

"${RL_PYTHON}" - "${RUN_DIR}" "${STATUS}" <<'PY'
import sys, time
from pathlib import Path
from agentic_rl.runtime.formal_state import atomic_write_json
run_dir = Path(sys.argv[1])
status = int(sys.argv[2])
errors = run_dir / "logs" / "errors.log"
tail = ""
if errors.is_file():
    tail = errors.read_text(encoding="utf-8", errors="replace")[-20000:]
atomic_write_json(
    run_dir / "state" / "trainer_exit.json",
    {
        "exit_code": status,
        "timestamp": time.time(),
        "error_tail": tail if status else "",
    },
)
PY
exit "${STATUS}"
