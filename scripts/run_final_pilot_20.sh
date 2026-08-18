#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
RL_PYTHON="${RL_ENV}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/pilot_20_final.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/pilot_20_final"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_final_pilot_20.sh [--dry-run]

Starts the isolated 20-successful-update final Pilot in a new tmux session.
It never starts formal training.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

test -f "${CONFIG}"
test "$(nvidia-smi -L | wc -l)" -eq 5
test -x "${RL_PYTHON}"
command -v tmux >/dev/null

"${RL_PYTHON}" - "${PROJECT_ROOT}" "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path
from agentic_rl.config import load_config
from agentic_rl.runtime.verl_runtime_adapter import assert_exact_ig_parity_gate

root = Path(sys.argv[1])
config = load_config(sys.argv[2])
gate = assert_exact_ig_parity_gate(config)
stage_sc = root / "runtime/stage_results/stage_sc.json"
sc = json.loads(stage_sc.read_text())
assert gate["allow_fast_path_training"] is True
assert sc["status"] == "PASS"
assert sc["stop_completion_count"] == 2 * sc["stop_state_count"]
assert sc["optimizer_steps"] == 0
assert sc["scheduler_steps"] == 0
assert sc["checkpoint_writes"] == 0
assert config["data"]["expected_rows"] == 150745
assert config["data"]["shuffle_seed"] == 20260724
assert config["pilot"]["successful_updates"] == 20
assert config["pilot"]["checkpoints"] == [20]
assert config["pilot"]["evaluations"] == []
assert config["evaluation"]["asynchronous"] is True
assert config["evaluation"]["physical_gpu"] == 0
assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
assert config["formal_schedule"]["fixed_eval_every_successful_updates"] == 20
assert config["advantage"]["lambda_ig"] == 0.3
assert config["advantage"]["lambda_task"] == 1.0
assert config["paths"]["actor_model"] == config["paths"]["reference_model"]
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'PILOT20_DRY_RUN=PASS\nconfig=%s\n' "${CONFIG}"
  exit 0
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf 'GPU compute processes are already present; Pilot is fail-closed.\n' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop the owning job first.\n' >&2
  exit 1
fi
AVAILABLE_GB="$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')"
if [[ "${AVAILABLE_GB}" -lt 80 ]]; then
  printf 'Pilot requires at least 80 GiB free; found %s GiB.\n' "${AVAILABLE_GB}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="pilot20_${TIMESTAMP}_lr2e7_t1_topP095_g16"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
SESSION="igpo_pilot20_${TIMESTAMP}"
if [[ -e "${RUN_DIR}" ]]; then
  printf 'Pilot run directory already exists: %s\n' "${RUN_DIR}" >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'Pilot tmux session already exists: %s\n' "${SESSION}" >&2
  exit 1
fi

mkdir -p \
  "${RUN_DIR}/configs" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/metrics" \
  "${RUN_DIR}/checkpoints" \
  "${RUN_DIR}/eval" \
  "${RUN_DIR}/state" \
  "${RUN_DIR}/artifacts/pids" \
  "${RUN_DIR}/reports"
cp "${CONFIG}" "${RUN_DIR}/configs/pilot_20_final.yaml"
PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" - "${CONFIG}" \
  "${RUN_DIR}/configs/resolved_config.yaml" <<'PY'
import sys
import yaml
from agentic_rl.config import load_config
config = load_config(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
{
  date -u '+timestamp_utc=%Y-%m-%dT%H:%M:%SZ'
  "${RL_PYTHON}" --version
  nvidia-smi
  "${RL_PYTHON}" - <<'PY'
import ray, torch, transformers, verl, vllm
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("ray=" + ray.__version__)
print("transformers=" + transformers.__version__)
print("verl=" + str(getattr(verl, "__version__", None)))
print("vllm=" + vllm.__version__)
PY
} >"${RUN_DIR}/configs/environment.txt" 2>&1
cp "${PROJECT_ROOT}/MANIFEST.sha256" \
  "${RUN_DIR}/configs/source_manifest.sha256"
printf '%q ' \
  bash "${PROJECT_ROOT}/scripts/_run_runtime_job.sh" \
  PILOT20 "${RUN_DIR}/configs/resolved_config.yaml" "${RUN_DIR}" \
  >"${RUN_DIR}/configs/launch_command.sh"
printf '\n' >>"${RUN_DIR}/configs/launch_command.sh"
ln -sfn "${RUN_ID}" "${OUTPUT_ROOT}/latest"

TMUX_COMMAND="bash '${PROJECT_ROOT}/scripts/_run_runtime_job.sh' PILOT20 '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}'; rc=\$?; printf '%s\n' \"\${rc}\" >'${RUN_DIR}/artifacts/exit_code'; exit \"\${rc}\""
tmux new-session -d -s "${SESSION}" "${TMUX_COMMAND}"
printf '%s\n' "${SESSION}" >"${RUN_DIR}/artifacts/tmux_session"

for _ in $(seq 1 240); do
  if [[ -s "${RUN_DIR}/artifacts/pids/driver.pid" ]]; then
    break
  fi
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    break
  fi
  sleep 1
done

DRIVER_PID="$(cat "${RUN_DIR}/artifacts/pids/driver.pid" 2>/dev/null || printf 'PENDING')"
RETRIEVER_PID="$(cat "${RUN_DIR}/artifacts/pids/retriever.pid" 2>/dev/null || printf 'PENDING')"
EVAL_PID="$(cat "${RUN_DIR}/artifacts/pids/eval_worker.pid" 2>/dev/null || printf 'PENDING')"
printf 'RUN_ID=%s\n' "${RUN_ID}"
printf 'RUN_DIR=%s\n' "${RUN_DIR}"
printf 'TMUX_SESSION=%s\n' "${SESSION}"
printf 'DRIVER_PID=%s\n' "${DRIVER_PID}"
printf 'RETRIEVER_PID=%s\n' "${RETRIEVER_PID}"
printf 'EVAL_WORKER_PID=%s\n' "${EVAL_PID}"
printf 'CONSOLE_LOG=%s\n' "${RUN_DIR}/logs/console.log"
printf 'METRICS_DIR=%s\n' "${RUN_DIR}/metrics"
printf 'CHECKPOINT_DIR=%s\n' "${RUN_DIR}/checkpoints"
printf 'EVAL_DIR=%s\n' "${RUN_DIR}/eval"
