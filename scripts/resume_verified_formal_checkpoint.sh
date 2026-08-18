#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
RL_PYTHON="${RL_ENV}/bin/python"
# Model/checkpoint validation is intentionally slow. Serialize launcher
# invocations so two callers cannot create concurrent recovery runs.
exec 9>"/tmp/igpo_mica_verified_resume_launcher.lock"
flock -n 9 || {
  printf 'Another verified resume launcher is already running.\n' >&2
  exit 1
}
CONFIG=""
CHECKPOINT=""
RESTORE_VALIDATION=""
SOURCE_RUN=""
CADENCE_MODEL_ARTIFACT=""
DRY_RUN=0
RESOLVED_INPUT_CONFIG=""

usage() {
  cat <<'EOF'
Usage: resume_verified_formal_checkpoint.sh \
  --config PATH --checkpoint PATH --restore-validation PATH --source-run PATH \
  [--cadence-model-artifact PATH] \
  [--dry-run]

Creates a new run and resumes only from a checkpoint that passed the fresh
distributed restore validation. The source run is never overwritten.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift ;;
    --restore-validation) RESTORE_VALIDATION="$2"; shift ;;
    --source-run) SOURCE_RUN="$2"; shift ;;
    --cadence-model-artifact) CADENCE_MODEL_ARTIFACT="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for value in CONFIG CHECKPOINT RESTORE_VALIDATION SOURCE_RUN; do
  [[ -n "${!value}" ]] || { usage >&2; exit 2; }
done
CONFIG="$(readlink -f "${CONFIG}")"
CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
RESTORE_VALIDATION="$(readlink -f "${RESTORE_VALIDATION}")"
SOURCE_RUN="$(readlink -f "${SOURCE_RUN}")"
if [[ -n "${CADENCE_MODEL_ARTIFACT}" ]]; then
  CADENCE_MODEL_ARTIFACT="$(readlink -f "${CADENCE_MODEL_ARTIFACT}")"
fi

# The checked-in child YAML intentionally leaves formal totals unresolved.
# Resume validation and Ray/FSDP construction must use the exact resolved
# config persisted by the source run, then change only the resume boundary.
SOURCE_RESOLVED_CONFIG="${SOURCE_RUN}/configs/resolved_config.yaml"
test -f "${SOURCE_RESOLVED_CONFIG}"
RESOLVED_INPUT_CONFIG="$(mktemp /tmp/mica-verified-resume-config.XXXXXX.yaml)"
cleanup_resolved_config() {
  rm -f "${RESOLVED_INPUT_CONFIG}"
}
trap cleanup_resolved_config EXIT
PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" - \
  "${SOURCE_RESOLVED_CONFIG}" "${CHECKPOINT}" "${RESOLVED_INPUT_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

from agentic_rl.config import load_config, validate_config
from agentic_rl.runtime.verl_config import assert_formal_hyperparameters_approved

source_config = load_config(sys.argv[1])
checkpoint = Path(sys.argv[2])
metadata = yaml.safe_load((checkpoint / "metadata.json").read_text())
step = int(metadata["successful_update_step"])
source_config["formal"]["fresh_start_required"] = False
source_config["formal"]["resume_from_successful_update"] = step
source_config["formal"]["total_successful_updates"] = 500
source_config["formal_schedule"]["total_successful_updates"] = 500
source_config["scheduler"]["total_successful_updates"] = 500
# The recovery run must preserve every complete cadence checkpoint. This is
# intentionally independent from checkpoint cadence (which remains 20).
source_config["checkpoint"]["formal_limit"] = None
validate_config(source_config)
assert_formal_hyperparameters_approved(source_config)
with Path(sys.argv[3]).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(source_config, handle, sort_keys=False)
PY
CONFIG="${RESOLVED_INPUT_CONFIG}"

PREPARE_EXTRA=()
if [[ -n "${CADENCE_MODEL_ARTIFACT}" ]]; then
  PREPARE_EXTRA+=(--cadence-model-artifact "${CADENCE_MODEL_ARTIFACT}")
fi

PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" \
  "${PROJECT_ROOT}/scripts/prepare_verified_resume_recovery.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --restore-validation "${RESTORE_VALIDATION}" \
  --source-run "${SOURCE_RUN}" \
  --run-dir "/tmp/verified-resume-validation-do-not-create" \
  --session "validation-only" \
  "${PREPARE_EXTRA[@]}" \
  --validate-only

test "$(nvidia-smi -L | wc -l)" -eq 4
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf 'GPU compute processes are already present; resume is fail-closed.\n' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; resume is fail-closed.\n' >&2
  exit 1
fi
AVAILABLE_GIB="$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')"
# The runtime preflight uses observed checkpoint/model sizes plus a margin.
# Keep this launcher floor conservative without relying on the stale 100-GiB
# assumption that blocked the verified U40 recovery on this 360-GiB host.
if [[ "${AVAILABLE_GIB}" -lt 60 ]]; then
  printf 'Verified formal resume requires at least 60 GiB free; found %s GiB.\n' \
    "${AVAILABLE_GIB}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'VERIFIED_FORMAL_RESUME_DRY_RUN=PASS\ncheckpoint=%s\n' "${CHECKPOINT}"
  exit 0
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RESUME_STEP="$(PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" - "${CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads((Path(sys.argv[1]) / "metadata.json").read_text())
print(int(metadata["successful_update_step"]))
PY
)"
RUN_ID="formal_resume_u$(printf '%03d' "${RESUME_STEP}")_to_u500_answer_ragen2_mica_ig_v1_g16_${TIMESTAMP}"
RUN_DIR="${PROJECT_ROOT}/outputs/formal_training/${RUN_ID}"
SESSION="igpo_mica_resume_u$(printf '%03d' "${RESUME_STEP}")_u500_${TIMESTAMP}"
if [[ -e "${RUN_DIR}" ]] || tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'Run directory or tmux session already exists.\n' >&2
  exit 1
fi

PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" \
  "${PROJECT_ROOT}/scripts/prepare_verified_resume_recovery.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --restore-validation "${RESTORE_VALIDATION}" \
  --source-run "${SOURCE_RUN}" \
  "${PREPARE_EXTRA[@]}" \
  --run-dir "${RUN_DIR}" \
  --session "${SESSION}"

ln -sfn "${RUN_ID}" "${PROJECT_ROOT}/outputs/formal_training/latest"
RESOLVED_CONFIG="${RUN_DIR}/configs/resolved_config.yaml"
CONTROLLER_COMMAND="bash '${PROJECT_ROOT}/scripts/_run_runtime_job.sh' FORMAL '${RESOLVED_CONFIG}' '${RUN_DIR}' '${CHECKPOINT}'; rc=\$?; printf '%s\n' \"\${rc}\" >'${RUN_DIR}/artifacts/exit_code'; exit \"\${rc}\""
tmux new-session -d -s "${SESSION}" -n controller "${CONTROLLER_COMMAND}"
tmux new-window -d -t "${SESSION}" -n eval \
  "tail -n 100 -F '${RUN_DIR}/logs/eval_worker.log'"
tmux new-window -d -t "${SESSION}" -n monitor \
  "'${RL_PYTHON}' -u '${PROJECT_ROOT}/scripts/runtime_guard.py' --run-dir '${RUN_DIR}' --kind monitor --interval 900"
tmux new-window -d -t "${SESSION}" -n watchdog \
  "'${RL_PYTHON}' -u '${PROJECT_ROOT}/scripts/runtime_guard.py' --run-dir '${RUN_DIR}' --kind watchdog --interval 60"

for _ in $(seq 1 360); do
  if [[ -s "${RUN_DIR}/artifacts/pids/driver.pid" && \
        -s "${RUN_DIR}/artifacts/pids/retriever.pid" && \
        -s "${RUN_DIR}/artifacts/pids/eval_worker.pid" && \
        -s "${RUN_DIR}/state/pids/monitor.pid" && \
        -s "${RUN_DIR}/state/pids/watchdog.pid" ]]; then
    break
  fi
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    break
  fi
  sleep 1
done

for path in \
  "${RUN_DIR}/artifacts/pids/driver.pid" \
  "${RUN_DIR}/artifacts/pids/retriever.pid" \
  "${RUN_DIR}/artifacts/pids/eval_worker.pid" \
  "${RUN_DIR}/state/pids/monitor.pid" \
  "${RUN_DIR}/state/pids/watchdog.pid"; do
  test -s "${path}"
done

printf 'VERIFIED_FORMAL_RESUME_STARTED=PASS\n'
printf 'run_id=%s\nrun_dir=%s\ntmux_session=%s\n' \
  "${RUN_ID}" "${RUN_DIR}" "${SESSION}"
printf 'checkpoint=%s\nconfig=%s\n' "${CHECKPOINT}" "${RESOLVED_CONFIG}"
printf 'controller_pid=%s\nretriever_pid=%s\neval_worker_pid=%s\nmonitor_pid=%s\nwatchdog_pid=%s\n' \
  "$(cat "${RUN_DIR}/artifacts/pids/driver.pid")" \
  "$(cat "${RUN_DIR}/artifacts/pids/retriever.pid")" \
  "$(cat "${RUN_DIR}/artifacts/pids/eval_worker.pid")" \
  "$(cat "${RUN_DIR}/state/pids/monitor.pid")" \
  "$(cat "${RUN_DIR}/state/pids/watchdog.pid")"
