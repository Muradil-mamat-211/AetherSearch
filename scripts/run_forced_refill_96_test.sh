#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
RL_PYTHON="${RL_ENV}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/forced_refill_96_test.yaml"
PILOT_RUN_DIR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_forced_refill_96_test.sh [--run-dir PILOT_RUN_DIR] [--dry-run]

Runs the isolated 64->96 refill test from the Pilot update_20 checkpoint.
The test performs no backward, optimizer step, scheduler step, or checkpoint write.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-dir)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      PILOT_RUN_DIR="$2"
      shift
      ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -z "${PILOT_RUN_DIR}" ]]; then
  PILOT_RUN_DIR="$(readlink -f "${PROJECT_ROOT}/outputs/pilot_20_final/latest")"
else
  PILOT_RUN_DIR="$(readlink -f "${PILOT_RUN_DIR}")"
fi
CHECKPOINT="${PILOT_RUN_DIR}/checkpoints/update_20"
PILOT_RESULT="${PILOT_RUN_DIR}/stage_results/stage_pilot20.json"
TEST_DIR="${PILOT_RUN_DIR}/forced_refill_96_test"
SESSION="igpo_refill96_$(basename "${PILOT_RUN_DIR}" | sed 's/^pilot20_//')"

test -f "${CONFIG}"
test -d "${CHECKPOINT}"
test -f "${CHECKPOINT}/metadata.json"
test -f "${CHECKPOINT}/integrity.sha256.json"
test -f "${PILOT_RESULT}"
"${RL_PYTHON}" - "${PILOT_RESULT}" "${CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

pilot = json.loads(Path(sys.argv[1]).read_text())
metadata = json.loads((Path(sys.argv[2]) / "metadata.json").read_text())
assert pilot["status"] == "PASS"
assert pilot["successful_updates"] == 20
assert pilot["optimizer_steps_total"] == 20
assert pilot["scheduler_steps_total"] == 20
assert metadata["successful_update_step"] == 20
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'FORCED_REFILL96_DRY_RUN=PASS\n'
  printf 'pilot_run_dir=%s\ncheckpoint=%s\noutput=%s\n' \
    "${PILOT_RUN_DIR}" "${CHECKPOINT}" "${TEST_DIR}"
  exit 0
fi

test "$(nvidia-smi -L | wc -l)" -eq 5
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf 'GPU compute processes are already present; refill test is fail-closed.\n' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop its owning job first.\n' >&2
  exit 1
fi
if [[ -e "${TEST_DIR}" ]]; then
  printf 'Forced-refill output already exists: %s\n' "${TEST_DIR}" >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'Forced-refill tmux session already exists: %s\n' "${SESSION}" >&2
  exit 1
fi

mkdir -p \
  "${TEST_DIR}/configs" \
  "${TEST_DIR}/logs" \
  "${TEST_DIR}/metrics" \
  "${TEST_DIR}/artifacts/pids" \
  "${TEST_DIR}/reports"
cp "${CONFIG}" "${TEST_DIR}/configs/forced_refill_96_test.yaml"
TMUX_COMMAND="bash '${PROJECT_ROOT}/scripts/_run_runtime_job.sh' FORCED_REFILL96 '${CONFIG}' '${TEST_DIR}' '${CHECKPOINT}'; rc=\$?; printf '%s\n' \"\${rc}\" >'${TEST_DIR}/artifacts/exit_code'; exit \"\${rc}\""
tmux new-session -d -s "${SESSION}" "${TMUX_COMMAND}"
printf '%s\n' "${SESSION}" >"${TEST_DIR}/artifacts/tmux_session"

printf 'PILOT_RUN_DIR=%s\n' "${PILOT_RUN_DIR}"
printf 'TEST_DIR=%s\n' "${TEST_DIR}"
printf 'TMUX_SESSION=%s\n' "${SESSION}"
printf 'CHECKPOINT=%s\n' "${CHECKPOINT}"
printf 'CONSOLE_LOG=%s\n' "${TEST_DIR}/logs/console.log"
