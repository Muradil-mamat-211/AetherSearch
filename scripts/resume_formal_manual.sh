#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
RUN_DIR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: resume_formal_manual.sh --run-dir /absolute/path/to/formal/RUN_ID [--dry-run]

Resumes the latest committed checkpoint in an existing formal run directory.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-dir)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      RUN_DIR="$2"
      shift
      ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${RUN_DIR}" = /* ]] || {
  printf '%s\n' '--run-dir must be an absolute path.' >&2
  exit 2
}
RUN_DIR="$(readlink -f "${RUN_DIR}")"
CONFIG="${RUN_DIR}/configs/resolved_config.yaml"
LATEST="${RUN_DIR}/checkpoints/latest_checkpoint.json"
test -f "${CONFIG}"
test -f "${LATEST}"
CHECKPOINT="$("${RL_ENV}/bin/python" - "${LATEST}" "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[2])
payload = json.loads(Path(sys.argv[1]).read_text())
print((root / "checkpoints" / payload["checkpoint"]).resolve())
PY
)"
test -d "${CHECKPOINT}"
test -f "${CHECKPOINT}/integrity.sha256.json"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'FORMAL_RESUME_DRY_RUN=PASS\nrun_dir=%s\ncheckpoint=%s\n' \
    "${RUN_DIR}" "${CHECKPOINT}"
  exit 0
fi
test "$(nvidia-smi -L | wc -l)" -eq 5
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf 'GPU compute processes are already present; resume is fail-closed.\n' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop its owning job first.\n' >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SESSION="igpo_formal_resume_${TIMESTAMP}"
TMUX_COMMAND="bash '${PROJECT_ROOT}/scripts/_run_runtime_job.sh' FORMAL '${CONFIG}' '${RUN_DIR}' '${CHECKPOINT}'; rc=\$?; printf '%s\n' \"\${rc}\" >'${RUN_DIR}/artifacts/exit_code'; exit \"\${rc}\""
tmux new-session -d -s "${SESSION}" "${TMUX_COMMAND}"
printf '%s\n' "${SESSION}" >"${RUN_DIR}/artifacts/tmux_session"
printf 'RUN_DIR=%s\nTMUX_SESSION=%s\nRESUME_CHECKPOINT=%s\nCONSOLE_LOG=%s\n' \
  "${RUN_DIR}" "${SESSION}" "${CHECKPOINT}" "${RUN_DIR}/logs/console.log"
