#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
RUN_DIR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: stop_formal_training.sh [--run-dir /absolute/path/to/RUN_ID] [--dry-run]

Sends SIGTERM only to PIDs recorded by the selected formal run.
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

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(readlink -f "${PROJECT_ROOT}/outputs/formal_training/latest")"
else
  RUN_DIR="$(readlink -f "${RUN_DIR}")"
fi
PID_DIR="${RUN_DIR}/state/pids"
SESSION_FILE="${RUN_DIR}/state/processes.json"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'FORMAL_STOP_DRY_RUN=PASS\nrun_dir=%s\n' "${RUN_DIR}"
  exit 0
fi

sent=0
for name in trainer eval_worker monitor watchdog retriever; do
  pid_file="${PID_DIR}/${name}.pid"
  if [[ -s "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}"
      printf 'SIGTERM sent to %s PID %s\n' "${name}" "${pid}"
      sent=1
    fi
  fi
done
if [[ -s "${SESSION_FILE}" ]]; then
  session="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("tmux_session", ""))' "${SESSION_FILE}")"
  if tmux has-session -t "${session}" 2>/dev/null; then
    printf 'Waiting for tmux session %s to exit cleanly.\n' "${session}"
  fi
fi
if [[ "${sent}" -eq 0 ]]; then
  printf 'No recorded live formal process exists for %s\n' "${RUN_DIR}"
fi
