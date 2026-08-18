#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
CONFIG="${PROJECT_ROOT}/configs/gate_calibration_role_localized.yaml"
RUN_DIR="${PROJECT_ROOT}/artifacts/role_localized_gate_calibration_20260804/runtime"
MANIFEST="${PROJECT_ROOT}/artifacts/role_localized_gate_calibration_20260804/gate_calibration_manifest.json"
DRY_RUN=0

usage() {
  printf '%s\n' 'Usage: run_role_localized_gate_calibration.sh [--dry-run] [--help]'
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
if [[ -e "${MANIFEST}" ]]; then
  printf 'Immutable calibration manifest already exists: %s\n' "${MANIFEST}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  AGENTIC_RL_RUNTIME_STAGE=GATE_CALIBRATION PYTHONPATH="${PROJECT_ROOT}/src" \
    /root/autodl-tmp/search-r1-workspace/envs/igpo-ragen2-fsdp2-vllm011/bin/python \
    -c 'from agentic_rl.config import load_config; import sys; load_config(sys.argv[1]); print("ROLE_GATE_CALIBRATION_DRY_RUN=PASS")' "${CONFIG}"
  printf 'config=%s\nrun_dir=%s\nmanifest=%s\n' "${CONFIG}" "${RUN_DIR}" "${MANIFEST}"
  exit 0
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf '%s\n' 'GPU compute processes are present; calibration is fail-closed.' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null 2>&1; then
  printf '%s\n' 'A Ray cluster is already active.' >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/artifacts/pids" "$(dirname "${MANIFEST}")"
exec bash "${PROJECT_ROOT}/scripts/_run_runtime_job.sh" GATE_CALIBRATION "${CONFIG}" "${RUN_DIR}"
