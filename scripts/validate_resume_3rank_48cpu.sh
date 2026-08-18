#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
CONFIG="${PROJECT_ROOT}/configs/formal_resume_u20_3rank_48cpu.yaml"
CHECKPOINT="${PROJECT_ROOT}/outputs/formal_training/formal_fresh_u000_to_u500_role_localized_gate_g16_lr2e7_kl1e2_20260808_133350/checkpoints/resume/update_020"
RUN_DIR="${PROJECT_ROOT}/outputs/3rank_resume_48cpu_validation"

usage() {
  printf 'Usage: %s [--dry-run]\n' "$0"
}

DRY_RUN=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "$#" -ne 0 ]]; then
  usage >&2
  exit 2
fi

test -f "${CONFIG}"
test -d "${CHECKPOINT}"
if [[ -e "${RUN_DIR}" && "${DRY_RUN}" -eq 0 ]]; then
  printf 'Validation run already exists: %s\n' "${RUN_DIR}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'CONFIG=%s\nCHECKPOINT=%s\nRUN_DIR=%s\nRL_CUDA_VISIBLE_DEVICES=1,2,3\n' \
    "${CONFIG}" "${CHECKPOINT}" "${RUN_DIR}"
  exit 0
fi

mkdir -p "${RUN_DIR}"
export AGENTIC_RL_RL_CUDA_VISIBLE_DEVICES=1,2,3
exec "${PROJECT_ROOT}/scripts/_run_runtime_job.sh" \
  RESUME_VALIDATE_3RANK "${CONFIG}" "${RUN_DIR}" "${CHECKPOINT}"
