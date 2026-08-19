#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

CONFIG="${PROJECT_ROOT}/recipes/rl/train_4x48gb.yaml"
RUN_DIR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: train_rl.sh [--config PATH] [--run-dir PATH] [--dry-run]

Launch the AetherSearch RL recipe. Source environment/env.local.sh or set
AETHERSEARCH_ENV_FILE before running. Training length and hardware topology
come from the selected recipe; this launcher contains no experiment constants.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      CONFIG="$2"
      shift
      ;;
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

CONFIG="$(readlink -f "${CONFIG}")"
test -f "${CONFIG}"
if [[ -z "${RUN_DIR}" ]]; then
  OUTPUT_ROOT="$("${RL_PYTHON}" -m agentic_rl.config \
    --config "${CONFIG}" --get paths.runtime_root)"
  RUN_DIR="${OUTPUT_ROOT}/aethersearch_rl_$(date -u +%Y%m%d_%H%M%S)"
fi
RUN_DIR="$(readlink -m "${RUN_DIR}")"
mkdir -p "${RUN_DIR}/configs" "${RUN_DIR}/logs" "${RUN_DIR}/artifacts/pids"
RESOLVED_CONFIG="${RUN_DIR}/configs/resolved_config.yaml"

"${RL_PYTHON}" "${PROJECT_ROOT}/scripts/resolve_mica_formal_config.py" \
  --input "${CONFIG}" \
  --output "${RESOLVED_CONFIG}" \
  --runtime-root "${RUN_DIR}"

"${RL_PYTHON}" - "${RESOLVED_CONFIG}" <<'PY'
import sys
from agentic_rl.config import load_config
from agentic_rl.qualification import qualification_mode, validate_reference_qualification

config = load_config(sys.argv[1])
if qualification_mode(config) in {"reference", "formal"}:
    validate_reference_qualification(config)
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${RL_PYTHON}" -m agentic_rl.config \
    --config "${RESOLVED_CONFIG}" --format yaml >/dev/null
  printf 'AETHERSEARCH_RL_DRY_RUN=PASS\n'
  printf 'recipe=%s\nresolved_config=%s\nrun_dir=%s\n' \
    "${CONFIG}" "${RESOLVED_CONFIG}" "${RUN_DIR}"
  printf 'services_started=false\n'
  exit 0
fi

"${RL_PYTHON}" "${PROJECT_ROOT}/scripts/preflight_mica_formal.py" \
  --config "${RESOLVED_CONFIG}" \
  --output "${RUN_DIR}/configs/preflight.json" >/dev/null

exec "${PROJECT_ROOT}/scripts/_run_runtime_job.sh" \
  FORMAL "${RESOLVED_CONFIG}" "${RUN_DIR}"
