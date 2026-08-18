#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
RL_PYTHON="${RL_ENV}/bin/python"
BASE_CONFIG="${PROJECT_ROOT}/configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/formal_training"
TOTAL_UPDATES=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: train_formal_manual.sh --total-successful-updates N [--config PATH] [--dry-run]

Creates a unique fresh U0 formal run and starts the existing production runtime
in tmux. This script is fail-closed unless the S/N/cumulative-IG/Probe-routing tests,
immutable inputs, and source manifest pass. --dry-run
loads no model and starts no Retriever, Ray process, or training process.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --total-successful-updates)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      TOTAL_UPDATES="$2"
      shift
      ;;
    --config)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      BASE_CONFIG="$(readlink -f "$2")"
      shift
      ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${TOTAL_UPDATES}" =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--total-successful-updates must be a positive integer.' >&2
  exit 2
}
test -f "${BASE_CONFIG}"
test -x "${RL_PYTHON}"
test "$(nvidia-smi -L | wc -l)" -eq 4
command -v tmux >/dev/null

TMP_CONFIG="$(mktemp /tmp/igpo-formal-resolved.XXXXXX.yaml)"
cleanup_tmp() { rm -f "${TMP_CONFIG}"; }
trap cleanup_tmp EXIT
PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" - \
  "${BASE_CONFIG}" "${TMP_CONFIG}" "${TOTAL_UPDATES}" <<'PY'
import sys
import yaml
from agentic_rl.config import load_config, validate_config
from agentic_rl.runtime.verl_config import assert_formal_hyperparameters_approved

config = load_config(sys.argv[1])
total = int(sys.argv[3])
config["formal"]["total_successful_updates"] = total
config["formal_schedule"]["total_successful_updates"] = total
config["scheduler"]["total_successful_updates"] = total
if total != 500:
    raise SystemExit("This locked fresh experiment must target U500")
if config["formal"].get("fresh_start_required") is not True:
    raise SystemExit("Formal config does not require a fresh start")
if int(config["formal"].get("resume_from_successful_update", -1)) != 0:
    raise SystemExit("Formal config is not locked to successful_update=0")
validate_config(config)
assert_formal_hyperparameters_approved(config)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

if [[ -n "${AGENTIC_RL_RESUME_CHECKPOINT:-}" ]]; then
  printf 'Fresh formal launch rejects AGENTIC_RL_RESUME_CHECKPOINT.\n' >&2
  exit 1
fi
unset AGENTIC_RL_RESUME_CHECKPOINT

PREFLIGHT_JSON="$(mktemp /tmp/igpo-formal-preflight.XXXXXX.json)"
FORMULA_AUDIT_JSON="$(mktemp /tmp/igpo-formal-formula-audit.XXXXXX.json)"
cleanup_tmp() {
  rm -f "${TMP_CONFIG}" "${PREFLIGHT_JSON}" "${FORMULA_AUDIT_JSON}"
}
trap cleanup_tmp EXIT

(
  cd "${PROJECT_ROOT}"
  sha256sum -c MANIFEST.sha256 >/dev/null
  PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" -m pytest -q \
    tests/test_mica_ig_v1.py \
    tests/test_answer_only_ragen2_mica_integration.py \
    tests/test_paper_ragen2_selection.py >/dev/null
  printf '{"result":"PASS","algorithm_mode":"answer_only_ragen2_mica_ig_v1_singleton_outcome","source":"production_tests"}\n' \
    >"${FORMULA_AUDIT_JSON}"
  PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" \
    scripts/preflight_mica_formal.py --config "${TMP_CONFIG}" \
    --output "${PREFLIGHT_JSON}" >/dev/null
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'ANSWER_ONLY_RAGEN2_MICA_IG_V1_PREFLIGHT=PASS\n'
  printf 'FORMAL_DRY_RUN=PASS\n'
  printf 'fresh_start_successful_update=0\n'
  printf 'total_successful_updates=%s\n' "${TOTAL_UPDATES}"
  printf 'base_config=%s\n' "${BASE_CONFIG}"
  printf 'no_model_or_service_started=true\n'
  exit 0
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf 'GPU compute processes are already present; formal launch is fail-closed.\n' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop its owning job first.\n' >&2
  exit 1
fi
AVAILABLE_GB="$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')"
if [[ "${AVAILABLE_GB}" -lt 60 ]]; then
  printf 'Formal launch requires at least 60 GiB free; found %s GiB.\n' \
    "${AVAILABLE_GB}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LAUNCH_MODE="$(PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" -c 'from agentic_rl.config import load_config; import sys; print(load_config(sys.argv[1])["advantage"]["search_task_mode"])' "${TMP_CONFIG}")"
SELECTION_MODE="$(PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" -c 'from agentic_rl.config import load_config; import sys; print(load_config(sys.argv[1])["selection"]["mode"])' "${TMP_CONFIG}")"
if [[ "${LAUNCH_MODE}" != "answer_only_ragen2_mica_ig_v1_singleton_outcome" ]]; then
  printf 'Unexpected formal algorithm mode: %s\n' "${LAUNCH_MODE}" >&2
  exit 1
fi
if [[ "${SELECTION_MODE}" == "answer_outcome_only_ragen2_paper_variance_top_p" ]]; then
  RUN_ID="formal_u000_answer_ragen2_paper_mica_ig_v1_g16_${TIMESTAMP}"
  SESSION="mica_ragen2_paper_formal"
else
  RUN_ID="formal_fresh_u000_to_u500_answer_ragen2_mica_ig_v1_g16_${TIMESTAMP}"
  SESSION="igpo_mica_ig_v1_u500_${TIMESTAMP}"
fi
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -e "${RUN_DIR}" ]] || tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'Formal run/session collision: %s / %s\n' "${RUN_DIR}" "${SESSION}" >&2
  exit 1
fi

mkdir -p \
  "${RUN_DIR}/configs" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/metrics" \
  "${RUN_DIR}/checkpoints" \
  "${RUN_DIR}/eval" \
  "${RUN_DIR}/artifacts/pids" \
  "${RUN_DIR}/reports"
cp "${BASE_CONFIG}" "${RUN_DIR}/configs/formal_train.yaml"
cp "${TMP_CONFIG}" "${RUN_DIR}/configs/resolved_config.yaml"
cp "${PROJECT_ROOT}/MANIFEST.sha256" \
  "${RUN_DIR}/configs/source_manifest.sha256"
cp "${PREFLIGHT_JSON}" "${RUN_DIR}/configs/preflight.json"
cp "${FORMULA_AUDIT_JSON}" \
  "${RUN_DIR}/configs/search_advantage_formula_audit.json"
sha256sum \
  "${RUN_DIR}/configs/formal_train.yaml" \
  "${RUN_DIR}/configs/resolved_config.yaml" \
  "${RUN_DIR}/configs/preflight.json" \
  "${RUN_DIR}/configs/search_advantage_formula_audit.json" \
  "${RUN_DIR}/configs/source_manifest.sha256" \
  >"${RUN_DIR}/configs/launch_inputs.sha256"
{
  date -u '+timestamp_utc=%Y-%m-%dT%H:%M:%SZ'
  "${RL_PYTHON}" --version
  nvidia-smi
} >"${RUN_DIR}/configs/environment.txt" 2>&1
printf '%q ' \
  bash "${PROJECT_ROOT}/scripts/_run_runtime_job.sh" \
  FORMAL "${RUN_DIR}/configs/resolved_config.yaml" "${RUN_DIR}" \
  >"${RUN_DIR}/configs/launch_command.sh"
printf '\n' >>"${RUN_DIR}/configs/launch_command.sh"
ln -sfn "${RUN_ID}" "${OUTPUT_ROOT}/latest"

TMUX_COMMAND="bash '${PROJECT_ROOT}/scripts/_run_runtime_job.sh' FORMAL '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}'; rc=\$?; printf '%s\n' \"\${rc}\" >'${RUN_DIR}/artifacts/exit_code'; exit \"\${rc}\""
tmux new-session -d -s "${SESSION}" "${TMUX_COMMAND}"
printf '%s\n' "${SESSION}" >"${RUN_DIR}/artifacts/tmux_session"

DRIVER_PID=""
for _ in $(seq 1 120); do
  if [[ -s "${RUN_DIR}/artifacts/pids/driver.pid" ]]; then
    DRIVER_PID="$(cat "${RUN_DIR}/artifacts/pids/driver.pid")"
    break
  fi
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    break
  fi
  sleep 1
done

printf 'RUN_ID=%s\nRUN_DIR=%s\nTMUX_SESSION=%s\nDRIVER_PID=%s\n' \
  "${RUN_ID}" "${RUN_DIR}" "${SESSION}" "${DRIVER_PID:-PENDING}"
printf 'CONSOLE_LOG=%s\n' "${RUN_DIR}/logs/console.log"
printf 'CHECKPOINT_DIR=%s\n' "${RUN_DIR}/checkpoints"
printf 'UPDATE_METRICS=%s\n' "${RUN_DIR}/metrics/update_metrics.jsonl"
