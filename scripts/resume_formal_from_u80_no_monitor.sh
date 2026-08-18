#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
SOURCE_RUN="${PROJECT_ROOT}/outputs/formal_training/formal_resume_u020_to_u500_3rank_48cpu_20260808_191223_role_localized_gate_g16"
CHECKPOINT="${SOURCE_RUN}/checkpoints/resume/update_080"
SOURCE_MODEL="${SOURCE_RUN}/checkpoints/models/update_080"
TEMPLATE_CONFIG="${SOURCE_RUN}/configs/resolved_config.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/formal_training"
RL_PYTHON="${RL_ENV}/bin/python"

test -d "${CHECKPOINT}"
test -d "${SOURCE_MODEL}"
test -f "${TEMPLATE_CONFIG}"
test "$(nvidia-smi -L | wc -l)" -eq 4
test -z "$(pgrep -x raylet || true)"
test -z "$(ps -eo args= | rg 'agentic_rl\.runtime\.entrypoint' | rg -v 'rg ' || true)"
"${RL_PYTHON}" -c "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')"
RETRIEVER_PID="$(pgrep -f 'hybrid_retrieval_server' | head -1 || true)"
test -n "${RETRIEVER_PID}"

readonly_report="/tmp/exact_ig_u80_resume_readonly_$$.json"
"${RL_PYTHON}" "${PROJECT_ROOT}/scripts/verify_checkpoint_readonly.py" \
  --checkpoint "${CHECKPOINT}" --expected-step 80 --expected-attempt 84 \
  --expected-cursor 9952 --output "${readonly_report}"
"${RL_PYTHON}" - "${TEMPLATE_CONFIG}" <<'PY'
import sys
from agentic_rl.config import load_config
config = load_config(sys.argv[1])
assert config["evaluation"]["asynchronous"] is True
assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
assert config["formal_schedule"]["fixed_eval_every_successful_updates"] == 20
assert config["rollout"]["data_parallel_size"] == 3
assert config["learner"]["world_size"] == 3
print("FORMAL_RESUME_CONFIG=PASS")
PY

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="formal_resume_u080_to_u500_no_monitor_${TIMESTAMP}_role_localized_gate_g16"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
SESSION="igpo_formal_u80_u500_no_monitor_${TIMESTAMP}"
if [[ -e "${RUN_DIR}" ]] || tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'Run directory or tmux session already exists.\n' >&2
  exit 1
fi

"${RL_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_formal_resume_from_checkpoint.py" \
  --config "${TEMPLATE_CONFIG}" --checkpoint "${CHECKPOINT}" \
  --source-model "${SOURCE_MODEL}" --source-run "${SOURCE_RUN}" \
  --run-dir "${RUN_DIR}" --session "${SESSION}" --retriever-pid "${RETRIEVER_PID}"
ln -sfn "${RUN_ID}" "${OUTPUT_ROOT}/latest"
printf '%s\n' "${SESSION}" >"${RUN_DIR}/state/tmux_session"

tmux new-session -d -s "${SESSION}" -n eval \
  "bash '${PROJECT_ROOT}/scripts/async_eval_gpu0_worker.sh' '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}'"
tmux new-window -d -t "${SESSION}" -n trainer \
  "AGENTIC_RL_RL_CUDA_VISIBLE_DEVICES=1,2,3 bash '${PROJECT_ROOT}/scripts/_formal_trainer_process.sh' '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}' '${CHECKPOINT}'"

for _ in $(seq 1 180); do
  if [[ -s "${RUN_DIR}/state/pids/eval_worker.pid" && -s "${RUN_DIR}/state/pids/trainer.pid" ]]; then
    break
  fi
  sleep 1
done
test -s "${RUN_DIR}/state/pids/eval_worker.pid"
test -s "${RUN_DIR}/state/pids/trainer.pid"
printf 'FORMAL_RESUME_U80_STARTED=PASS\n'
printf 'run_id=%s\nrun_dir=%s\ntmux_session=%s\n' "${RUN_ID}" "${RUN_DIR}" "${SESSION}"
printf 'windows=eval,trainer\ncheckpoint=%s\nretriever_pid=%s\n' "${CHECKPOINT}" "${RETRIEVER_PID}"
printf 'trainer_pid=%s\neval_worker_pid=%s\n' "$(cat "${RUN_DIR}/state/pids/trainer.pid")" "$(cat "${RUN_DIR}/state/pids/eval_worker.pid")"
printf 'monitor=disabled\nwatchdog=disabled\nconfig=%s\n' "${RUN_DIR}/configs/resolved_config.yaml"
