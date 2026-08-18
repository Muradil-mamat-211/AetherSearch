#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

PROFILE="${PROJECT_ROOT}/configs/formal_resume_u20_3rank_48cpu.yaml"
CHECKPOINT="${PROJECT_ROOT}/outputs/formal_training/formal_fresh_u000_to_u500_role_localized_gate_g16_lr2e7_kl1e2_20260808_133350/checkpoints/resume/update_020"
SOURCE_RUN="${PROJECT_ROOT}/outputs/formal_training/formal_fresh_u000_to_u500_role_localized_gate_g16_lr2e7_kl1e2_20260808_133350"
SOURCE_MODEL="${SOURCE_RUN}/checkpoints/models/update_020"
VALIDATION_RESULT="${PROJECT_ROOT}/outputs/3rank_resume_48cpu_validation/stage_results/stage_resume_validate_3rank.json"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/formal_training"
MIN_FREE_GIB=80
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: resume_formal_u20_3rank_48cpu.sh [--dry-run]

Resume global Update 20 -> 500 on physical GPUs 1,2,3 with FSDP2 world size
3. GPU0 runs Retriever and the independent async evaluator in tmux window
"eval". A new RUN_ID and output directory are always created.
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

for required in "${PROFILE}" "${CHECKPOINT}/metadata.json" \
  "${CHECKPOINT}/integrity.sha256.json" "${CHECKPOINT}/controller/state.json" \
  "${SOURCE_MODEL}/training_metadata.json" "${SOURCE_RUN}/eval/update_020/metrics.json" \
  "${SOURCE_RUN}/eval/update_020/COMPLETED" "${VALIDATION_RESULT}"; do
  test -e "${required}"
done
command -v tmux >/dev/null
test "$(nvidia-smi -L | wc -l)" -eq 4
GPU_INDEXES="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -)"
test "${GPU_INDEXES}" = "0,1,2,3"

"${RL_ENV}/bin/python" "${PROJECT_ROOT}/scripts/prepare_formal_resume_run.py" \
  --profile "${PROFILE}" \
  --checkpoint "${CHECKPOINT}" \
  --source-model "${SOURCE_MODEL}" \
  --source-run "${SOURCE_RUN}" \
  --validation-result "${VALIDATION_RESULT}" \
  --check-only

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'RESUME_3RANK_48CPU_DRY_RUN=PASS\n'
  printf 'profile=%s\ncheckpoint=%s\ntarget_successful_updates=500\n' \
    "${PROFILE}" "${CHECKPOINT}"
  printf 'rl_physical_gpus=1,2,3\nretriever_gpu=0\nasync_eval_tmux_window=eval\n'
  exit 0
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
  printf 'GPU compute processes already exist; refusing to start a second runtime.\n' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; refusing to start a second trainer.\n' >&2
  exit 1
fi
if pgrep -f 'hybrid_retrieval_server' >/dev/null 2>&1; then
  printf 'A Retriever server is already running; refusing to start a second runtime.\n' >&2
  exit 1
fi
AVAILABLE_GIB="$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')"
if [[ "${AVAILABLE_GIB}" -lt "${MIN_FREE_GIB}" ]]; then
  printf 'Need at least %s GiB free; found %s GiB.\n' "${MIN_FREE_GIB}" "${AVAILABLE_GIB}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="formal_resume_u020_to_u500_3rank_48cpu_${TIMESTAMP}_role_localized_gate_g16"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
SESSION="igpo_formal_u20_u500_3rank_${TIMESTAMP}"
if [[ -e "${RUN_DIR}" ]] || tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'Formal run destination or tmux session already exists.\n' >&2
  exit 1
fi

"${RL_ENV}/bin/python" "${PROJECT_ROOT}/scripts/prepare_formal_resume_run.py" \
  --profile "${PROFILE}" \
  --checkpoint "${CHECKPOINT}" \
  --source-model "${SOURCE_MODEL}" \
  --source-run "${SOURCE_RUN}" \
  --validation-result "${VALIDATION_RESULT}" \
  --run-dir "${RUN_DIR}" \
  --session "${SESSION}"

ln -sfn "${RUN_ID}" "${OUTPUT_ROOT}/latest"

tmux new-session -d -s "${SESSION}" -n retriever \
  "bash '${PROJECT_ROOT}/scripts/_formal_retriever_process.sh' '${RUN_DIR}'"
retriever_ready=0
for _ in $(seq 1 360); do
  if "${RL_ENV}/bin/python" -c \
    "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
    >/dev/null 2>&1; then
    retriever_ready=1
    break
  fi
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    break
  fi
  sleep 2
done
if [[ "${retriever_ready}" -ne 1 ]]; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  printf 'Retriever did not reach health=PASS; no trainer was started.\n' >&2
  exit 1
fi

# The evaluator is deliberately an independent tmux window and process.
tmux new-window -d -t "${SESSION}" -n eval \
  "bash '${PROJECT_ROOT}/scripts/async_eval_gpu0_worker.sh' '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}'"
tmux new-window -d -t "${SESSION}" -n monitor \
  "bash '${PROJECT_ROOT}/scripts/monitor_formal_training_10min.sh' '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}'"
tmux new-window -d -t "${SESSION}" -n trainer \
  "AGENTIC_RL_RL_CUDA_VISIBLE_DEVICES=1,2,3 bash '${PROJECT_ROOT}/scripts/_formal_trainer_process.sh' '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}' '${CHECKPOINT}'"
tmux new-window -d -t "${SESSION}" -n watchdog \
  "AGENTIC_RL_RL_CUDA_VISIBLE_DEVICES=1,2,3 bash '${PROJECT_ROOT}/scripts/formal_training_watchdog.sh' '${RUN_DIR}/configs/resolved_config.yaml' '${RUN_DIR}' '${CHECKPOINT}'"

for _ in $(seq 1 300); do
  if [[ -s "${RUN_DIR}/state/pids/eval_worker.pid" && \
        -s "${RUN_DIR}/state/pids/monitor.pid" && \
        -s "${RUN_DIR}/state/pids/trainer.pid" && \
        -s "${RUN_DIR}/state/pids/watchdog.pid" ]]; then
    break
  fi
  sleep 1
done

for pid_file in eval_worker monitor trainer watchdog; do
  test -s "${RUN_DIR}/state/pids/${pid_file}.pid"
done
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'tmux session exited during startup.\n' >&2
  exit 1
fi

printf 'FORMAL_RESUME_STARTED=PASS\n'
printf 'run_id=%s\nrun_dir=%s\ntmux_session=%s\n' "${RUN_ID}" "${RUN_DIR}" "${SESSION}"
printf 'checkpoint=%s\nconfig=%s\n' "${CHECKPOINT}" "${RUN_DIR}/configs/resolved_config.yaml"
printf 'windows=retriever,eval,monitor,trainer,watchdog\n'
printf 'eval_window=eval\n'
printf 'trainer_pid=%s\neval_worker_pid=%s\nmonitor_pid=%s\nwatchdog_pid=%s\n' \
  "$(cat "${RUN_DIR}/state/pids/trainer.pid")" \
  "$(cat "${RUN_DIR}/state/pids/eval_worker.pid")" \
  "$(cat "${RUN_DIR}/state/pids/monitor.pid")" \
  "$(cat "${RUN_DIR}/state/pids/watchdog.pid")"
