#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

CONFIG="${PROJECT_ROOT}/configs/formal_resume_u20_to_u500.yaml"
PILOT_RUN="${PROJECT_ROOT}/outputs/pilot_20_final/pilot20_20260801_093720_lr2e7_t1_topP095_g16"
PILOT_CHECKPOINT="${PILOT_RUN}/checkpoints/update_20"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/formal_training"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: train_formal_from_pilot20_to_500.sh [--dry-run]

Resumes the verified Pilot Update 20 state and stops at global successful
Update 500. The trainer, GPU0 evaluator, monitor and watchdog run in tmux.
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

test -f "${CONFIG}"
test -f "${PILOT_CHECKPOINT}/metadata.json"
test -f "${PILOT_CHECKPOINT}/integrity.sha256.json"
test -f "${PILOT_RUN}/stage_results/stage_pilot20.json"
test -f "${PROJECT_ROOT}/outputs/exact_ig_precision_gate.json"
test -x "${RL_ENV}/bin/python"
command -v tmux >/dev/null
test "$(nvidia-smi -L | wc -l)" -eq 5

"${RL_ENV}/bin/python" - "${CONFIG}" "${PILOT_RUN}" "${PILOT_CHECKPOINT}" <<'PY'
import json, sys
from pathlib import Path
from agentic_rl.config import load_config

config = load_config(sys.argv[1])
pilot = Path(sys.argv[2])
checkpoint = Path(sys.argv[3])
stage = json.loads((pilot / "stage_results/stage_pilot20.json").read_text())
metadata = json.loads((checkpoint / "metadata.json").read_text())
gate = json.loads(Path(config["exact_ig"]["numerical_gate_path"]).read_text())
assert stage["status"] == "PASS"
assert stage["successful_updates"] == 20
assert stage["optimizer_steps_total"] == 20
assert stage["scheduler_steps_total"] == 20
assert stage["checkpoint_reload"]["status"] == "PASS"
assert metadata["successful_update_step"] == 20
assert metadata["attempt_id"] == stage["attempts"]
assert metadata["data_cursor"] > 0
assert metadata["ig_channel"]["valid_success_count"] == 20
assert metadata["outcome_channel"]["valid_success_count"] == 20
assert metadata["ig_channel"]["health_reference"] is not None
assert metadata["outcome_channel"]["health_reference"] is not None
assert gate["allow_fast_path_training"] is True
assert gate["exact_ig_version"] == "exact_ig_official_offset_fp32_no_anchor_v4"
for required_gate in (
    "TARGET_CONTRACT",
    "ATTENTION_MASK_EXHAUSTIVE",
    "LOGICAL_POSITION_IDS",
    "P_MINUS_ONE_SHIFT",
    "FUTURE_LEAKAGE",
    "FP32_RUNTIME",
    "FSDP_RESTORE",
    "MODEL_CHECKSUM_UNCHANGED",
    "RAGEN_SELECTED_SET_PARITY",
):
    assert gate["gates"][required_gate] is True, required_gate
assert config["formal_schedule"]["total_successful_updates"] == 500
assert config["formal_schedule"]["warmup"] == 2
assert config["formal_schedule"]["checkpoint_every_successful_updates"] == 20
assert config["evaluation"]["asynchronous"] is True
assert config["data"]["expected_rows"] == 150745
assert config["data"]["shuffle_seed"] == 20260724
assert config["rollout"]["group_size"] == 16
assert config["rollout"]["candidate_prompts_max"] == 128
assert config["rollout"]["max_num_seqs"] == 64
assert config["rollout"]["gpu_memory_utilization"] == 0.46
assert config["formal_schedule"]["learner_micro_batch_size"] == 6
assert config["advantage"]["lambda_ig"] == 0.3
assert config["advantage"]["lambda_task"] == 1.0
assert config["advantage"]["lambda_outcome"] == 1.0
assert config["advantage"]["lambda_format"] == 1.0
assert config["exact_ig"]["exact_ig_version"] == gate["exact_ig_version"]
assert config["exact_ig"]["production_precision_mode"] == "fp32_exact_ig"
algorithm = metadata["algorithm_config"]
assert algorithm["advantage"]["lambda_ig"] == config["advantage"]["lambda_ig"]
assert algorithm["candidate_pool"]["max_prompts"] == config["rollout"]["candidate_prompts_max"]
assert algorithm["exact_ig"]["exact_ig_version"] == config["exact_ig"]["exact_ig_version"]
assert config["policy"]["optimizer_steps_per_successful_update"] == 1
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'FORMAL_RESUME_DRY_RUN=PASS\nconfig=%s\ncheckpoint=%s\ntarget=500\n' \
    "${CONFIG}" "${PILOT_CHECKPOINT}"
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
AVAILABLE_GIB="$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')"
if [[ "${AVAILABLE_GIB}" -lt 300 ]]; then
  printf 'Formal run requires at least 300 GiB free; found %s GiB.\n' "${AVAILABLE_GIB}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="formal_resume_u020_to_u500_${TIMESTAMP}_lr2e7_t1_topP095_g16"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
SESSION="igpo_formal_u20_u500_${TIMESTAMP}"
if [[ -e "${RUN_DIR}" ]] || tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'Formal run destination/session already exists.\n' >&2
  exit 1
fi

mkdir -p \
  "${RUN_DIR}/configs" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/metrics" \
  "${RUN_DIR}/checkpoints/models" \
  "${RUN_DIR}/checkpoints/resume" \
  "${RUN_DIR}/eval" \
  "${RUN_DIR}/monitor" \
  "${RUN_DIR}/state/pids" \
  "${RUN_DIR}/final"
touch \
  "${RUN_DIR}/logs/console.log" \
  "${RUN_DIR}/logs/train_rank0.log" \
  "${RUN_DIR}/logs/retriever.log" \
  "${RUN_DIR}/logs/ray_driver.log" \
  "${RUN_DIR}/logs/eval_worker.log" \
  "${RUN_DIR}/logs/watchdog.log" \
  "${RUN_DIR}/logs/errors.log" \
  "${RUN_DIR}/monitor/monitor_10min.log" \
  "${RUN_DIR}/monitor/monitor_10min.jsonl" \
  "${RUN_DIR}/monitor/alerts.log"

cp "${CONFIG}" "${RUN_DIR}/configs/formal_resume_u20_to_u500.yaml"
PYTHONPATH="${PROJECT_ROOT}/src" "${RL_ENV}/bin/python" - "${CONFIG}" "${RUN_DIR}/configs/resolved_config.yaml" <<'PY'
import sys, yaml
from agentic_rl.config import load_config
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    yaml.safe_dump(load_config(sys.argv[1]), handle, sort_keys=False)
PY
cp "${PROJECT_ROOT}/MANIFEST.sha256" "${RUN_DIR}/configs/source_manifest.sha256"
{
  date -u '+timestamp_utc=%Y-%m-%dT%H:%M:%SZ'
  "${RL_ENV}/bin/python" --version
  nvidia-smi
  "${RL_ENV}/bin/python" - <<'PY'
import ray, torch, transformers, verl, vllm
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("ray=" + ray.__version__)
print("transformers=" + transformers.__version__)
print("verl=" + str(getattr(verl, "__version__", None)))
print("vllm=" + vllm.__version__)
PY
} >"${RUN_DIR}/configs/environment.txt" 2>&1

cp -a "${PILOT_RUN}/eval/update_020" "${RUN_DIR}/eval/update_020"
"${RL_ENV}/bin/python" - "${RUN_DIR}" "${PILOT_CHECKPOINT}" "${PILOT_RUN}" "${SESSION}" <<'PY'
import json, sys, time
from pathlib import Path
from agentic_rl.runtime.formal_state import (
    append_jsonl, atomic_write_json, seed_completed_eval,
)

run = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
pilot = Path(sys.argv[3])
session = sys.argv[4]
summary = json.loads((run / "eval/update_020/metrics.json").read_text())
metadata = json.loads((checkpoint / "metadata.json").read_text())
stage = json.loads((pilot / "stage_results/stage_pilot20.json").read_text())
overall = next(row for row in summary["metrics"] if row["domain"] == "overall")
checksum = overall["actor_checksum"]
reload_checksums = {
    item["distributed_reload"]["actor_checksum_after"]
    for item in stage["checkpoint_reload"]["results"]
}
if reload_checksums != {checksum}:
    raise SystemExit("Pilot U20 Eval/checkpoint Actor checksum mismatch")
for row in summary["metrics"]:
    append_jsonl(run / "metrics/eval_metrics.jsonl", row)
seed_completed_eval(
    run,
    update=20,
    model_path=checkpoint,
    actor_checksum=checksum,
)
atomic_write_json(
    run / "state/processes.json",
    {
        "run_id": run.name,
        "tmux_session": session,
        "started_at": time.time(),
        "source_pilot_run": str(pilot),
        "source_checkpoint": str(checkpoint),
    },
)
atomic_write_json(run / "state/fatal_status.json", {"fatal": False, "timestamp": time.time()})
atomic_write_json(
    run / "state/training_progress.json",
    {
        "status": "starting",
        "attempt_id": int(metadata["attempt_id"]),
        "successful_update_step": 20,
        "successful_updates_since_resume": 0,
        "data_cursor": int(metadata["data_cursor"]),
        "optimizer_steps_total": 20,
        "scheduler_steps_total": 20,
        "latest_resume_checkpoint": str(checkpoint),
        "latest_model_checkpoint": None,
        "actor_checksum": checksum,
        "updated_at": time.time(),
    },
)
atomic_write_json(
    run / "state/latest_resume_checkpoint.json",
    {"successful_update_step": 20, "path": str(checkpoint), "actor_checksum": checksum},
)
(run / "state/latest_successful_update").write_text("20\n", encoding="utf-8")
(run / "state/latest_checkpoint").write_text(str(checkpoint) + "\n", encoding="utf-8")
PY

printf '%q ' bash "${PROJECT_ROOT}/scripts/train_formal_from_pilot20_to_500.sh" \
  >"${RUN_DIR}/configs/launch_command.sh"
printf '\n' >>"${RUN_DIR}/configs/launch_command.sh"
ln -sfn "${RUN_ID}" "${OUTPUT_ROOT}/latest"

tmux new-session -d -s "${SESSION}" -n retriever \
  "bash '${PROJECT_ROOT}/scripts/_formal_retriever_process.sh' '${RUN_DIR}'"
for _ in $(seq 1 360); do
  if "${RL_ENV}/bin/python" -c \
    "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
    >/dev/null 2>&1; then
    break
  fi
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    printf 'Retriever tmux session exited during startup.\n' >&2
    exit 1
  fi
  sleep 2
done
"${RL_ENV}/bin/python" -c \
  "from agentic_rl.retriever.health import query_health; query_health('http://127.0.0.1:8000')" \
  >/dev/null

tmux new-window -d -t "${SESSION}" -n eval \
  "bash '${PROJECT_ROOT}/scripts/async_eval_gpu0_worker.sh' '${CONFIG}' '${RUN_DIR}'"
tmux new-window -d -t "${SESSION}" -n monitor \
  "bash '${PROJECT_ROOT}/scripts/monitor_formal_training_10min.sh' '${CONFIG}' '${RUN_DIR}'"
tmux new-window -d -t "${SESSION}" -n trainer \
  "bash '${PROJECT_ROOT}/scripts/_formal_trainer_process.sh' '${CONFIG}' '${RUN_DIR}' '${PILOT_CHECKPOINT}'"
tmux new-window -d -t "${SESSION}" -n watchdog \
  "bash '${PROJECT_ROOT}/scripts/formal_training_watchdog.sh' '${CONFIG}' '${RUN_DIR}' '${PILOT_CHECKPOINT}'"

for _ in $(seq 1 300); do
  [[ -s "${RUN_DIR}/state/pids/trainer.pid" ]] && \
  [[ -s "${RUN_DIR}/state/pids/eval_worker.pid" ]] && \
  [[ -s "${RUN_DIR}/state/pids/monitor.pid" ]] && \
  [[ -s "${RUN_DIR}/state/pids/watchdog.pid" ]] && break
  sleep 1
done

printf 'RUN_ID=%s\n' "${RUN_ID}"
printf 'RUN_DIR=%s\n' "${RUN_DIR}"
printf 'TMUX_SESSION=%s\n' "${SESSION}"
for name in trainer retriever eval_worker monitor watchdog; do
  printf '%s_PID=%s\n' "${name^^}" "$(cat "${RUN_DIR}/state/pids/${name}.pid" 2>/dev/null || printf PENDING)"
done
printf 'CONSOLE_LOG=%s\n' "${RUN_DIR}/logs/console.log"
printf 'MONITOR_LOG=%s\n' "${RUN_DIR}/monitor/monitor_10min.log"
printf 'EVAL_QUEUE=%s\n' "${RUN_DIR}/state/eval_queue.json"
printf 'MODEL_CHECKPOINT_ROOT=%s\n' "${RUN_DIR}/checkpoints/models"
printf 'RESUME_CHECKPOINT_ROOT=%s\n' "${RUN_DIR}/checkpoints/resume"
