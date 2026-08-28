#!/usr/bin/env bash
set -euo pipefail

sft_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "${sft_root}/.." && pwd)"
trainer="${sft_root}/scripts/train_sft_2000.py"

python_bin="${PYTHON_BIN:-python}"
data_file="${DATA_FILE:-${sft_root}/final_sft_2000.jsonl}"
default_model="Qwen/Qwen2.5-3B-Instruct"
default_model_revision="aa8e72537993ba99e69dfaafa59ed015b17504d1"
model_name_or_path="${MODEL_NAME_OR_PATH:-${default_model}}"
model_revision="${MODEL_REVISION:-}"
if [[ -z "${model_revision}" && "${model_name_or_path}" == "${default_model}" ]]; then
  model_revision="${default_model_revision}"
fi
output_dir="${OUTPUT_DIR:-${project_root}/outputs/sft/qwen2p5_3b_instruct_sft_2000}"
deepspeed_config="${DEEPSPEED_CONFIG:-${sft_root}/configs/ds_zero3_bf16.json}"
expected_sha256="${EXPECTED_SHA256:-fec609652d3832c7a6c0ee2861c6f946b6cf7c3d3d40fc5d9be9b75df6325dcb}"
expected_num_samples="${EXPECTED_NUM_SAMPLES:-2000}"
max_seq_len="${MAX_SEQ_LEN:-4096}"
tokenization_batch_size="${TOKENIZATION_BATCH_SIZE:-64}"
per_device_batch="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-8}"
dataloader_workers="${DATALOADER_NUM_WORKERS:-2}"
resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-}"

if [[ "${python_bin}" == */* ]]; then
  if [[ ! -x "${python_bin}" ]]; then
    echo "[ERROR] PYTHON_BIN is not executable: ${python_bin}" >&2
    exit 1
  fi
elif ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "[ERROR] Python command is unavailable: ${python_bin}" >&2
  exit 1
fi

for required_file in "${data_file}" "${trainer}" "${deepspeed_config}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[ERROR] required file is missing: ${required_file}" >&2
    exit 1
  fi
done
if [[ -e "${output_dir}/final_model" ]]; then
  echo "[ERROR] refusing to overwrite existing final model: ${output_dir}/final_model" >&2
  exit 1
fi
if [[ -e "${output_dir}/.final_model.incomplete" ]]; then
  echo "[ERROR] incomplete final-model export requires inspection: ${output_dir}/.final_model.incomplete" >&2
  exit 1
fi
if [[ -n "${resume_from_checkpoint}" && ! -d "${resume_from_checkpoint}" ]]; then
  echo "[ERROR] resume checkpoint directory is missing: ${resume_from_checkpoint}" >&2
  exit 1
fi
if [[ -d "${output_dir}" && -z "${resume_from_checkpoint}" ]] && \
   [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] OUTPUT_DIR is non-empty; choose a new directory or set RESUME_FROM_CHECKPOINT" >&2
  exit 1
fi

for numeric_value in \
  "${expected_num_samples}" \
  "${max_seq_len}" \
  "${tokenization_batch_size}" \
  "${per_device_batch}" \
  "${gradient_accumulation}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] expected a positive integer, got: ${numeric_value}" >&2
    exit 1
  fi
done
if [[ ! "${dataloader_workers}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] DATALOADER_NUM_WORKERS must be a non-negative integer" >&2
  exit 1
fi
for boolean_value in "${TRUST_REMOTE_CODE:-0}" "${GROUP_BY_LENGTH:-1}"; do
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "[ERROR] TRUST_REMOTE_CODE and GROUP_BY_LENGTH must be 0 or 1" >&2
    exit 1
  fi
done

if ! "${python_bin}" -c \
  'import importlib.util as u; required=("torch","transformers","accelerate","deepspeed","tensorboard"); missing=[m for m in required if u.find_spec(m) is None]; assert not missing, f"missing Python packages: {missing}"'; then
  echo "[ERROR] install the dependencies listed in sft/requirements.txt" >&2
  exit 1
fi
visible_gpu_count="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
nproc_per_node="${NPROC_PER_NODE:-${visible_gpu_count}}"
if [[ ! "${nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] no CUDA device is visible; set CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi
if [[ ! "${visible_gpu_count}" =~ ^[0-9]+$ ]] || \
   [[ "${nproc_per_node}" -gt "${visible_gpu_count}" ]]; then
  echo "[ERROR] requested ${nproc_per_node} processes but only ${visible_gpu_count} CUDA devices are visible" >&2
  exit 1
fi
if ! "${python_bin}" -c \
  'import torch; assert torch.cuda.is_bf16_supported(), "visible CUDA hardware does not support BF16"'; then
  echo "[ERROR] this ZeRO-3 recipe requires BF16-capable CUDA hardware" >&2
  exit 1
fi

mkdir -p "${output_dir}"
available_kb="$(df -Pk "${output_dir}" | awk 'NR == 2 {print $4}')"
minimum_free_kb="${MINIMUM_FREE_KB:-7000000}"
if [[ ! "${minimum_free_kb}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MINIMUM_FREE_KB must be a positive integer" >&2
  exit 1
fi
if [[ "${available_kb}" -lt "${minimum_free_kb}" ]]; then
  echo "[ERROR] insufficient output-disk space: available=${available_kb}KB required=${minimum_free_kb}KB" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

common_args=(
  --model_name_or_path "${model_name_or_path}"
  --train_file "${data_file}"
  --output_dir "${output_dir}"
  --max_seq_len "${max_seq_len}"
  --long_sample_policy error
  --expected_num_samples "${expected_num_samples}"
  --tokenization_batch_size "${tokenization_batch_size}"
)
if [[ -n "${model_revision}" ]]; then
  common_args+=(--model_revision "${model_revision}")
fi
if [[ -n "${expected_sha256}" ]]; then
  common_args+=(--expected_sha256 "${expected_sha256}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  common_args+=(--trust_remote_code)
fi

echo "[INFO] Running strict data and loss-mask preflight"
tee_args=()
if [[ -n "${resume_from_checkpoint}" ]]; then
  tee_args=(-a)
fi
"${python_bin}" "${trainer}" \
  "${common_args[@]}" \
  --check_data_only \
  --audit_report_path "${output_dir}/sft_2000_data_audit.json" \
  2>&1 | tee "${tee_args[@]}" "${output_dir}/preflight.log"

effective_global_batch=$((nproc_per_node * per_device_batch * gradient_accumulation))
echo "[INFO] Starting training with nproc_per_node=${nproc_per_node} effective_global_batch=${effective_global_batch}"
train_args=(
  --deepspeed "${deepspeed_config}"
  --precision bf16
  --tf32
  --per_device_train_batch_size "${per_device_batch}"
  --gradient_accumulation_steps "${gradient_accumulation}"
  --dataloader_num_workers "${dataloader_workers}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1.0}"
  --learning_rate "${LEARNING_RATE:-2e-6}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --weight_decay "${WEIGHT_DECAY:-0.0}"
  --seed "${SEED:-42}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --save_strategy "${SAVE_STRATEGY:-no}"
  --save_steps "${SAVE_STEPS:-100}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-1}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
  train_args+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${resume_from_checkpoint}" ]]; then
  train_args+=(--resume_from_checkpoint "${resume_from_checkpoint}")
fi
if [[ "${GROUP_BY_LENGTH:-1}" == "0" ]]; then
  train_args+=(--no-group_by_length)
fi

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${nproc_per_node}" \
  "${trainer}" \
  "${common_args[@]}" \
  "${train_args[@]}" \
  2>&1 | tee "${tee_args[@]}" "${output_dir}/train.log"
