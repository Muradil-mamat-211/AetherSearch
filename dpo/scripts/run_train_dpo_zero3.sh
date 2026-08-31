#!/usr/bin/env bash
set -euo pipefail

dpo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "${dpo_root}/.." && pwd)"
trainer="${dpo_root}/scripts/train_dpo.py"

# Data, model, and optimization defaults define the public recipe. Hardware
# topology and machine-local paths are environment inputs; this launcher never
# assigns physical device IDs or assumes a server filesystem layout.
python_bin="${PYTHON_BIN:-python}"
data_file="${DATA_FILE:-${dpo_root}/train.jsonl}"
default_model="muradil211/AetherSearch_SFT"
default_model_revision="437aca474d3966e57e82af565db95d0ad64aa24d"
model_name_or_path="${MODEL_NAME_OR_PATH:-${default_model}}"
model_revision="${MODEL_REVISION:-}"
if [[ -z "${model_revision}" && "${model_name_or_path}" == "${default_model}" ]]; then
  model_revision="${default_model_revision}"
fi
ref_model_name_or_path="${REFERENCE_MODEL_NAME_OR_PATH:-${model_name_or_path}}"
ref_model_revision="${REFERENCE_MODEL_REVISION:-}"
if [[ -z "${ref_model_revision}" && \
      "${ref_model_name_or_path}" == "${model_name_or_path}" ]]; then
  ref_model_revision="${model_revision}"
fi
output_dir="${OUTPUT_DIR:-${project_root}/outputs/dpo/aethersearch_dpo}"
deepspeed_config="${DEEPSPEED_CONFIG:-${dpo_root}/configs/ds_zero3_bf16.json}"
canonical_data_sha256="c42adcb0f194cff3126134b37afd85e4b89aa9917e5c98dda4b09904509f61e9"
canonical_num_samples=2126
max_seq_len="${MAX_SEQ_LEN:-4096}"
tokenization_batch_size="${TOKENIZATION_BATCH_SIZE:-64}"
per_device_batch="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
global_batch_size="${GLOBAL_BATCH_SIZE:-12}"
dataloader_workers="${DATALOADER_NUM_WORKERS:-2}"
forward_mode="${FORWARD_MODE:-sequential}"
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
  echo "[ERROR] refusing to overwrite final model: ${output_dir}/final_model" >&2
  exit 1
fi
if [[ -e "${output_dir}/.final_model.incomplete" ]]; then
  echo "[ERROR] incomplete final-model export requires inspection" >&2
  exit 1
fi
if [[ -n "${resume_from_checkpoint}" && ! -d "${resume_from_checkpoint}" ]]; then
  echo "[ERROR] resume checkpoint directory is missing: ${resume_from_checkpoint}" >&2
  exit 1
fi
if [[ -d "${output_dir}" && -z "${resume_from_checkpoint}" ]] && \
   [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] OUTPUT_DIR is non-empty; choose a new directory" >&2
  exit 1
fi

for numeric_value in \
  "${max_seq_len}" \
  "${tokenization_batch_size}" \
  "${per_device_batch}" \
  "${global_batch_size}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] expected a positive integer, got: ${numeric_value}" >&2
    exit 1
  fi
done
if [[ ! "${dataloader_workers}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] DATALOADER_NUM_WORKERS must be a non-negative integer" >&2
  exit 1
fi
if [[ "${forward_mode}" != "sequential" && "${forward_mode}" != "concatenated" ]]; then
  echo "[ERROR] FORWARD_MODE must be sequential or concatenated" >&2
  exit 1
fi
for boolean_value in "${TRUST_REMOTE_CODE:-0}" "${TF32:-1}"; do
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "[ERROR] TRUST_REMOTE_CODE and TF32 must be 0 or 1" >&2
    exit 1
  fi
done

if ! "${python_bin}" -c \
  'import importlib.util as u; required=("torch","transformers","accelerate","deepspeed","tensorboard"); missing=[m for m in required if u.find_spec(m) is None]; assert not missing, f"missing Python packages: {missing}"'; then
  echo "[ERROR] install dependencies from dpo/requirements.txt" >&2
  exit 1
fi
visible_gpu_count="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
nproc_per_node="${NPROC_PER_NODE:-${visible_gpu_count}}"
if [[ ! "${nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] no CUDA device is visible; configure device visibility externally" >&2
  exit 1
fi
if [[ ! "${visible_gpu_count}" =~ ^[0-9]+$ ]] || \
   [[ "${nproc_per_node}" -gt "${visible_gpu_count}" ]]; then
  echo "[ERROR] requested ${nproc_per_node} workers but ${visible_gpu_count} devices are visible" >&2
  exit 1
fi

world_size="${nproc_per_node}"
micro_batch_across_workers=$((world_size * per_device_batch))
if (( global_batch_size % micro_batch_across_workers != 0 )); then
  echo "[ERROR] GLOBAL_BATCH_SIZE=${global_batch_size} must be divisible by" \
       "NPROC_PER_NODE*PER_DEVICE_TRAIN_BATCH_SIZE=${micro_batch_across_workers}" >&2
  exit 1
fi
gradient_accumulation=$((global_batch_size / micro_batch_across_workers))
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ ! "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] GRADIENT_ACCUMULATION_STEPS must be a positive integer" >&2
    exit 1
  fi
  if [[ "${GRADIENT_ACCUMULATION_STEPS}" != "${gradient_accumulation}" ]]; then
    echo "[ERROR] GRADIENT_ACCUMULATION_STEPS conflicts with the derived value" >&2
    exit 1
  fi
fi
if ! "${python_bin}" -c \
  'import torch; assert torch.cuda.is_bf16_supported(), "visible hardware does not support BF16"'; then
  echo "[ERROR] this recipe requires BF16-capable CUDA hardware" >&2
  exit 1
fi

mkdir -p "${output_dir}"
available_kb="$(df -Pk "${output_dir}" | awk 'NR == 2 {print $4}')"
minimum_free_kb="${MINIMUM_FREE_KB:-0}"
if [[ ! "${minimum_free_kb}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] MINIMUM_FREE_KB must be a non-negative integer" >&2
  exit 1
fi
if [[ "${minimum_free_kb}" -gt 0 && "${available_kb}" -lt "${minimum_free_kb}" ]]; then
  echo "[ERROR] insufficient output-disk space" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

common_args=(
  --model_name_or_path "${model_name_or_path}"
  --ref_model_name_or_path "${ref_model_name_or_path}"
  --train_file "${data_file}"
  --output_dir "${output_dir}"
  --max_seq_len "${max_seq_len}"
  --long_sample_policy error
  --expected_num_samples "${canonical_num_samples}"
  --expected_sha256 "${canonical_data_sha256}"
  --tokenization_batch_size "${tokenization_batch_size}"
)
if [[ -n "${model_revision}" ]]; then
  common_args+=(--model_revision "${model_revision}")
fi
if [[ -n "${ref_model_revision}" ]]; then
  common_args+=(--ref_model_revision "${ref_model_revision}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  common_args+=(--trust_remote_code)
fi

echo "[INFO] Running strict data and mask preflight"
tee_args=()
if [[ -n "${resume_from_checkpoint}" ]]; then
  tee_args=(-a)
fi
"${python_bin}" "${trainer}" \
  "${common_args[@]}" \
  --check_data_only \
  --audit_report_path "${output_dir}/dpo_data_audit.json" \
  2>&1 | tee "${tee_args[@]}" "${output_dir}/preflight.log"

effective_global_batch=$((world_size * per_device_batch * gradient_accumulation))
echo "[INFO] Starting single-node training with nproc_per_node=${nproc_per_node}" \
     "world_size=${world_size}" \
     "per_device_batch=${per_device_batch}" \
     "gradient_accumulation=${gradient_accumulation}" \
     "effective_global_batch=${effective_global_batch}"

train_args=(
  --deepspeed "${deepspeed_config}"
  --precision bf16
  --forward_mode "${forward_mode}"
  --beta "${BETA:-0.1}"
  --per_device_train_batch_size "${per_device_batch}"
  --gradient_accumulation_steps "${gradient_accumulation}"
  --dataloader_num_workers "${dataloader_workers}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1.0}"
  --learning_rate "${LEARNING_RATE:-5e-7}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --weight_decay "${WEIGHT_DECAY:-0.0}"
  --seed "${SEED:-42}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --save_strategy "${SAVE_STRATEGY:-no}"
  --save_steps "${SAVE_STEPS:-100}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-1}"
)
if [[ "${TF32:-1}" == "1" ]]; then
  train_args+=(--tf32)
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  train_args+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${resume_from_checkpoint}" ]]; then
  train_args+=(--resume_from_checkpoint "${resume_from_checkpoint}")
fi

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${nproc_per_node}" \
  "${trainer}" \
  "${common_args[@]}" \
  "${train_args[@]}" \
  2>&1 | tee "${tee_args[@]}" "${output_dir}/train.log"
