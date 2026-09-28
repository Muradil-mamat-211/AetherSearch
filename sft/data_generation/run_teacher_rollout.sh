#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export AETHERSEARCH_SFT_WORKSPACE="$(cd "${AETHERSEARCH_SFT_WORKSPACE:-$repo_root}" && pwd -P)"
runner="$repo_root/sft/data_generation/search_sft_teacher/deepseek_rollout.py"
python_bin="${PYTHON_BIN:-$AETHERSEARCH_SFT_WORKSPACE/envs/retriever/bin/python}"
questions_file="${QUESTIONS_FILE:-}"
db_file="${DB_FILE:-$repo_root/logs/search_sft_teacher/deepseek_flat_v22.sqlite}"

if [[ -z "$questions_file" || "$questions_file" != /* || ! -f "$questions_file" ]]; then
  printf 'Set QUESTIONS_FILE to an existing absolute path to verified train QA JSONL.\n' >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Retriever Python is missing or not executable: %s\n' "$python_bin" >&2
  exit 2
fi

cd "$repo_root"
doctor_output="$("$python_bin" "$runner" --doctor)"
printf '%s\n' "$doctor_output"
if ! printf '%s\n' "$doctor_output" | "$python_bin" -c 'import json, sys; sys.exit(0 if json.load(sys.stdin)["ready"] else 1)'; then
  printf 'Retriever preflight is not ready; no teacher API request was made.\n' >&2
  exit 1
fi

args=(
  "$python_bin" "$runner" --run
  --model "${TEACHER_MODEL:-deepseek-flash}"
  --thinking "${TEACHER_THINKING:-disabled}"
  --questions "$questions_file"
  --db "$db_file"
  --max-examples "${MAX_EXAMPLES:-10}"
  --max-searches "${MAX_SEARCHES:-5}"
  --max-api-requests "${MAX_API_REQUESTS:-100}"
  --concurrency "${CONCURRENCY:-8}"
  --retrieval-batch-queries "${RETRIEVAL_BATCH_QUERIES:-8}"
  --retrieval-batch-wait-ms "${RETRIEVAL_BATCH_WAIT_MS:-5}"
  --public-id-start "${PUBLIC_ID_START:-500001}"
)
if [[ -n "${REASONING_EFFORT:-}" ]]; then
  args+=(--reasoning-effort "$REASONING_EFFORT")
fi
if [[ -n "${MAX_TOKENS:-}" ]]; then
  args+=(--max-tokens "$MAX_TOKENS")
fi
if [[ -n "${RETRY_IDS:-}" ]]; then
  args+=(--retry-ids "$RETRY_IDS")
fi
exec "${args[@]}"
