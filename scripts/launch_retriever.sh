#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  printf 'Usage: launch_retriever.sh CONFIG [RUN_DIR]\n' >&2
  exit 2
fi

CONFIG="$(readlink -f "$1")"
RUN_DIR="${2:-}"
test -f "${CONFIG}"

mapfile -d '' -t VALUES < <(
  "${RL_PYTHON}" - "${CONFIG}" <<'PY'
import sys
from urllib.parse import urlparse
from agentic_rl.config import load_config
from agentic_rl.runtime.environment import retriever_runtime_options

config = load_config(sys.argv[1])
paths = config["paths"]
retriever = config["retriever"]
runtime = retriever_runtime_options(config)
service = urlparse(retriever["service_url"])
if service.scheme not in {"http", "https"} or not service.hostname or not service.port:
    raise ValueError("retriever.service_url must include scheme, host, and port")
values = (
    paths["retriever_python"],
    retriever["server_source"],
    paths["runtime_root"],
    retriever["bm25_index_path"],
    retriever["dense_index_path"],
    retriever["corpus_path"],
    retriever["dense_encoder_name"],
    retriever["dense_encoder_path"],
    retriever["top_k"],
    retriever["bm25_top_n"],
    retriever["dense_top_n"],
    retriever["fusion_alpha"],
    retriever["dense_query_batch_size"],
    retriever["bm25_workers"],
    retriever["request_batch_wait_ms"],
    retriever["request_batch_max_queries"],
    retriever["timeout_seconds"],
    runtime["faiss_gpu_device"],
    retriever["service_url"],
    service.hostname,
    service.port,
    runtime["query_max_length"],
    runtime["retrieval_use_fp16"],
    runtime["faiss_gpu"],
    runtime["require_faiss_gpu"],
    runtime["faiss_gpu_stream_flat"],
    runtime["faiss_gpu_use_fp16"],
    runtime["faiss_temp_memory_mb"],
    runtime["faiss_add_batch_size"],
    runtime["dense_device"],
)
for value in values:
    print(value, end="\0")
PY
)

if [[ "${#VALUES[@]}" -ne 30 ]]; then
  printf 'Failed to resolve the retriever configuration.\n' >&2
  exit 1
fi

RETRIEVER_PYTHON="${VALUES[0]}"
SERVER="${VALUES[1]}"
RUNTIME_ROOT="${VALUES[2]}"
LOG_ROOT="${RUN_DIR:-${RUNTIME_ROOT}}"
test -x "${RETRIEVER_PYTHON}"
test -f "${SERVER}"
mkdir -p "${LOG_ROOT}/logs"

mapfile -d '' -t RETRIEVER_ENV_ASSIGNMENTS < <(
  "${RL_PYTHON}" - "${CONFIG}" <<'PY'
import sys
from agentic_rl.config import load_config
from agentic_rl.runtime.environment import runtime_retriever_environment

config = load_config(sys.argv[1])
for key, value in runtime_retriever_environment(config).items():
    print(f"{key}={value}", end="\0")
PY
)
if [[ "${#RETRIEVER_ENV_ASSIGNMENTS[@]}" -eq 0 ]]; then
  printf 'Failed to resolve retriever runtime environment.\n' >&2
  exit 1
fi

RETRIEVER_ENV="$(cd "$(dirname "${RETRIEVER_PYTHON}")/.." && pwd)"
if [[ -d "${RETRIEVER_ENV}/lib/server" ]]; then
  export JAVA_HOME="${RETRIEVER_ENV}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${JAVA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

for assignment in "${RETRIEVER_ENV_ASSIGNMENTS[@]}"; do
  export "${assignment%%=*}=${assignment#*=}"
done

RETRIEVER_ARGS=(
  --bm25-index-path "${VALUES[3]}" \
  --dense-index-path "${VALUES[4]}" \
  --corpus-path "${VALUES[5]}" \
  --retriever-name "${VALUES[6]}" \
  --retriever-model "${VALUES[7]}" \
  --topk "${VALUES[8]}" \
  --bm25-topn "${VALUES[9]}" \
  --dense-topn "${VALUES[10]}" \
  --alpha "${VALUES[11]}" \
  --query-max-length "${VALUES[21]}" \
  --dense-query-batch-size "${VALUES[12]}" \
  --bm25-workers "${VALUES[13]}" \
  --request-batch-wait-ms "${VALUES[14]}" \
  --request-batch-max-queries "${VALUES[15]}" \
  --request-wait-timeout-seconds "${VALUES[16]}" \
  --faiss-gpu-device "${VALUES[17]}" \
  --faiss-temp-memory-mb "${VALUES[27]}" \
  --faiss-add-batch-size "${VALUES[28]}" \
  --dense-device "${VALUES[29]}" \
  --host "${VALUES[19]}" \
  --port "${VALUES[20]}" \
  --log-file "${LOG_ROOT}/logs/retriever.log"
)
[[ "${VALUES[22]}" == "True" ]] && RETRIEVER_ARGS+=(--retrieval-use-fp16)
[[ "${VALUES[23]}" == "True" ]] && RETRIEVER_ARGS+=(--faiss-gpu)
[[ "${VALUES[24]}" == "True" ]] && RETRIEVER_ARGS+=(--require-faiss-gpu)
[[ "${VALUES[25]}" == "True" ]] && RETRIEVER_ARGS+=(--faiss-gpu-stream-flat)
[[ "${VALUES[26]}" == "True" ]] && RETRIEVER_ARGS+=(--faiss-gpu-use-fp16)

exec "${RETRIEVER_PYTHON}" "${SERVER}" "${RETRIEVER_ARGS[@]}"
