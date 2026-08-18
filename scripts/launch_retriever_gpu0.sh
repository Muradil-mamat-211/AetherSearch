#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

export CUDA_VISIBLE_DEVICES=0
RETRIEVER_PYTHON="/root/autodl-tmp/search-r1-workspace/envs/retriever/bin/python"
export JAVA_HOME="/root/autodl-tmp/search-r1-workspace/envs/retriever"
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${JAVA_HOME}/lib:${LD_LIBRARY_PATH:-}"
# Pyserini imports its optional OpenAI encoder module eagerly. Recent OpenAI
# clients require a syntactically present key during construction even though
# this local E5/BM25 service never uses that encoder or makes OpenAI requests.
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-pyserini-import-only}"
# BM25 already has a 16-worker pool. Bound per-process math libraries so the
# hybrid service uses host cores without multiplying 16 workers by 100 threads.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
SERVER="${PROJECT_ROOT}/runtime_assets/retriever/hybrid_retrieval_server.py"
RUNTIME="${PROJECT_ROOT}/runtime"

test -x "${RETRIEVER_PYTHON}"
test -f "${SERVER}"
mkdir -p "${RUNTIME}/logs"

exec "${RETRIEVER_PYTHON}" "${SERVER}" \
  --bm25-index-path /root/autodl-tmp/search-r1-workspace/data/wiki18_bm25/bm25 \
  --dense-index-path /root/autodl-tmp/search-r1-workspace/data/nq_search/e5_Flat.index \
  --corpus-path /root/autodl-tmp/search-r1-workspace/data/wiki18_corpus/wiki-18.jsonl \
  --retriever-name e5 \
  --retriever-model /root/autodl-tmp/search-r1-workspace/models/e5-base-v2 \
  --topk 3 \
  --bm25-topn 20 \
  --dense-topn 20 \
  --alpha 0.5 \
  --query-max-length 256 \
  --dense-query-batch-size 64 \
  --bm25-workers 16 \
  --request-batch-wait-ms 5 \
  --request-batch-max-queries 256 \
  --request-wait-timeout-seconds 180 \
  --retrieval-use-fp16 \
  --faiss-gpu \
  --require-faiss-gpu \
  --faiss-gpu-stream-flat \
  --faiss-gpu-device 0 \
  --faiss-gpu-use-fp16 \
  --faiss-temp-memory-mb 256 \
  --faiss-add-batch-size 0 \
  --dense-device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --log-file "${RUNTIME}/logs/retriever_gpu0.log"
