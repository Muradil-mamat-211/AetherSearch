#!/usr/bin/env bash
set -euo pipefail

# Copy this file to environment/env.local.sh, replace every /path/to value,
# then source it before resolving or launching the public RL recipe.
AETHERSEARCH_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AETHERSEARCH_PROJECT_ROOT=${AETHERSEARCH_PROJECT_ROOT:-$(dirname "${AETHERSEARCH_ENV_DIR}")}
export AETHERSEARCH_WORKSPACE=${AETHERSEARCH_WORKSPACE:-/path/to/aethersearch-workspace}

# Python environments. The RL environment contains PyTorch, veRL, vLLM, Ray,
# and this package; the retriever environment contains FAISS GPU and Pyserini.
export AETHERSEARCH_RL_PYTHON=${AETHERSEARCH_RL_PYTHON:-/path/to/rl-env/bin/python}
export AETHERSEARCH_RETRIEVER_PYTHON=${AETHERSEARCH_RETRIEVER_PYTHON:-/path/to/retriever-env/bin/python}
export AETHERSEARCH_ENV_SCRIPT=${AETHERSEARCH_ENV_SCRIPT:-${BASH_SOURCE[0]}}

# Model and Search-R1 data. Actor and Reference may point to the same immutable
# starting model, matching the released formal run.
export AETHERSEARCH_ACTOR_MODEL=${AETHERSEARCH_ACTOR_MODEL:-/path/to/actor-model}
export AETHERSEARCH_REFERENCE_MODEL=${AETHERSEARCH_REFERENCE_MODEL:-${AETHERSEARCH_ACTOR_MODEL}}
export AETHERSEARCH_TRAIN_DATA=${AETHERSEARCH_TRAIN_DATA:-/path/to/train.parquet}
export AETHERSEARCH_VALIDATION_DATA=${AETHERSEARCH_VALIDATION_DATA:-/path/to/test.parquet}
export AETHERSEARCH_SEARCH_R1_ROOT=${AETHERSEARCH_SEARCH_R1_ROOT:-/path/to/Search-R1}
export AETHERSEARCH_RUNTIME_ROOT=${AETHERSEARCH_RUNTIME_ROOT:-${AETHERSEARCH_WORKSPACE}/outputs/rl}

# Hybrid Wikipedia retriever assets.
export AETHERSEARCH_CORPUS_PATH=${AETHERSEARCH_CORPUS_PATH:-/path/to/wiki-18.jsonl}
export AETHERSEARCH_BM25_INDEX_PATH=${AETHERSEARCH_BM25_INDEX_PATH:-/path/to/bm25-index}
export AETHERSEARCH_DENSE_INDEX_PATH=${AETHERSEARCH_DENSE_INDEX_PATH:-/path/to/e5_Flat.index}
export AETHERSEARCH_DENSE_ENCODER_PATH=${AETHERSEARCH_DENSE_ENCODER_PATH:-/path/to/e5-base-v2}

export HF_HOME=${HF_HOME:-${AETHERSEARCH_WORKSPACE}/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}

# Required only when downloading/uploading private HuggingFace assets.
# export HF_TOKEN=hf_xxx
