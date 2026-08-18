#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEARCH_R1_ENV="/root/autodl-tmp/search-r1-workspace/env.sh"
RL_ENV="/root/autodl-tmp/search-r1-workspace/envs/igpo-ragen2-fsdp2-vllm011"

test -f "${SEARCH_R1_ENV}"
test -x "${RL_ENV}/bin/python"
source "${SEARCH_R1_ENV}"

export PATH="${RL_ENV}/bin:${PATH}"
# RL must import the installed veRL 0.6.1 package. Adding the legacy
# Search-R1 source tree here shadows it with Search-R1's bundled veRL 0.1.
export PYTHONPATH="${PROJECT_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export PROJECT_ROOT
export RL_ENV
