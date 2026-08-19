#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AETHERSEARCH_PROJECT_ROOT="${AETHERSEARCH_PROJECT_ROOT:-${PROJECT_ROOT}}"

ENV_FILE="${AETHERSEARCH_ENV_FILE:-${PROJECT_ROOT}/environment/env.local.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

RL_PYTHON="${AETHERSEARCH_RL_PYTHON:-$(command -v python)}"
RETRIEVER_PYTHON="${AETHERSEARCH_RETRIEVER_PYTHON:-${RL_PYTHON}}"
test -x "${RL_PYTHON}"
test -x "${RETRIEVER_PYTHON}"

RL_ENV="$(cd "$(dirname "${RL_PYTHON}")/.." && pwd)"
export PATH="$(dirname "${RL_PYTHON}"):${PATH}"
# Do not add a legacy Search-R1 source tree here: it can shadow the installed
# veRL version selected by the user's RL environment.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PROJECT_ROOT
export RL_ENV
export RL_PYTHON
export RETRIEVER_PYTHON
