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

mapfile -d '' -t RESOLVED < <(
  "${RL_PYTHON}" - "${CONFIG}" "${RUN_DIR}" <<'PY'
import sys
from agentic_rl.config import load_config
from agentic_rl.runtime.retriever_command import build_retriever_command

config = load_config(sys.argv[1])
log_root = sys.argv[2] or config["paths"]["runtime_root"]
print(config["paths"]["runtime_root"], end="\0")
for value in build_retriever_command(config, log_root=log_root):
    print(value, end="\0")
PY
)

if [[ "${#RESOLVED[@]}" -lt 4 ]]; then
  printf 'Failed to resolve the retriever configuration.\n' >&2
  exit 1
fi

RUNTIME_ROOT="${RESOLVED[0]}"
LOG_ROOT="${RUN_DIR:-${RUNTIME_ROOT}}"
RETRIEVER_COMMAND=("${RESOLVED[@]:1}")
RETRIEVER_PYTHON="${RETRIEVER_COMMAND[0]}"
SERVER="${RETRIEVER_COMMAND[1]}"
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

exec "${RETRIEVER_COMMAND[@]}"
