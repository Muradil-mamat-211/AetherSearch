#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

"${RL_PYTHON}" -m compileall -q \
  "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/scripts" \
  "${PROJECT_ROOT}/tests"
"${RL_PYTHON}" -c "import agentic_rl; import agentic_rl.config"

while IFS= read -r script; do
  bash -n "${script}"
done < <(find "${PROJECT_ROOT}/scripts" -maxdepth 1 -type f -name '*.sh' -print | sort)

for test_file in \
  test_public_config.py \
  test_config_schema.py \
  test_strict_one_step_contract.py; do
  PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" -m pytest -q \
    "${PROJECT_ROOT}/tests/${test_file}"
done

printf 'AETHERSEARCH_STATIC_VALIDATION=PASS\n'
