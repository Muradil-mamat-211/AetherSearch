#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

if [[ "$#" -gt 0 ]]; then
  exec env PYTHONPATH="${PROJECT_ROOT}/src" \
    "${RL_PYTHON}" -m pytest -q "$@"
fi

# Isolate test modules so large ML imports are released between files. This
# keeps the default suite usable in memory-constrained containers.
while IFS= read -r test_file; do
  printf 'AETHERSEARCH_TEST_FILE=%s\n' "${test_file}"
  env PYTHONPATH="${PROJECT_ROOT}/src" \
    "${RL_PYTHON}" -m pytest -q "${test_file}"
done < <(find "${PROJECT_ROOT}/tests" -maxdepth 1 -type f \
  -name 'test_*.py' -print | sort)
