#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

"${RL_PYTHON}" -m compileall -q \
  "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/scripts" \
  "${PROJECT_ROOT}/sft/scripts" \
  "${PROJECT_ROOT}/tests"
"${RL_PYTHON}" -c "import agentic_rl; import agentic_rl.config"
markdown_files=()
while IFS= read -r -d '' relative_path; do
  if [[ -f "${PROJECT_ROOT}/${relative_path}" ]]; then
    markdown_files+=("${PROJECT_ROOT}/${relative_path}")
  fi
done < <(
  git -C "${PROJECT_ROOT}" ls-files -z \
    --cached --others --exclude-standard -- '*.md' | sort -z
)
"${RL_PYTHON}" "${PROJECT_ROOT}/scripts/validate_readme.py" "${markdown_files[@]}"

while IFS= read -r script; do
  bash -n "${script}"
done < <(
  find "${PROJECT_ROOT}/scripts" "${PROJECT_ROOT}/sft/scripts" \
    -maxdepth 1 -type f -name '*.sh' -print | sort
)

for test_file in \
  test_public_config.py \
  test_fixed_eval_full_manifest.py \
  test_config_schema.py \
  test_strict_one_step_contract.py \
  test_topology_decoupling.py \
  test_runtime_ownership.py; do
  PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" -m pytest -q \
    "${PROJECT_ROOT}/tests/${test_file}"
done

PYTHONPATH="${PROJECT_ROOT}/src" "${RL_PYTHON}" -m pytest -q \
  "${PROJECT_ROOT}/tests/test_sft_2000_trainer.py"

printf 'AETHERSEARCH_STATIC_VALIDATION=PASS\n'
