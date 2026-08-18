#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"
PYTHON="/root/autodl-tmp/search-r1-workspace/envs/igpo-ragen2-fsdp2-vllm011/bin/python"

"${PYTHON}" -m compileall -q \
  "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/scripts" \
  "${PROJECT_ROOT}/tests"
"${PYTHON}" -c "import agentic_rl; import agentic_rl.controller.update_controller"
"${PROJECT_ROOT}/scripts/print_resolved_config.sh" >/dev/null
"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_static.py" >/dev/null
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_algorithm_boundary.py" >/dev/null
"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_stop_continue_search_advantage.py" >/dev/null

while IFS= read -r test_file; do
  "${PYTHON}" -m pytest -q "${test_file}"
done < <(find "${PROJECT_ROOT}/tests" -maxdepth 1 -type f -name 'test_*.py' -print | sort)

while IFS= read -r script; do
  bash -n "${script}"
done < <(find "${PROJECT_ROOT}/scripts" -maxdepth 1 -type f -name '*.sh' -print | sort)

"${PYTHON}" "${PROJECT_ROOT}/scripts/build_manifest.py"
(
  cd "${PROJECT_ROOT}"
  sha256sum -c MANIFEST.sha256 >/dev/null
)
