#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1"

for service in rl retriever; do
  pid_file="${PROJECT_ROOT}/runtime/pids/${service}.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(<"${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}"
    fi
    rm -f "${pid_file}"
  fi
done
