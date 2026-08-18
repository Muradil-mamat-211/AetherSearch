#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

exec /root/autodl-tmp/search-r1-workspace/envs/igpo-ragen2-fsdp2-vllm011/bin/python \
  -m agentic_rl.config \
  --config "${PROJECT_ROOT}/configs/base.yaml" \
  --format yaml
