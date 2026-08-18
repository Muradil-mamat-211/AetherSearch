#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

export CUDA_VISIBLE_DEVICES=1,2,3,4
export RAY_memory_monitor_refresh_ms=1000
export RAY_memory_usage_threshold=0.80
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RL_PYTHON="/root/autodl-tmp/search-r1-workspace/envs/igpo-ragen2-fsdp2-vllm011/bin/python"
CONFIG="${PROJECT_ROOT}/configs/base.yaml"

test -x "${RL_PYTHON}"
exec "${RL_PYTHON}" -m agentic_rl.runtime.entrypoint --config "${CONFIG}"
