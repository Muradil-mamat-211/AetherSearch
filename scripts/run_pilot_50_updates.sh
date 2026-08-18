#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

RL_PYTHON="${RL_ENV}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/base.yaml"

"${RL_PYTHON}" - "${PROJECT_ROOT}" "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path
from agentic_rl.config import load_config
from agentic_rl.runtime.verl_config import unresolved_formal_fields

root = Path(sys.argv[1])
for stage in ("a", "b", "c", "d"):
    path = root / "runtime" / "stage_results" / f"stage_{stage}.json"
    if not path.is_file() or json.loads(path.read_text()).get("status") != "PASS":
        raise SystemExit(f"Runtime Stage {stage.upper()} gate is not PASS: {path}")
unresolved = unresolved_formal_fields(load_config(sys.argv[2]))
if unresolved:
    raise SystemExit("Formal hyperparameters are unresolved: " + ", ".join(unresolved))
PY

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_memory_monitor_refresh_ms=1000
export RAY_memory_usage_threshold=0.80
export AGENTIC_RL_RUNTIME_STAGE=E

exec "${RL_PYTHON}" -m agentic_rl.runtime.entrypoint --config "${CONFIG}"
