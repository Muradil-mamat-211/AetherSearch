#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1"
source "${PROJECT_ROOT}/scripts/bootstrap_env.sh"

GATE="${PROJECT_ROOT}/artifacts/exact_ig_official_alignment_v3_20260730/EXACT_IG_FAST_SEQUENTIAL_PARITY_V3.json"
"${RL_ENV}/bin/python" - "${GATE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Exact-IG precision gate is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("gate_pass") is not True:
    raise SystemExit(f"Exact-IG precision gate is FAIL: {path}")
if payload.get("oracle_validated") is not True:
    raise SystemExit(f"Sequential Oracle is not validated: {path}")
if payload.get("allow_next_stage") is not True:
    raise SystemExit(f"Exact-IG V3 does not allow runtime stages: {path}")
if payload.get("selected_mode") != "OFFICIAL_BF16_FAST_FULL_LOGITS":
    raise SystemExit(f"Official BF16 full-logits Fast mode is absent: {path}")
equivalence = payload.get("ragen", {})
if equivalence.get("selected_ids_equal") is not True:
    raise SystemExit(f"Fast/Oracle selected IDs differ: {path}")
PY
"${PROJECT_ROOT}/scripts/run_stage_a_tito.sh"
"${PROJECT_ROOT}/scripts/run_stage_b_one_update.sh"
"${PROJECT_ROOT}/scripts/run_stage_c_five_updates.sh"
"${PROJECT_ROOT}/scripts/run_stage_d_full_shape_one_update.sh"
