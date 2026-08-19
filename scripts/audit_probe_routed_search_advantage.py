#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from agentic_rl.advantage.a2tgpo import (
    SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
    SUFFICIENCY_NOVELTY_LOCAL_IG_MODE,
    _rebuild_sufficiency_novelty_cumulative_ig_probe_routed_outcome,
    rebuild_search_advantages,
)
from agentic_rl.config import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    advantage = config["advantage"]
    source = inspect.getsource(
        _rebuild_sufficiency_novelty_cumulative_ig_probe_routed_outcome
    )
    dispatcher = inspect.getsource(rebuild_search_advantages)
    checks = {
        "new_mode_selected": advantage["search_task_mode"]
        == SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
        "old_mode_distinct": SUFFICIENCY_NOVELTY_LOCAL_IG_MODE
        != SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
        "dispatcher_explicit": (
            "_rebuild_sufficiency_novelty_cumulative_ig_probe_routed_outcome"
            in dispatcher
        ),
        "s_priority": 'if sufficient:\n                actual = -1.0' in source,
        "n_priority": 'elif no_new:\n                actual = -1.0' in source,
        "n_does_not_break": "encountered_masked_n = True" in source,
        "s_after_break": "if s_after.get(future_index, False):" in source,
        "effective_sqrt_count": "math.fsum(values) / math.sqrt(len(values))"
        in source,
        "positive_probe_route": "route = max(z_outcome, 0.0)" in source,
        "negative_probe_route": "route = min(z_outcome, 0.0)" in source,
        "neutral_probe_route": "route = 0.0" in source,
        "delta_not_direct_term": "actual = float(d_ig_eff + route)" in source,
        "answer_copy_assertion": (
            "Probe-routed Search reconstruction changed A_answer" in source
        ),
        "a_sc_absent": "stop_continue_by_search_index={}" in source,
        "formula_config": advantage["search_advantage_formula"]
        == "-1.0 if S_before else -1.0 if N else D_ig_eff + O_route",
        "probe_epsilon": float(advantage["probe_epsilon"]) == 1.0e-6,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    files = [
        ROOT / "src/agentic_rl/advantage/a2tgpo.py",
        ROOT / "src/agentic_rl/runtime/stop_branching.py",
        ROOT / "src/agentic_rl/runtime/verl_runtime_adapter.py",
        ROOT / "configs/base.yaml",
        ROOT / "tests/test_sufficiency_novelty_cumulative_probe_routed.py",
    ]
    payload = {
        "result": "PASS" if not failed else "FAIL",
        "mode": SUFFICIENCY_NOVELTY_CUMULATIVE_IG_PROBE_ROUTED_OUTCOME_MODE,
        "checks": checks,
        "failed_checks": failed,
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in files
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
