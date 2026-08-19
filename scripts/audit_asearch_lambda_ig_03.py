#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from agentic_rl.advantage.a2tgpo import SEARCH_IG_COEFFICIENT
from agentic_rl.config import load_config, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
FORMAL_CONFIG = PROJECT_ROOT / "configs" / "formal_train.yaml"
BEFORE_RESOLVED = REPORT_DIR / "ASEARCH_LAMBDA_IG_03_RESOLVED_BEFORE.yaml"
AFTER_RESOLVED = REPORT_DIR / "ASEARCH_LAMBDA_IG_03_RESOLVED_AFTER.yaml"
RUNTIME_MANIFEST = REPORT_DIR / "ASEARCH_LAMBDA_IG_03_RUNTIME_MANIFEST.json"
SEARCH_RESULTS = REPORT_DIR / "ASEARCH_LAMBDA_IG_03_FULL_REPO_SEARCH.txt"
REPORT_JSON = REPORT_DIR / "ASEARCH_LAMBDA_IG_03_AUDIT.json"
REPORT_MD = REPORT_DIR / "ASEARCH_LAMBDA_IG_03_AUDIT.md"
FULL_TEST_LOG = REPORT_DIR / "ASEARCH_LAMBDA_IG_03_FULL_TESTS.log"

PRODUCTION_SCRIPTS = frozenset(
    {
        "scripts/train_formal_manual.sh",
        "scripts/_run_runtime_job.sh",
        "scripts/preflight_fresh_formal_sc.py",
        "scripts/audit_stop_continue_search_advantage.py",
        "scripts/check_algorithm_boundary.py",
    }
)
MODIFIED_FILES = (
    "configs/base.yaml",
    "configs/formal_train.yaml",
    "src/agentic_rl/advantage/a2tgpo.py",
    "src/agentic_rl/runtime/learner_batch.py",
    "src/agentic_rl/config.py",
    "scripts/preflight_fresh_formal_sc.py",
    "scripts/audit_stop_continue_search_advantage.py",
    "scripts/check_algorithm_boundary.py",
    "scripts/audit_final_asearch_production.py",
    "scripts/audit_asearch_lambda_ig_03.py",
    "tests/test_a2tgpo_advantage.py",
    "tests/test_stop_continue_advantage.py",
    "tests/test_config_schema.py",
    "tests/test_fresh_formal_sc_launch.py",
    "tests/test_final_asearch_production_contract.py",
    "STOP_CONTINUE_SEARCH_ADVANTAGE_ALGORITHM_REPORT.md",
    "FINAL_ALGORITHM_SPEC_V2_2.md",
)
TEST_FILES = (
    "tests/test_a2tgpo_advantage.py",
    "tests/test_stop_continue_advantage.py",
    "tests/test_final_asearch_production_contract.py",
    "tests/test_config_schema.py",
    "tests/test_fresh_formal_sc_launch.py",
    "tests/test_selection_math.py",
    "tests/test_selection_boundaries.py",
)
OLD_ASSIGNMENT = re.compile(
    r"lambda[_ ]?(?:ig|IG)[^\n]{0,50}(?::|=)\s*1(?:\.0)?(?:[,\s]|$)"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line(path: Path, needle: str) -> str:
    for number, value in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in value:
            return f"{path.relative_to(PROJECT_ROOT)}:{number}"
    raise RuntimeError(f"Missing production evidence {needle!r} in {path}")


def _load_before_config() -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in (PROJECT_ROOT / "outputs").glob("**/configs/resolved_config.yaml"):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            coefficient = float(payload["advantage"]["lambda_ig"])
        except (KeyError, TypeError, ValueError, yaml.YAMLError):
            continue
        if coefficient == 1.0:
            candidates.append((path.stat().st_mtime_ns, path, payload))
    if not candidates:
        raise RuntimeError("No immutable pre-change resolved config with lambda_ig=1.0")
    _, path, payload = max(candidates, key=lambda item: item[0])
    return path, payload


def _resolve_after_config() -> dict[str, Any]:
    config = load_config(FORMAL_CONFIG)
    for section in ("formal", "formal_schedule", "scheduler"):
        config[section]["total_successful_updates"] = 500
    validate_config(config)
    return config


def _repo_assignment_matches() -> list[dict[str, Any]]:
    command = [
        "rg",
        "-n",
        "--hidden",
        "--no-ignore",
        "--glob",
        "!*.pyc",
        "--glob",
        "!*.safetensors",
        "--glob",
        "!*.pt",
        "--glob",
        "!*.bin",
        "--glob",
        "!runtime/ray/**",
        "--glob",
        "!outputs/**/logs/**",
        "--glob",
        "!outputs/**/metrics/**",
        "--glob",
        "!reports/ASEARCH_LAMBDA_IG_03_*",
        "-e",
        OLD_ASSIGNMENT.pattern,
        ".",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "rg failed")
    matches: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        relative, line_number, text = raw.removeprefix("./").split(":", 2)
        if relative.startswith("tests/"):
            category = "test_prechange_comparator"
        elif relative.startswith("outputs/"):
            category = "immutable_historical_run"
        elif relative in {
            "FINAL_ALGORITHM_SPEC_V2_1.md",
            "EFFECTIVE_ALGORITHM_FROM_CODE_V2_1.md",
            "scripts/audit_final_algorithm_v2_1.py",
        }:
            category = "frozen_v2_1_audit"
        elif relative.startswith("src/") or relative.startswith("configs/"):
            category = "production_violation"
        elif relative in PRODUCTION_SCRIPTS:
            category = "production_violation"
        else:
            category = "nonproduction_historical_document"
        matches.append(
            {
                "path": relative,
                "line": int(line_number),
                "text": text.strip(),
                "category": category,
            }
        )
    return matches


def _production_chain() -> list[dict[str, str]]:
    evidence = (
        (
            "formal entry selects the production overlay",
            "scripts/train_formal_manual.sh",
            'BASE_CONFIG="${PROJECT_ROOT}/configs/formal_train.yaml"',
        ),
        (
            "formal entry resolves inherited configuration",
            "scripts/train_formal_manual.sh",
            "config = load_config(sys.argv[1])",
        ),
        (
            "configuration validator locks 0.3",
            "src/agentic_rl/config.py",
            '_require_equal(advantage["lambda_ig"], 0.3',
        ),
        (
            "preflight runtime manifest records lambda_ig",
            "scripts/preflight_fresh_formal_sc.py",
            '"lambda_ig": float(config["advantage"]["lambda_ig"]),',
        ),
        (
            "formal run persists the runtime manifest",
            "scripts/train_formal_manual.sh",
            'cp "${PREFLIGHT_JSON}" "${RUN_DIR}/configs/preflight.json"',
        ),
        (
            "controller waits for selected-only Stop branches",
            "src/agentic_rl/controller/update_controller.py",
            '"prepare_selected_stop_branches"',
        ),
        (
            "controller requests selected learner microbatches",
            "src/agentic_rl/controller/update_controller.py",
            "runtime.selected_microbatches(selected_groups)",
        ),
        (
            "runtime passes resolved advantage config",
            "src/agentic_rl/runtime/verl_runtime_adapter.py",
            'advantage_config=dict(self.config["advantage"])',
        ),
        (
            "learner builder passes lambda_ig",
            "src/agentic_rl/runtime/learner_batch.py",
            'config.get("lambda_ig", SEARCH_IG_COEFFICIENT)',
        ),
        (
            "Search advantage is rebuilt",
            "src/agentic_rl/advantage/a2tgpo.py",
            "new_value = float(lambda_ig * a_ig + lambda_task * sc.task_advantage)",
        ),
        (
            "rebuilt values enter rank payload",
            "src/agentic_rl/runtime/learner_batch.py",
            '"advantage_by_turn": [',
        ),
        (
            "FSDP loss consumes rank payload",
            "src/agentic_rl/runtime/fsdp_worker.py",
            'microbatch["advantage_by_turn"][batch_index]',
        ),
    )
    return [
        {
            "step": step,
            "location": _line(PROJECT_ROOT / relative, needle),
            "needle": needle,
        }
        for step, relative, needle in evidence
    ]


def _run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *TEST_FILES]
    environment = {**dict(os.environ), "PYTHONPATH": str(PROJECT_ROOT / "src")}
    collection = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *TEST_FILES],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    test_count = sum(
        int(match.group(1))
        for match in re.finditer(r"\.py:\s+([0-9]+)$", collection.stdout, re.MULTILINE)
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = (completed.stdout + completed.stderr).strip()
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "test_count": test_count,
        "output": output,
    }


def _modified_line_evidence() -> list[dict[str, str]]:
    evidence = (
        ("base production coefficient", "configs/base.yaml", "lambda_ig: 0.3"),
        ("formal overlay coefficient", "configs/formal_train.yaml", "lambda_ig: 0.3"),
        (
            "code fallback and lock constant",
            "src/agentic_rl/advantage/a2tgpo.py",
            "SEARCH_IG_COEFFICIENT = 0.3",
        ),
        (
            "final Search formula",
            "src/agentic_rl/advantage/a2tgpo.py",
            "new_value = float(lambda_ig * a_ig + lambda_task * sc.task_advantage)",
        ),
        (
            "runtime learner config handoff",
            "src/agentic_rl/runtime/learner_batch.py",
            'config.get("lambda_ig", SEARCH_IG_COEFFICIENT)',
        ),
        (
            "resolved config hard lock",
            "src/agentic_rl/config.py",
            '_require_equal(advantage["lambda_ig"], 0.3',
        ),
        (
            "fixed tensor regression",
            "tests/test_final_asearch_production_contract.py",
            "expected_search = 0.3 * a_ig + expected_task",
        ),
        (
            "RAGEN before/after comparator",
            "tests/test_final_asearch_production_contract.py",
            'production_config["advantage"]["lambda_ig"] = 0.3',
        ),
    )
    return [
        {"description": description, "location": _line(PROJECT_ROOT / path, needle)}
        for description, path, needle in evidence
    ]


def _coefficient_view(config: dict[str, Any]) -> dict[str, float]:
    advantage = config["advantage"]
    return {
        "lambda_ig": float(advantage["lambda_ig"]),
        "lambda_task": float(advantage["lambda_task"]),
        "lambda_outcome": float(advantage["lambda_outcome"]),
        "lambda_format": float(advantage["lambda_format"]),
    }


def _full_test_log_status() -> dict[str, Any]:
    if not FULL_TEST_LOG.is_file():
        return {"passed": False, "test_count": 0, "path": str(FULL_TEST_LOG)}
    text = FULL_TEST_LOG.read_text(encoding="utf-8")
    test_count = 0
    for line in text.splitlines():
        match = re.match(r"^(\.+)\s+\[\s*[0-9]+%\]$", line)
        if match:
            test_count += len(match.group(1))
    passed = (
        test_count == 246
        and "[100%]" in text
        and " failed" not in text.lower()
        and " error" not in text.lower()
    )
    return {
        "passed": passed,
        "test_count": test_count,
        "path": str(FULL_TEST_LOG),
        "sha256": _sha256_file(FULL_TEST_LOG),
    }


def _assert_invariants(config: dict[str, Any]) -> dict[str, bool]:
    advantage = config["advantage"]
    selection = config["selection"]
    policy = config["policy"]
    return {
        "production_constant_is_point_three": SEARCH_IG_COEFFICIENT == 0.3,
        "resolved_lambda_ig_is_point_three": float(advantage["lambda_ig"]) == 0.3,
        "lambda_task_unchanged": float(advantage["lambda_task"]) == 1.0,
        "lambda_outcome_unchanged": float(advantage["lambda_outcome"]) == 1.0,
        "lambda_format_unchanged": float(advantage["lambda_format"]) == 1.0,
        "ragen_alpha_ig_unchanged": float(selection["alpha_ig"]) == 0.5,
        "ragen_alpha_outcome_unchanged": float(selection["alpha_outcome"]) == 0.5,
        "adaptive_clip_beta_unchanged": float(policy["adaptive_clip_beta"]) == 0.3,
        "adaptive_clip_epsilon_low_unchanged": (
            float(policy["adaptive_clip_epsilon_low"]) == 0.003
        ),
        "adaptive_clip_epsilon_high_unchanged": (
            float(policy["adaptive_clip_epsilon_high"]) == 0.004
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Search Advantage lambda_IG=0.3 Production Audit",
        "",
        f"Result: **{payload['result']}**",
        "",
        "## Effective Formula",
        "",
        "```text",
        "A_task = A_SC if sc_clear else z_O",
        "A_search = 0.3 * A_IG + 1.0 * A_task",
        "A_answer = 1.0 * z_O + 1.0 * A_format",
        "```",
        "",
        "## Resolved Configuration",
        "",
        f"- Before source: `{payload['resolved_config']['before_source']}`",
        f"- Before snapshot: `{payload['resolved_config']['before_snapshot']}`",
        f"- After snapshot: `{payload['resolved_config']['after_snapshot']}`",
        f"- Runtime manifest: `{payload['runtime_manifest']}`",
        "",
        "| Field | Before | After |",
        "|---|---:|---:|",
    ]
    before = payload["resolved_config"]["before_coefficients"]
    after = payload["resolved_config"]["after_coefficients"]
    for key in ("lambda_ig", "lambda_task", "lambda_outcome", "lambda_format"):
        lines.append(f"| `{key}` | {before[key]} | {after[key]} |")
    lines.extend(["", "## Production Call Chain", ""])
    for index, item in enumerate(payload["production_call_chain"], 1):
        lines.append(f"{index}. {item['step']}: `{item['location']}`")
    lines.extend(["", "## Modified Line Evidence", ""])
    for item in payload["modified_line_evidence"]:
        lines.append(f"- {item['description']}: `{item['location']}`")
    lines.extend(["", "## Locked Invariants", "", "| Check | Result |", "|---|---|"])
    for key, value in payload["checks"].items():
        lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Regression Tests",
            "",
            f"- Command: `{payload['tests']['command']}`",
            f"- Result: **{'PASS' if payload['tests']['passed'] else 'FAIL'}**",
            f"- Tests: **{payload['tests']['test_count']} passed**",
            f"- Output: `{payload['tests']['output']}`",
            f"- Full project suite: **{payload['full_suite']['test_count']} passed**",
            f"- Full suite log: `{payload['full_suite']['path']}`",
            "",
            "The fixed-tensor Search formula, byte-identical Answer advantage, "
            "RAGEN selected-set/hash invariance, and production max-over-alias "
            "task reward are executable tests in the listed suite.",
            "",
            "## Full Repository Search",
            "",
            f"- Search artifact: `{payload['full_repo_search']['path']}`",
            f"- Production-path `lambda_ig=1.0` matches: "
            f"**{payload['full_repo_search']['production_violation_count']}**",
            "- Remaining matches are immutable old run/config/checkpoint evidence, "
            "frozen V2.1 material, or the explicit pre-change test comparator.",
            "",
            "## Side Effects",
            "",
            "No rollout, backward, optimizer step, scheduler step, checkpoint write, "
            "or formal training process was started by this audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    before_path, before = _load_before_config()
    after = _resolve_after_config()
    BEFORE_RESOLVED.write_text(
        yaml.safe_dump(before, sort_keys=False), encoding="utf-8"
    )
    AFTER_RESOLVED.write_text(
        yaml.safe_dump(after, sort_keys=False), encoding="utf-8"
    )

    matches = _repo_assignment_matches()
    SEARCH_RESULTS.write_text(
        "\n".join(
            f"{row['category']}\t{row['path']}:{row['line']}\t{row['text']}"
            for row in matches
        )
        + ("\n" if matches else ""),
        encoding="utf-8",
    )
    production_violations = [
        row for row in matches if row["category"] == "production_violation"
    ]
    checks = _assert_invariants(after)
    checks["no_production_lambda_ig_one"] = not production_violations
    tests = _run_tests()
    checks["regression_tests"] = bool(tests["passed"])
    full_suite = _full_test_log_status()
    checks["full_project_test_suite"] = bool(full_suite["passed"])

    manifest = {
        "schema": "asearch_lambda_ig_03_runtime_manifest_v1",
        "resolved_config": str(AFTER_RESOLVED),
        "resolved_config_sha256": _sha256_file(AFTER_RESOLVED),
        "lambda_ig": 0.3,
        "lambda_task": 1.0,
        "lambda_outcome": 1.0,
        "lambda_format": 1.0,
        "search_formula": "0.3 * A_IG + 1.0 * A_task",
        "answer_formula": "1.0 * z_O + 1.0 * A_format",
        "production_call_chain": _production_chain(),
        "source_hashes": {
            relative: _sha256_file(PROJECT_ROOT / relative)
            for relative in MODIFIED_FILES
            if (PROJECT_ROOT / relative).is_file()
        },
    }
    RUNTIME_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "result": result,
        "checks": checks,
        "resolved_config": {
            "before_source": str(before_path),
            "before_source_sha256": _sha256_file(before_path),
            "before_snapshot": str(BEFORE_RESOLVED),
            "before_snapshot_sha256": _sha256_file(BEFORE_RESOLVED),
            "before_coefficients": _coefficient_view(before),
            "after_snapshot": str(AFTER_RESOLVED),
            "after_snapshot_sha256": _sha256_file(AFTER_RESOLVED),
            "after_coefficients": _coefficient_view(after),
        },
        "production_call_chain": manifest["production_call_chain"],
        "modified_line_evidence": _modified_line_evidence(),
        "runtime_manifest": str(RUNTIME_MANIFEST),
        "runtime_manifest_sha256": _sha256_file(RUNTIME_MANIFEST),
        "modified_files": [
            {
                "path": relative,
                "sha256": _sha256_file(PROJECT_ROOT / relative),
            }
            for relative in MODIFIED_FILES
            if (PROJECT_ROOT / relative).is_file()
        ],
        "tests": tests,
        "full_suite": full_suite,
        "full_repo_search": {
            "path": str(SEARCH_RESULTS),
            "match_count": len(matches),
            "production_violation_count": len(production_violations),
            "matches": matches,
        },
        "training_side_effects": {
            "rollout_started": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "formal_training_started": False,
        },
    }
    REPORT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    REPORT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if result != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
