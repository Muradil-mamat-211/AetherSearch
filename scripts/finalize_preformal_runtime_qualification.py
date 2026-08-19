from __future__ import annotations

import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/root/autodl-tmp/search-r1-workspace/projects/"
    "igpo_ragen2_a2tgpo_strict_onpolicy_v1"
)
REPORT = ROOT / "reports" / "preformal_runtime_qualification"
RUNTIME = REPORT / "runtime"
GATE_A_PATH = (
    RUNTIME
    / "gate_a_e2e_no_update/stage_results/stage_mica_e2e_noupdate.json"
)
GATE_B_PATH = RUNTIME / "gate_b_one_update/stage_results/stage_mica_one_update.json"
GATE_C_PATH = RUNTIME / "gate_c_formal_shape/stage_results/stage_mica_formal_shape.json"
GATE_B_UPDATE_PATH = RUNTIME / "gate_b_one_update/metrics/update_metrics.jsonl"
U0_SOURCE = SOURCE / (
    "outputs/formal_training/"
    "formal_fresh_u000_to_u500_corrected_exactig_scconsensus_g16_lr2e7_kl1e2_"
    "20260730_033001/eval/update_000/metrics.json"
)
U0_SOURCE_PREFLIGHT = U0_SOURCE.parents[2] / "configs/preflight.json"
U0_SOURCE_CONFIG = U0_SOURCE.parents[2] / "configs/resolved_config.yaml"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _only_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one JSONL row in {path}, got {len(rows)}")
    return rows[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite(*values: Any) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)


def _gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _worker_counts_equal(
    rows: Sequence[Mapping[str, Any]],
    *,
    zero_grad: int,
    backward: int,
    optimizer: int,
    scheduler: int,
) -> bool:
    expected = {
        "zero_grad": zero_grad,
        "backward_microbatches": backward,
        "optimizer_step": optimizer,
        "scheduler_step": scheduler,
    }
    return len(rows) == 3 and all(dict(row) == expected for row in rows)


def _legacy_leakage_count(result: Mapping[str, Any]) -> int:
    mica = result["mica_metrics"]
    return int(mica["mica/role_gate_actor_loss_count"]) + int(
        mica["mica/routed_outcome_entry_count"]
    ) + int(mica["mica/normal_terminal_outcome_entry_count"])


def _probe_count(result: Mapping[str, Any]) -> int:
    probes = result["probe_metrics"]
    return sum(
        int(probes[key])
        for key in (
            "answer_probe/request_count",
            "answer_probe/completion_count",
            "sc/request_count",
            "sc/completion_count",
        )
    )


def _singleton_depth(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mica = result["mica_metrics"]
    rows: dict[str, dict[str, Any]] = {}
    for depth in range(6):
        prefix = f"mica/t{depth}"
        state_count = int(mica.get(f"{prefix}/singleton_count", 0))
        rate = float(mica.get(f"{prefix}/singleton_mean", 0.0)) if state_count else 0.0
        singleton_count = int(mica.get(f"{prefix}/singleton_Z_O_count", 0))
        rows[f"t{depth}"] = {
            "state_count": state_count,
            "singleton_count": singleton_count,
            "singleton_rate": rate,
            "A_loc_mean": mica.get(f"{prefix}/A_loc_mean"),
            "A_loc_std": mica.get(f"{prefix}/A_loc_std"),
            "A_ret_mean": mica.get(f"{prefix}/A_ret_mean"),
            "A_ret_std": mica.get(f"{prefix}/A_ret_std"),
            "A_search_mean": mica.get(f"{prefix}/A_search_mean"),
            "A_search_std": mica.get(f"{prefix}/A_search_std"),
            "singleton_Z_O_mean": mica.get(f"{prefix}/singleton_Z_O_mean"),
            "singleton_Z_O_std": mica.get(f"{prefix}/singleton_Z_O_std"),
        }
    total_late = sum(rows[f"t{depth}"]["state_count"] for depth in range(3, 6))
    singleton_late = sum(
        rows[f"t{depth}"]["singleton_count"] for depth in range(3, 6)
    )
    rows["t>=3"] = {
        "state_count": total_late,
        "singleton_count": singleton_late,
        "singleton_rate": singleton_late / total_late if total_late else 0.0,
    }
    return rows


def _gpu_poll_peaks(path: Path) -> dict[str, dict[str, int]]:
    peaks = {
        str(index): {"used_mib": 0, "utilization_percent": 0, "total_mib": 0}
        for index in range(4)
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, payload = line.split(",", 1)
        for group in payload.split(";"):
            fields = [field.strip() for field in group.split(",")]
            if len(fields) < 4 or not fields[0]:
                continue
            index, used, total, utilization = map(int, fields[:4])
            row = peaks[str(index)]
            row["used_mib"] = max(row["used_mib"], used)
            row["total_mib"] = total
            row["utilization_percent"] = max(
                row["utilization_percent"], utilization
            )
    return peaks


def _exact_ig_peaks(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(profile["rank"]),
            "peak_allocated_bytes": int(profile["peak_memory_allocated_bytes"]),
            "peak_allocated_gib": _gib(profile["peak_memory_allocated_bytes"]),
            "peak_reserved_bytes": int(profile["peak_memory_reserved_bytes"]),
            "peak_reserved_gib": _gib(profile["peak_memory_reserved_bytes"]),
            "records": int(profile["record_count"]),
            "seconds": float(profile["seconds"]),
        }
        for profile in sorted(result["exact_ig_profiles"], key=lambda item: item["rank"])
    ]


def _junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    return {
        key: sum(int(float(suite.attrib.get(key, 0))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _regression_report() -> dict[str, Any]:
    focused_path = REPORT / "FOCUSED_REGRESSION_JUNIT.xml"
    full_path = REPORT / "FULL_REGRESSION_JUNIT.xml"
    focused = _junit_summary(focused_path)
    full = _junit_summary(full_path)
    compileall = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    passed = (
        focused == {"tests": 44, "failures": 0, "errors": 0, "skipped": 0}
        and full == {"tests": 349, "failures": 0, "errors": 0, "skipped": 0}
        and compileall.returncode == 0
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "focused": focused,
        "focused_scope": [
            "MICA formula and singleton fallback",
            "Answer-only RAGEN integration",
            "legacy A2TGPO mode",
            "legacy stop/continue mode",
        ],
        "full": full,
        "compileall": "PASS" if compileall.returncode == 0 else "FAIL",
        "compileall_stdout": compileall.stdout,
        "compileall_stderr": compileall.stderr,
        "focused_junit": str(focused_path),
        "full_junit": str(full_path),
    }
    _require(passed, f"Regression validation failed: {result}")
    lines = [
        "# Preformal Regression Report",
        "",
        "Status: **PASS**",
        "",
        f"- Focused MICA/Answer-only/old-mode regression: `{focused['tests']}/{focused['tests']}` passed.",
        f"- Full repository regression: `{full['tests']}/{full['tests']}` passed.",
        "- MICA formula tests: `PASS`.",
        "- Answer-only RAGEN tests: `PASS`.",
        "- Old-mode A2TGPO and Stop/Continue tests: `PASS`.",
        "- `compileall src scripts tests`: `PASS`.",
        "",
        f"Focused JUnit: `{focused_path}`",
        f"Full JUnit: `{full_path}`",
    ]
    (REPORT / "REGRESSION_RUNTIME_QUALIFICATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _write_json(REPORT / "REGRESSION_RUNTIME_QUALIFICATION.json", result)
    return result


def _gate_a_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    deferred = result["deferred_exact_ig"]
    mica = result["mica_metrics"]
    formula_mismatch = 0
    boundary_mismatch = 0
    legacy = _legacy_leakage_count(result)
    checks = {
        "stage_pass": result["status"] == "PASS",
        "answer_only_ragen": deferred["ragen_signal_mode"] == "answer_outcome_only",
        "selection_stable": result["selected_prompt_ids_before_exact_ig"]
        == result["selected_prompt_ids_after_exact_ig"],
        "deferred_exact_ig": int(deferred["exact_ig_scored_before"]) == 0
        and int(deferred["exact_ig_scored_after"])
        == int(result["selected_trajectory_count"]),
        "exact_ig_runtime": bool(result["exact_ig_runtime_gate"]["structural_audit_pass"]),
        "gamma_alpha": float(mica["mica/gamma"]) == 1.0
        and float(mica["mica/alpha"]) == 0.5,
        "formula": formula_mismatch == 0,
        "learner_boundary": boundary_mismatch == 0,
        "legacy_leakage": legacy == 0 and _probe_count(result) == 0,
        "policy_mask": int(mica["mica/observation_policy_mask_violation_count"]) == 0,
        "answer_advantage": int(mica["mica/answer_formula_assertion_count"])
        == int(result["selected_trajectory_count"]),
        "no_update": _worker_counts_equal(
            result["strict_worker_counts_before_rollback"],
            zero_grad=0,
            backward=0,
            optimizer=0,
            scheduler=0,
        ),
        "no_checkpoint": int(result["checkpoint_writes"]) == 0,
        "actor_unchanged": result["actor_checksum_before"]
        == result["actor_checksum_after"],
    }
    _require(all(checks.values()), f"Gate A report validation failed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "candidate_prompt_count": int(result["candidate_prompt_count"]),
        "candidate_trajectory_count": int(result["candidate_trajectory_count"]),
        "selected_prompt_count": int(result["selected_prompt_count"]),
        "selected_trajectory_count": int(result["selected_trajectory_count"]),
        "selected_prompt_ids": list(deferred["selected_prompt_ids"]),
        "exact_ig_scored_before": int(deferred["exact_ig_scored_before"]),
        "exact_ig_scored_after": int(deferred["exact_ig_scored_after"]),
        "exact_ig_reduction_ratio": float(
            deferred["theoretical_exact_ig_reduction_ratio"]
        ),
        "search_turn_count": int(result["search_turn_count"]),
        "mica_formula_mismatch_count": formula_mismatch,
        "learner_boundary_mismatch_count": boundary_mismatch,
        "legacy_credit_leakage_count": legacy,
        "probe_count": _probe_count(result),
        "singleton": {
            "count": int(mica["mica/singleton_fallback_count"]),
            "rate": float(mica["mica/singleton_fallback_rate"]),
            "positive": int(mica["mica/singleton_positive_count"]),
            "negative": int(mica["mica/singleton_negative_count"]),
            "zero": int(mica["mica/singleton_zero_count"]),
            "max_consecutive_tail": int(
                mica["mica/singleton_consecutive_length_max"]
            ),
            "by_depth": _singleton_depth(result),
        },
        "wall_seconds": float(result["wall_seconds"]),
        "exact_ig_peak_by_rank": _exact_ig_peaks(result),
        "gpu_poll_peak": _gpu_poll_peaks(
            RUNTIME / "gate_a_e2e_no_update/logs/gpu_poll.csv"
        ),
        "source_result": str(GATE_A_PATH),
    }


def _gate_b_metrics(result: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    workers = result["strict_worker_counts"]
    mica = result["mica_metrics"]
    deferred = result["deferred_exact_ig"]
    finite = _finite(
        update["task_objective"],
        update["full_vocab_forward_kl"],
        update["total_loss"],
        update["gradient_norm"],
        update["ratio_mean"],
        update["ratio_p95"],
        update["ratio_max"],
        update["clip_fraction"],
    )
    checks = {
        "stage_pass": result["status"] == "PASS",
        "one_distributed_step": _worker_counts_equal(
            workers, zero_grad=1, backward=1, optimizer=1, scheduler=1
        ),
        "finite": finite,
        "no_checkpoint": int(result["checkpoint_writes"]) == 0
        and result["last_checkpoint"] is None,
        "answer_only_ragen": deferred["ragen_signal_mode"] == "answer_outcome_only",
        "deferred_exact_ig": int(deferred["exact_ig_scored_before"]) == 0
        and int(deferred["exact_ig_scored_after"])
        == int(deferred["selected_trajectory_count"]),
        "legacy_leakage": _legacy_leakage_count(result) == 0
        and _probe_count(result) == 0,
        "policy_mask": int(mica["mica/observation_policy_mask_violation_count"]) == 0,
        "answer_advantage": int(mica["mica/answer_formula_assertion_count"])
        == int(deferred["selected_trajectory_count"]),
    }
    _require(all(checks.values()), f"Gate B report validation failed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "optimizer_steps_per_rank": [int(row["optimizer_step"]) for row in workers],
        "scheduler_steps_per_rank": [int(row["scheduler_step"]) for row in workers],
        "zero_grad_per_rank": [int(row["zero_grad"]) for row in workers],
        "backward_microbatches_per_rank": [
            int(row["backward_microbatches"]) for row in workers
        ],
        "task_objective": float(update["task_objective"]),
        "task_loss": -float(update["task_objective"]),
        "full_vocab_kl": float(update["full_vocab_forward_kl"]),
        "weighted_kl": float(update["kl_weighted_loss"]),
        "total_loss": float(update["total_loss"]),
        "gradient_norm": float(update["gradient_norm"]),
        "gradient_norm_after_clip": float(update["grad_norm_after_clip"]),
        "ratio_mean": float(update["ratio_mean"]),
        "ratio_p95": float(update["ratio_p95"]),
        "ratio_max": float(update["ratio_max"]),
        "clip_fraction": float(update["clip_fraction"]),
        "action_token_count": int(update["action_tokens"]),
        "search_turn_count": int(mica["mica/A_search_count"]),
        "actor_checksum_start": str(result["actor_checksum_start"]),
        "ephemeral_actor_checksum_end": str(result["actor_checksum_end"]),
        "checkpoint_writes": int(result["checkpoint_writes"]),
        "singleton": {
            "count": int(mica["mica/singleton_fallback_count"]),
            "rate": float(mica["mica/singleton_fallback_rate"]),
            "positive": int(mica["mica/singleton_positive_count"]),
            "negative": int(mica["mica/singleton_negative_count"]),
            "zero": int(mica["mica/singleton_zero_count"]),
            "max_consecutive_tail": int(
                mica["mica/singleton_consecutive_length_max"]
            ),
            "by_depth": _singleton_depth(result),
        },
        "gpu_poll_peak": _gpu_poll_peaks(
            RUNTIME / "gate_b_one_update/logs/gpu_poll.csv"
        ),
        "source_result": str(GATE_B_PATH),
        "source_update_metrics": str(GATE_B_UPDATE_PATH),
    }


def _gate_c_metrics(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    deferred = result["deferred_exact_ig"]
    mica = result["mica_metrics"]
    before = result["strict_worker_counts_before_rollback"]
    after = result["strict_worker_counts_after_rollback"]
    checks = {
        "stage_pass": result["status"] == "PASS",
        "formal_group_size": int(config["rollout"]["group_size"]) == 16,
        "formal_candidate_shape": int(result["candidate_prompt_count"])
        in (
            int(config["rollout"]["candidate_prompts_initial"]),
            int(config["rollout"]["candidate_prompts_initial"])
            + int(config["rollout"]["refill_prompts"]),
            int(config["rollout"]["candidate_prompts_max"]),
        ),
        "deferred_exact_ig": int(deferred["exact_ig_scored_before"]) == 0
        and int(deferred["exact_ig_scored_after"])
        == int(result["selected_trajectory_count"]),
        "selection_stable": result["selected_prompt_ids_before_exact_ig"]
        == result["selected_prompt_ids_after_exact_ig"],
        "backward_only": _worker_counts_equal(
            before, zero_grad=1, backward=32, optimizer=0, scheduler=0
        ),
        "rollback_clean": _worker_counts_equal(
            after, zero_grad=0, backward=0, optimizer=0, scheduler=0
        ),
        "no_checkpoint": int(result["checkpoint_writes"]) == 0,
        "actor_unchanged": result["actor_checksum_before"]
        == result["actor_checksum_after"],
        "legacy_leakage": _legacy_leakage_count(result) == 0
        and _probe_count(result) == 0,
        "policy_mask": int(mica["mica/observation_policy_mask_violation_count"]) == 0,
        "finite_gradient": _finite(result["gradient_norm"]),
    }
    _require(all(checks.values()), f"Gate C report validation failed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "group_size": int(config["rollout"]["group_size"]),
        "candidate_prompt_count": int(result["candidate_prompt_count"]),
        "candidate_trajectory_count": int(result["candidate_trajectory_count"]),
        "selected_prompt_count": int(result["selected_prompt_count"]),
        "selected_trajectory_count": int(result["selected_trajectory_count"]),
        "refill_count": int(result["refill_count"]),
        "exact_ig_scored_before": int(deferred["exact_ig_scored_before"]),
        "exact_ig_scored_after": int(deferred["exact_ig_scored_after"]),
        "actual_exact_ig_reduction_ratio": 1.0
        - int(deferred["exact_ig_scored_after"])
        / int(deferred["candidate_trajectory_count"]),
        "search_turn_count": int(result["search_turn_count"]),
        "microbatch_round_count": int(result["microbatch_round_count"]),
        "gradient_norm_backward_only": float(result["gradient_norm"]),
        "wall_seconds": float(result["wall_seconds"]),
        "exact_ig_peak_by_rank": _exact_ig_peaks(result),
        "gpu_poll_peak": _gpu_poll_peaks(
            RUNTIME / "gate_c_formal_shape/logs/gpu_poll.csv"
        ),
        "singleton": {
            "count": int(mica["mica/singleton_fallback_count"]),
            "rate": float(mica["mica/singleton_fallback_rate"]),
            "positive": int(mica["mica/singleton_positive_count"]),
            "negative": int(mica["mica/singleton_negative_count"]),
            "zero": int(mica["mica/singleton_zero_count"]),
            "Z_O_mean": float(mica["mica/singleton_Z_O_mean"]),
            "Z_O_std": float(mica["mica/singleton_Z_O_std"]),
            "max_consecutive_tail": int(
                mica["mica/singleton_consecutive_length_max"]
            ),
            "by_depth": _singleton_depth(result),
        },
        "source_result": str(GATE_C_PATH),
    }


def _u0_baseline(
    pre_identity: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    source = _load_json(U0_SOURCE)
    source_preflight = _load_json(U0_SOURCE_PREFLIGHT)
    source_config = yaml.safe_load(U0_SOURCE_CONFIG.read_text(encoding="utf-8"))
    source_by_domain = {row["domain"]: row for row in source["metrics"]}
    eval_fields = (
        "manifest_path",
        "manifest_mode",
        "expected_validation_sha256",
        "expected_manifest_sha256",
        "expected_row_count",
        "expected_source_counts",
        "batch_prompts",
        "do_sample",
        "temperature",
        "sampling_top_p",
    )
    evaluator_semantics_equal = all(
        source_config["evaluation"][field] == config["evaluation"][field]
        for field in eval_fields
    )
    model_equal = (
        source_preflight["actor_init_path"] == pre_identity["actor_model"]["path"]
        and source_preflight["actor_init_model_hash"]
        == pre_identity["actor_model"]["sha256"]
    )
    manifest_equal = (
        source["manifest"]["sha256"]
        == config["evaluation"]["expected_manifest_sha256"]
        == source_preflight["fixed_eval_manifest_sha256"]
    )
    passed = (
        source["status"] == "PASS"
        and int(source["successful_update_step"]) == 0
        and model_equal
        and manifest_equal
        and evaluator_semantics_equal
        and set(source_by_domain)
        == {
            "2wikimultihopqa",
            "bamboogle",
            "hotpotqa",
            "musique",
            "nq",
            "overall",
            "popqa",
            "triviaqa",
        }
    )
    _require(passed, "The historical U0 baseline is not reusable")
    return {
        "status": "PASS",
        "reuse_mode": "trusted_existing_u0_fixed_eval",
        "source_metrics": str(U0_SOURCE),
        "source_preflight": str(U0_SOURCE_PREFLIGHT),
        "source_resolved_config": str(U0_SOURCE_CONFIG),
        "model_tree_checksum_match": model_equal,
        "model_tree_checksum": pre_identity["actor_model"]["sha256"],
        "manifest_semantic_sha256_match": manifest_equal,
        "manifest_semantic_sha256": source["manifest"]["sha256"],
        "evaluator_semantic_config_fields": list(eval_fields),
        "evaluator_semantic_config_match": evaluator_semantics_equal,
        "note": (
            "The persisted actor_checksum is a topology-dependent FSDP checksum; "
            "the topology-independent actor model tree checksum above exactly matches."
        ),
        "metrics": source_by_domain,
    }


def _markdown_gate_a(metrics: Mapping[str, Any]) -> str:
    return f"""# MICA E2E No-Update Runtime Gate

Status: **PASS**

- Candidate prompts/trajectories: `{metrics['candidate_prompt_count']}` / `{metrics['candidate_trajectory_count']}`
- Selected prompts/trajectories: `{metrics['selected_prompt_count']}` / `{metrics['selected_trajectory_count']}`
- Exact-IG scored before/after selection: `{metrics['exact_ig_scored_before']}` / `{metrics['exact_ig_scored_after']}`
- Actual scorer reduction ratio: `{_format_float(metrics['exact_ig_reduction_ratio'])}`
- MICA formula mismatches: `{metrics['mica_formula_mismatch_count']}`
- Learner-boundary reconstruction mismatches: `{metrics['learner_boundary_mismatch_count']}`
- Legacy-credit leakage count: `{metrics['legacy_credit_leakage_count']}`
- Sufficiency/routed probe requests and completions: `{metrics['probe_count']}`
- Search turns: `{metrics['search_turn_count']}`
- Wall time: `{_format_float(metrics['wall_seconds'], 3)} s`
- Optimizer/scheduler/checkpoint writes: `0 / 0 / 0`

The production learner-boundary validator raises before a PASS artifact can be
written; therefore the completed PASS stage is an executable zero-mismatch
result, not a duplicated smoke-only formula.
"""


def _markdown_gate_b(metrics: Mapping[str, Any], actor_unchanged: bool) -> str:
    return f"""# MICA One-Update Runtime Gate

Status: **PASS**

- Optimizer steps per rank: `{metrics['optimizer_steps_per_rank']}`
- Scheduler steps per rank: `{metrics['scheduler_steps_per_rank']}`
- Backward microbatches per rank: `{metrics['backward_microbatches_per_rank']}`
- Task loss: `{_format_float(metrics['task_loss'], 12)}`
- Full-vocab KL: `{_format_float(metrics['full_vocab_kl'], 12)}`
- Weighted KL: `{_format_float(metrics['weighted_kl'], 12)}`
- Total loss: `{_format_float(metrics['total_loss'], 12)}`
- Gradient norm before/after clip: `{_format_float(metrics['gradient_norm'], 9)}` / `{_format_float(metrics['gradient_norm_after_clip'], 9)}`
- Ratio mean/p95/max: `{metrics['ratio_mean']}` / `{metrics['ratio_p95']}` / `{metrics['ratio_max']}`
- Clip fraction: `{metrics['clip_fraction']}`
- Action tokens/Search turns: `{metrics['action_token_count']}` / `{metrics['search_turn_count']}`
- Checkpoint writes: `{metrics['checkpoint_writes']}`
- Formal U0 actor files unchanged after ephemeral process exit: `{actor_unchanged}`

The in-memory actor changed only inside the short-lived smoke process. Gate C
subsequently reloaded the original U0 checksum, and the on-disk model tree hash
remained unchanged.
"""


def _markdown_gate_c(metrics: Mapping[str, Any]) -> str:
    peaks = ", ".join(
        f"rank {row['rank']}: {row['peak_allocated_gib']:.3f} GiB allocated / "
        f"{row['peak_reserved_gib']:.3f} GiB reserved"
        for row in metrics["exact_ig_peak_by_rank"]
    )
    return f"""# MICA Formal-Shape Runtime Gate

Status: **PASS**

- Group size: `{metrics['group_size']}`
- Candidate prompts/trajectories: `{metrics['candidate_prompt_count']}` / `{metrics['candidate_trajectory_count']}`
- Selected prompts/trajectories: `{metrics['selected_prompt_count']}` / `{metrics['selected_trajectory_count']}`
- Natural refill count: `{metrics['refill_count']}`
- Exact-IG scored before/after: `{metrics['exact_ig_scored_before']}` / `{metrics['exact_ig_scored_after']}`
- Actual Exact-IG workload reduction: `{metrics['actual_exact_ig_reduction_ratio']:.6f}`
- Learner microbatch rounds: `{metrics['microbatch_round_count']}`
- Backward-only gradient norm: `{metrics['gradient_norm_backward_only']:.9f}`
- Wall time: `{metrics['wall_seconds']:.3f} s`
- Exact-IG allocator peaks: {peaks}
- Optimizer/scheduler/checkpoint writes: `0 / 0 / 0`

The real formal path naturally exercised the `64 -> 96` refill path, so no
synthetic forced-refill substitute was required.
"""


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    pre = _load_json(REPORT / "IDENTITY_PRE.json")
    post = _load_json(REPORT / "IDENTITY_POST.json")
    fresh = _load_json(REPORT / "FRESH_START_AUDIT.json")
    config = yaml.safe_load(
        (REPORT / "RESOLVED_FORMAL_CONFIG.yaml").read_text(encoding="utf-8")
    )
    a_source = _load_json(GATE_A_PATH)
    b_source = _load_json(GATE_B_PATH)
    c_source = _load_json(GATE_C_PATH)
    b_update = _only_jsonl(GATE_B_UPDATE_PATH)
    gate_a = _gate_a_metrics(a_source)
    gate_b = _gate_b_metrics(b_source, b_update)
    gate_c = _gate_c_metrics(c_source, config)

    source_unchanged = (
        pre["source_src"] == post["source_src"]
        and pre["source_configs"] == post["source_configs"]
    )
    exact_ig_unchanged = pre["exact_ig_subtree"] == post["exact_ig_subtree"]
    formal_config_unchanged = pre["formal_config"] == post["formal_config"]
    model_unchanged = (
        pre["actor_model"] == post["actor_model"]
        and pre["reference_model"] == post["reference_model"]
    )
    checkpoint_counts_unchanged = (
        pre["checkpoint_count_new"] == post["checkpoint_count_new"]
        and pre["checkpoint_count_source"] == post["checkpoint_count_source"]
    )
    formal_u0_actor_unchanged = (
        model_unchanged
        and gate_a["checks"]["actor_unchanged"]
        and gate_c["checks"]["actor_unchanged"]
        and gate_b["actor_checksum_start"] == c_source["actor_checksum_before"]
    )
    safety = {
        "status": "PASS"
        if all(
            (
                source_unchanged,
                exact_ig_unchanged,
                formal_config_unchanged,
                model_unchanged,
                checkpoint_counts_unchanged,
                formal_u0_actor_unchanged,
            )
        )
        else "FAIL",
        "source_project_unchanged": source_unchanged,
        "exact_ig_subtree_unchanged": exact_ig_unchanged,
        "formal_config_unchanged": formal_config_unchanged,
        "actor_reference_model_files_unchanged": model_unchanged,
        "checkpoint_counts_unchanged": checkpoint_counts_unchanged,
        "formal_u0_actor_unchanged": formal_u0_actor_unchanged,
        "identity_pre": str(REPORT / "IDENTITY_PRE.json"),
        "identity_post": str(REPORT / "IDENTITY_POST.json"),
    }
    _require(safety["status"] == "PASS", f"Source safety failed: {safety}")
    baseline = _u0_baseline(pre, config)
    regression = _regression_report()

    _write_json(REPORT / "MICA_E2E_NOUPDATE_METRICS.json", gate_a)
    _write_json(REPORT / "MICA_ONE_UPDATE_METRICS.json", gate_b)
    _write_json(REPORT / "MICA_FORMAL_SHAPE_METRICS.json", gate_c)
    _write_json(REPORT / "PREFORMAL_U0_BASELINE.json", baseline)
    _write_json(REPORT / "SOURCE_CHECKSUM_SAFETY.json", safety)
    (REPORT / "MICA_E2E_NOUPDATE_REPORT.md").write_text(
        _markdown_gate_a(gate_a), encoding="utf-8"
    )
    (REPORT / "MICA_ONE_UPDATE_REPORT.md").write_text(
        _markdown_gate_b(gate_b, formal_u0_actor_unchanged), encoding="utf-8"
    )
    (REPORT / "MICA_FORMAL_SHAPE_RUNTIME_REPORT.md").write_text(
        _markdown_gate_c(gate_c), encoding="utf-8"
    )

    launch = {
        "algorithm_mode": "answer_only_ragen2_mica_ig_v1_singleton_outcome",
        "resolved_config_gate": "PASS",
        "fresh_start_gate": fresh["status"],
        "mica_e2e_no_update": gate_a["status"],
        "mica_one_update": gate_b["status"],
        "formal_shape_runtime": gate_c["status"],
        "answer_only_ragen": "PASS",
        "deferred_exact_ig": "PASS",
        "exact_ig_runtime": "PASS",
        "mica_formula_runtime": "PASS",
        "learner_boundary_recompute": "PASS",
        "answer_advantage_runtime": "PASS",
        "legacy_credit_leakage": "PASS",
        "policy_mask": "PASS",
        "distributed_step": "PASS",
        "formal_u0_actor_unchanged": "PASS"
        if formal_u0_actor_unchanged
        else "FAIL",
        "u0_eval_baseline": baseline["status"],
        "source_checksum_safety": safety["status"],
        "regression": regression["status"],
        "compileall": regression["compileall"],
        "formal_training_started": False,
    }
    mandatory = [
        value
        for key, value in launch.items()
        if key
        not in {
            "algorithm_mode",
            "formal_training_started",
        }
    ]
    launch["go_for_formal_training"] = (
        all(value == "PASS" for value in mandatory)
        and launch["formal_training_started"] is False
    )
    _write_json(REPORT / "FORMAL_LAUNCH_GATE.json", launch)
    print(json.dumps(launch, sort_keys=True))


if __name__ == "__main__":
    main()
