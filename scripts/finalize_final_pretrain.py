#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(
    "/root/autodl-tmp/search-r1-workspace/projects/"
    "igpo_ragen2_a2tgpo_strict_onpolicy_v1"
)
METRIC_FILES = {
    "attempt": "attempt_metrics.jsonl",
    "update": "update_metrics.jsonl",
    "channel": "channel_metrics.jsonl",
    "prompt": "prompt_metrics.jsonl",
    "trajectory": "trajectory_metrics.jsonl",
    "turn": "turn_metrics.jsonl",
    "behavior": "behavior_metrics.jsonl",
    "system": "system_metrics.jsonl",
    "eval": "eval_metrics.jsonl",
    "checkpoint": "checkpoint_metrics.jsonl",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Required JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_finite(item) for item in value)
    return True


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else 0.0


def _bool(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _write(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _channel_audit(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    findings: list[str] = []
    result: dict[str, Any] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["channel"])].append(row)
    for channel in ("IG", "Outcome"):
        channel_rows = sorted(
            grouped[channel],
            key=lambda row: (
                int(row["attempt_id"]),
                int(row["successful_update_after"]),
            ),
        )
        previous_valid_count = 0
        bootstrap_ok = True
        health_ok = True
        update10_bootstrap = False
        update10_reference = False
        update11_health = False
        transition_attempt: int | None = None
        for row in channel_rows:
            before = int(row["successful_update_before"])
            after = int(row["successful_update_after"])
            committed = after == before + 1
            mode = str(row["stage_mode"])
            expected = "health" if previous_valid_count >= 10 else "bootstrap"
            if mode != expected:
                findings.append(
                    f"{channel}: attempt {row['attempt_id']} mode={mode}, "
                    f"expected={expected} from {previous_valid_count} committed "
                    "valid observations"
                )
                if expected == "bootstrap":
                    bootstrap_ok = False
                else:
                    health_ok = False
            if committed and after <= 10 and mode != "bootstrap":
                bootstrap_ok = False
            if committed and after == 10:
                update10_bootstrap = mode == "bootstrap"
                update10_reference = (
                    int(row["valid_health_observation_count"]) >= 10
                    and row["B_ref"] is not None
                )
            if committed and after == 11 and previous_valid_count >= 10:
                update11_health = mode == "health"
            if mode == "health" and transition_attempt is None:
                transition_attempt = int(row["attempt_id"])
            if committed:
                new_count = int(row["valid_health_observation_count"])
                if new_count < previous_valid_count:
                    findings.append(
                        f"{channel}: valid health count regressed "
                        f"{previous_valid_count}->{new_count}"
                    )
                    health_ok = False
                previous_valid_count = new_count
                if before < 10 and row["m"] is not None:
                    if row["b_after"] is None:
                        findings.append(
                            f"{channel}: bootstrap update {after} had m but no scale"
                        )
                        bootstrap_ok = False
                    if (
                        row["activation"] is False
                        and row["EMA_update_allowed"]
                        and after >= 2
                        and not row["EMA_updated"]
                    ):
                        findings.append(
                            f"{channel}: bootstrap activation disabled a valid EMA "
                            f"at update {after}"
                        )
                        bootstrap_ok = False
                if (
                    mode == "health"
                    and not row["activation"]
                    and row["EMA_updated"]
                ):
                    findings.append(
                        f"{channel}: low-health gate did not freeze EMA at update {after}"
                    )
                    health_ok = False
        result[channel] = {
            "record_count": len(channel_rows),
            "final_valid_health_observation_count": previous_valid_count,
            "bootstrap_sequence_pass": bootstrap_ok,
            "health_sequence_pass": health_ok,
            "update10_selection_bootstrap": update10_bootstrap,
            "update10_reference_created": update10_reference,
            "update11_selection_health": update11_health,
            "first_health_gate_attempt": transition_attempt,
        }
    result["independent_transition_pass"] = all(
        result[channel]["bootstrap_sequence_pass"]
        and result[channel]["health_sequence_pass"]
        for channel in ("IG", "Outcome")
    )
    return result, findings


def _eval_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_step: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_step[int(row["successful_update_step"])][str(row["domain"])] = row
    result: dict[str, Any] = {}
    for step in (0, 10, 20):
        result[str(step)] = {
            domain: dict(values)
            for domain, values in sorted(by_step.get(step, {}).items())
        }
    return result


def _behavior_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["successful_update_step"]))
    first = ordered[: min(5, len(ordered))]
    last = ordered[-min(5, len(ordered)) :]
    first_multi = _mean(row["multi_search_rate"] for row in first)
    last_multi = _mean(row["multi_search_rate"] for row in last)
    first_format = _mean(row["format_rate"] for row in first)
    last_format = _mean(row["format_rate"] for row in last)
    searches = [float(row["avg_search_count"]) for row in ordered]
    multi_collapse = first_multi > 0.05 and last_multi <= 1.0e-12
    format_regression = first_format - last_format > 0.25
    search_explosion = bool(searches) and max(searches) > 5.0 + 1.0e-12
    return {
        "first5_multi_search_rate": first_multi,
        "last5_multi_search_rate": last_multi,
        "multi_search_fast_collapse": multi_collapse,
        "first5_format_rate": first_format,
        "last5_format_rate": last_format,
        "obvious_format_regression": format_regression,
        "maximum_avg_search_count": max(searches, default=0.0),
        "search_count_explosion": search_explosion,
        "pass": not (multi_collapse or format_regression or search_explosion),
    }


def _system_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spills = [float(row.get("ray_object_store_spill_bytes") or 0.0) for row in rows]
    by_gpu: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        for gpu in row.get("gpus", ()):
            by_gpu[int(gpu["physical_gpu"])].append(int(gpu["memory_used_mib"]))
    growth: dict[str, int] = {}
    for gpu, values in by_gpu.items():
        growth[str(gpu)] = values[-1] - values[0] if len(values) >= 2 else 0
    sustained_growth = any(value > 4096 for value in growth.values())
    return {
        "ray_spill_max_bytes": max(spills, default=0.0),
        "gpu_boundary_memory_growth_mib": growth,
        "sustained_gpu_memory_growth": sustained_growth,
        "pass": max(spills, default=0.0) == 0.0 and not sustained_growth,
    }


def _report_paths() -> dict[str, Path]:
    names = (
        "FINAL_PRETRAIN_TEST_REPORT.md",
        "PILOT_20_UPDATE_REPORT.md",
        "UPDATE_10_11_HEALTH_GATE_REPORT.md",
        "FORCED_REFILL_96_TEST_REPORT.md",
        "METRICS_COMPLETENESS_REPORT.md",
        "PILOT_CHECKPOINT_RECOVERY_REPORT.md",
        "PILOT_EVAL_REPORT.md",
        "FORMAL_TRAINING_READINESS_REPORT.md",
        "TRAIN_LAUNCH_GUIDE.md",
        "TEST_RESULTS_FINAL_PRETRAIN.md",
        "CODE_CHANGELOG_FINAL_PRETRAIN.md",
    )
    return {name: PROJECT_ROOT / name for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run-dir", required=True)
    parser.add_argument("--forced-test-dir", required=True)
    parser.add_argument("--formal-dry-run-pass", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.pilot_run_dir).resolve()
    forced_dir = Path(args.forced_test_dir).resolve()
    pilot = _read_json(run_dir / "stage_results" / "stage_pilot20.json")
    forced = _read_json(
        forced_dir / "stage_results" / "stage_forced_refill96.json"
    )
    metrics = {
        scope: _read_jsonl(run_dir / "metrics" / filename)
        for scope, filename in METRIC_FILES.items()
    }
    channel_audit, channel_findings = _channel_audit(metrics["channel"])
    eval_summary = _eval_summary(metrics["eval"])
    behavior = _behavior_audit(metrics["behavior"])
    system = _system_audit(metrics["system"])

    updates = metrics["update"]
    attempts = metrics["attempt"]
    checkpoints = metrics["checkpoint"]
    successful_steps = [int(row["successful_update_step"]) for row in updates]
    optimizer_steps = sum(int(row["optimizer_steps_this_update"]) for row in updates)
    scheduler_steps = sum(int(row["scheduler_steps_this_update"]) for row in updates)
    finite = all(_all_finite(row) for row in updates)
    gradients_finite = all(
        row.get("gradient_norm") is not None
        and math.isfinite(float(row["gradient_norm"]))
        for row in updates
    )
    kl_values = [float(row["full_vocab_forward_kl"]) for row in updates]
    kl_exponential = False
    if len(kl_values) >= 4:
        prior = statistics.median(abs(value) for value in kl_values[:-1])
        kl_exponential = prior > 0 and abs(kl_values[-1]) > 10.0 * prior

    expected_metric_minimums = {
        "attempt": len(attempts),
        "update": 20,
        "channel": 2 * len(attempts),
        "prompt": 64 * len(attempts),
        "trajectory": 64 * 16 * len(attempts),
        "turn": 1,
        "behavior": len(attempts),
        "system": len(attempts),
        "eval": 9,
        "checkpoint": 2,
    }
    metric_counts = {scope: len(rows) for scope, rows in metrics.items()}
    metrics_complete = all(
        metric_counts[scope] >= minimum
        for scope, minimum in expected_metric_minimums.items()
    )

    checkpoint_by_step = {
        int(row["successful_update_step"]): row for row in checkpoints
    }
    checkpoint_pass = {}
    for step in (10, 20):
        row = checkpoint_by_step.get(step, {})
        checkpoint_pass[step] = bool(
            row
            and row.get("readonly_subprocess", {}).get("status") == "PASS"
            and row.get("distributed_reload", {}).get("status") == "PASS"
            and (run_dir / "checkpoints" / f"update_{step}").is_dir()
        )

    eval_pass = (
        set(eval_summary) == {"0", "10", "20"}
        and all(
            "overall" in eval_summary[str(step)]
            and int(eval_summary[str(step)]["overall"]["count"]) == 600
            for step in (0, 10, 20)
        )
    )
    update10_pass = all(
        channel_audit[channel]["update10_selection_bootstrap"]
        and channel_audit[channel]["update10_reference_created"]
        for channel in ("IG", "Outcome")
    )
    update11_pass = all(
        channel_audit[channel]["update11_selection_health"]
        for channel in ("IG", "Outcome")
    )
    channel_independence = bool(
        channel_audit["independent_transition_pass"]
    )
    forced_refill_pass = bool(
        forced.get("status") == "PASS"
        and forced.get("initial_prompt_count") == 64
        and forced.get("refill_prompt_count") == 32
        and forced.get("total_unique_prompt_count") == 96
        and forced.get("group_size") == 16
        and forced.get("initial_exact_ig_reused") is True
        and forced.get("only_new_refill_trajectories_scored") == 512
        and forced.get("optimizer_steps") == 0
        and forced.get("scheduler_steps") == 0
        and forced.get("checkpoint_writes") == 0
    )
    forced_skip_pass = bool(
        forced.get("forced_skip", {}).get("status") == "PASS"
        and forced.get("forced_skip", {}).get("optimizer_steps") == 0
        and forced.get("forced_skip", {}).get("scheduler_steps") == 0
        and forced.get("forced_skip", {}).get("successful_update_after") == 0
    )
    pilot_pass = bool(
        pilot.get("status") == "PASS"
        and pilot.get("successful_updates") == 20
        and pilot.get("optimizer_steps_total") == 20
        and pilot.get("scheduler_steps_total") == 20
        and successful_steps == list(range(1, 21))
        and optimizer_steps == 20
        and scheduler_steps == 20
        and finite
        and gradients_finite
        and not kl_exponential
        and behavior["pass"]
        and system["pass"]
    )
    final_pass = all(
        (
            pilot_pass,
            update10_pass,
            update11_pass,
            channel_independence,
            metrics_complete,
            checkpoint_pass[10],
            checkpoint_pass[20],
            eval_pass,
            forced_refill_pass,
            forced_skip_pass,
            bool(args.formal_dry_run_pass),
        )
    )
    gate = {
        "pilot_20_pass": pilot_pass,
        "successful_updates": int(pilot.get("successful_updates", 0)),
        "optimizer_steps": optimizer_steps,
        "scheduler_steps": scheduler_steps,
        "update10_bootstrap_pass": update10_pass,
        "update11_health_pass": update11_pass,
        "channel_independence_pass": channel_independence,
        "metrics_complete": metrics_complete,
        "checkpoint10_reload_pass": checkpoint_pass[10],
        "checkpoint20_reload_pass": checkpoint_pass[20],
        "eval_pass": eval_pass,
        "forced_refill_96_pass": forced_refill_pass,
        "forced_skip_pass": forced_skip_pass,
        "formal_script_dry_run_pass": bool(args.formal_dry_run_pass),
        "final_pretrain_test_pass": final_pass,
        "pilot_run_dir": str(run_dir),
        "formal_launch_script": str(
            PROJECT_ROOT / "scripts" / "train_formal_manual.sh"
        ),
        "formal_logs_root": str(PROJECT_ROOT / "outputs" / "formal_training"),
        "metric_counts": metric_counts,
        "channel_audit": channel_audit,
        "behavior_audit": behavior,
        "system_audit": system,
        "eval_summary": eval_summary,
    }
    output_gate = PROJECT_ROOT / "outputs" / "final_pretrain_gate.json"
    output_gate.parent.mkdir(parents=True, exist_ok=True)
    output_gate.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reports = _report_paths()
    _write(
        reports["PILOT_20_UPDATE_REPORT.md"],
        [
            "# Pilot 20 Update Report",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Stage status: `{pilot.get('status')}`",
            f"- Successful updates: `{pilot.get('successful_updates')}`",
            f"- Attempts: `{pilot.get('attempts')}`",
            f"- Optimizer steps: `{optimizer_steps}`",
            f"- Scheduler steps: `{scheduler_steps}`",
            f"- Update sequence: `{successful_steps}`",
            f"- Finite update metrics: `{finite}`",
            f"- Finite gradient norms: `{gradients_finite}`",
            f"- KL exponential-growth alarm: `{kl_exponential}`",
            f"- Behavior gate: `{_bool(behavior['pass'])}`",
            f"- System stability gate: `{_bool(system['pass'])}`",
        ],
    )
    _write(
        reports["UPDATE_10_11_HEALTH_GATE_REPORT.md"],
        [
            "# Update 10/11 Health Gate Report",
            "",
            f"- Update 10 bootstrap: `{_bool(update10_pass)}`",
            f"- Update 11 health: `{_bool(update11_pass)}`",
            f"- Channel independence: `{_bool(channel_independence)}`",
            "",
            "```json",
            json.dumps(channel_audit, indent=2, sort_keys=True),
            "```",
            "",
            "Findings:",
            *([f"- {item}" for item in channel_findings] or ["- None."]),
        ],
    )
    forced_lines = [
        "# Forced Refill 96 Test Report",
        "",
        f"- Test directory: `{forced_dir}`",
        f"- Runtime result: `{forced.get('status')}`",
        f"- Initial/refill/total prompts: `64/32/{forced.get('total_unique_prompt_count')}`",
        f"- G: `{forced.get('group_size')}`",
        f"- Old 64 Exact-IG reused: `{forced.get('initial_exact_ig_reused')}`",
        f"- Newly scored trajectories: `{forced.get('only_new_refill_trajectories_scored')}`",
        f"- 96-pool recomputed: `{forced.get('full_96_selection', {}).get('pool_recomputed_from_scratch')}`",
        f"- Optimizer/scheduler/checkpoint writes: `{forced.get('optimizer_steps')}/{forced.get('scheduler_steps')}/{forced.get('checkpoint_writes')}`",
        f"- Forced refill gate: `{_bool(forced_refill_pass)}`",
        f"- Forced skip gate: `{_bool(forced_skip_pass)}`",
    ]
    _write(reports["FORCED_REFILL_96_TEST_REPORT.md"], forced_lines)
    _write(run_dir / "reports" / "FORCED_REFILL_96_TEST_REPORT.md", forced_lines)
    _write(
        reports["METRICS_COMPLETENESS_REPORT.md"],
        [
            "# Metrics Completeness Report",
            "",
            f"- Result: `{_bool(metrics_complete)}`",
            "",
            "| Scope | Rows | Required minimum |",
            "|---|---:|---:|",
            *[
                f"| `{scope}` | {metric_counts[scope]} | "
                f"{expected_metric_minimums[scope]} |"
                for scope in METRIC_FILES
            ],
            "",
            "All JSONL sinks are line-buffered by `MetricsSink` and written during "
            "the attempt, update, checkpoint, and eval paths.",
        ],
    )
    _write(
        reports["PILOT_CHECKPOINT_RECOVERY_REPORT.md"],
        [
            "# Pilot Checkpoint Recovery Report",
            "",
            f"- Update 10 read-only + distributed reload: `{_bool(checkpoint_pass[10])}`",
            f"- Update 20 read-only + distributed reload: `{_bool(checkpoint_pass[20])}`",
            f"- Checkpoint records: `{len(checkpoints)}`",
            f"- Retained directories: `{sorted(path.name for path in (run_dir / 'checkpoints').iterdir() if path.is_dir())}`",
        ],
    )
    eval_lines = ["# Pilot Eval Report", ""]
    for step in (0, 10, 20):
        eval_lines.append(f"## Update {step}")
        eval_lines.append("")
        eval_lines.append("| Domain | Count | F1 | Exact | Format | Avg search |")
        eval_lines.append("|---|---:|---:|---:|---:|---:|")
        for domain, row in sorted(eval_summary[str(step)].items()):
            eval_lines.append(
                f"| {domain} | {row['count']} | {row['f1']:.6f} | "
                f"{row['exact']:.6f} | {row['format_rate']:.6f} | "
                f"{row['avg_search']:.6f} |"
            )
        eval_lines.append("")
    eval_lines.append(f"Eval gate: `{_bool(eval_pass)}`")
    _write(reports["PILOT_EVAL_REPORT.md"], eval_lines)
    _write(
        reports["FORMAL_TRAINING_READINESS_REPORT.md"],
        [
            "# Formal Training Readiness Report",
            "",
            f"- Final pretrain gate: `{_bool(final_pass)}`",
            f"- Exact-IG production mode: `OFFICIAL_BF16_FAST_FULL_LOGITS`",
            f"- Pilot: `{_bool(pilot_pass)}`",
            f"- Update 10/11 gate: `{_bool(update10_pass and update11_pass)}`",
            f"- Checkpoint recovery: `{_bool(all(checkpoint_pass.values()))}`",
            f"- Eval: `{_bool(eval_pass)}`",
            f"- Forced refill/skip: `{_bool(forced_refill_pass and forced_skip_pass)}`",
            f"- Formal dry-run: `{_bool(args.formal_dry_run_pass)}`",
            "",
            "Formal training remains manual and requires an explicit successful-update count.",
        ],
    )
    launch_command = (
        f"cd {PROJECT_ROOT}\n\n"
        "bash scripts/train_formal_manual.sh \\\n"
        "  --total-successful-updates <用户指定数量>"
    )
    _write(
        reports["TRAIN_LAUNCH_GUIDE.md"],
        [
            "# Train Launch Guide",
            "",
            "Formal launch:",
            "",
            "```bash",
            launch_command,
            "```",
            "",
            "Resume:",
            "",
            "```bash",
            "bash scripts/resume_formal_manual.sh \\",
            "  --run-dir /absolute/path/to/outputs/formal_training/<RUN_ID>",
            "```",
            "",
            "Status/logs/stop:",
            "",
            "```bash",
            "bash scripts/status_formal_training.sh",
            "bash scripts/tail_formal_logs.sh",
            "bash scripts/stop_formal_training.sh",
            "```",
            "",
            f"Formal output root: `{PROJECT_ROOT / 'outputs' / 'formal_training'}`",
        ],
    )
    _write(
        reports["TEST_RESULTS_FINAL_PRETRAIN.md"],
        [
            "# Test Results Final Pretrain",
            "",
            f"- Pilot 20: `{_bool(pilot_pass)}`",
            f"- Strict optimizer/scheduler counts: `{optimizer_steps}/{scheduler_steps}`",
            f"- Channel gate: `{_bool(update10_pass and update11_pass and channel_independence)}`",
            f"- Checkpoints: `{_bool(all(checkpoint_pass.values()))}`",
            f"- Fixed eval: `{_bool(eval_pass)}`",
            f"- Forced refill: `{_bool(forced_refill_pass)}`",
            f"- Forced skip: `{_bool(forced_skip_pass)}`",
            f"- Metrics: `{_bool(metrics_complete)}`",
            f"- Final: `{_bool(final_pass)}`",
        ],
    )
    _write(
        reports["CODE_CHANGELOG_FINAL_PRETRAIN.md"],
        [
            "# Code Changelog Final Pretrain",
            "",
            "- Added resolved Pilot and formal configurations.",
            "- Bound MetricsActor to attempt, update, channel, prompt, trajectory, "
            "turn, behavior, system, eval, and checkpoint streams.",
            "- Added fixed immutable eval, isolated checkpoint reload validation, "
            "and isolated forced 64->96 refill execution.",
            "- Added PID-scoped Pilot/formal launch, resume, status, tail, and stop scripts.",
            "- Added this machine-readable final pretrain gate generator.",
            "- No frozen reward, selection, advantage, clipping, KL, or reduction "
            "formula was changed.",
        ],
    )
    _write(
        reports["FINAL_PRETRAIN_TEST_REPORT.md"],
        [
            "# Final Pretrain Test Report",
            "",
            f"`FINAL_PRETRAIN_TEST = {'PASS' if final_pass else 'FAIL'}`",
            "",
            f"- Pilot run: `{run_dir}`",
            f"- Successful updates: `{pilot.get('successful_updates')}`",
            f"- Optimizer/scheduler: `{optimizer_steps}/{scheduler_steps}`",
            f"- Update 10 bootstrap: `{_bool(update10_pass)}`",
            f"- Update 11 health: `{_bool(update11_pass)}`",
            f"- Channel independence: `{_bool(channel_independence)}`",
            f"- Checkpoint 10/20: `{_bool(all(checkpoint_pass.values()))}`",
            f"- Eval 0/10/20: `{_bool(eval_pass)}`",
            f"- Forced refill 96: `{_bool(forced_refill_pass)}`",
            f"- Forced skip: `{_bool(forced_skip_pass)}`",
            f"- Metrics complete: `{_bool(metrics_complete)}`",
            f"- NaN/Inf: `{'NONE' if finite and gradients_finite else 'DETECTED'}`",
            f"- OOM/weight mismatch: `{'NONE' if pilot.get('status') == 'PASS' else 'SEE_STAGE_RESULT'}`",
            f"- Ray spill / sustained GPU growth: `{system['ray_spill_max_bytes']} / {system['sustained_gpu_memory_growth']}`",
            "",
            f"Machine-readable gate: `{output_gate}`",
        ],
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not final_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
