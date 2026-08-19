from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from agentic_rl.config import load_config
from agentic_rl.retriever.health import query_health

from .formal_state import append_jsonl, atomic_write_json, eval_queue_snapshot, read_json
from .resource_guard import read_runtime_resource_snapshot


def _tail_jsonl(path: Path, maximum: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=maximum)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def _processes(run_dir: Path) -> dict[str, Any]:
    payload = read_json(run_dir / "state" / "processes.json", {})
    result = dict(payload)
    for name in ("trainer", "retriever", "eval_worker", "monitor", "watchdog"):
        pid_path = run_dir / "state" / "pids" / f"{name}.pid"
        pid = (
            pid_path.read_text(encoding="utf-8").strip()
            if pid_path.is_file()
            else payload.get(f"{name}_pid")
        )
        result[f"{name}_pid"] = int(pid) if str(pid).isdigit() else None
        result[f"{name}_alive"] = _alive(pid)
    return result


def _nvidia_snapshot() -> list[dict[str, Any]]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    processes: dict[str, list[dict[str, Any]]] = {}
    for line in apps.stdout.splitlines():
        values = [item.strip() for item in line.split(",", maxsplit=3)]
        if len(values) == 4:
            processes.setdefault(values[0], []).append(
                {"pid": values[1], "name": values[2], "used_memory_mib": values[3]}
            )
    rows = []
    for line in gpu.stdout.splitlines():
        values = [item.strip() for item in line.split(",", maxsplit=7)]
        if len(values) != 8:
            continue
        rows.append(
            {
                "physical_gpu": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "utilization_percent": int(values[3]),
                "memory_used_mib": int(values[4]),
                "memory_total_mib": int(values[5]),
                "temperature_c": int(values[6]),
                "power_watts": float(values[7]),
                "processes": processes.get(values[2], []),
            }
        )
    return rows


def _host_snapshot(run_dir: Path) -> dict[str, Any]:
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", maxsplit=1)
        memory[key] = int(value.strip().split()[0]) * 1024
    disk = os.statvfs(run_dir)
    load1, load5, load15 = os.getloadavg()
    cgroup = read_runtime_resource_snapshot()
    return {
        "ram_total_bytes": memory.get("MemTotal"),
        "ram_available_bytes": memory.get("MemAvailable"),
        "ram_used_bytes": memory.get("MemTotal", 0) - memory.get("MemAvailable", 0),
        "load_average": [load1, load5, load15],
        "cpu_count": os.cpu_count(),
        "cgroup_memory_limit_bytes": cgroup.get("memory_limit_bytes"),
        "cgroup_memory_current_bytes": cgroup.get("memory_current_bytes"),
        "cgroup_cpu_quota_cores": cgroup.get("cpu_quota_cores"),
        "cgroup_memory_events": cgroup.get("memory_events", {}),
        "disk_free_bytes": disk.f_bavail * disk.f_frsize,
        "disk_total_bytes": disk.f_blocks * disk.f_frsize,
    }


def _mean_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = fmean(values) if values else None
    return result


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latest_eval(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    queue = eval_queue_snapshot(run_dir)
    completed = sorted(
        (item for item in queue["tasks"] if item["status"] == "completed"),
        key=lambda item: int(item["update"]),
    )
    latest = None
    if completed:
        update = int(completed[-1]["update"])
        metrics_path = run_dir / "eval" / f"update_{update:03d}" / "metrics.json"
        latest = read_json(metrics_path)
    return latest, queue


def _eval_section(run_dir: Path, processes: Mapping[str, Any]) -> dict[str, Any]:
    latest, queue = _latest_eval(run_dir)
    pending = [int(item["update"]) for item in queue["tasks"] if item["status"] == "pending"]
    running = [int(item["update"]) for item in queue["tasks"] if item["status"] == "running"]
    completed = [int(item["update"]) for item in queue["tasks"] if item["status"] == "completed"]
    latest_metrics = {}
    latest_update = max(completed) if completed else None
    if latest:
        latest_metrics = {str(item["domain"]): item for item in latest.get("metrics", [])}
    baseline = read_json(run_dir / "eval" / "update_020" / "metrics.json", {})
    baseline_by_domain = {
        str(item["domain"]): item for item in baseline.get("metrics", [])
    }
    previous_update = sorted(completed)[-2] if len(completed) >= 2 else None
    previous = (
        read_json(run_dir / "eval" / f"update_{previous_update:03d}" / "metrics.json", {})
        if previous_update is not None
        else {}
    )
    previous_by_domain = {
        str(item["domain"]): item for item in previous.get("metrics", [])
    }
    deltas: dict[str, Any] = {}
    for domain, row in latest_metrics.items():
        deltas[domain] = {
            "exact_vs_update20": float(row["exact"]) - float(baseline_by_domain.get(domain, row)["exact"]),
            "f1_vs_update20": float(row["f1"]) - float(baseline_by_domain.get(domain, row)["f1"]),
            "exact_vs_previous": float(row["exact"]) - float(previous_by_domain.get(domain, row)["exact"]),
            "f1_vs_previous": float(row["f1"]) - float(previous_by_domain.get(domain, row)["f1"]),
        }
    wait_reason = next(
        (item.get("wait_reason") for item in queue["tasks"] if item["status"] == "pending" and item.get("wait_reason")),
        None,
    )
    last_error = next(
        (item.get("last_error") for item in reversed(queue["tasks"]) if item.get("last_error")),
        None,
    )
    return {
        "latest_completed_eval_update": latest_update,
        "currently_running_eval_update": running[0] if running else None,
        "pending_eval_updates": pending,
        "eval_worker_pid": processes.get("eval_worker_pid"),
        "eval_worker_alive": processes.get("eval_worker_alive"),
        "GPU0_eval_wait_reason": wait_reason,
        "latest_eval_wall_seconds": latest.get("wall_seconds") if latest else None,
        "latest_eval_error": last_error,
        "latest_metrics": latest_metrics,
        "deltas": deltas,
    }


def build_snapshot(config: Mapping[str, Any], run_dir: Path) -> tuple[dict[str, Any], str]:
    now = time.time()
    metrics = run_dir / "metrics"
    updates = _tail_jsonl(metrics / "update_metrics.jsonl", 20)
    attempts = _tail_jsonl(metrics / "attempt_metrics.jsonl", 20)
    behaviors = _tail_jsonl(metrics / "behavior_metrics.jsonl", 20)
    channels = _tail_jsonl(metrics / "channel_metrics.jsonl", 40)
    latest_update = updates[-1] if updates else {}
    latest_attempt = attempts[-1] if attempts else {}
    attempt_id = int(latest_attempt.get("attempt_id", 20))
    prompts = [
        row for row in _tail_jsonl(metrics / "prompt_metrics.jsonl", 3000)
        if int(row.get("attempt_id", -1)) == attempt_id
    ]
    trajectories = [
        row for row in _tail_jsonl(metrics / "trajectory_metrics.jsonl", 5000)
        if int(row.get("attempt_id", -1)) == attempt_id
    ]
    turns = [
        row for row in _tail_jsonl(metrics / "turn_metrics.jsonl", 16000)
        if int(row.get("attempt_id", -1)) == attempt_id
    ]
    system_rows = _tail_jsonl(metrics / "system_metrics.jsonl", 2)
    system = system_rows[-1] if system_rows else {}
    process = _processes(run_dir)
    current_attempt = read_json(run_dir / "state" / "current_attempt.json", {})
    progress = read_json(run_dir / "state" / "training_progress.json", {})
    successful = int(progress.get("successful_update_step", latest_update.get("successful_update_step", 20)))
    target = int(config["monitoring"]["target_successful_update"])
    update_times = [float(row.get("attempt_wall_time", 0.0)) for row in attempts if row.get("successful_update_after", 0) > row.get("successful_update_before", 0)]
    average5 = fmean(update_times[-5:]) if update_times else None
    average20 = fmean(update_times[-20:]) if update_times else None
    eta = (target - successful) * average20 if average20 is not None else None
    timestamps = [float(row["timestamp_unix"]) for row in updates if row.get("timestamp_unix")]
    lifecycle = {
        "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "RUN_ID": run_dir.name,
        "training_status": "RUNNING" if process.get("trainer_alive") else read_json(run_dir / "state" / "trainer_result.json", {}).get("status", "STARTING"),
        "driver_pid": process.get("trainer_pid"),
        "tmux_session": process.get("tmux_session"),
        "uptime_seconds": now - float(process.get("started_at", now)),
        "current_attempt": current_attempt.get("attempt_id_before", attempt_id),
        "current_successful_update": successful,
        "target_successful_update": target,
        "successful_updates_since_resume": successful - 20,
        "data_cursor": int(progress.get("data_cursor", latest_attempt.get("data_cursor_after", 1600))),
        "skipped_attempts_total": sum(row.get("skip_reason") is not None for row in attempts),
        "last_successful_update_time": timestamps[-1] if timestamps else None,
        "current_attempt_elapsed": now - float(current_attempt.get("started_at", now)),
        "average_update_seconds_last_5": average5,
        "average_update_seconds_last_20": average20,
        "ETA_to_update_500_seconds": eta,
        "latest_model_checkpoint": progress.get("latest_model_checkpoint"),
        "latest_resume_checkpoint": progress.get("latest_resume_checkpoint"),
        "next_checkpoint_update": min(target, ((successful // 20) + 1) * 20),
    }
    evaluation = _eval_section(run_dir, process)

    policy = {
        "learning_rate": latest_update.get("learning_rate_used"),
        "optimizer_steps_total": latest_update.get("optimizer_steps_total", 20),
        "scheduler_steps_total": latest_update.get("scheduler_steps_total", 20),
        "optimizer_steps_this_update": latest_update.get("optimizer_steps_this_update"),
        "scheduler_steps_this_update": latest_update.get("scheduler_steps_this_update"),
        "policy_loss": (-float(latest_update["task_objective"]) if latest_update.get("task_objective") is not None else None),
        "reference_kl_raw": latest_update.get("full_vocab_forward_kl"),
        "weighted_kl_loss": latest_update.get("kl_weighted_loss"),
        "total_loss": latest_update.get("total_loss"),
        "grad_norm_before_clip": latest_update.get("grad_norm_before_clip", latest_update.get("gradient_norm")),
        "grad_norm_after_clip": latest_update.get("grad_norm_after_clip"),
        "max_grad_norm": latest_update.get("max_grad_norm", 1.0),
        "ratio_mean": latest_update.get("ratio_mean"),
        "ratio_std": latest_update.get("ratio_std"),
        "ratio_p95": latest_update.get("ratio_p95"),
        "ratio_min": latest_update.get("ratio_min"),
        "ratio_max": latest_update.get("ratio_max"),
        "clipfrac_low": latest_update.get("clipfrac_low"),
        "clipfrac_high": latest_update.get("clipfrac_high"),
        "action_tokens": latest_update.get("action_tokens"),
        "selected_action_tokens": sum(int(row.get("action_token_count", 0)) for row in trajectories if any(prompt.get("prompt_global_id") == row.get("prompt_global_id") and prompt.get("selected") for prompt in prompts)),
    }

    phi = [float(value) for row in turns for value in (row.get("Phi_before"), row.get("Phi_after")) if value is not None]
    raw_ig = [float(row["raw_IG"]) for row in turns if row.get("raw_IG") is not None]
    by_search: dict[str, list[float]] = {}
    for row in turns:
        if row.get("raw_IG") is not None:
            by_search.setdefault(str(row.get("turn_id")), []).append(float(row["raw_IG"]))
    canary_rows = system.get("exact_ig_oracle_canary_by_rank", []) or []
    exact_ig = {
        "Phi": {"mean": fmean(phi) if phi else None, "std": float(np.std(phi)) if phi else None, "min": min(phi) if phi else None, "max": max(phi) if phi else None},
        "raw_IG": {"mean": fmean(raw_ig) if raw_ig else None, "std": float(np.std(raw_ig)) if raw_ig else None, "min": min(raw_ig) if raw_ig else None, "max": max(raw_ig) if raw_ig else None},
        "positive_IG_ratio": sum(value > 1e-12 for value in raw_ig) / len(raw_ig) if raw_ig else None,
        "negative_IG_ratio": sum(value < -1e-12 for value in raw_ig) / len(raw_ig) if raw_ig else None,
        "near_zero_IG_ratio": sum(abs(value) <= 1e-12 for value in raw_ig) / len(raw_ig) if raw_ig else None,
        "IG_by_search_index": {key: _mean_rows([{"value": value} for value in values], ["value"])["value"] for key, values in by_search.items()},
        "Fast_Oracle_canary_count": sum(int(row.get("checks", 0)) for row in canary_rows),
        "canary_max_Phi_error": max((float(row.get("phi_max_abs_error", 0.0)) for row in canary_rows), default=0.0),
        "canary_max_IG_error": max((float(row.get("ig_max_abs_error", 0.0)) for row in canary_rows), default=0.0),
        "canary_hard_failures": sum(int(row.get("hard_failures", 0)) for row in canary_rows),
        "telescoping_max_error": max((abs(float(row.get("Phi_after", 0.0)) - float(row.get("Phi_before", 0.0)) - float(row.get("raw_IG", 0.0))) for row in turns if row.get("raw_IG") is not None), default=0.0),
        "records_per_second": system.get("exact_ig_records_per_second"),
        "trajectories_per_second": system.get("exact_ig_trajectories_per_second"),
        "GPU_seconds": system.get("exact_ig_gpu_time"),
        "peak_memory_bytes": system.get("exact_ig_peak_memory_bytes"),
        "context_overflow_count": 0,
        "nonfinite_metadata_count": sum(not math.isfinite(value) for value in (*phi, *raw_ig)),
    }

    channel_latest = {
        str(row["channel"]): row for row in channels if int(row.get("attempt_id", -1)) == attempt_id
    }
    selected_prompts = [row for row in prompts if row.get("selected")]
    ragen = {
        "candidate_pool_size": latest_attempt.get("pool_size"),
        "refill_used": latest_attempt.get("refill_used"),
        "selected_prompt_count": latest_attempt.get("selected_prompt_count"),
        "selected_trajectory_count": latest_attempt.get("selected_trajectory_count"),
        "variance_mass_top_p": float(config["selection"]["top_p_mass"]),
        "channels": channel_latest,
        "score_p50": _percentile([row.get("S", 0.0) for row in prompts], 0.50),
        "score_p95": _percentile([row.get("S", 0.0) for row in prompts], 0.95),
        "score_max": max((float(row.get("S", 0.0)) for row in prompts), default=None),
        "selection_boundary_margin": min((abs(float(row.get("selection_boundary_distance", 0.0))) for row in selected_prompts), default=None),
        "skip_reason": latest_attempt.get("skip_reason"),
        "selection_membership_note": "scaled fused-score selection; IG-only/outcome-only membership is not defined",
    }

    behavior_keys = (
        "answer_rate", "format_rate", "no_answer_rate", "task_f1_mean",
        "avg_search_count", "multi_search_rate", "repeat_query_rate",
        "gold_seen_then_search_rate", "max_turn_rate", "query_diversity",
        "template_similarity", "malformed_rate", "system_invalid_rate",
    )
    behavior = {
        "current": {key: behaviors[-1].get(key) for key in behavior_keys} if behaviors else {},
        "moving_average_5": _mean_rows(behaviors[-5:], behavior_keys),
        "moving_average_20": _mean_rows(behaviors[-20:], behavior_keys),
        "candidate_exact_rate": sum(math.isclose(float(row.get("R_task", 0.0)), 1.0, abs_tol=1e-12) for row in trajectories) / len(trajectories) if trajectories else None,
        "search_count_histogram": {str(value): sum(int(row.get("search_count", -1)) == value for row in trajectories) for value in range(6)},
        "avg_action_tokens": fmean([int(row.get("action_token_count", 0)) for row in trajectories]) if trajectories else None,
        "trajectory_action_tokens_p50": _percentile([row.get("action_token_count", 0) for row in trajectories], 0.50),
        "trajectory_action_tokens_p95": _percentile([row.get("action_token_count", 0) for row in trajectories], 0.95),
        "trajectory_action_tokens_p99": _percentile([row.get("action_token_count", 0) for row in trajectories], 0.99),
    }

    try:
        retriever = query_health(str(config["retriever"]["service_url"])).raw
        retriever_error = None
    except BaseException as exc:
        retriever = None
        retriever_error = f"{type(exc).__name__}: {exc}"
    host = _host_snapshot(run_dir)
    live_system = {
        "gpus": _nvidia_snapshot(),
        "GPU0_retriever_health": retriever,
        "GPU0_retriever_error": retriever_error,
        "retriever_requests": system.get("retriever_requests"),
        "retriever_latency_p50": system.get("retriever_p50_latency"),
        "retriever_latency_p95": system.get("retriever_p95_latency"),
        "retriever_latency_p99": system.get("retriever_p99_latency"),
        "ray_object_store_used_bytes": system.get("ray_object_store_used_bytes"),
        "ray_object_store_spill_bytes": system.get("ray_object_store_spill_bytes"),
        "phase_seconds": {key: system.get(key) for key in ("rollout_time", "exact_ig_gpu_time", "selection_time", "old_logprob_time", "reference_kl_backward_time", "optimizer_time", "weight_sync_time", "checkpoint_time", "vllm_sleep_wake_time")},
        "host": host,
        "processes": process,
    }

    red: list[str] = []
    yellow: list[str] = []
    fatal = read_json(run_dir / "state" / "fatal_status.json", {})
    if fatal.get("fatal"):
        red.append(f"persisted_fatal:{fatal.get('error')}")
    numeric_policy = [value for value in (policy.get("total_loss"), policy.get("reference_kl_raw"), policy.get("grad_norm_before_clip")) if value is not None]
    if any(not math.isfinite(float(value)) for value in numeric_policy):
        red.append("nonfinite_policy_metric")
    if exact_ig["nonfinite_metadata_count"]:
        red.append("nonfinite_exact_ig_metadata")
    if successful > 20 and (
        int(policy.get("optimizer_steps_total") or -1) != successful
        or int(policy.get("scheduler_steps_total") or -1) != successful
    ):
        red.append("optimizer_scheduler_successful_update_mismatch")
    if channel_latest and not any(bool(row.get("activation")) for row in channel_latest.values()):
        red.append("both_selection_channels_inactive")
    if host["disk_free_bytes"] < int(config["monitoring"]["minimum_disk_free_gib"]) * 1024**3:
        red.append("disk_space_below_safety_floor")
    if not process.get("trainer_alive") and not read_json(run_dir / "state" / "trainer_result.json", {}).get("status") == "PASS":
        yellow.append("trainer_not_yet_alive_or_exited")
    if evaluation["pending_eval_updates"]:
        yellow.append("eval_queue_pending")
    if retriever_error:
        yellow.append("retriever_health_unavailable")
    if live_system["ray_object_store_spill_bytes"] not in (None, 0, 0.0):
        yellow.append("ray_object_store_spill_detected")
    health = "RED" if red else ("YELLOW" if yellow else "GREEN")
    safety = {"HEALTH": health, "red_reasons": red, "yellow_reasons": yellow}
    snapshot = {
        "lifecycle": lifecycle,
        "eval": evaluation,
        "policy": policy,
        "exact_ig": exact_ig,
        "ragen": ragen,
        "behavior": behavior,
        "system": live_system,
        "safety": safety,
    }
    rendered = "\n".join(
        [
            "=" * 88,
            f"FORMAL TRAINING MONITOR {lifecycle['timestamp']}",
            *(
                f"[{name}] {json.dumps(value, sort_keys=True, ensure_ascii=False)}"
                for name, value in snapshot.items()
            ),
            f"HEALTH = {health}",
            "=" * 88,
        ]
    )
    return snapshot, rendered


def run_monitor(config_path: Path, run_dir: Path, *, once: bool) -> int:
    config = load_config(config_path)
    interval = int(config["monitoring"]["interval_seconds"])
    stopped = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stopped:
        snapshot, rendered = build_snapshot(config, run_dir)
        print(rendered, flush=True)
        append_jsonl(run_dir / "monitor" / "monitor_10min.jsonl", snapshot)
        atomic_write_json(run_dir / "state" / "training_state.json", snapshot)
        if snapshot["safety"]["HEALTH"] == "RED":
            atomic_write_json(
                run_dir / "state" / "fatal_status.json",
                {
                    "fatal": True,
                    "source": "monitor",
                    "error": snapshot["safety"]["red_reasons"],
                    "timestamp": time.time(),
                },
            )
            pid = snapshot["system"]["processes"].get("trainer_pid")
            if _alive(pid):
                os.kill(int(pid), signal.SIGTERM)
            return 2
        trainer = read_json(run_dir / "state" / "trainer_result.json", {})
        evaluator = read_json(run_dir / "state" / "eval_worker_result.json", {})
        if trainer.get("status") == "PASS" and evaluator.get("status") == "PASS":
            atomic_write_json(
                run_dir / "state" / "monitor_result.json",
                {"status": "PASS", "timestamp": time.time(), "final": snapshot},
            )
            return 0
        if once:
            return 0
        time.sleep(interval)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed ten-minute formal monitor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(
        run_monitor(
            Path(arguments.config).resolve(),
            Path(arguments.run_dir).resolve(),
            once=arguments.once,
        )
    )


if __name__ == "__main__":
    main()
