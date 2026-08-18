#!/usr/bin/env python3
"""Export domain-split training metrics for all committed updates in the MICA run lineage."""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
from datetime import datetime, timezone


PROJECT = pathlib.Path(__file__).resolve().parents[1]
RUNS = PROJECT / "outputs" / "formal_training"
REPORT_DIR = PROJECT / "reports" / "update1_to_current_nq_hotpotqa"
CSV_PATH = REPORT_DIR / "UPDATE1_TO_CURRENT_NQ_HOTPOTQA_TRAIN_METRICS.csv"
MD_PATH = REPORT_DIR / "UPDATE1_TO_CURRENT_NQ_HOTPOTQA_TRAIN_METRICS.md"

# These are the committed lineage segments. U180 is a verified checkpoint boundary
# without a standard per-update trajectory/behavior row, so it is reported as NA.
SEGMENTS = [
    (1, 39, "formal_u000_answer_ragen2_paper_mica_ig_v1_g16_20260811_130634"),
    (40, 40, "formal_resume_u020_to_u500_answer_ragen2_mica_ig_v1_g16_20260811_075718"),
    (41, 179, "formal_resume_u040_to_u500_answer_ragen2_mica_ig_v1_g16_20260812_030537"),
    (181, 196, "formal_resume_u180_to_u500_answer_ragen2_mica_ig_v1_g16_20260813_004642"),
]
DOMAINS = ("nq", "hotpotqa")
FIELDS = [
    "update",
    "domain",
    "attempt_id",
    "source_run",
    "trajectory_count",
    "train_f1_mean",
    "avg_search",
    "multi_search_rate",
    "repeat_query_rate",
    "global_full_vocab_forward_kl",
    "global_kl_weighted_loss",
    "status",
]


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def committed_steps(run_dir: pathlib.Path) -> dict[int, int]:
    mapping = {}
    for event in load_jsonl(run_dir / "events" / "successful_updates.jsonl"):
        attempt = event.get("attempt_id")
        step = event.get("successful_update_step")
        if isinstance(attempt, int) and isinstance(step, int):
            mapping[attempt] = step
    return mapping


def segment_for(update: int) -> tuple[pathlib.Path, str] | None:
    for lo, hi, name in SEGMENTS:
        if lo <= update <= hi:
            return RUNS / name, name
    return None


def build_rows() -> list[dict]:
    rows_by_update: dict[int, dict] = {}
    trajectories_by_update: dict[int, list[dict]] = {}

    for lo, hi, run_name in SEGMENTS:
        run_dir = RUNS / run_name
        valid_attempt_to_step = committed_steps(run_dir)
        updates = load_jsonl(run_dir / "metrics" / "update_metrics.jsonl")
        behavior = load_jsonl(run_dir / "metrics" / "behavior_metrics.jsonl")
        trajectories = load_jsonl(run_dir / "metrics" / "trajectory_metrics.jsonl")

        for record in updates:
            attempt = record.get("attempt_id")
            step = record.get("successful_update_step")
            if not isinstance(attempt, int) or not isinstance(step, int):
                continue
            if not (lo <= step <= hi) or valid_attempt_to_step.get(attempt) != step:
                continue
            rows_by_update[step] = {
                "update": step,
                "attempt_id": attempt,
                "source_run": run_name,
                "kl": record.get("full_vocab_forward_kl"),
                "kl_weighted": record.get("kl_weighted_loss"),
            }

        # Behavior is retained as a consistency/reference record. Domain rows are
        # reconstructed from trajectory records below; the persisted behavior row
        # is not used as a substitute for domain data.
        for record in behavior:
            attempt = record.get("attempt_id")
            step = record.get("successful_update_step")
            if not isinstance(attempt, int) or not isinstance(step, int):
                continue
            if not (lo <= step <= hi) or valid_attempt_to_step.get(attempt) != step:
                continue
            rows_by_update.setdefault(step, {}).update(
                {
                    "behavior_f1": record.get("task_f1_mean"),
                    "behavior_avg_search": record.get("avg_search_count"),
                    "behavior_multi": record.get("multi_search_rate"),
                    "behavior_repeat": record.get("repeat_query_rate"),
                }
            )

        for record in trajectories:
            attempt = record.get("attempt_id")
            if not isinstance(attempt, int) or not (lo <= attempt <= hi):
                continue
            if valid_attempt_to_step.get(attempt) != attempt:
                continue
            trajectories_by_update.setdefault(attempt, []).append(record)

    result = []
    for update in range(1, 197):
        update_meta = rows_by_update.get(update, {})
        source = update_meta.get("source_run", "")
        for domain in DOMAINS:
            records = [
                record
                for record in trajectories_by_update.get(update, [])
                if str(record.get("prompt_global_id", "")).startswith(domain + ":")
            ]
            kl = update_meta.get("kl")
            kl_weighted = update_meta.get("kl_weighted")
            if records:
                f1_values = [r["R_task"] for r in records if isinstance(r.get("R_task"), (int, float))]
                search_values = [r["search_count"] for r in records if isinstance(r.get("search_count"), (int, float))]
                repeat_values = [r.get("exact_query_repeat_count", 0) > 0 for r in records]
                result.append(
                    {
                        "update": update,
                        "domain": domain,
                        "attempt_id": update_meta.get("attempt_id", update),
                        "source_run": source,
                        "trajectory_count": len(records),
                        "train_f1_mean": statistics.fmean(f1_values) if f1_values else None,
                        "avg_search": statistics.fmean(search_values) if search_values else None,
                        "multi_search_rate": sum(value >= 2 for value in search_values) / len(search_values)
                        if search_values
                        else None,
                        "repeat_query_rate": sum(repeat_values) / len(repeat_values) if repeat_values else None,
                        "global_full_vocab_forward_kl": kl,
                        "global_kl_weighted_loss": kl_weighted,
                        "status": "PASS",
                    }
                )
            else:
                result.append(
                    {
                        "update": update,
                        "domain": domain,
                        "attempt_id": update_meta.get("attempt_id", update),
                        "source_run": source,
                        "trajectory_count": 0,
                        "train_f1_mean": None,
                        "avg_search": None,
                        "multi_search_rate": None,
                        "repeat_query_rate": None,
                        "global_full_vocab_forward_kl": kl,
                        "global_kl_weighted_loss": kl_weighted,
                        "status": "NA_checkpoint_boundary_no_domain_trajectory_row",
                    }
                )
    return result


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def main() -> None:
    rows = build_rows()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pass_rows = [row for row in rows if row["status"] == "PASS"]
    na_rows = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# U1-U196 NQ / HotpotQA Training Metrics",
        "",
        f"Snapshot UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is a read-only aggregation of persisted successful-update artifacts.",
        "`AvgS = mean(search_count)`, `Multi-search = P(search_count >= 2)`,",
        "`Repeat-query = P(exact_query_repeat_count > 0)`, and `Train F1 = mean(R_task)`.",
        "KL is persisted only at selected-learner global scope; the same global value is",
        "shown on both domain rows and must not be interpreted as domain-specific KL.",
        "",
        f"Rows: {len(rows)} ({len(pass_rows)} with domain trajectories, {len(na_rows)} NA rows).",
        "U180 is a verified checkpoint boundary with no standard domain trajectory metric row; it is intentionally NA.",
        "",
        "Complete data is in the CSV. The table below contains every update and both domains.",
        "",
        "| U | Domain | N | Train F1 | AvgS | Multi | Repeat | Global KL | KL loss | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            "| {update} | {domain} | {trajectory_count} | {train_f1_mean} | {avg_search} | "
            "{multi_search_rate} | {repeat_query_rate} | {global_full_vocab_forward_kl} | "
            "{global_kl_weighted_loss} | {status} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(CSV_PATH)
    print(MD_PATH)
    print(f"rows={len(rows)} pass={len(pass_rows)} na={len(na_rows)}")


if __name__ == "__main__":
    main()
