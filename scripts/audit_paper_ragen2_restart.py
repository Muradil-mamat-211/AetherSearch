from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from agentic_rl.selection.paper_ragen2 import (
    select_ragen2_raw_variance_mass_top_p,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "paper_ragen2_restart"
FRESH_RUN = (
    PROJECT_ROOT
    / "outputs/formal_training/"
    "formal_fresh_u000_to_u500_answer_ragen2_mica_ig_v1_g16_20260810_184159"
)
LATEST_RUN = (
    PROJECT_ROOT
    / "outputs/formal_training/"
    "formal_resume_u020_to_u500_answer_ragen2_mica_ig_v1_g16_20260811_075718"
)
REQUESTS = (
    (1, 1, FRESH_RUN),
    (10, 10, FRESH_RUN),
    # U20 prompt metrics were not committed; U19 is the nearest complete pool.
    (20, 19, FRESH_RUN),
    (30, 30, LATEST_RUN),
    (40, 40, LATEST_RUN),
    (41, 41, LATEST_RUN),
)


def _read_matching(path: Path, attempt_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("attempt_id", -1)) == attempt_id:
                rows.append(row)
    return rows


def _mean(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return fmean(values) if values else None


def _domain_counts(
    selected_ids: set[str],
    prompt_rows: list[dict[str, Any]],
) -> tuple[int, int]:
    domain_by_prompt = {
        str(row["prompt_global_id"]): str(row.get("domain", ""))
        for row in prompt_rows
    }
    nq = sum(domain_by_prompt[prompt_id] == "nq" for prompt_id in selected_ids)
    hotpot = sum(
        domain_by_prompt[prompt_id] == "hotpotqa" for prompt_id in selected_ids
    )
    return nq, hotpot


def _trajectory_summary(
    trajectory_rows: list[dict[str, Any]],
    selected_ids: set[str],
) -> tuple[float | None, float | None]:
    selected = [
        row
        for row in trajectory_rows
        if str(row["prompt_global_id"]) in selected_ids
    ]
    return _mean(selected, "search_count"), _mean(selected, "R_task")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for requested_update, actual_update, run_dir in REQUESTS:
        prompt_rows = _read_matching(
            run_dir / "metrics/prompt_metrics.jsonl",
            actual_update,
        )
        if not prompt_rows:
            raise RuntimeError(
                f"No prompt records for actual U{actual_update}: {run_dir}"
            )
        trajectory_rows = _read_matching(
            run_dir / "metrics/trajectory_metrics.jsonl",
            actual_update,
        )
        channel_rows = _read_matching(
            run_dir / "metrics/channel_metrics.jsonl",
            actual_update,
        )
        outcome_channel = next(
            (
                row
                for row in channel_rows
                if str(row.get("channel")) == "Outcome"
            ),
            {},
        )

        variance = {
            str(row["prompt_global_id"]): float(row["V_Outcome"])
            for row in prompt_rows
        }
        paper_raw = select_ragen2_raw_variance_mass_top_p(variance, rho=0.9)
        paper_final_ids = set(paper_raw.selected_ids[:36])
        old_ids = {
            str(row["prompt_global_id"])
            for row in prompt_rows
            if bool(row["selected"])
        }
        intersection = old_ids & paper_final_ids
        union = old_ids | paper_final_ids
        old_nq, old_hotpot = _domain_counts(old_ids, prompt_rows)
        paper_nq, paper_hotpot = _domain_counts(paper_final_ids, prompt_rows)
        old_avg_search, old_task = _trajectory_summary(
            trajectory_rows,
            old_ids,
        )
        paper_avg_search, paper_task = _trajectory_summary(
            trajectory_rows,
            paper_final_ids,
        )
        candidate_avg_search, candidate_task = _trajectory_summary(
            trajectory_rows,
            set(variance),
        )
        needs_refill = len(paper_final_ids) < 32 and len(prompt_rows) < 128
        output_rows.append(
            {
                "requested_update": requested_update,
                "actual_complete_update": actual_update,
                "source_run": run_dir.name,
                "u20_substitution_reason": (
                    "U20 prompt metrics missing; nearest complete U19 used"
                    if requested_update == 20
                    else ""
                ),
                "candidate_prompt_count": len(prompt_rows),
                "old_selected_count": len(old_ids),
                "paper_raw_k_star": len(paper_raw.selected_ids),
                "paper_selected_count_on_persisted_pool": len(paper_final_ids),
                "paper_requires_unobserved_refill": needs_refill,
                "paper_final_count_exactly_reconstructable": not needs_refill,
                "intersection": len(intersection),
                "jaccard_old_paper": len(intersection) / len(union) if union else 1.0,
                "old_health_gate_active": outcome_channel.get("activation"),
                "old_health_gate_reason": outcome_channel.get("activation_reason"),
                "old_scale_used": outcome_channel.get("b_use"),
                "paper_raw_variance_total_mass": paper_raw.total_mass,
                "paper_raw_selected_mass_ratio": paper_raw.selected_mass_ratio,
                "paper_capped_selected_mass_ratio": (
                    sum(variance[prompt_id] for prompt_id in paper_final_ids)
                    / paper_raw.total_mass
                    if paper_raw.total_mass > 0.0
                    else 0.0
                ),
                "candidate_mean_outcome_variance": fmean(variance.values()),
                "candidate_mean_old_ig_variance": _mean(prompt_rows, "V_IG"),
                "old_selected_nq": old_nq,
                "old_selected_hotpotqa": old_hotpot,
                "paper_selected_nq": paper_nq,
                "paper_selected_hotpotqa": paper_hotpot,
                "candidate_avg_search": candidate_avg_search,
                "old_selected_avg_search": old_avg_search,
                "paper_selected_avg_search": paper_avg_search,
                "candidate_mean_task_outcome": candidate_task,
                "old_selected_mean_task_outcome": old_task,
                "paper_selected_mean_task_outcome": paper_task,
                "paper_health_gate_selection_call_count": 0,
                "paper_scale_selection_call_count": 0,
            }
        )

    path = OUTPUT / "PAPER_RAGEN2_SHADOW_SELECTION.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
