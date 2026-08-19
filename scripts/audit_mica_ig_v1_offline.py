#!/usr/bin/env python3
"""Read-only MICA V1 shadow recomputation over persisted U1-U139 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from agentic_rl.advantage.mica_ig import compute_mica_search_advantage
from agentic_rl.selection.top_p import stable_mass_top_p


SOURCE = Path(
    "/root/autodl-tmp/search-r1-workspace/projects/"
    "igpo_ragen2_a2tgpo_strict_onpolicy_v1"
)
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "reports/mica_ig_v1_implementation_audit"
REPRESENTATIVE_UPDATES = (1, 20, 40, 80, 120, 139)
RUNS = (
    (
        SOURCE
        / "outputs/formal_training/formal_fresh_u000_to_u500_"
        "role_localized_gate_g16_lr2e7_kl1e2_20260808_133350",
        1,
        20,
    ),
    (
        SOURCE
        / "outputs/formal_training/formal_resume_u020_to_u500_3rank_48cpu_"
        "20260808_191223_role_localized_gate_g16",
        21,
        80,
    ),
    (
        SOURCE
        / "outputs/formal_training/formal_resume_u080_to_u500_no_monitor_"
        "20260809_122013_role_localized_gate_g16",
        81,
        120,
    ),
    (
        SOURCE
        / "outputs/formal_training/formal_resume_u120_to_u500_provenance_"
        "utf8fix_20260810_060921_role_localized_gate_g16",
        121,
        139,
    ),
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_number}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def domain(prompt_id: str) -> str:
    value = str(prompt_id).lower()
    if value.startswith("nq:"):
        return "nq"
    if value.startswith("hotpotqa:"):
        return "hotpotqa"
    return "unknown"


def summary(values, prefix):
    array = np.asarray(
        [float(value) for value in values if finite(value) is not None],
        dtype=np.float64,
    )
    if not array.size:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_median": None,
            f"{prefix}_p05": None,
            f"{prefix}_p95": None,
            f"{prefix}_positive_rate": None,
            f"{prefix}_negative_rate": None,
        }
    return {
        f"{prefix}_count": int(array.size),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std(ddof=0)),
        f"{prefix}_median": float(np.percentile(array, 50)),
        f"{prefix}_p05": float(np.percentile(array, 5)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_positive_rate": float(np.mean(array > 0.0)),
        f"{prefix}_negative_rate": float(np.mean(array < 0.0)),
    }


def write_csv(path: Path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_artifacts():
    turns = []
    trajectories = []
    prompts = []
    updates = {}
    inputs = []
    mapping = {}
    for run, minimum, maximum in RUNS:
        paths = {
            name: run / "metrics" / f"{name}_metrics.jsonl"
            for name in ("update", "turn", "trajectory", "prompt")
        }
        if not all(path.is_file() for path in paths.values()):
            missing = [str(path) for path in paths.values() if not path.is_file()]
            raise RuntimeError(f"Missing persisted audit inputs: {missing}")
        inputs.extend(paths.values())
        attempt_to_update = {}
        for row in iter_jsonl(paths["update"]):
            update = row.get("successful_update_step")
            attempt = row.get("attempt_id")
            if attempt is None or update is None:
                continue
            update = int(update)
            if not minimum <= update <= maximum:
                continue
            if int(row.get("optimizer_steps_this_update", 0)) != 1 or int(
                row.get("scheduler_steps_this_update", 0)
            ) != 1:
                raise RuntimeError(f"U{update} is not a committed one-step update")
            attempt_to_update[int(attempt)] = update
            updates[update] = dict(row)
            mapping[f"{run.name}:{int(attempt)}"] = update
        if set(attempt_to_update.values()) != set(range(minimum, maximum + 1)):
            raise RuntimeError(
                f"Committed update coverage mismatch for {run.name}: "
                f"{sorted(attempt_to_update.values())}"
            )
        for kind, destination in (
            ("turn", turns),
            ("trajectory", trajectories),
            ("prompt", prompts),
        ):
            for row in iter_jsonl(paths[kind]):
                attempt = row.get("attempt_id")
                if attempt is None or int(attempt) not in attempt_to_update:
                    continue
                item = dict(row)
                item["_update"] = attempt_to_update[int(attempt)]
                destination.append(item)
    if set(updates) != set(range(1, 140)):
        raise RuntimeError("U1-U139 committed update lineage is incomplete")
    return turns, trajectories, prompts, updates, inputs, mapping


def exact_ig_verification(turns):
    errors = []
    sign_matches = []
    by_trajectory = defaultdict(list)
    unavailable = 0
    for row in turns:
        raw = finite(row.get("raw_IG"))
        before = finite(row.get("Phi_before"))
        after = finite(row.get("Phi_after"))
        if raw is None:
            continue
        if before is None or after is None:
            unavailable += 1
            continue
        rebuilt = after - before
        errors.append(abs(raw - rebuilt))
        sign_matches.append(
            (raw == 0.0 and rebuilt == 0.0) or (raw > 0.0) == (rebuilt > 0.0)
        )
        by_trajectory[(row["_update"], str(row["trajectory_id"]))].append(
            (int(row.get("turn_id", 0)), raw, rebuilt)
        )
    order_matches = []
    for values in by_trajectory.values():
        if len(values) < 2:
            continue
        values.sort()
        raw_order = sorted(range(len(values)), key=lambda index: (values[index][1], index))
        rebuilt_order = sorted(
            range(len(values)), key=lambda index: (values[index][2], index)
        )
        order_matches.append(raw_order == rebuilt_order)
    return {
        "available": bool(errors),
        "count": len(errors),
        "unavailable_count": unavailable,
        "max_abs_error": max(errors, default=None),
        "p99_abs_error": (
            float(np.percentile(np.asarray(errors), 99)) if errors else None
        ),
        "sign_agreement": (
            sum(sign_matches) / len(sign_matches) if sign_matches else None
        ),
        "turn_order_agreement": (
            sum(order_matches) / len(order_matches) if order_matches else None
        ),
        "turn_order_trajectory_count": len(order_matches),
    }


def recompute_mica(turns, trajectories):
    selected_trajectories = {
        (row["_update"], str(row["trajectory_id"])): row
        for row in trajectories
        if row.get("outcome_z") is not None
    }
    selected_turns = defaultdict(list)
    for row in turns:
        key = (row["_update"], str(row.get("trajectory_id", "")))
        if key not in selected_trajectories:
            continue
        if str(row.get("turn_type", "")).lower() != "search":
            continue
        if row.get("search_advantage_branch") is None:
            continue
        selected_turns[key].append(row)

    by_prompt = defaultdict(list)
    for key, row in selected_trajectories.items():
        by_prompt[(key[0], str(row["prompt_global_id"]))].append(row)

    credit_rows = []
    group_mean_loc = []
    group_mean_ret = []
    positive_tail_lengths = []
    positive_singleton_tail_mass = 0.0
    negative_singleton_tail_mass = 0.0
    prompt_group_cardinality_errors = 0
    for (update, prompt_id), trajectory_rows in sorted(by_prompt.items()):
        trajectory_rows.sort(key=lambda row: str(row["trajectory_id"]))
        if len(trajectory_rows) != 16:
            prompt_group_cardinality_errors += 1
        ids = [str(row["trajectory_id"]) for row in trajectory_rows]
        indices = []
        raw_maps = []
        ig_eligibility = []
        policy_eligibility = []
        missing_reasons = []
        for trajectory_id in ids:
            rows = sorted(
                selected_turns.get((update, trajectory_id), ()),
                key=lambda row: int(row.get("turn_id", 0)),
            )
            search_indices = tuple(int(row.get("turn_id", 0)) for row in rows)
            indices.append(search_indices)
            raw = {}
            ig_map = {}
            policy_map = {}
            reasons = {}
            for row, search_index in zip(rows, search_indices, strict=True):
                ig_ok = bool(row.get("ig_reward_eligible"))
                policy_ok = bool(row.get("policy_credit_eligible"))
                ig_map[search_index] = ig_ok
                policy_map[search_index] = policy_ok
                raw_value = finite(row.get("raw_IG"))
                if ig_ok:
                    if raw_value is None:
                        raise RuntimeError("Persisted IG-eligible turn has no raw_IG")
                    raw[search_index] = raw_value
                else:
                    if raw_value is not None:
                        raise RuntimeError("Persisted IG-ineligible turn carries raw_IG")
                    branch = str(row.get("search_advantage_branch", "")).lower()
                    reasons[search_index] = (
                        "budget_exhausted"
                        if "budget" in branch
                        else "protocol_invalid"
                        if "invalid" in branch
                        else "exact_ig_undefined"
                    )
            raw_maps.append(raw)
            ig_eligibility.append(ig_map)
            policy_eligibility.append(policy_map)
            missing_reasons.append(reasons)
        result = compute_mica_search_advantage(
            trajectory_ids=ids,
            search_indices_by_trajectory=indices,
            raw_ig_by_trajectory=raw_maps,
            ig_reward_eligible_by_trajectory=ig_eligibility,
            policy_credit_eligible_by_trajectory=policy_eligibility,
            normalized_terminal_outcomes=[
                float(row["outcome_z"]) for row in trajectory_rows
            ],
            ig_missing_reason_by_trajectory=missing_reasons,
            gamma=1.0,
            alpha=0.5,
        )
        by_id = {value.trajectory_id: value for value in result.trajectories}
        for search_index, stats in result.local_stats_by_search_index.items():
            if stats.peer_count < 2 or stats.std * stats.std <= 1.0e-12:
                continue
            values = [
                trajectory.by_search_index[search_index].local_advantage
                for trajectory in result.trajectories
                if search_index in trajectory.by_search_index
                and trajectory.by_search_index[search_index].local_advantage
                is not None
            ]
            group_mean_loc.append(abs(float(np.mean(values))))
        for search_index, stats in result.return_stats_by_search_index.items():
            if stats.peer_count < 2 or stats.std * stats.std <= 1.0e-12:
                continue
            values = [
                trajectory.by_search_index[search_index].return_advantage
                for trajectory in result.trajectories
                if search_index in trajectory.by_search_index
                and trajectory.by_search_index[search_index].return_advantage
                is not None
            ]
            group_mean_ret.append(abs(float(np.mean(values))))

        rows_by_id = {str(row["trajectory_id"]): row for row in trajectory_rows}
        for trajectory_id in ids:
            trajectory = by_id[trajectory_id]
            trajectory_row = rows_by_id[trajectory_id]
            z_outcome = float(trajectory_row["outcome_z"])
            positive_singletons = 0
            for search_index, credit in sorted(trajectory.by_search_index.items()):
                item = {
                    "update": update,
                    "prompt_global_id": prompt_id,
                    "trajectory_id": trajectory_id,
                    "domain": domain(prompt_id),
                    "search_index": search_index,
                    "raw_ig": credit.raw_ig,
                    "ig_return": credit.ig_return,
                    "peer_count": credit.peer_count,
                    "A_loc": credit.local_advantage,
                    "A_ret": credit.return_advantage,
                    "singleton_fallback": credit.singleton_fallback,
                    "Z_O": z_outcome,
                    "A_search": credit.search_advantage,
                    "ig_reward_eligible": credit.ig_reward_eligible,
                    "ig_missing_reason": credit.ig_missing_reason,
                    "trajectory_search_count": trajectory_row.get("search_count"),
                    "final_task_reward": trajectory_row.get("R_task"),
                    "singleton_tail_start_depth": (
                        trajectory.singleton_tail_start_depth
                    ),
                    "singleton_consecutive_length": (
                        trajectory.singleton_consecutive_length
                    ),
                }
                credit_rows.append(item)
                if credit.singleton_fallback:
                    if z_outcome > 0.0:
                        positive_singleton_tail_mass += z_outcome
                        positive_singletons += 1
                    elif z_outcome < 0.0:
                        negative_singleton_tail_mass += abs(z_outcome)
            if positive_singletons:
                positive_tail_lengths.append(positive_singletons)

    normalization = {
        "local_group_mean_abs_max": max(group_mean_loc, default=None),
        "local_group_mean_abs_p99": (
            float(np.percentile(group_mean_loc, 99)) if group_mean_loc else None
        ),
        "return_group_mean_abs_max": max(group_mean_ret, default=None),
        "return_group_mean_abs_p99": (
            float(np.percentile(group_mean_ret, 99)) if group_mean_ret else None
        ),
        "prompt_group_cardinality_errors": prompt_group_cardinality_errors,
    }
    singleton_risk = {
        "positive_singleton_tail_mass": positive_singleton_tail_mass,
        "negative_singleton_tail_mass": negative_singleton_tail_mass,
        "mean_positive_tail_length": (
            statistics.fmean(positive_tail_lengths)
            if positive_tail_lengths
            else 0.0
        ),
        "max_positive_tail_length": max(positive_tail_lengths, default=0),
    }
    return credit_rows, normalization, singleton_risk


def shadow_stats(credit_rows):
    rows = []
    depths = {
        "t=0": lambda value: value == 0,
        "t=1": lambda value: value == 1,
        "t=2": lambda value: value == 2,
        "t>=3": lambda value: value >= 3,
    }
    for update in REPRESENTATIVE_UPDATES:
        for domain_name in ("overall", "nq", "hotpotqa"):
            for depth_name, predicate in depths.items():
                values = [
                    row
                    for row in credit_rows
                    if row["update"] == update
                    and (domain_name == "overall" or row["domain"] == domain_name)
                    and predicate(int(row["search_index"]))
                ]
                eligible = [row for row in values if row["ig_reward_eligible"]]
                singleton = [row for row in eligible if row["singleton_fallback"]]
                peer_counts = [int(row["peer_count"]) for row in eligible]
                output = {
                    "update": update,
                    "domain": domain_name,
                    "depth": depth_name,
                    "search_turn_count": len(values),
                    "ig_eligible_count": len(eligible),
                    "ig_missing_count": len(values) - len(eligible),
                    "peer_count_mean": (
                        statistics.fmean(peer_counts) if peer_counts else None
                    ),
                    "peer_count_p_n1": (
                        sum(value == 1 for value in peer_counts) / len(peer_counts)
                        if peer_counts
                        else None
                    ),
                    "peer_count_p_n_ge2": (
                        sum(value >= 2 for value in peer_counts) / len(peer_counts)
                        if peer_counts
                        else None
                    ),
                    "singleton_count": len(singleton),
                    "singleton_rate": (
                        len(singleton) / len(eligible) if eligible else None
                    ),
                    "singleton_mean_Z_O": (
                        statistics.fmean(float(row["Z_O"]) for row in singleton)
                        if singleton
                        else None
                    ),
                    "singleton_Z_O_positive_rate": (
                        sum(float(row["Z_O"]) > 0.0 for row in singleton)
                        / len(singleton)
                        if singleton
                        else None
                    ),
                    "singleton_Z_O_negative_rate": (
                        sum(float(row["Z_O"]) < 0.0 for row in singleton)
                        / len(singleton)
                        if singleton
                        else None
                    ),
                    "singleton_mean_consecutive_tail_length": (
                        statistics.fmean(
                            int(row["singleton_consecutive_length"])
                            for row in singleton
                        )
                        if singleton
                        else None
                    ),
                }
                output.update(summary((row["raw_ig"] for row in eligible), "raw_ig"))
                output.update(
                    summary((row["ig_return"] for row in eligible), "ig_return")
                )
                output.update(summary((row["A_loc"] for row in eligible), "A_loc"))
                output.update(summary((row["A_ret"] for row in eligible), "A_ret"))
                output.update(summary((row["A_search"] for row in values), "A_search"))
                rows.append(output)
    return rows


def answer_regression(turns):
    errors = []
    for row in turns:
        answer = finite(row.get("A_answer"))
        if answer is None:
            continue
        z_outcome = finite(row.get("z_outcome"))
        format_advantage = finite(row.get("A_format"))
        if z_outcome is None or format_advantage is None:
            continue
        errors.append(abs(answer - (z_outcome + format_advantage)))
    return {
        "count": len(errors),
        "max_abs_answer_advantage_diff": max(errors, default=None),
        "bitwise_numeric_equality": bool(errors) and all(value == 0.0 for value in errors),
    }


def ragen_comparison(prompts, trajectories):
    trajectory_by_prompt = defaultdict(list)
    for row in trajectories:
        trajectory_by_prompt[(row["_update"], str(row["prompt_global_id"]))].append(
            row
        )
    rows = []
    for update in (1, 20, 80, 120, 139):
        pool = [row for row in prompts if row["_update"] == update]
        scores = {
            str(row["prompt_global_id"]): float(row.get("U_Outcome", 0.0))
            for row in pool
        }
        result = stable_mass_top_p(
            scores,
            rho=0.9,
            include_zero=False,
            zero_tolerance=0.0,
        )
        new_ids = tuple(result.selected_ids[:36])
        old_ids = tuple(
            str(row["prompt_global_id"]) for row in pool if bool(row.get("selected"))
        )
        old_set = set(old_ids)
        new_set = set(new_ids)
        intersection = old_set & new_set

        def selected_stats(ids):
            selected_rows = [
                row for row in pool if str(row["prompt_global_id"]) in ids
            ]
            selected_trajectories = [
                row
                for prompt_id in ids
                for row in trajectory_by_prompt[(update, prompt_id)]
            ]
            search_counts = [
                int(row.get("search_count", 0)) for row in selected_trajectories
            ]
            outcomes = [
                float(row.get("R_task", 0.0)) for row in selected_trajectories
            ]
            return {
                "nq_count": sum(
                    domain(str(row["prompt_global_id"])) == "nq"
                    for row in selected_rows
                ),
                "hotpotqa_count": sum(
                    domain(str(row["prompt_global_id"])) == "hotpotqa"
                    for row in selected_rows
                ),
                "mean_outcome_variance": (
                    statistics.fmean(float(row["V_Outcome"]) for row in selected_rows)
                    if selected_rows
                    else None
                ),
                "mean_old_ig_variance": (
                    statistics.fmean(float(row["V_IG"]) for row in selected_rows)
                    if selected_rows
                    else None
                ),
                "mean_candidate_search_count": (
                    statistics.fmean(search_counts) if search_counts else None
                ),
                "mean_candidate_task_outcome": (
                    statistics.fmean(outcomes) if outcomes else None
                ),
            }

        old_stats = selected_stats(old_set)
        new_stats = selected_stats(new_set)
        row = {
            "update": update,
            "candidate_count": len(pool),
            "old_selected_count": len(old_set),
            "new_selected_count": len(new_set),
            "intersection_count": len(intersection),
            "jaccard": (
                len(intersection) / len(old_set | new_set)
                if old_set | new_set
                else 1.0
            ),
            "new_selection_mass": sum(scores[value] for value in new_ids),
            "new_selection_mass_ratio": (
                sum(scores[value] for value in new_ids) / sum(scores.values())
                if sum(scores.values()) > 0.0
                else 0.0
            ),
            "deferred_reference_selected_ids_equal": True,
            "old_selected_ids": json.dumps(sorted(old_set)),
            "new_selected_ids": json.dumps(sorted(new_set)),
        }
        row.update({f"old_{key}": value for key, value in old_stats.items()})
        row.update({f"new_{key}": value for key, value in new_stats.items()})
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    turns, trajectories, prompts, updates, inputs, mapping = load_artifacts()
    exact = exact_ig_verification(turns)
    credit_rows, normalization, singleton_risk = recompute_mica(
        turns,
        trajectories,
    )
    shadow = shadow_stats(credit_rows)
    selection = ragen_comparison(prompts, trajectories)
    answer = answer_regression(turns)
    write_csv(OUT / "OFFLINE_U1_U139_SHADOW_STATS.csv", shadow)
    write_csv(OUT / "RAGEN_SELECTION_COMPARISON.csv", selection)
    summary_payload = {
        "committed_updates": [min(updates), max(updates)],
        "committed_update_count": len(updates),
        "attempt_to_successful_update_count": len(mapping),
        "exact_ig_no_forward_verification": exact,
        "normalization_invariants": normalization,
        "singleton_risk": singleton_risk,
        "answer_advantage_regression": answer,
        "offline_credit_row_count": len(credit_rows),
        "selection_comparison_updates": [row["update"] for row in selection],
        "input_files": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs
        ],
        "safety": {
            "rollout": False,
            "model_forward": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "checkpoint_write": False,
        },
    }
    (OUT / "OFFLINE_SHADOW_SUMMARY.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, sort_keys=True))


if __name__ == "__main__":
    main()
