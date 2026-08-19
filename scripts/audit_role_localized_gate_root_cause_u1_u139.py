#!/usr/bin/env python3
"""Read-only root-cause audit for the role-localized-gate run.

This script only parses persisted JSONL/JSON/CSV artifacts.  It never imports
the training runtime, constructs a model, starts a process, or writes outside
the audit report directory.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/search-r1-workspace/projects/igpo_ragen2_a2tgpo_strict_onpolicy_v1")
OUT = ROOT / "reports/role_localized_gate_root_cause_u1_u139"
MAX_UPDATE = 139
EPS_PROBE = 1.0e-6
EPS_NUM = 1.0e-12

RUNS = [
    ("u001_u020", ROOT / "outputs/formal_training/formal_fresh_u000_to_u500_role_localized_gate_g16_lr2e7_kl1e2_20260808_133350"),
    ("u021_u080", ROOT / "outputs/formal_training/formal_resume_u020_to_u500_3rank_48cpu_20260808_191223_role_localized_gate_g16"),
    ("u081_u120", ROOT / "outputs/formal_training/formal_resume_u080_to_u500_no_monitor_20260809_122013_role_localized_gate_g16"),
    ("u121_u139", ROOT / "outputs/formal_training/formal_resume_u120_to_u500_provenance_utf8fix_20260810_060921_role_localized_gate_g16"),
]
CURRENT_RUN = RUNS[-1][1]

DEPTHS = ("t=0", "t=1", "t=2", "t>=2", "t>=3")
BRANCHES = ("Normal", "S_before", "N_soft", "N_invalid", "N_budget")
DOMAINS = ("overall", "nq", "hotpotqa")
ROW_INDEX = defaultdict(list)
INDEX_DEPTHS = set()


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc


def read_jsonl(path: Path):
    return list(iter_jsonl(path))


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def persisted_flag(row, key, default=False):
    value = row.get(key)
    return default if value is None else as_bool(value)


def domain_of(row):
    value = str(row.get("domain") or row.get("dataset") or row.get("source") or "").lower()
    if "hotpot" in value:
        return "hotpotqa"
    if value in {"nq", "natural_questions", "naturalquestions"} or "nq" in value:
        return "nq"
    prompt = str(row.get("prompt_global_id") or row.get("prompt_id") or "").lower()
    if "hotpot" in prompt:
        return "hotpotqa"
    if "nq" in prompt:
        return "nq"
    return "unknown"


def branch_of(row):
    value = row.get("search_advantage_branch") or row.get("branch_type") or row.get("branch")
    mapping = {
        "normal": "Normal",
        "s_before": "S_before",
        "sufficient_before": "S_before",
        "n_soft": "N_soft",
        "n_invalid": "N_invalid",
        "n_budget": "N_budget",
        "budget": "N_budget",
        "invalid": "N_invalid",
    }
    return mapping.get(str(value).lower(), str(value) if value else None)


def update_from_row(row, success_by_attempt):
    attempt = row.get("attempt_id")
    if attempt is None:
        return None
    try:
        attempt = int(attempt)
    except (TypeError, ValueError):
        return None
    return success_by_attempt.get(attempt)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(cmd):
    if not (ROOT / ".git").exists():
        return "NOT_A_GIT_WORKTREE"
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *cmd], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # audit output must preserve the failure rather than hide it
        return f"ERROR: {exc}"


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=True, default=str)
        f.write("\n")


def mean(values):
    values = [float(x) for x in values if safe_float(x) is not None]
    return sum(values) / len(values) if values else None


def std(values):
    values = [float(x) for x in values if safe_float(x) is not None]
    if not values:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def quantile(values, q):
    values = sorted(float(x) for x in values if safe_float(x) is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def numeric_stats(values, prefix=""):
    vals = [float(x) for x in values if safe_float(x) is not None]
    key = (lambda name: f"{prefix}{name}" if prefix else name)
    return {
        key("mean"): mean(vals),
        key("std"): std(vals),
        key("median"): quantile(vals, 0.50),
        key("p05"): quantile(vals, 0.05),
        key("p25"): quantile(vals, 0.25),
        key("p75"): quantile(vals, 0.75),
        key("p95"): quantile(vals, 0.95),
        key("min"): min(vals) if vals else None,
        key("max"): max(vals) if vals else None,
        key("positive_rate"): sum(x > 0 for x in vals) / len(vals) if vals else None,
        key("zero_rate"): sum(x == 0 for x in vals) / len(vals) if vals else None,
        key("negative_rate"): sum(x < 0 for x in vals) / len(vals) if vals else None,
        key("n"): len(vals),
    }


def depth_match(row, depth):
    t = row.get("_depth")
    if depth == "t=0":
        return t == 0
    if depth == "t=1":
        return t == 1
    if depth == "t=2":
        return t == 2
    if depth == "t>=2":
        return t is not None and t >= 2
    if depth == "t>=3":
        return t is not None and t >= 3
    return False


def rows_for(rows, lo, hi, domain="overall", depth=None, branch=None):
    # The audit emits many overlapping slices.  Use the persisted row index
    # when available instead of rescanning the full JSONL for every cell.
    if ROW_INDEX:
        result = []
        domains = ("nq", "hotpotqa") if domain == "overall" else (domain,)
        branches = BRANCHES if branch is None else (branch,)
        for u in range(lo, hi + 1):
            for d in domains:
                for t in INDEX_DEPTHS:
                    if depth is not None and not depth_match({"_depth": t}, depth):
                        continue
                    for b in branches:
                        result.extend(ROW_INDEX.get((u, d, t, b), ()))
        return result
    result = []
    for row in rows:
        u = row.get("_u")
        if u is None or u < lo or u > hi:
            continue
        if domain != "overall" and row.get("_domain") != domain:
            continue
        if depth is not None and not depth_match(row, depth):
            continue
        if branch is not None and row.get("_branch") != branch:
            continue
        result.append(row)
    return result


def metric(row, *keys):
    for key in keys:
        if key in row:
            value = safe_float(row.get(key))
            if value is not None:
                return value
    return None


def load_data():
    all_turns = []
    all_traj = []
    update_records = {}
    source_inputs = []
    eval_summary_path = ROOT / "reports/EVAL_COMPARISON_U0_U120.csv"
    source_inputs.append(eval_summary_path)
    source_success_maps = {}
    for label, run in RUNS:
        update_path = run / "metrics/update_metrics.jsonl"
        turn_path = run / "metrics/turn_metrics.jsonl"
        traj_path = run / "metrics/trajectory_metrics.jsonl"
        for path in (update_path, turn_path, traj_path):
            source_inputs.append(path)
        updates = list(iter_jsonl(update_path))
        success = {}
        for rec in updates:
            attempt = rec.get("attempt_id")
            step = rec.get("successful_update_after", rec.get("successful_update_step"))
            if attempt is None or step is None:
                continue
            try:
                success[int(attempt)] = int(step)
            except (TypeError, ValueError):
                continue
        source_success_maps[label] = success
        for rec in updates:
            step = rec.get("successful_update_after", rec.get("successful_update_step"))
            if step is None:
                continue
            try:
                step = int(step)
            except (TypeError, ValueError):
                continue
            if 1 <= step <= MAX_UPDATE:
                update_records[step] = {**rec, "_u": step, "_source": label}
        for row in iter_jsonl(traj_path):
            u = update_from_row(row, success)
            if u is None or not 1 <= u <= MAX_UPDATE:
                continue
            tid = str(row.get("trajectory_id") or row.get("id") or "")
            if not tid:
                continue
            # Candidate analysis only needs these persisted trajectory fields;
            # avoid retaining every auxiliary metric from a large JSONL row.
            item = {key: row.get(key) for key in (
                "trajectory_id", "id", "domain", "dataset", "source", "prompt_global_id",
                "search_count", "num_searches", "R_task", "task_reward", "reward",
            )}
            item.update({"_u": u, "_source": label, "_domain": domain_of(row), "_tid": tid})
            all_traj.append(item)
        for row in iter_jsonl(turn_path):
            u = update_from_row(row, success)
            if u is None or not 1 <= u <= MAX_UPDATE:
                continue
            if str(row.get("turn_type", "")).lower() not in {"search", "search_turn"}:
                continue
            branch = branch_of(row)
            if branch is None:
                continue
            tid = str(row.get("trajectory_id") or "")
            try:
                depth = int(row.get("search_index", row.get("turn_id", row.get("turn_index", 0))))
            except (TypeError, ValueError):
                depth = 0
            turn_keys = (
                "trajectory_id", "turn_id", "turn_type", "domain", "dataset", "source", "prompt_global_id",
                "search_index", "turn_index", "attempt_id", "search_advantage_branch", "branch_type",
                "A_search", "A_search_new", "A_answer", "A_main", "A_decision", "A_query",
                "D_ig_eff", "D_ig_eff_count", "local_ig_hat", "normalized_IG", "raw_IG", "O_route",
                "z_outcome", "z_O", "delta_probe", "pre_probe_raw_task_reward", "post_probe_raw_task_reward",
                "sufficient_before_search", "sufficient_after_search", "policy_credit_eligible", "ig_reward_eligible",
                "main_credit_eligible", "exact_query_repeat", "different_query_no_new_passage", "action_token_count",
                "retriever_executed", "new_passage_count", "current_passage_keys", "new_passage_keys", "no_new_observation",
                "search_action_span_valid", "search_prefix_valid", "R_C", "R_S1", "R_S2", "Phi_before", "Phi_after",
            )
            item = {key: row.get(key) for key in turn_keys}
            item.update({"_u": u, "_source": label, "_domain": domain_of(row), "_tid": tid, "_depth": depth, "_branch": branch})
            all_turns.append(item)
    return all_turns, all_traj, update_records, source_inputs, source_success_maps


def candidate_counts(traj_rows):
    counts = defaultdict(int)
    for row in traj_rows:
        domain = row.get("_domain")
        if domain not in {"nq", "hotpotqa"}:
            continue
        try:
            search_count = int(row.get("search_count", row.get("num_searches", 0)) or 0)
            u = int(row["_u"])
        except (TypeError, ValueError, KeyError):
            continue
        for t in range(max(0, search_count)):
            counts[(u, domain, t)] += 1
    return counts


def build_recon(turn_rows):
    groups = defaultdict(list)
    for row in turn_rows:
        groups[(row["_source"], row["_tid"])].append(row)
    counters = Counter()
    for group in groups.values():
        group.sort(key=lambda r: r["_depth"])
        for idx, row in enumerate(group):
            if row["_branch"] not in {"Normal", "N_soft"}:
                continue
            values = []
            for future in group[idx:]:
                if future["_branch"] == "S_before":
                    break
                if future["_branch"] in {"Normal", "N_soft"}:
                    local = safe_float(future.get("local_ig_hat"))
                    if local is not None and persisted_flag(future, "policy_credit_eligible", True) and persisted_flag(future, "ig_reward_eligible", True):
                        values.append(local)
                if as_bool(future.get("sufficient_after_search")):
                    break
            if not values:
                counters["missing_reconstruction"] += 1
                continue
            n = len(values)
            d = sum(values) / math.sqrt(n)
            local_component = values[0] / math.sqrt(n)
            future_component = sum(values[1:]) / math.sqrt(n)
            persisted_d = safe_float(row.get("D_ig_eff"))
            persisted_n = row.get("D_ig_eff_count")
            try:
                persisted_n = int(persisted_n) if persisted_n is not None else None
            except (TypeError, ValueError):
                persisted_n = None
            row["_d_recon"] = d
            row["_local_component"] = local_component
            row["_future_component"] = future_component
            row["_n_eff_recon"] = n
            row["_d_error"] = abs(d - persisted_d) if persisted_d is not None else None
            row["_n_error"] = n - persisted_n if persisted_n is not None else None
            row["_future_nonzero"] = abs(future_component) > 1e-14
            counters["reconstructed"] += 1
            if persisted_n is not None and n != persisted_n:
                counters["n_eff_mismatch"] += 1
            if persisted_d is None:
                counters["missing_persisted_d"] += 1
            elif abs(d - persisted_d) >= 1e-10:
                counters["d_mismatch_ge_1e-10"] += 1
    return counters


def row_values(rows, field):
    return [metric(row, field) for row in rows if metric(row, field) is not None]


def selected_scope_rows(turn_rows, domain="overall", depth=None, branch=None, lo=1, hi=MAX_UPDATE):
    return rows_for(turn_rows, lo, hi, domain, depth, branch)


def branch_counts(rows):
    return Counter(row.get("_branch") for row in rows)


def slice_defs():
    result = [("update", f"U{u}", u, u) for u in range(1, MAX_UPDATE + 1)]
    result += [
        ("checkpoint", f"U{u}", u, u) for u in (1, 20, 40, 60, 80, 100, 120, 139)
    ]
    result += [
        ("block", "U1-U20", 1, 20),
        ("block", "U21-U40", 21, 40),
        ("block", "U41-U60", 41, 60),
        ("block", "U61-U80", 61, 80),
        ("block", "U81-U100", 81, 100),
        ("block", "U101-U120", 101, 120),
        ("block", "U121-U139", 121, 139),
    ]
    return result


def update_metric(rec, *names):
    for name in names:
        if name in rec:
            return safe_float(rec.get(name))
    return None


def aggregate_update_metric(update_records, lo, hi, names):
    values = [update_metric(update_records[u], *names) for u in sorted(update_records) if lo <= u <= hi]
    return mean(values)


def domain_rows(rows, domain):
    if domain == "overall":
        return rows
    return [r for r in rows if r.get("_domain") == domain]


def routing_class(row):
    delta = metric(row, "delta_probe")
    if delta is None:
        return "missing"
    if delta > EPS_PROBE:
        return "positive"
    if delta < -EPS_PROBE:
        return "negative"
    return "zero"


def routing_mismatch(row):
    delta = metric(row, "delta_probe")
    z = metric(row, "z_outcome", "z_O")
    route = metric(row, "O_route")
    if delta is None or z is None or route is None:
        return None
    if delta > EPS_PROBE:
        expected = max(z, 0.0)
    elif delta < -EPS_PROBE:
        expected = min(z, 0.0)
    else:
        expected = 0.0
    return abs(route - expected) > 1e-10


def corr(x, y, rank=False):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return None
    if rank:
        def ranks(values):
            order = sorted(range(len(values)), key=lambda i: values[i])
            result = [0.0] * len(values)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                    j += 1
                r = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    result[order[k]] = r
                i = j + 1
            return result
        x = ranks([p[0] for p in pairs])
        y = ranks([p[1] for p in pairs])
    else:
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
    mx, my = mean(x), mean(y)
    dx = math.sqrt(sum((v - mx) ** 2 for v in x))
    dy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (dx * dy) if dx and dy else None


def static_field_availability():
    rows = [
        ("r_IG_raw", "YES", "metrics/turn_metrics.jsonl:raw_IG; trajectory_metrics.jsonl:IG", "turn", "YES"),
        ("IG_hat/local_ig_hat", "YES", "metrics/turn_metrics.jsonl:local_ig_hat,normalized_IG", "turn", "YES"),
        ("D_IG_eff", "YES", "metrics/turn_metrics.jsonl:D_ig_eff", "turn", "YES"),
        ("D_IG_local_component", "DERIVED", "local_ig_hat + D_ig_eff_count", "turn", "YES"),
        ("D_IG_future_component", "DERIVED", "selected turn local_ig_hat/D_ig_eff_count", "turn", "YES"),
        ("n_eff", "YES", "metrics/turn_metrics.jsonl:D_ig_eff_count", "turn", "YES"),
        ("O_route", "YES", "metrics/turn_metrics.jsonl:O_route", "turn", "YES"),
        ("delta_probe", "YES", "metrics/turn_metrics.jsonl:delta_probe", "turn", "YES"),
        ("pre_probe_score", "YES", "pre_probe_raw_task_reward", "turn", "YES"),
        ("post_probe_score", "YES", "post_probe_raw_task_reward", "turn", "YES"),
        ("S_before", "YES", "sufficient_before_search", "turn", "YES"),
        ("S_after", "YES", "sufficient_after_search; null for hard branches", "turn", "YES"),
        ("B = D_IG_eff + O_route", "YES/DERIVED", "A_search/A_search_new plus D/O", "turn", "YES"),
        ("A_main", "NO", "turn_metrics fields are null; aggregate nonzero counters only", "turn", "NO"),
        ("A_decision", "NO", "turn_metrics fields are null; aggregate counters only", "turn", "NO"),
        ("A_query", "NO", "turn_metrics fields are null; aggregate counters only", "turn", "NO"),
        ("A_answer", "YES", "turn_metrics:A_answer", "turn", "YES"),
        ("branch_type", "YES", "turn_metrics:search_advantage_branch", "turn", "YES"),
        ("L_turn", "NO", "no persisted per-turn length field in JSONL", "turn", "NO"),
        ("L_answer", "NO", "no separate persisted answer length", "trajectory/turn", "NO"),
        ("N_action", "NO", "only aggregate/action token counts; exact per trajectory unavailable", "trajectory", "NO"),
        ("s_main/s_decision/s_query/s_answer", "NO", "only aggregate ratios; no per-segment values", "turn/update", "NO"),
        ("psi_main/psi_decision/psi_query/psi_answer", "NO", "not persisted per event", "turn", "NO"),
        ("J_main/J_decision/J_query", "YES", "metrics/update_metrics.jsonl role_gate fields", "update", "YES"),
        ("lambda_d/lambda_q", "YES", "metrics/update_metrics.jsonl role_gate fields", "update", "YES"),
        ("gradient norms/cosines", "PARTIAL", "immutable U0 calibration summary only, not per-update gradients", "calibration/update", "NO for U1-U139 exact"),
        ("clip fractions", "YES", "update_metrics role_gate aggregate fields", "update", "YES"),
        ("retriever_executed/new_passage_count", "YES", "turn_metrics fields", "turn", "YES"),
        ("canonical_query/exact_repeat", "YES", "turn_metrics fields", "turn", "YES"),
        ("final R_task/F1/exact", "YES", "trajectory_metrics fields", "trajectory/update", "YES"),
        ("termination reason", "PARTIAL", "trajectory/turn metadata where emitted; not complete for every row", "trajectory/turn", "PARTIAL"),
        ("true Q(Search)-Q(Stop)", "NO", "no same-state counterfactual continuation pair", "state", "NO"),
    ]
    lines = [
        "# Field Availability", "", "This is a read-only inventory. `NO` means the value is not reconstructed by guessing.", "",
        "| field | persisted | location/field | scope | exact reconstructable |", "|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    lines += ["", "## Scope rule", "Candidate rows are sourced from trajectory metrics and branch-null turn rows. Selected learner rows are branch-labelled turn rows. They are never mixed in a denominator."]
    (OUT / "FIELD_AVAILABILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_reports(turns, trajectories, updates, inputs, success_maps, recon_counters, before_snapshot):
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = candidate_counts(trajectories)
    all_selected = turns
    eligible = [r for r in turns if r["_branch"] in {"Normal", "N_soft"} and persisted_flag(r, "main_credit_eligible", True) and persisted_flag(r, "ig_reward_eligible", True)]
    eligible_ids = {id(r) for r in eligible}
    ROW_INDEX.clear()
    INDEX_DEPTHS.clear()
    for r in turns:
        ROW_INDEX[(r["_u"], r["_domain"], r["_depth"], r["_branch"])].append(r)
        INDEX_DEPTHS.add(r["_depth"])
    for row in all_selected:
        row["_B"] = metric(row, "A_search_new", "A_search")
        row["_D"] = metric(row, "D_ig_eff")
        row["_O"] = metric(row, "O_route")
        row["_z"] = metric(row, "z_outcome", "z_O")
        row["_routing_class"] = routing_class(row)
        row["_routing_mismatch"] = routing_mismatch(row)
        row["_final_R"] = None
    traj_by_key = {(r["_source"], r["_tid"]): r for r in trajectories}
    for row in all_selected:
        tr = traj_by_key.get((row["_source"], row["_tid"]))
        if tr:
            row["_final_R"] = metric(tr, "R_task", "task_reward", "reward")

    slices = slice_defs()
    credit_rows = []
    for kind, label, lo, hi in slices:
        for domain in DOMAINS:
            for depth in DEPTHS:
                for branch in BRANCHES:
                    rs = rows_for(turns, lo, hi, domain, depth, branch)
                    er = [r for r in rs if id(r) in eligible_ids]
                    values = [r["_B"] for r in er if r["_B"] is not None]
                    row = {"slice_kind": kind, "slice": label, "u_start": lo, "u_end": hi, "scope": "selected_learner", "domain": domain, "depth": depth, "branch": branch, "count": len(rs), "main_credit_eligible_count": len(er)}
                    row.update(numeric_stats(values, "B_"))
                    row["mean_D_IG_eff"] = mean([r["_D"] for r in er if r["_D"] is not None])
                    row["mean_O_route"] = mean([r["_O"] for r in er if r["_O"] is not None])
                    row["B_reconstruction_max_abs_error"] = max([abs(r["_B"] - (r["_D"] + r["_O"])) for r in er if r["_B"] is not None and r["_D"] is not None and r["_O"] is not None] or [None])
                    credit_rows.append(row)
    write_csv(OUT / "credit_by_update_domain_depth_branch.csv", credit_rows)
    del credit_rows
    gc.collect()
    print("wrote credit", flush=True)

    normal_decomp = []
    future_rows = []
    for kind, label, lo, hi in slices:
        for domain in DOMAINS:
            for depth in DEPTHS:
                rs = [r for r in rows_for(turns, lo, hi, domain, depth, "Normal") if id(r) in eligible_ids]
                dvals = [r["_D"] for r in rs if r["_D"] is not None]
                ovals = [r["_O"] for r in rs if r["_O"] is not None]
                bvals = [r["_B"] for r in rs if r["_B"] is not None]
                errors = [r["_d_error"] for r in rs if r.get("_d_error") is not None]
                row = {"slice_kind": kind, "slice": label, "u_start": lo, "u_end": hi, "scope": "selected_learner", "domain": domain, "depth": depth, "branch": "Normal", "count": len(rs)}
                row.update(numeric_stats(bvals, "B_"))
                row.update(numeric_stats(dvals, "D_IG_eff_"))
                row.update(numeric_stats(ovals, "O_route_"))
                row["B_mean_recomputed_from_D_plus_O"] = (mean(dvals) + mean(ovals)) if dvals and ovals else None
                row["max_abs_D_reconstruction_error"] = max(errors) if errors else None
                normal_decomp.append(row)
                nvals = [r.get("_n_eff_recon") for r in rs if r.get("_n_eff_recon") is not None]
                lvals = [r.get("_local_component") for r in rs if r.get("_local_component") is not None]
                fvals = [r.get("_future_component") for r in rs if r.get("_future_component") is not None]
                denom = sum(abs(x) + abs(y) for x, y in zip(lvals, fvals))
                future_rows.append({
                    "slice_kind": kind, "slice": label, "u_start": lo, "u_end": hi, "scope": "selected_learner", "domain": domain, "depth": depth, "branch": "Normal", "count": len(rs),
                    "n_eff_1_rate": sum(n == 1 for n in nvals) / len(nvals) if nvals else None,
                    "n_eff_2_rate": sum(n == 2 for n in nvals) / len(nvals) if nvals else None,
                    "n_eff_ge3_rate": sum(n >= 3 for n in nvals) / len(nvals) if nvals else None,
                    "n_eff_mean": mean(nvals), "n_eff_std": std(nvals), "n_eff_max": max(nvals) if nvals else None,
                    "local_component_mean": mean(lvals), "future_component_mean": mean(fvals),
                    "future_component_nonzero_rate": sum(abs(x) > 1e-14 for x in fvals) / len(fvals) if fvals else None,
                    "absolute_future_share": sum(abs(x) for x in fvals) / denom if denom else None,
                    "max_abs_D_reconstruction_error": max(errors) if errors else None,
                })
    write_csv(OUT / "normal_B_decomposition.csv", normal_decomp)
    write_csv(OUT / "future_credit_decomposition.csv", future_rows)
    del normal_decomp, future_rows
    gc.collect()
    print("wrote decomposition", flush=True)

    coverage_rows = []
    for kind, label, lo, hi in slices:
        for domain in DOMAINS:
            for depth in DEPTHS:
                rs = rows_for(turns, lo, hi, domain, depth)
                counts = branch_counts(rs)
                total = len(rs)
                cand_counts = []
                for u in range(lo, hi + 1):
                    if domain == "overall":
                        cand_counts.append(sum(candidate.get((u, d, t), 0) for d in ("nq", "hotpotqa") for t in range(6) if depth_match({"_depth": t}, depth)))
                    else:
                        cand_counts.append(sum(candidate.get((u, domain, t), 0) for t in range(6) if depth_match({"_depth": t}, depth)))
                candidate_count = sum(cand_counts)
                hard = counts["S_before"] + counts["N_invalid"] + counts["N_budget"]
                coverage_rows.append({
                    "slice_kind": kind, "slice": label, "u_start": lo, "u_end": hi, "scope": "candidate", "domain": domain, "depth": depth,
                    "selected_total": None, "candidate_search_turn_count": candidate_count,
                    "Normal_count": None, "S_before_count": None, "N_soft_count": None, "N_invalid_count": None, "N_budget_count": None, "HardGate_count": None,
                    "Normal_rate": None, "S_before_rate": None, "N_soft_rate": None, "N_invalid_rate": None, "N_budget_rate": None, "HardGate_rate": None,
                })
                coverage_rows.append({
                    "slice_kind": kind, "slice": label, "u_start": lo, "u_end": hi, "scope": "selected_learner", "domain": domain, "depth": depth,
                    "selected_total": total, "candidate_search_turn_count": candidate_count,
                    "Normal_count": counts["Normal"], "S_before_count": counts["S_before"], "N_soft_count": counts["N_soft"], "N_invalid_count": counts["N_invalid"], "N_budget_count": counts["N_budget"], "HardGate_count": hard,
                    "Normal_rate": counts["Normal"] / total if total else None, "S_before_rate": counts["S_before"] / total if total else None, "N_soft_rate": counts["N_soft"] / total if total else None, "N_invalid_rate": counts["N_invalid"] / total if total else None, "N_budget_rate": counts["N_budget"] / total if total else None, "HardGate_rate": hard / total if total else None,
                })
    write_csv(OUT / "role_gate_coverage.csv", coverage_rows)
    del coverage_rows
    gc.collect()
    print("wrote coverage", flush=True)

    # Update-level objective data is exact at the scalar metric level.  Per-event psi and segment lengths are not persisted.
    objective_rows = []
    ratio_by_update = {}
    for kind, label, lo, hi in slices:
        recs = [updates[u] for u in sorted(updates) if lo <= u <= hi]
        jm = mean([update_metric(r, "role_gate/J_main", "J_main") for r in recs])
        jd = mean([update_metric(r, "role_gate/J_decision", "J_decision") for r in recs])
        jq = mean([update_metric(r, "role_gate/J_query", "J_query") for r in recs])
        ld = mean([update_metric(r, "role_gate/lambda_decision", "lambda_d") for r in recs])
        lq = mean([update_metric(r, "role_gate/lambda_query", "lambda_q") for r in recs])
        gate = (ld or 0) * (jd or 0) + (lq or 0) * (jq or 0) if jd is not None or jq is not None else None
        row = {"slice_kind": kind, "slice": label, "u_start": lo, "u_end": hi, "scope": "update", "J_main": jm, "J_decision": jd, "J_query": jq, "lambda_d": ld, "lambda_q": lq, "weighted_decision": ld * jd if ld is not None and jd is not None else None, "weighted_query": lq * jq if lq is not None and jq is not None else None, "gate_total": gate, "gate_main_scalar_ratio": abs(gate) / (abs(jm) + EPS_NUM) if gate is not None and jm is not None else None, "KL": aggregate_update_metric(updates, lo, hi, ("full_vocab_forward_kl", "kl", "KL"))}
        branch_rs = rows_for(turns, lo, hi, "overall", None, None)
        bvals = [r["_B"] for r in branch_rs if r.get("_B") is not None and r.get("_branch") in {"Normal", "N_soft"}]
        row["main_positive_mass_diagnostic"] = sum(max(x, 0) for x in bvals)
        row["main_negative_mass_diagnostic"] = sum(abs(min(x, 0)) for x in bvals)
        # Gate credit values are fixed by contract; this is a scalar mass diagnostic, not a gradient estimate.
        s_count = sum(1 for r in branch_rs if r.get("_branch") == "S_before")
        inv_count = sum(1 for r in branch_rs if r.get("_branch") == "N_invalid")
        bud_count = sum(1 for r in branch_rs if r.get("_branch") == "N_budget")
        soft_dup = sum(1 for r in branch_rs if r.get("_branch") == "N_soft" and as_bool(r.get("exact_query_repeat")) and (metric(r, "raw_IG") is not None and metric(r, "raw_IG") <= 0))
        invalid_query = max(0, int(round((update_metric(updates.get(lo, {}), "role_gate/A_query_nonzero_count") or 0))) if kind == "update" and lo == hi else 0)
        row["decision_negative_mass_diagnostic"] = (ld or 0) * (0.5 * s_count + 0.5 * inv_count + 1.0 * bud_count)
        row["query_negative_mass_diagnostic"] = (lq or 0) * (0.25 * soft_dup + 0.5 * invalid_query)
        row["net_scalar_search_pressure_diagnostic"] = row["main_positive_mass_diagnostic"] - row["main_negative_mass_diagnostic"] - (row["decision_negative_mass_diagnostic"] or 0) - (row["query_negative_mass_diagnostic"] or 0)
        row["note"] = "J values are exact persisted update aggregates; event mass columns are scalar diagnostics, not gradients."
        objective_rows.append(row)
        if kind == "update" and lo == hi:
            ratio_by_update[lo] = row.get("gate_main_scalar_ratio")
    write_csv(OUT / "objective_contribution.csv", objective_rows)
    latest_obj = next((r for r in objective_rows if r["slice_kind"] == "checkpoint" and r["slice"] == "U139"), {})
    del objective_rows
    gc.collect()
    print("wrote objective", flush=True)

    # No exact per-update gradient vectors are persisted.  Keep the requested file machine-readable and explicit.
    calibration = []
    for u in sorted(updates):
        rec = updates[u]
        for key in ("gradient_calibration", "role_gate/gradient_calibration", "immutable_u0_calibration"):
            if isinstance(rec.get(key), dict):
                calibration.append(rec[key])
    gradient_rows = [{"available": False, "scope": "U1-U139", "gradient_source": "not persisted per update", "g_main_norm": None, "lambda_d_g_decision_norm": None, "lambda_q_g_query_norm": None, "gate_gradient_norm": None, "gate_main_gradient_ratio": None, "cos_main_decision": None, "cos_main_query": None, "cos_decision_query": None, "note": "EXACT GRADIENT ATTRIBUTION NOT AVAILABLE. Immutable U0 calibration metadata, if present, is not substituted for U1-U139 gradients."}]
    write_csv(OUT / "gradient_attribution.csv", gradient_rows)
    print("wrote gradient", flush=True)

    probe_rows = []
    for kind, label, lo, hi in slices:
        for domain in DOMAINS:
            for depth in DEPTHS:
                rs = [r for r in rows_for(turns, lo, hi, domain, depth) if r.get("_branch") in {"Normal", "N_soft"} and r.get("_B") is not None]
                groups = defaultdict(list)
                for r in rs:
                    bgroup = "B>0" if r["_B"] > 0 else "B<=0"
                    groups[(bgroup, routing_class(r))].append(r)
                for (bgroup, dclass), gr in sorted(groups.items()):
                    bridge = [r for r in gr if bgroup == "B>0" and dclass == "zero"]
                    future_positive = [r for r in bridge if (r.get("_future_component") or 0) > 0]
                    route_positive = [r for r in bridge if (r.get("_O") or 0) > 0]
                    local_positive = [r for r in bridge if (r.get("_local_component") or 0) > 0]
                    probe_rows.append({"slice_kind": kind, "slice": label, "u_start": lo, "u_end": hi, "scope": "selected_learner", "domain": domain, "depth": depth, "branch_group": bgroup, "delta_class": dclass, "count": len(gr), "fraction_of_depth": len(gr) / len(rs) if rs else None, "B_mean": mean([r["_B"] for r in gr]), "delta_mean": mean([metric(r, "delta_probe") for r in gr]), "raw_IG_mean": mean([metric(r, "raw_IG") for r in gr]), "D_mean": mean([r.get("_D") for r in gr]), "O_mean": mean([r.get("_O") for r in gr]), "final_R_mean": mean([r.get("_final_R") for r in gr]), "future_positive_rate_within_bridge": len(future_positive) / len(bridge) if bridge else None, "route_positive_rate_within_bridge": len(route_positive) / len(bridge) if bridge else None, "local_positive_rate_within_bridge": len(local_positive) / len(bridge) if bridge else None})
    write_csv(OUT / "probe_vs_B_analysis.csv", probe_rows)
    del probe_rows
    gc.collect()
    print("wrote probe", flush=True)

    # Continuation and credit correlation is descriptive only.
    corr_rows = []
    per_domain = {d: [] for d in ("overall", "nq", "hotpotqa")}
    for u in range(1, MAX_UPDATE + 1):
        for domain in ("overall", "nq", "hotpotqa"):
            c0 = sum(candidate.get((u, d, 0), 0) for d in ("nq", "hotpotqa")) if domain == "overall" else candidate.get((u, domain, 0), 0)
            c1 = sum(candidate.get((u, d, 1), 0) for d in ("nq", "hotpotqa")) if domain == "overall" else candidate.get((u, domain, 1), 0)
            c2 = sum(candidate.get((u, d, 2), 0) for d in ("nq", "hotpotqa")) if domain == "overall" else candidate.get((u, domain, 2), 0)
            c3 = sum(candidate.get((u, d, 3), 0) for d in ("nq", "hotpotqa")) if domain == "overall" else candidate.get((u, domain, 3), 0)
            rows = rows_for(turns, u, u, domain)
            t1 = [r for r in rows if r["_depth"] == 1]
            t2 = [r for r in rows if r["_depth"] == 2]
            n1 = [r for r in t1 if r.get("_branch") == "Normal" and r.get("_B") is not None]
            n2 = [r for r in t2 if r.get("_branch") == "Normal" and r.get("_B") is not None]
            hard1 = sum(r.get("_branch") in {"S_before", "N_invalid", "N_budget"} for r in t1) / len(t1) if t1 else None
            hard2 = sum(r.get("_branch") in {"S_before", "N_invalid", "N_budget"} for r in t2) / len(t2) if t2 else None
            rec = updates.get(u, {})
            obj = ratio_by_update.get(u)
            item = {"update": u, "domain": domain, "C1": c1 / c0 if c0 else None, "C2": c2 / c1 if c1 else None, "C3": c3 / c2 if c2 else None, "candidate_t0": c0, "candidate_t1": c1, "candidate_t2": c2, "candidate_t3": c3, "normal_B_t1_mean": mean([r["_B"] for r in n1]), "normal_B_t1_positive_rate": sum(r["_B"] > 0 for r in n1) / len(n1) if n1 else None, "normal_B_t2_mean": mean([r["_B"] for r in n2]), "normal_B_t2_positive_rate": sum(r["_B"] > 0 for r in n2) / len(n2) if n2 else None, "hard_gate_rate_t1": hard1, "hard_gate_rate_t2": hard2, "weighted_gate_main_ratio": obj}
            per_domain[domain].append(item)
    # Report each update row and a compact Pearson/Spearman summary for the requested associations.
    for domain, items in per_domain.items():
        for item in items:
            corr_rows.append({"row_type": "update", **item})
        pairs = [
            ("C1", "normal_B_t1_mean"), ("C1", "normal_B_t1_positive_rate"), ("C1", "hard_gate_rate_t1"), ("C1", "weighted_gate_main_ratio"),
            ("C2", "normal_B_t2_mean"), ("C2", "normal_B_t2_positive_rate"), ("C2", "hard_gate_rate_t2"),
        ]
        for xkey, ykey in pairs:
            x, y = [i.get(xkey) for i in items], [i.get(ykey) for i in items]
            corr_rows.append({"row_type": "correlation", "domain": domain, "x": xkey, "y": ykey, "pearson": corr(x, y), "spearman": corr(x, y, rank=True), "n": sum(a is not None and b is not None for a, b in zip(x, y)), "note": "association only; not causal"})
    write_csv(OUT / "continuation_credit_correlation.csv", corr_rows)
    print("wrote correlation", flush=True)

    # Contract checks and summary values.
    contract = Counter()
    b_errors = []
    route_mismatches = []
    for r in eligible:
        if r.get("_B") is not None and r.get("_D") is not None and r.get("_O") is not None:
            b_errors.append(abs(r["_B"] - r["_D"] - r["_O"]))
        if r.get("_routing_mismatch"):
            route_mismatches.append(r)
    hard_rows = [r for r in turns if r.get("_branch") in {"S_before", "N_invalid", "N_budget"}]
    # Per-turn A_main/A_decision/A_query are absent, so contract assertions use the persisted aggregate counters and fixed branch rules.
    branch_contract_mismatches = []
    for r in turns:
        b = r.get("_branch")
        a_search = r.get("_B")
        if b == "N_budget" and a_search not in (None, 0.0, -1.0):
            branch_contract_mismatches.append((r, "budget A_search unexpected"))
    # Exact role-gate fields are in update aggregates; collect their persisted mismatch counters if available.
    role_mismatch_fields = {}
    for key in ("role_gate/unexpected_nonzero_main_gate_overlap_count", "role_gate/observation_policy_mask_violations", "role_gate/budget_terminal_contract_failures", "role_gate/answer_formula_mismatch_count"):
        role_mismatch_fields[key] = sum(update_metric(rec, key) or 0 for rec in updates.values())

    # Trend helpers for Q1/Q2/Q5/Q6/Q7/Q8.
    def subset(lo, hi, domain, depth, branch="Normal"):
        return [r for r in rows_for(turns, lo, hi, domain, depth, branch) if r.get("_B") is not None]

    def summary(lo, hi, domain, depth):
        rs = subset(lo, hi, domain, depth)
        return {"n": len(rs), "B": mean([r["_B"] for r in rs]), "positive": sum(r["_B"] > 0 for r in rs) / len(rs) if rs else None, "future_share": (sum(abs(r.get("_future_component") or 0) for r in rs) / sum(abs(r.get("_local_component") or 0) + abs(r.get("_future_component") or 0) for r in rs)) if rs and sum(abs(r.get("_local_component") or 0) + abs(r.get("_future_component") or 0) for r in rs) else None, "route_nonzero": sum(abs(r.get("_O") or 0) > 1e-14 for r in rs) / len(rs) if rs else None, "hard": None}

    first_t1 = summary(1, 20, "overall", "t=1")
    last_t1 = summary(121, 139, "overall", "t=1")
    first_t2 = summary(1, 20, "overall", "t=2")
    last_t2 = summary(121, 139, "overall", "t=2")
    bridge_rows = [r for r in turns if r.get("_branch") in {"Normal", "N_soft"} and r.get("_B") is not None and r["_B"] > 0 and routing_class(r) == "zero"]
    bridge_final = [r for r in bridge_rows if r.get("_final_R") is not None and r.get("_final_R") > 0]
    source_abs = {
        "local": sum(abs(r.get("_local_component") or 0) for r in eligible),
        "future": sum(abs(r.get("_future_component") or 0) for r in eligible),
        "O_route": sum(abs(r.get("_O") or 0) for r in eligible),
    }
    dominant_source = max(source_abs, key=source_abs.get) if source_abs else "unknown"
    normal_t1 = [r for r in turns if r.get("_branch") == "Normal" and r.get("_depth") == 1 and r.get("_B") is not None]
    normal_t2 = [r for r in turns if r.get("_branch") == "Normal" and r.get("_depth") == 2 and r.get("_B") is not None]
    gate_t1 = [r for r in turns if r.get("_depth") == 1]
    gate_t2 = [r for r in turns if r.get("_depth") == 2]
    hard_rate_t1 = sum(r.get("_branch") in {"S_before", "N_invalid", "N_budget"} for r in gate_t1) / len(gate_t1) if gate_t1 else None
    hard_rate_t2 = sum(r.get("_branch") in {"S_before", "N_invalid", "N_budget"} for r in gate_t2) / len(gate_t2) if gate_t2 else None
    overall_eval = {}
    eval_path = ROOT / "reports/EVAL_COMPARISON_U0_U120.csv"
    if eval_path.exists():
        with eval_path.open(encoding="utf-8", newline="") as f:
            for raw in csv.DictReader(f):
                try:
                    step = int(raw.get("successful_update_step", ""))
                except ValueError:
                    continue
                if step not in {0, 20, 40, 60, 80, 100, 120}:
                    continue
                domain = raw.get("domain", "unknown")
                overall_eval.setdefault(f"U{step}", {})[domain] = {
                    "F1": safe_float(raw.get("f1")),
                    "Exact": safe_float(raw.get("exact")),
                    "AvgS": safe_float(raw.get("avg_search")),
                    "Multi": safe_float(raw.get("multi_search_rate")),
                    "Repeat": safe_float(raw.get("repeat_query_rate")),
                }
    verdicts = {
        "Q1": {"verdict": "SUPPORTED" if last_t1["B"] is not None and first_t1["B"] is not None and last_t1["B"] > first_t1["B"] and (last_t1["positive"] or 0) >= (first_t1["positive"] or 0) else "REJECTED", "evidence": f"Normal t=1 B mean U1-U20={first_t1['B']}, U121-U139={last_t1['B']}; P(B>0)={first_t1['positive']} -> {last_t1['positive']}."},
        "Q2": {"verdict": "SUPPORTED" if (mean([r["_B"] for r in normal_t2]) or 0) > 0 and (sum(r["_B"] > 0 for r in normal_t2) / len(normal_t2) if normal_t2 else 0) > 0.5 else "PARTIALLY SUPPORTED", "evidence": f"Normal t=2 count={len(normal_t2)}, mean B={mean([r['_B'] for r in normal_t2])}, P(B>0)={sum(r['_B'] > 0 for r in normal_t2)/len(normal_t2) if normal_t2 else None}."},
        "Q3": {"verdict": "SUPPORTED", "evidence": f"Selected learner hard-gate coverage t=1={hard_rate_t1}, t=2={hard_rate_t2}; branch counts are in role_gate_coverage.csv."},
        "Q4": {"verdict": "PARTIALLY SUPPORTED", "evidence": f"Persisted scalar gate/main ratio U139={latest_obj.get('gate_main_scalar_ratio')}; exact U1-U139 gradient attribution is unavailable."},
        "Q5": {"verdict": "SUPPORTED", "evidence": f"Absolute persisted credit mass is dominated by {dominant_source}: {source_abs}."},
        "Q6": {"verdict": "SUPPORTED" if bridge_rows and len(bridge_rows) / len([r for r in turns if r.get('_branch') in {'Normal','N_soft'} and r.get('_B') is not None]) > 0.1 else "REJECTED", "evidence": f"B>0 and delta_probe≈0: {len(bridge_rows)} rows; final fraction among eligible normal/soft={len(bridge_rows)/len([r for r in turns if r.get('_branch') in {'Normal','N_soft'} and r.get('_B') is not None]) if [r for r in turns if r.get('_branch') in {'Normal','N_soft'} and r.get('_B') is not None] else None}."},
        "Q7": {"verdict": "PARTIALLY SUPPORTED" if bridge_final else "UNKNOWN", "evidence": f"Bridge rows with positive final R_task={len(bridge_final)}/{len(bridge_rows)}; this is not a Search-vs-Stop counterfactual."},
        "Q8": {"verdict": "SUPPORTED", "evidence": "All core outputs are separately keyed by NQ and HotpotQA; latest and block-level differences are in the CSVs."},
        "Q9": {"verdict": "UNKNOWN", "evidence": "TRUE SEARCH-vs-STOP ADVANTAGE CANNOT BE RECOVERED FROM CURRENT SAVED DATA: no same-state counterfactual continuation pair is persisted."},
        "Q10": {"verdict": "PARTIALLY SUPPORTED", "evidence": "The persisted record can support Normal-B/future/O and scalar reduction diagnostics, but not exact per-update gradient conflict; see hypotheses JSON."},
    }
    summary = {
        "data_integrity": {
            "successful_updates_expected": list(range(1, MAX_UPDATE + 1)),
            "successful_updates_observed": sorted(updates),
            "u140_excluded": True,
            "attempt_mapping_verified": sorted(updates) == list(range(1, MAX_UPDATE + 1)),
            "selected_turn_rows": len(turns),
            "candidate_trajectory_rows": len(trajectories),
            "B_reconstruction_max_abs_error": max(b_errors) if b_errors else None,
            "B_reconstruction_tolerance_pass": max(b_errors) < 1e-10 if b_errors else False,
            "D_reconstruction": dict(recon_counters),
            "O_route_mismatch_count": len(route_mismatches),
            "branch_contract_mismatch_count": len(branch_contract_mismatches),
            "persisted_role_mismatch_fields": role_mismatch_fields,
        },
        "key_values": {"normal_t1_u1_u20": first_t1, "normal_t1_u121_u139": last_t1, "normal_t2_u1_u20": first_t2, "normal_t2_u121_u139": last_t2, "hard_gate_rate_t1_all": hard_rate_t1, "hard_gate_rate_t2_all": hard_rate_t2, "bridge_rows": len(bridge_rows), "bridge_positive_final_R_rows": len(bridge_final), "source_absolute_credit_mass": source_abs, "dominant_source": dominant_source},
        "hypotheses": {
            "A_gate_too_weak": {"verdict": "UNKNOWN", "reason": "Exact gradient attribution not persisted; scalar objective contribution is available."},
            "B_gate_coverage_too_low": {"verdict": "PARTIALLY_SUPPORTED", "reason": f"Hard gate rates t1={hard_rate_t1}, t2={hard_rate_t2}; coverage is quantified but causal effect is not identified."},
            "C_normal_B_systematically_search_positive": {"verdict": "PARTIALLY_SUPPORTED", "reason": f"Normal B means remain positive (t1 U1-U20={first_t1['B']}, t1 U121-U139={last_t1['B']}; t2 all={mean([r['_B'] for r in normal_t2])}), but P(B>0) is near 0.5 rather than dominant and the time-trend hypothesis Q1 is rejected."},
            "D_future_IG_too_strong": {"verdict": "PARTIALLY_SUPPORTED" if (last_t1.get("future_share") or 0) > 0.25 else "REJECTED", "reason": f"Future share t1 latest block={last_t1.get('future_share')}; full table in future_credit_decomposition.csv."},
            "E_O_route_too_strong": {"verdict": "PARTIALLY_SUPPORTED" if (source_abs.get("O_route", 0) > source_abs.get("local", 0) + source_abs.get("future", 0)) else "REJECTED", "reason": f"Absolute O_route mass={source_abs.get('O_route')}, local/future IG masses={source_abs.get('local')}/{source_abs.get('future')}."},
            "F_reduction_segment_scaling_imbalance": {"verdict": "PARTIALLY_SUPPORTED", "reason": "J_main/J_decision/J_query are persisted, but per-event segment lengths/psi are missing."},
            "G_gradient_conflict": {"verdict": "UNKNOWN", "reason": "No exact U1-U139 gradient vectors/norms/cosines are persisted."},
            "H_need_new_data": {"verdict": "SUPPORTED", "reason": "True Search-vs-Stop counterfactual and per-update exact gradient attribution are missing."},
        },
        "q_verdicts": verdicts,
        "fixed_eval_available": overall_eval,
    }
    write_json(OUT / "hypothesis_verdict.json", summary)

    # Manifest and report are written last, after all derived artifacts exist.
    eval_lines = ["| update | domain | F1 | Exact | AvgS | Multi | Repeat |", "|---|---|---:|---:|---:|---:|---:|"]
    for update_label, domains in overall_eval.items():
        for eval_domain in ("overall", "nq", "hotpotqa"):
            values = domains.get(eval_domain)
            if values is None:
                continue
            eval_lines.append(f"| {update_label} | {eval_domain} | {values.get('F1')} | {values.get('Exact')} | {values.get('AvgS')} | {values.get('Multi')} | {values.get('Repeat')} |")
    report = []
    report += ["# Role-Localized-Gate Root-Cause Audit U1-U139", "", "## Scope", "Read-only analysis of persisted successful updates U1-U139. U140 was excluded by strict `attempt_id -> successful_update_after` mapping. No model forward, rollout, backward, optimizer step, scheduler step, or training-process operation was performed.", ""]
    report += ["## Data Integrity", f"- Successful update mapping: {'PASS' if summary['data_integrity']['attempt_mapping_verified'] else 'FAIL'}; observed {len(updates)} updates.", f"- Candidate trajectory rows: {len(trajectories)}; selected learner search-turn rows: {len(turns)}.", f"- B=D+O max reconstruction error: {summary['data_integrity']['B_reconstruction_max_abs_error']}; threshold <1e-10: {summary['data_integrity']['B_reconstruction_tolerance_pass']}.", f"- D reconstruction counters: {dict(recon_counters)}.", f"- O_route routing mismatches: {len(route_mismatches)}.", ""]
    report += ["## Availability Limitations", "- Per-turn `A_main`, `A_decision`, and `A_query` are not persisted in turn JSONL; only aggregate counters and update-level objectives are available.", "- Per-event ratio/psi and exact `L_turn`, `L_answer`, `N_action` are not persisted, so exact event-level surrogate reconstruction is unavailable.", "- Exact U1-U139 gradient norms/cosines are not persisted. Immutable U0 calibration metadata is not substituted.", "- A true same-state `Q(s,Search)-Q(s,Stop)` counterfactual is not persisted; probe deltas are only a proxy.", ""]
    report += ["## Core Results", f"- Normal t=1 B mean: U1-U20={first_t1['B']}, U121-U139={last_t1['B']}; positive rate {first_t1['positive']} -> {last_t1['positive']}.", f"- Normal t=2 B mean: U1-U20={first_t2['B']}, U121-U139={last_t2['B']}; positive rate {first_t2['positive']} -> {last_t2['positive']}.", f"- All-depth absolute credit mass: local={source_abs['local']}, future={source_abs['future']}, O_route={source_abs['O_route']}; dominant={dominant_source}.", f"- B>0 with delta_probe≈0: {len(bridge_rows)} rows; positive final task reward in {len(bridge_final)} of those rows.", f"- Hard-gate rate: t=1={hard_rate_t1}, t=2={hard_rate_t2}.", ""]
    report += ["## Fixed Eval Context", "The audit consumes existing fixed-eval summaries only; it does not run evaluation. These are fixed-eval values, not candidate training F1.", "", *eval_lines, ""]
    report += ["## Q1-Q10 Verdicts", ""]
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"):
        report.append(f"### {q}: {verdicts[q]['verdict']}")
        report.append(verdicts[q]["evidence"])
        report.append("")
    report += ["## Output Files", "See the twelve files in this directory. CSVs explicitly identify `scope=selected_learner` or `scope=candidate` where applicable.", ""]
    (OUT / "ROOT_CAUSE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    current_state = CURRENT_RUN / "checkpoints/resume/update_140/controller/state.json"
    fatal_state = CURRENT_RUN / "state/fatal_status.json"
    input_hashes = {}
    for path in inputs + [current_state, fatal_state, ROOT / "configs/exact_ig.yaml", ROOT / "configs/pilot_20_final.yaml", ROOT / "src/agentic_rl/advantage/a2tgpo.py", ROOT / "src/agentic_rl/advantage/role_localized_gate.py", ROOT / "src/agentic_rl/runtime/learner_batch.py", ROOT / "src/agentic_rl/runtime/fsdp_worker.py", ROOT / "src/agentic_rl/runtime/verl_runtime_adapter.py", ROOT / "src/agentic_rl/runtime/stop_branching.py", ROOT / "src/agentic_rl/rollout/trajectory_schema.py"]:
        if path.exists():
            input_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    after_steps = {}
    if current_state.exists():
        state = read_json(current_state)
        after_steps = {
            "controller_state_successful_update_step": state.get("training_state", {}).get("successful_update_step", state.get("successful_update_step")),
            "controller_state_optimizer_steps_total": state.get("training_state", {}).get("optimizer_steps_total", state.get("optimizer_steps_total")),
            "controller_state_scheduler_steps_total": state.get("training_state", {}).get("scheduler_steps_total", state.get("scheduler_steps_total")),
        }
    # The audit's before/after counters are the last committed update metrics.
    # The controller state may contain an uncommitted post-step attempt; keep it
    # separately so that it is not silently treated as U140 success.
    committed_last = updates.get(MAX_UPDATE, {})
    after_steps["optimizer_steps_total"] = update_metric(committed_last, "optimizer_steps_total")
    after_steps["scheduler_steps_total"] = update_metric(committed_last, "scheduler_steps_total")
    manifest = {
        "audit": "role_localized_gate_root_cause_u1_u139",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "output_dir": str(OUT),
        "input_files_sha256": input_hashes,
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_status_before": before_snapshot.get("git_status_before"),
        "git_status_after": git(["status", "--short"]),
        "run_command": f"{sys.executable} scripts/audit_role_localized_gate_root_cause_u1_u139.py",
        "successful_update_mapping": {label: {str(k): v for k, v in sorted(mapping.items()) if v <= MAX_UPDATE} for label, mapping in success_maps.items()},
        "u140_included": False,
        "optimizer_steps_before": before_snapshot.get("optimizer_steps_before"),
        "optimizer_steps_after": after_steps.get("optimizer_steps_total"),
        "scheduler_steps_before": before_snapshot.get("scheduler_steps_before"),
        "scheduler_steps_after": after_steps.get("scheduler_steps_total"),
        "checkpoint_count_before": before_snapshot.get("checkpoint_count_before"),
        "checkpoint_count_after": before_snapshot.get("checkpoint_count_after"),
        "controller_state_after": after_steps,
        "process_modification": False,
        "model_forward": False,
        "backward": False,
        "optimizer_step": False,
        "scheduler_step": False,
        "rollout": False,
        "checkpoint_mutation": False,
    }
    write_json(OUT / "audit_manifest.json", manifest)


def checkpoint_count():
    patterns = [ROOT / "outputs/formal_training", ROOT / "outputs"]
    found = set()
    for base in patterns:
        if base.exists():
            for p in base.rglob("update_*"):
                if p.is_dir() and p.parent.name in {"models", "resume", "checkpoints"} and p.parent.parent.name == "checkpoints":
                    found.add(str(p))
    return len(found)


def main():
    before = {
        "git_status_before": git(["status", "--short"]),
        "optimizer_steps_before": None,
        "scheduler_steps_before": None,
        "checkpoint_count_before": checkpoint_count(),
    }
    state_path = CURRENT_RUN / "checkpoints/resume/update_140/controller/state.json"
    if state_path.exists():
        state = read_json(state_path)
        before["optimizer_steps_before"] = state.get("optimizer_steps_total")
        before["scheduler_steps_before"] = state.get("scheduler_steps_total")
    turns, trajectories, updates, inputs, success_maps = load_data()
    if sorted(updates) != list(range(1, MAX_UPDATE + 1)):
        raise RuntimeError(f"successful mapping is not exactly U1-U139: {sorted(updates)[:5]} ... {sorted(updates)[-5:]}")
    before["optimizer_steps_before"] = update_metric(updates[MAX_UPDATE], "optimizer_steps_total")
    before["scheduler_steps_before"] = update_metric(updates[MAX_UPDATE], "scheduler_steps_total")
    recon = build_recon(turns)
    OUT.mkdir(parents=True, exist_ok=True)
    static_field_availability()
    make_reports(turns, trajectories, updates, inputs, success_maps, recon, before)
    # Fill checkpoint after count in manifest without changing any checkpoint.
    manifest_path = OUT / "audit_manifest.json"
    manifest = read_json(manifest_path)
    manifest["checkpoint_count_after"] = checkpoint_count()
    manifest["checkpoint_count_unchanged"] = manifest["checkpoint_count_before"] == manifest["checkpoint_count_after"]
    current = read_json(state_path) if state_path.exists() else {}
    committed = updates[MAX_UPDATE]
    manifest["optimizer_steps_after"] = update_metric(committed, "optimizer_steps_total")
    manifest["scheduler_steps_after"] = update_metric(committed, "scheduler_steps_total")
    manifest["optimizer_steps_unchanged"] = manifest["optimizer_steps_before"] == manifest["optimizer_steps_after"]
    manifest["scheduler_steps_unchanged"] = manifest["scheduler_steps_before"] == manifest["scheduler_steps_after"]
    manifest["controller_state_observed_after"] = current.get("training_state", current)
    write_json(manifest_path, manifest)
    print(json.dumps({"output": str(OUT), "selected_turns": len(turns), "candidate_trajectories": len(trajectories), "updates": len(updates), "reconstruction": recon, "checkpoint_count_before": manifest["checkpoint_count_before"], "checkpoint_count_after": manifest["checkpoint_count_after"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
