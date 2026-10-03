#!/usr/bin/env python3
"""Verify the recorded U325 cases and rebuild their documentation figures.

Run from any directory with Python, NumPy and Matplotlib installed.
This script reads committed evidence; it does not rescore or train a model.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
OUTPUT = ROOT / "assets" / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#8064A2"
GRAY = "#667085"


def verify_evidence() -> tuple[list[dict], dict]:
    evidence = json.loads((DATA / "search-credit-u325-trajectories.json").read_text())
    numeric = {
        "R_task", "Z_outcome", "IG", "G", "A_loc", "A_ret", "A_search",
        "IG_mean", "IG_std", "G_mean", "G_std",
    }
    integer = {"search_index", "search_ordinal", "peer_count"}
    boolean = {"exact_query_repeat", "no_new_passage", "singleton_fallback"}
    with (DATA / "search-credit-u325-peers.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key]) if row[key] else None
        for key in integer:
            row[key] = int(row[key])
        for key in boolean:
            assert row[key] in {"True", "False"}, (key, row[key])
            row[key] = row[key] == "True"
    config = evidence["normalization"]
    assert config == dict(gamma=1.0, mix_weight=0.5, epsilon=1e-6,
                          variance_tolerance=1e-12, ddof=0)
    trajectories = {item["trajectory_id"]: item for item in evidence["trajectories"]}

    def equal(actual: float, expected: float) -> None:
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), (actual, expected)

    for group_name, group in evidence["peer_groups"].items():
        peers = [row for row in rows if row["peer_group"] == group_name]
        assert len(peers) == group["peer_count"]
        assert len({row["trajectory_id"] for row in peers}) == len(peers)
        for signal, mean_key, std_key, advantage_key in [
            ("IG", "IG_mean", "IG_std", "A_loc"),
            ("G", "G_mean", "G_std", "A_ret"),
        ]:
            values = np.array([row[signal] for row in peers], dtype=np.float64)
            mean, std = float(values.mean()), float(values.std(ddof=0))
            for row in peers:
                equal(row[mean_key], mean)
                equal(row[std_key], std)
                if len(peers) >= 2:
                    expected = 0.0 if std * std <= 1e-12 else (row[signal] - mean) / (std + 1e-6)
                    equal(row[advantage_key], expected)
                else:
                    assert row[advantage_key] is None
        for row in peers:
            assert row["peer_count"] == len(peers)
            assert row["search_index"] == group["search_index"]
            assert row["search_ordinal"] == row["search_index"] + 1
            assert row["prompt_global_id"] == group["prompt_global_id"]
            tr = trajectories[row["trajectory_id"]]
            equal(row["R_task"], tr["R_task"])
            equal(row["Z_outcome"], tr["outcome_z"])
            turn = next(t for t in tr["search_turns"] if t["search_index"] == row["search_index"])
            assert turn["ig_reward_eligible"] and turn["policy_credit_eligible"]
            equal(row["IG"], turn["IG"])
            for key in ("exact_query_repeat", "no_new_passage"):
                assert row[key] == turn[key]
            assert turn["no_new_passage"] == (len(turn["new_passage_keys"]) == 0)
            suffix = [t["IG"] for t in tr["search_turns"]
                      if t["search_index"] >= row["search_index"] and t["ig_reward_eligible"]]
            equal(row["G"], math.fsum(suffix))
            assert row["singleton_fallback"] == (len(peers) == 1)
            expected = row["Z_outcome"] if len(peers) == 1 else 0.5 * (row["A_loc"] + row["A_ret"])
            equal(row["A_search"], expected)
    assert len(rows) == 38 and len(trajectories) == 37
    print("Verified 38 peer rows, five complete peer groups, and 37 trajectory records.")
    return rows, evidence


def style(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#344054", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)


def labeled_bars(ax: plt.Axes, x, values, *, width=0.6, color=None, label=None):
    colors = color if color else [BLUE if value >= 0 else ORANGE for value in values]
    bars = ax.bar(x, values, width=width, color=colors, label=label, zorder=3)
    ax.bar_label(bars, labels=[f"{value:+.4f}" for value in values], padding=4, fontsize=10)
    return bars


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180, facecolor="white")
    svg = OUTPUT / f"{stem}.svg"
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    # Matplotlib adds trailing spaces to path lines; keep repository text clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


def same_depth(rows: list[dict]) -> None:
    peers = [r for r in rows if r["peer_group"] == "same_depth"]
    selected = [next(r for r in peers if r["trajectory_number"] == n) for n in ["04", "06", "08", "14"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})
    fig.suptitle("Same question, second Search: process credit differs from terminal outcome", fontsize=15)
    x = np.arange(4)
    labels = [f"T{r['trajectory_number']}\nTask F1 = {r['R_task']:.0f}" for r in selected]
    ax = axes[0]
    labeled_bars(ax, x, [r["IG"] for r in selected])
    ax.axhline(peers[0]["IG_mean"], color=GRAY, linestyle="--", label="Mean of all 16 peers")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.78)
    ax.set_title("Immediate retrieval gain")
    style(ax, "Raw IG (mean log-probability change)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax = axes[1]
    for offset, key, label, color in [(-0.27, "A_loc", "Local", TEAL), (0, "A_ret", "Return", PURPLE), (0.27, "A_search", "Mixed Search", BLUE)]:
        labeled_bars(ax, x + offset, [r[key] for r in selected], width=0.25, color=color, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylim(-2.25, 3.7)
    ax.set_title("Separate normalization, then a 50/50 mixture")
    style(ax, "Normalized advantage")
    ax.legend(loc="upper left", ncols=3, frameon=False, fontsize=10)
    fig.supxlabel("U325 / snapshot 324  |  Selected cases; normalization uses the full 16-peer group", fontsize=10, color=GRAY)
    save(fig, "search-credit-same-depth")


def delayed_return(rows: list[dict], evidence: dict) -> None:
    row = next(r for r in rows if r["peer_group"] == "delayed_return" and r["trajectory_number"] == "10")
    tr = next(t for t in evidence["trajectories"] if t["trajectory_id"] == row["trajectory_id"])
    suffix = [t["IG"] for t in tr["search_turns"] if t["search_index"] >= 1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    fig.suptitle("Delayed credit: a weak immediate step shares a strong observed suffix", fontsize=15)
    ax = axes[0]
    labeled_bars(ax, np.arange(3), suffix)
    ax.set_xticks(np.arange(3), ["Search 2\nRepeated query; no new passage", "Search 3", "Search 4"])
    ax.set_title(f"Raw suffix return G = {row['G']:.4f}")
    ax.set_ylim(-0.45, 3.35)
    style(ax, "Raw IG (mean log-probability change)")
    ax = axes[1]
    labeled_bars(ax, np.arange(3), [row["A_loc"], row["A_ret"], row["A_search"]])
    ax.set_xticks(np.arange(3), ["Local", "Return", "Mixed Search"])
    ax.set_title("Search 2 credit, normalized among 14 peers")
    ax.set_ylim(-0.65, 2.65)
    style(ax, "Normalized advantage")
    fig.supxlabel("T10  |  Task F1 = 0.6000  |  Later gain does not establish that Search 2 was necessary", fontsize=10, color=GRAY)
    save(fig, "search-credit-delayed-return")


def boundaries(rows: list[dict]) -> None:
    tail = [r for r in rows if r["peer_group"] == "negative_tail"]
    singleton = [next(r for r in rows if r["peer_group"] == g) for g in ["singleton_search_4", "singleton_search_5"]]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    fig.suptitle("Recorded boundaries: relative terminal credit and singleton outcome fallback", fontsize=15)
    x = np.arange(len(tail))
    labels = [f"T{r['trajectory_number']}" for r in tail]
    ax = axes[0, 0]
    labeled_bars(ax, x, [r["IG"] for r in tail])
    ax.axhline(tail[0]["IG_mean"], color=GRAY, linestyle="--", label=f"Peer mean = {tail[0]['IG_mean']:.4f}")
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.5, 0.12)
    ax.set_title("Final valid-IG Search: all six IG values are negative")
    style(ax, "Raw IG = raw G")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax = axes[0, 1]
    labeled_bars(ax, x, [r["A_search"] for r in tail])
    ax.set_xticks(x, labels)
    ax.set_ylim(-2.65, 1.15)
    ax.set_title("Same group: five mixed advantages are positive")
    style(ax, "Normalized Search advantage")
    ax = axes[1, 0]
    x = np.arange(2)
    labeled_bars(ax, x - 0.18, [r["IG"] for r in singleton], width=0.32, color=TEAL, label="Immediate IG")
    labeled_bars(ax, x + 0.18, [r["G"] for r in singleton], width=0.32, color=PURPLE, label="Suffix G")
    ax.set_xticks(x, ["Search 4\nNo new passage", "Search 5\nTwo new passages"])
    ax.set_ylim(-0.45, 2.2)
    ax.set_title("Singleton T15: distinct process signals")
    style(ax, "Raw retrieval signal")
    ax.legend(loc="upper left", ncols=2, frameon=False, fontsize=10)
    ax = axes[1, 1]
    labeled_bars(ax, x, [r["A_search"] for r in singleton])
    ax.set_xticks(x, ["Search 4\nOne peer", "Search 5\nOne peer"])
    ax.set_ylim(0, 2.2)
    ax.set_title("Both use the same logged outcome fallback")
    style(ax, "Search advantage = normalized outcome")
    ax.text(0.5, 0.91, r"$A^{search} = Z^O = +1.5269$", transform=ax.transAxes, ha="center", fontsize=12)
    fig.supxlabel("U325 / snapshot 324  |  These cases expose credit limitations; they do not prove causal over-search", fontsize=10, color=GRAY)
    save(fig, "search-credit-boundaries")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                         "svg.fonttype": "none", "svg.hashsalt": "aethersearch-u325"})
    rows, evidence = verify_evidence()
    same_depth(rows)
    delayed_return(rows, evidence)
    boundaries(rows)
    print(f"Rebuilt three PNG figures and three SVG figures in {OUTPUT}.")


if __name__ == "__main__":
    main()
