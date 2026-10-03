#!/usr/bin/env python3
"""Illustrate the over-search improvement under assumed labels, without model or judge inference."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
FIGURES = ROOT / "assets" / "figures"


def main() -> None:
    with (DATA / "search-credit-u325-peers.csv").open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    scenarios = [
        ("terminal_search_5", "negative_tail", "10", "OVER"),
        ("singleton_search_4", "singleton_search_4", "15", "OVER"),
        ("singleton_search_5", "singleton_search_5", "15", "NECESSARY"),
        ("delayed_search_2", "delayed_return", "10", "OVER"),
    ]
    bases = {}
    output = []
    for name, group, number, label in scenarios:
        matches = [r for r in source if r["peer_group"] == group and r["trajectory_number"] == number]
        if len(matches) != 1:
            raise ValueError(f"Expected one historical base row for {name}")
        row = matches[0]
        base = float(row["A_search"])
        bases[name] = base
        for beta in (0.5, 1.0, 2.0):
            correction = -beta if label == "OVER" else 0.0
            output.append({
                "scenario_id": name,
                "source_peer_group": group,
                "trajectory_id": row["trajectory_id"],
                "search_ordinal": row["search_ordinal"],
                "recorded_base_advantage": base,
                "assumed_label": label,
                "beta_over": beta,
                "illustrative_correction": correction,
                "illustrative_new_advantage": base + correction,
                "evidence_kind": "conditional_arithmetic_not_a_judge_or_training_result",
            })
    with (DATA / "over-search-conditional-examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "aethersearch-over-search"})
    fig, ax = plt.subplots(figsize=(11.8, 5.8), layout="constrained")
    beta = np.linspace(0, 2.2, 221)
    for name, label, color in [
        ("terminal_search_5", "Terminal Search 5: base = +0.5324", "#0072B2"),
        ("singleton_search_4", "Singleton Search 4: base = +1.5269", "#D55E00"),
    ]:
        base = bases[name]
        ax.plot(beta, base - beta, color=color, linewidth=2.4, label=label)
        ax.scatter([base], [0], color=color, s=60, zorder=5)
        ax.annotate(f"Zero at beta = {base:.4f}", xy=(base, 0), xytext=(base + 0.04, 0.19),
                    color=color, fontsize=10)
        ax.scatter([1], [base - 1], color=color, s=48, zorder=5)
        ax.annotate(f"At beta = 1: {base - 1:+.4f}", xy=(1, base - 1),
                    xytext=(1.07, base - 1 + 0.16), fontsize=10, color=color)
    ax.axhline(0, color="#344054", linewidth=1.1)
    ax.axvline(1, color="#98A2B3", linestyle="--", linewidth=1)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-1.9, 2.1)
    ax.set_xlabel(r"Illustrative penalty coefficient $\beta_{over}$")
    ax.set_ylabel("Corrected Search advantage")
    ax.set_title("Conditional arithmetic: both actions are assumed to be labeled OVER", pad=14, fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.16)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.supxlabel("Recorded U325 base values + hypothetical labels | No judge inference or training experiment", fontsize=10, color="#667085")
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "over-search-conditional-correction"
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print("Wrote 12 conditional scenarios and the PNG/SVG illustration; no labels were measured.")
    for row in output:
        print(f"{row['scenario_id']}: assumed {row['assumed_label']}, "
              f"beta={row['beta_over']:g}, new={row['illustrative_new_advantage']:+.4f}")


if __name__ == "__main__":
    main()
