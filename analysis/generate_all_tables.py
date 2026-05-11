"""
generate_all_tables.py
======================
Master script: run all sub-analyses and produce paper-ready tables and figures.

Requires: pandas, matplotlib, scipy (pip install pandas matplotlib scipy)

Tables produced
---------------
  Table 1  — Win rates per model pair contrast  (7 rows)
  Table 2  — Win rates by workshop × pair       (7 × N_workshops rows)
  Table 3  — Spearman ρ correlation matrix (human vote vs automated NDKL)

Figures produced
----------------
  figure1_forest.png  — Forest plot: effect sizes per contrast
  figure2_heatmap.png — Heatmap: workshop × contrast win-rate
  figure3_scatter.png — Scatter: automated NDKL delta vs human win-rate

Inputs (defaults look in current directory, then ../data)
---------
  win_rates_overall.csv, win_rates_by_workshop.csv,
  alignment_results.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nRun: pip install pandas matplotlib scipy numpy")


# ── Helpers ───────────────────────────────────────────────────────────────

def find(name: str, candidates: list[str]) -> str | None:
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def load(name: str) -> pd.DataFrame:
    path = find(name, [name, f"../{name}", f"../data/{name}"])
    if path is None:
        raise FileNotFoundError(
            f"{name} not found. Run win_rates.py and alignment_correlation.py first."
        )
    return pd.read_csv(path)


# ── Table 1: Overall win rates ────────────────────────────────────────────

def make_table1(df: pd.DataFrame, out: str = "table1_win_rates.csv"):
    cols = ["pair", "n", "pct_left", "ci_left_lo", "ci_left_hi",
            "pct_tie", "pct_right", "ci_right_lo", "ci_right_hi"]
    t1 = df[cols].copy()
    t1.to_csv(out, index=False)
    print(f"Table 1 → {out}")
    return t1


# ── Table 2: Win rates by workshop ────────────────────────────────────────

def make_table2(df: pd.DataFrame, out: str = "table2_by_workshop.csv"):
    cols = ["pair", "workshop_name", "community_context", "n",
            "pct_left", "pct_tie", "pct_right"]
    available = [c for c in cols if c in df.columns]
    t2 = df[available].copy()
    t2.to_csv(out, index=False)
    print(f"Table 2 → {out}")
    return t2


# ── Table 3: Correlation matrix ───────────────────────────────────────────

def make_table3(df: pd.DataFrame, out: str = "table3_correlations.csv"):
    overall = df[df["scope"] == "overall"][["metric", "rho", "p_value", "n"]].copy()
    overall.to_csv(out, index=False)
    print(f"Table 3 → {out}")
    return overall


# ── Figure 1: Forest plot ─────────────────────────────────────────────────

def fig_forest(df: pd.DataFrame, out: str = "figure1_forest.png"):
    pairs = df["pair"].tolist()
    pcts  = df["pct_left"].astype(float).tolist()
    los   = df["ci_left_lo"].astype(float).tolist()
    his   = df["ci_left_hi"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, max(4, len(pairs) * 0.55)))
    y = list(range(len(pairs)))

    for i, (p, lo, hi) in enumerate(zip(pcts, los, his)):
        if math.isnan(p):
            continue
        ax.plot([lo, hi], [i, i], color="#4a90d9", lw=2)
        ax.plot(p, i, "o", color="#1a5ea8", zorder=3)

    ax.axvline(0.5, color="gray", linestyle="--", lw=1, label="No preference (50%)")
    ax.set_yticks(y)
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel("Left-side win rate (95% Wilson CI)")
    ax.set_title("Figure 1 — Effect sizes per model pair contrast")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Figure 1 → {out}")


# ── Figure 2: Heatmap workshop × contrast ────────────────────────────────

def fig_heatmap(df: pd.DataFrame, out: str = "figure2_heatmap.png"):
    if "workshop_name" not in df.columns or df["workshop_name"].isna().all():
        print("Figure 2 skipped — no workshop data available.")
        return

    pivot = df.pivot_table(
        index="workshop_name", columns="pair", values="pct_left", aggfunc="mean"
    )
    if pivot.empty:
        print("Figure 2 skipped — pivot is empty.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4), max(3, len(pivot) * 0.7)))
    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Left-side win rate")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if not math.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.25 < v < 0.75 else "white")
    ax.set_title("Figure 2 — Left-side win rate: workshop × model pair")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Figure 2 → {out}")


# ── Figure 3: Scatter automated vs human ─────────────────────────────────

def fig_scatter(align_df: pd.DataFrame, out: str = "figure3_scatter.png"):
    overall = align_df[align_df["scope"] == "overall"].copy()
    if overall.empty:
        print("Figure 3 skipped — no alignment data.")
        return

    metrics = overall["metric"].unique().tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        row = overall[overall["metric"] == metric]
        if row.empty:
            continue
        rho  = row["rho"].values[0]
        pval = row["p_value"].values[0]
        n    = row["n"].values[0]
        label = f"ρ={rho:.3f}, p={pval:.3f}, n={n}"
        ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(f"{metric} alignment")
        ax.set_xlabel("NDKL delta (left − right)")
        ax.set_ylabel("Human left-win rate")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.axvline(0,   color="gray", lw=0.8, ls="--")

    fig.suptitle("Figure 3 — Automated NDKL vs human preference alignment")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Figure 3 → {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=".", help="Output directory for tables/figures")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def out(name: str) -> str:
        return str(outdir / name)

    overall_df   = load("win_rates_overall.csv")
    workshop_df  = load("win_rates_by_workshop.csv")
    alignment_df = load("alignment_results.csv")

    make_table1(overall_df,   out("table1_win_rates.csv"))
    make_table2(workshop_df,  out("table2_by_workshop.csv"))
    make_table3(alignment_df, out("table3_correlations.csv"))

    fig_forest(overall_df,   out("figure1_forest.png"))
    fig_heatmap(workshop_df, out("figure2_heatmap.png"))
    fig_scatter(alignment_df, out("figure3_scatter.png"))

    print("\nAll tables and figures generated.")


if __name__ == "__main__":
    main()
