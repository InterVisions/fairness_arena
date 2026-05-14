"""
community_analysis.py
=====================
Test whether different sessions perceive fairness differently.

For each model pair contrast, performs a chi-square test on the 3xN contingency
table (vote outcome: left/tie/right) x sessions.

Inputs
------
  --votes   Path to analysis export CSV  (default: ../data/analysis_export.csv)

Output: session_differences.csv
  pair, session_a, session_b, chi2, p_value, cramers_v, n, significant_05
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

_DEFAULT_VOTES = str(Path(__file__).parent.parent / "data" / "analysis_export.csv")


def chi2_test(observed: list[list[int]]) -> tuple[float, float]:
    """Two-way chi-square test. Returns (chi2, p_value)."""
    rows = len(observed)
    cols = len(observed[0])
    row_totals = [sum(r) for r in observed]
    col_totals = [sum(observed[r][c] for r in range(rows)) for c in range(cols)]
    grand = sum(row_totals)

    if grand == 0:
        return (float("nan"), float("nan"))

    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            expected = row_totals[r] * col_totals[c] / grand
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected

    df = (rows - 1) * (cols - 1)
    p = chi2_pvalue(chi2, df)
    return chi2, p


def chi2_pvalue(chi2: float, df: int) -> float:
    if math.isnan(chi2) or df <= 0:
        return float("nan")
    try:
        from scipy.stats import chi2 as scipy_chi2
        return float(scipy_chi2.sf(chi2, df))
    except ImportError:
        pass
    x = chi2
    k = df
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return _norm_sf(z)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def cramers_v(chi2: float, n: int, rows: int, cols: int) -> float:
    if n == 0 or math.isnan(chi2):
        return float("nan")
    return math.sqrt(chi2 / (n * (min(rows, cols) - 1)))


def outcome(row: dict) -> str:
    winner = row.get("winner", "")
    pos_a  = row.get("position_a", "left")
    if winner == "tie":
        return "tie"
    if (winner == "A" and pos_a == "left") or (winner == "B" and pos_a == "right"):
        return "left"
    return "right"


OUTCOMES = ["left", "tie", "right"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--votes", default=_DEFAULT_VOTES)
    args = p.parse_args()

    with open(args.votes, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    pair_workshop_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {o: 0 for o in OUTCOMES})
    )

    for row in all_rows:
        pair   = f"{row.get('model_a','')} vs {row.get('model_b','')}"
        slabel = row.get("session_name") or row.get("session_id") or "unknown"
        oc     = outcome(row)
        pair_workshop_counts[pair][slabel][oc] += 1

    results = []

    for pair, workshop_data in sorted(pair_workshop_counts.items()):
        workshops = sorted(workshop_data.keys())
        if len(workshops) < 2:
            continue

        for wa, wb in combinations(workshops, 2):
            counts_a = [workshop_data[wa][o] for o in OUTCOMES]
            counts_b = [workshop_data[wb][o] for o in OUTCOMES]
            observed = [counts_a, counts_b]
            n = sum(counts_a) + sum(counts_b)

            chi2, pval = chi2_test(observed)
            cv = cramers_v(chi2, n, rows=2, cols=3)

            results.append({
                "pair":      pair,
                "session_a": wa,
                "session_b": wb,
                "n_a":            sum(counts_a),
                "n_b":            sum(counts_b),
                "n":              n,
                "chi2":           round(chi2, 4) if not math.isnan(chi2) else "",
                "p_value":        round(pval, 4) if not math.isnan(pval) else "",
                "cramers_v":      round(cv, 4)   if not math.isnan(cv)   else "",
                "significant_05": (pval < 0.05)  if not math.isnan(pval) else "",
            })

    outpath = "session_differences.csv"
    if results:
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Written {len(results)} rows -> {outpath}")
    else:
        print("No pairwise comparisons found (need >= 2 sessions with data).")


if __name__ == "__main__":
    main()
