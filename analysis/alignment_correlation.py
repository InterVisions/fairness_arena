"""
alignment_correlation.py
========================
Measure how well automated fairness metrics (NDKL) align with human votes.

For each vote: human preferred the left side (1) or right side (0) or tied (0.5).
Automated signal: ndkl_left - ndkl_right (negative = left is fairer by NDKL).

Compute Spearman ρ between human_vote_signal and ndkl_delta, overall and per workshop.

Inputs
------
  --votes    analysis_export.csv  (default: ../data/analysis_export.csv)
  --metrics  automated_metrics.csv (default: automated_metrics.csv)

Output: alignment_results.csv
  scope, workshop_id, workshop_name, community_context,
  metric, rho, p_value, n
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict


# ── Spearman ρ (no scipy dependency) ─────────────────────────────────────

def _rank(values: list[float]) -> list[float]:
    """Assign average ranks, handling ties."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (rho, p_value). p_value uses t-distribution approximation."""
    n = len(x)
    if n < 3:
        return (float("nan"), float("nan"))
    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n)) *
        sum((ry[i] - my) ** 2 for i in range(n))
    )
    rho = num / den if den else float("nan")
    if math.isnan(rho):
        return (float("nan"), float("nan"))
    # t-approximation
    t_stat = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    # two-sided p-value via normal approximation for large n
    pval = 2.0 * _norm_sf(abs(t_stat))
    return rho, pval


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


# ── Human vote signal ─────────────────────────────────────────────────────

def human_signal(winner: str, position_a: str) -> float:
    """1 = left preferred, 0 = right preferred, 0.5 = tie."""
    if winner == "tie":
        return 0.5
    left_wins = (winner == "A" and position_a == "left") or (winner == "B" and position_a == "right")
    return 1.0 if left_wins else 0.0


AUTOMATED_METRICS = [
    "ndkl_gender", "ndkl_race", "ndkl_age",
    "maxskew_gender", "maxskew_race", "maxskew_age",
]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--votes",   default="../data/analysis_export.csv")
    p.add_argument("--metrics", default="automated_metrics.csv")
    args = p.parse_args()

    with open(args.votes, newline="", encoding="utf-8") as f:
        votes = {row["vote_id"]: row for row in csv.DictReader(f)}

    with open(args.metrics, newline="", encoding="utf-8") as f:
        metrics = {row["vote_id"]: row for row in csv.DictReader(f)}

    def safe_float(v: str) -> float | None:
        try:
            x = float(v)
            return None if math.isnan(x) else x
        except (ValueError, TypeError):
            return None

    # Build joined records
    joined = []
    for vid, vote in votes.items():
        m = metrics.get(vid)
        if m is None:
            continue
        hs = human_signal(vote.get("winner", ""), vote.get("position_a", "left"))
        record = {
            "vote_id":           vid,
            "human_signal":      hs,
            "workshop_id":       vote.get("workshop_id", ""),
            "workshop_name":     vote.get("workshop_name", ""),
            "community_context": vote.get("community_context", ""),
        }
        for metric in AUTOMATED_METRICS:
            nl = safe_float(m.get(f"{metric}_left",  ""))
            nr = safe_float(m.get(f"{metric}_right", ""))
            record[f"delta_{metric}"] = (nl - nr) if (nl is not None and nr is not None) else None
        joined.append(record)

    print(f"Joined {len(joined)} votes with automated metrics")

    results = []

    def compute_scope(scope_label: str, scope_extra: dict, rows: list[dict]):
        for metric in AUTOMATED_METRICS:
            col = f"delta_{metric}"
            pairs = [
                (r["human_signal"], r[col])
                for r in rows
                if r.get(col) is not None
            ]
            if not pairs:
                continue
            hs_vals   = [p[0] for p in pairs]
            auto_vals = [p[1] for p in pairs]
            rho, pval = spearman(hs_vals, auto_vals)
            results.append({
                "scope":             scope_label,
                **scope_extra,
                "metric":            metric,
                "rho":               round(rho,  4) if not math.isnan(rho)  else "",
                "p_value":           round(pval, 4) if not math.isnan(pval) else "",
                "n":                 len(pairs),
            })

    # Overall
    compute_scope("overall", {"workshop_id": "", "workshop_name": "", "community_context": ""}, joined)

    # Per workshop
    by_workshop: dict[str, list] = defaultdict(list)
    for r in joined:
        key = r["workshop_id"] or "none"
        by_workshop[key].append(r)

    for wid, rows in sorted(by_workshop.items()):
        wname = rows[0]["workshop_name"]
        ctx   = rows[0]["community_context"]
        compute_scope("workshop", {"workshop_id": wid, "workshop_name": wname, "community_context": ctx}, rows)

    outpath = "alignment_results.csv"
    if results:
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Written {len(results)} rows → {outpath}")
    else:
        print("No results produced — check that metric and vote files are aligned.")


if __name__ == "__main__":
    main()
