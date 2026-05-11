"""
win_rates.py
============
Compute win rates (left wins / ties / right wins) per model pair,
stratified by workshop/community_context and by query_category.

Inputs
------
  --votes   Path to analysis export CSV  (default: ../data/analysis_export.csv)

Outputs
-------
  win_rates_by_workshop.csv  - 7 pairs x N workshops
  win_rates_by_query.csv     - 7 pairs x 4 query categories
  win_rates_overall.csv      - 7 pairs overall

Wilson score 95% CI is included for each win/loss/tie proportion.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

_DEFAULT_VOTES = str(Path(__file__).parent.parent / "data" / "analysis_export.csv")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def pair_key(model_a: str, model_b: str) -> str:
    return f"{model_a} vs {model_b}"


def aggregate(rows: list[dict]) -> dict:
    counts: dict[str, dict] = defaultdict(lambda: {"left": 0, "tie": 0, "right": 0, "total": 0})
    for row in rows:
        ma, mb = row.get("model_a", ""), row.get("model_b", "")
        winner = row.get("winner", "")
        pos_a = row.get("position_a", "left")
        key = pair_key(ma, mb)
        counts[key]["total"] += 1
        if winner == "tie":
            counts[key]["tie"] += 1
        elif (winner == "A" and pos_a == "left") or (winner == "B" and pos_a == "right"):
            counts[key]["left"] += 1
        else:
            counts[key]["right"] += 1
    return counts


def counts_to_rows(counts: dict, extra_cols: dict | None = None) -> list[dict]:
    out = []
    for pair, c in sorted(counts.items()):
        n = c["total"]
        lw, tw, rw = c["left"], c["tie"], c["right"]
        l_lo, l_hi = wilson_ci(lw, n)
        t_lo, t_hi = wilson_ci(tw, n)
        r_lo, r_hi = wilson_ci(rw, n)
        row = {
            "pair":        pair,
            "n":           n,
            "left_wins":   lw,
            "ties":        tw,
            "right_wins":  rw,
            "pct_left":    round(lw / n, 4) if n else float("nan"),
            "pct_tie":     round(tw / n, 4) if n else float("nan"),
            "pct_right":   round(rw / n, 4) if n else float("nan"),
            "ci_left_lo":  round(l_lo, 4),
            "ci_left_hi":  round(l_hi, 4),
            "ci_tie_lo":   round(t_lo, 4),
            "ci_tie_hi":   round(t_hi, 4),
            "ci_right_lo": round(r_lo, 4),
            "ci_right_hi": round(r_hi, 4),
        }
        if extra_cols:
            row.update(extra_cols)
        out.append(row)
    return out


def write_csv(rows: list[dict], path: str):
    if not rows:
        print(f"  (no data - skipping {path})")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--votes", default=_DEFAULT_VOTES)
    args = p.parse_args()

    with open(args.votes, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    print(f"Loaded {len(all_rows)} votes from {args.votes}")

    overall_rows = counts_to_rows(aggregate(all_rows))
    write_csv(overall_rows, "win_rates_overall.csv")

    by_workshop: dict[str, list] = defaultdict(list)
    for row in all_rows:
        wid   = row.get("workshop_id") or "none"
        wname = row.get("workshop_name") or "unknown"
        ctx   = row.get("community_context") or "unknown"
        key   = f"{wid}|{wname}|{ctx}"
        by_workshop[key].append(row)

    workshop_out = []
    for key, rows in by_workshop.items():
        wid, wname, ctx = key.split("|", 2)
        counts = aggregate(rows)
        workshop_out.extend(counts_to_rows(
            counts,
            extra_cols={"workshop_id": wid, "workshop_name": wname, "community_context": ctx},
        ))
    write_csv(workshop_out, "win_rates_by_workshop.csv")

    by_cat: dict[str, list] = defaultdict(list)
    for row in all_rows:
        cat = row.get("query_category") or "unknown"
        by_cat[cat].append(row)

    cat_out = []
    for cat, rows in sorted(by_cat.items()):
        counts = aggregate(rows)
        cat_out.extend(counts_to_rows(counts, extra_cols={"query_category": cat}))
    write_csv(cat_out, "win_rates_by_query.csv")


if __name__ == "__main__":
    main()
