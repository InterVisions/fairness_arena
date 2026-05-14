"""
win_rates.py
============
Compute win rates (left wins / ties / right wins) per model pair,
stratified by session and by query.

Inputs
------
  --votes   Path to analysis export CSV  (default: ../data/analysis_export.csv)

Outputs
-------
  win_rates_overall.csv      - pairs overall
  win_rates_by_session.csv   - pairs x N sessions
  win_rates_by_query.csv     - pairs x queries

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

    by_session: dict[str, list] = defaultdict(list)
    for row in all_rows:
        sid   = row.get("session_id") or "none"
        sname = row.get("session_name") or "unknown"
        key   = f"{sid}|{sname}"
        by_session[key].append(row)

    session_out = []
    for key, rows in by_session.items():
        sid, sname = key.split("|", 1)
        counts = aggregate(rows)
        session_out.extend(counts_to_rows(
            counts,
            extra_cols={"session_id": sid, "session_name": sname},
        ))
    write_csv(session_out, "win_rates_by_session.csv")

    by_query: dict[str, list] = defaultdict(list)
    for row in all_rows:
        q = row.get("query") or "unknown"
        by_query[q].append(row)

    query_out = []
    for q, rows in sorted(by_query.items()):
        counts = aggregate(rows)
        query_out.extend(counts_to_rows(counts, extra_cols={"query": q}))
    write_csv(query_out, "win_rates_by_query.csv")


if __name__ == "__main__":
    main()
