"""
stats_from_db.py
================
Query the arena SQLite database directly and print two LaTeX tables:

  Table 1 — Vote counts for the 10 fixed queries by workshop session.
  Table 2 — Win rates per model pair, queries grouped by contrast.

Usage
-----
    python analysis/stats_from_db.py [--db PATH]

Default DB path: data/arena.db (relative to the repo root).
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

FIXED_QUERIES = [
    "nurse", "doctor", "terrorist", "caretaker", "activist",
    "attractive", "criminal", "hero", "homeless", "rich",
]

DB_DEFAULT = Path(__file__).parent.parent / "data" / "arena.db"


# ── Table 1: vote counts by session ─────────────────────────────────────────

def fetch_counts(db_path: Path) -> tuple[list[str], dict[str, dict[str, int]], dict[str, int]]:
    """Return session names (ordered), {query: {session: count}}, and {query: total} dicts.

    Totals are computed independently so votes with no session_id are included.
    """
    con = sqlite3.connect(db_path)
    cur = con.execute("SELECT name FROM sessions ORDER BY started_at")
    sessions = [r[0] for r in cur.fetchall()]

    placeholders = ",".join("?" * len(FIXED_QUERIES))

    # Per-session counts (votes without a session_id are excluded here)
    cur = con.execute(
        f"""
        SELECT v.query, s.name, COUNT(*) AS cnt
        FROM votes v
        JOIN sessions s ON s.id = v.session_id
        WHERE v.query IN ({placeholders})
        GROUP BY v.query, s.name
        """,
        FIXED_QUERIES,
    )
    counts: dict[str, dict[str, int]] = {q: {} for q in FIXED_QUERIES}
    for query, session, cnt in cur.fetchall():
        counts[query][session] = cnt

    # Totals include all votes regardless of session assignment
    cur = con.execute(
        f"SELECT query, COUNT(*) FROM votes WHERE query IN ({placeholders}) GROUP BY query",
        FIXED_QUERIES,
    )
    totals = {query: cnt for query, cnt in cur.fetchall()}
    con.close()

    return sessions, counts, totals


def print_latex_counts(sessions: list[str], counts: dict[str, dict[str, int]], totals: dict[str, int]) -> None:
    col_spec = "l" + "r" * len(sessions) + "r"
    header_cols = " & ".join(sessions) + " & \\textbf{Total}"

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Query}} & {header_cols} \\\\",
        "\\midrule",
    ]

    session_totals = {s: 0 for s in sessions}
    grand_total = 0

    sorted_queries = sorted(FIXED_QUERIES, key=lambda q: totals.get(q, 0), reverse=True)

    for query in sorted_queries:
        row_counts = [counts[query].get(s, 0) for s in sessions]
        row_total = totals.get(query, 0)
        for s, c in zip(sessions, row_counts):
            session_totals[s] += c
        grand_total += row_total
        cells = " & ".join(str(c) for c in row_counts)
        lines.append(f"{query} & {cells} & {row_total} \\\\")

    lines.append("\\midrule")
    footer_cells = " & ".join(str(session_totals[s]) for s in sessions)
    lines.append(f"\\textbf{{Total}} & {footer_cells} & {grand_total} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Vote counts for the 10 fixed queries by workshop session.}",
        "\\label{tab:votes_by_query_session}",
        "\\end{table}",
    ]

    print("\n".join(lines))


# ── Wilson CI ────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Table 2: win rates by model-pair contrast ────────────────────────────────

def fetch_win_rates(db_path: Path) -> list[tuple[str, str, list[tuple[str, int, int, int]]]]:
    """Return a list of (model_a, model_b, [(query, a_wins, ties, b_wins), ...])
    sorted by contrast pair and then by total votes descending within each group.
    """
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(FIXED_QUERIES))

    # Contrast assignment per query
    cur = con.execute(
        f"SELECT query, model_a, model_b FROM query_contrasts WHERE query IN ({placeholders})",
        FIXED_QUERIES,
    )
    contrasts: dict[str, tuple[str, str]] = {q: (ma, mb) for q, ma, mb in cur.fetchall()}

    # Win counts per query
    cur = con.execute(
        f"""
        SELECT query,
               SUM(winner = 'A') AS a_wins,
               SUM(winner = 'tie') AS ties,
               SUM(winner = 'B') AS b_wins
        FROM votes
        WHERE query IN ({placeholders})
        GROUP BY query
        """,
        FIXED_QUERIES,
    )
    win_counts: dict[str, tuple[int, int, int]] = {
        q: (int(a), int(t), int(b)) for q, a, t, b in cur.fetchall()
    }
    con.close()

    # Group queries by contrast pair (preserving a canonical order)
    groups: dict[tuple[str, str], list[tuple[str, int, int, int]]] = {}
    for query in FIXED_QUERIES:
        if query not in contrasts:
            continue
        pair = contrasts[query]
        a, t, b = win_counts.get(query, (0, 0, 0))
        groups.setdefault(pair, []).append((query, a, t, b))

    # Sort queries within each group by total votes descending
    result = []
    for pair, rows in groups.items():
        rows.sort(key=lambda r: r[1] + r[2] + r[3], reverse=True)
        result.append((pair[0], pair[1], rows))

    # Sort groups by model_a then model_b for a stable order
    result.sort(key=lambda g: (g[0], g[1]))
    return result


def fmt_pct_ci(k: int, n: int) -> str:
    """Format a proportion as 'XX (lo--hi)' with Wilson 95% CI, all rounded to int %."""
    if n == 0:
        return "---"
    lo, hi = wilson_ci(k, n)
    return f"{round(100 * k / n)} ({round(100 * lo)}--{round(100 * hi)})"


def print_latex_win_rates(groups: list[tuple[str, str, list[tuple[str, int, int, int]]]]) -> None:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "\\textbf{Query} & \\textbf{A\\,\\%} & \\textbf{Tie\\,\\%} & \\textbf{B\\,\\%} & \\textbf{N} \\\\",
        "& \\multicolumn{3}{c}{\\scriptsize value (95\\% CI)} & \\\\",
    ]

    for model_a, model_b, rows in groups:
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{5}}{{l}}{{\\textit{{\\texttt{{{model_a}}} vs \\texttt{{{model_b}}}}}}}\\\\"
        )
        lines.append("\\midrule")

        g_a = g_t = g_b = 0
        for query, a_wins, ties, b_wins in rows:
            n = a_wins + ties + b_wins
            g_a += a_wins; g_t += ties; g_b += b_wins
            lines.append(
                f"\\quad {query}"
                f" & {fmt_pct_ci(a_wins, n)}"
                f" & {fmt_pct_ci(ties, n)}"
                f" & {fmt_pct_ci(b_wins, n)}"
                f" & {n} \\\\"
            )

        g_n = g_a + g_t + g_b
        lines.append(
            f"\\quad \\textit{{subtotal}}"
            f" & {fmt_pct_ci(g_a, g_n)}"
            f" & {fmt_pct_ci(g_t, g_n)}"
            f" & {fmt_pct_ci(g_b, g_n)}"
            f" & {g_n} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Win rates with Wilson 95\\% confidence intervals for the 10 fixed queries, "
        "grouped by model-pair contrast. "
        "Model A and Model B refer to the canonical assignment in the \\texttt{query\\_contrasts} table.}",
        "\\label{tab:win_rates_by_contrast}",
        "\\end{table}",
    ]

    print("\n".join(lines))


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print fixed-query vote stats as LaTeX tables."
    )
    parser.add_argument("--db", type=Path, default=DB_DEFAULT, help="Path to arena.db")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}")

    print("% ── Table 1: vote counts by workshop session ────────────────────────────\n")
    sessions, counts, totals = fetch_counts(args.db)
    print_latex_counts(sessions, counts, totals)

    print("\n\n% ── Table 2: win rates grouped by model-pair contrast ───────────────────\n")
    groups = fetch_win_rates(args.db)
    print_latex_win_rates(groups)


if __name__ == "__main__":
    main()
