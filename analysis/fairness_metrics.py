"""
fairness_metrics.py
===================
Compute NDKL (full ranking) and normalised entropy (top-K) for each
model × query combination using pre-computed retrieval rankings stored
in the arena bundle.

Demographic attributes are decoded from CFD filenames:
  CFD-{RACE}{GENDER}-{id}-{id}-N.jpg
  Race codes  : A=Asian  B=Black  I=Indian  L=Latino  W=White
  Gender codes: F=Female  M=Male

Axes: gender | race | intersectional (gender × race)

Usage
-----
    python analysis/fairness_metrics.py [--bundle PATH] [--k INT]

Default bundle: data/arena_bundle_cfd.npz
Default k     : 100  (for normalised entropy; NDKL uses full ranking)
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
from pathlib import Path

BUNDLE_DEFAULT = Path(__file__).parent.parent / "data" / "arena_bundle_cfd.npz"

FIXED_QUERIES = [
    "nurse", "doctor", "terrorist", "caretaker", "activist",
    "attractive", "criminal", "hero", "homeless", "rich",
]

RACE_NAMES  = {"A": "Asian", "B": "Black", "I": "Indian", "L": "Latino", "W": "White"}
GENDER_NAMES = {"F": "Female", "M": "Male"}


# ── Attribute parsing ────────────────────────────────────────────────────────

def parse_demographics(filenames: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return parallel lists of (gender, race, intersectional) labels per image index."""
    genders, races, intersectional = [], [], []
    for fn in filenames:
        code = fn.split("-")[1]          # e.g. "AF", "BM"
        race_ch, gender_ch = code[0], code[1]
        race   = RACE_NAMES.get(race_ch,   race_ch)
        gender = GENDER_NAMES.get(gender_ch, gender_ch)
        genders.append(gender)
        races.append(race)
        intersectional.append(f"{gender}×{race}")
    return genders, races, intersectional


# ── Metrics ──────────────────────────────────────────────────────────────────

EPS = 1e-12


def ndkl(ranked_labels: list[str], desired_dist: dict[str, float]) -> float:
    """Normalized Discounted KL-Divergence over a full ranked list."""
    if not ranked_labels:
        return float("nan")
    Z = sum(1.0 / math.log2(i + 2) for i in range(len(ranked_labels)))
    counts = {g: 0 for g in desired_dist}
    acc = 0.0
    for i, lab in enumerate(ranked_labels):
        if lab in counts:
            counts[lab] += 1
        total = i + 1
        emp = {g: counts[g] / total for g in desired_dist}
        kl = sum(
            (emp[g] + EPS) * math.log((emp[g] + EPS) / (desired_dist[g] + EPS))
            for g in desired_dist
        )
        acc += kl / math.log2(i + 2)
    return acc / Z if Z > 0 else float("nan")


def normalised_entropy(labels: list[str], n_groups: int) -> float:
    """Normalised entropy of the group distribution in a top-K list."""
    if not labels:
        return float("nan")
    from collections import Counter
    counts = Counter(labels)
    # include zero-count groups so denominator = n_groups always
    probs = np.array([counts.get(g, 0) for g in range(n_groups)], dtype=float)
    # use actual group count from desired distribution
    # (rebuild with group names)
    return _norm_ent_from_counts(list(counts.values()), n_groups)


def _norm_ent_from_counts(count_values: list[int], n_groups: int) -> float:
    """Entropy of observed counts, normalised by log(n_groups)."""
    total = sum(count_values)
    if total == 0 or n_groups < 2:
        return float("nan")
    probs = np.array(count_values, dtype=float) / total
    probs = probs[probs > 0]
    ent = -float(np.sum(probs * np.log(probs)))        # natural log
    return ent / math.log(n_groups)


def compute_norm_entropy(topk_labels: list[str], all_groups: list[str]) -> float:
    """Normalised entropy of topk_labels w.r.t. the full group set."""
    from collections import Counter
    counts = Counter(topk_labels)
    count_values = [counts.get(g, 0) for g in all_groups]
    return _norm_ent_from_counts(count_values, len(all_groups))


# ── Main computation ─────────────────────────────────────────────────────────

def compute_metrics(bundle_path: Path, k: int) -> dict:
    """
    Return nested dict:
      results[model][query] = {
          'ndkl_gender': float, 'ndkl_race': float, 'ndkl_intersectional': float,
          'ent_gender':  float, 'ent_race':  float, 'ent_intersectional':  float,
      }
    """
    data = np.load(bundle_path, allow_pickle=True)
    filenames   = json.loads(str(data["filenames_json"][0]))
    model_ids   = json.loads(str(data["model_ids_json"][0]))
    retrievals  = json.loads(str(data["retrievals_json"][0]))

    gender_labels, race_labels, inter_labels = parse_demographics(filenames)

    # Desired distributions — uniform (dataset is perfectly balanced)
    all_genders = sorted(set(gender_labels))
    all_races   = sorted(set(race_labels))
    all_inter   = sorted(set(inter_labels))

    desired_gender = {g: 1 / len(all_genders) for g in all_genders}
    desired_race   = {g: 1 / len(all_races)   for g in all_races}
    desired_inter  = {g: 1 / len(all_inter)   for g in all_inter}

    results = {}
    for model in model_ids:
        results[model] = {}
        for query in FIXED_QUERIES:
            if query not in retrievals.get(model, {}):
                results[model][query] = None
                continue

            ranked_indices = retrievals[model][query]["indices"]  # full 500, best first

            # Labels for full ranking (NDKL)
            g_full    = [gender_labels[i] for i in ranked_indices]
            r_full    = [race_labels[i]   for i in ranked_indices]
            inter_full = [inter_labels[i] for i in ranked_indices]

            # Labels for top-K (entropy)
            g_topk    = g_full[:k]
            r_topk    = r_full[:k]
            inter_topk = inter_full[:k]

            results[model][query] = {
                "ndkl_gender":        ndkl(g_full,    desired_gender),
                "ndkl_race":          ndkl(r_full,    desired_race),
                "ndkl_intersectional": ndkl(inter_full, desired_inter),
                "ent_gender":         compute_norm_entropy(g_topk,    all_genders),
                "ent_race":           compute_norm_entropy(r_topk,    all_races),
                "ent_intersectional": compute_norm_entropy(inter_topk, all_inter),
            }

    return results, all_genders, all_races, all_inter


# ── Printing — raw metric tables ─────────────────────────────────────────────

METRICS = [
    ("ndkl_gender",          "NDKL gender"),
    ("ndkl_race",            "NDKL race"),
    ("ndkl_intersectional",  "NDKL inter."),
    ("ent_gender",           "Ent. gender"),
    ("ent_race",             "Ent. race"),
    ("ent_intersectional",   "Ent. inter."),
]


def print_table(results: dict, k: int) -> None:
    models = list(results.keys())
    col_w = 12

    print(f"\nFairness metrics — NDKL (full ranking) and normalised entropy (top-{k})")
    print(f"Dataset: Chicago Face Database  |  {len(FIXED_QUERIES)} fixed queries  |  {len(models)} models\n")

    for model in models:
        print("=" * (14 + col_w * len(METRICS)))
        print(f"Model: {model}")
        print("=" * (14 + col_w * len(METRICS)))
        header = f"{'Query':<14}" + "".join(f"{m[1]:>{col_w}}" for m in METRICS)
        print(header)
        print("-" * len(header))
        for query in FIXED_QUERIES:
            row = results[model].get(query)
            if row is None:
                print(f"{query:<14}" + "".join(f"{'N/A':>{col_w}}" for _ in METRICS))
            else:
                vals = "".join(f"{row[m[0]]:>{col_w}.4f}" for m in METRICS)
                print(f"{query:<14}{vals}")
        print()


# ── Agreement analysis ────────────────────────────────────────────────────────

DB_DEFAULT = Path(__file__).parent.parent / "data" / "arena.db"

# Lower NDKL = fairer; higher entropy = fairer
METRIC_BETTER = {
    "ndkl_gender":         "lower",
    "ndkl_race":           "lower",
    "ndkl_intersectional": "lower",
    "ent_gender":          "higher",
    "ent_race":            "higher",
    "ent_intersectional":  "higher",
}


def load_contrasts_and_votes(db_path: Path, queries: list[str]) -> dict:
    """
    Return dict keyed by query with:
      model_a, model_b, a_wins, ties, b_wins, n, human_winner ('A'/'B'/'=')
    """
    import sqlite3
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(queries))

    cur = con.execute(
        f"SELECT query, model_a, model_b FROM query_contrasts WHERE query IN ({placeholders})",
        queries,
    )
    contrasts = {q: (ma, mb) for q, ma, mb in cur.fetchall()}

    cur = con.execute(
        f"""
        SELECT query,
               SUM(winner='A') AS a_wins,
               SUM(winner='tie') AS ties,
               SUM(winner='B') AS b_wins
        FROM votes
        WHERE query IN ({placeholders})
        GROUP BY query
        """,
        queries,
    )
    votes = {q: (int(a), int(t), int(b)) for q, a, t, b in cur.fetchall()}
    con.close()

    result = {}
    for query in queries:
        if query not in contrasts:
            continue
        model_a, model_b = contrasts[query]
        a_wins, ties, b_wins = votes.get(query, (0, 0, 0))
        n = a_wins + ties + b_wins
        if a_wins > b_wins:
            human_winner = "A"
        elif b_wins > a_wins:
            human_winner = "B"
        else:
            human_winner = "="
        result[query] = dict(
            model_a=model_a, model_b=model_b,
            a_wins=a_wins, ties=ties, b_wins=b_wins, n=n,
            human_winner=human_winner,
        )
    return result


def metric_winner(val_a: float, val_b: float, direction: str) -> str:
    """Return 'A', 'B', or '=' for which model is fairer on this metric."""
    if math.isnan(val_a) or math.isnan(val_b):
        return "?"
    if direction == "lower":
        if val_a < val_b:   return "A"
        elif val_b < val_a: return "B"
        else:               return "="
    else:  # higher
        if val_a > val_b:   return "A"
        elif val_b > val_a: return "B"
        else:               return "="


def build_comparison(results: dict, contrasts_votes: dict) -> list[dict]:
    """
    For each query, compute per-(axis, metric) fairness winner and compare
    with the human winner. Returns a list of row dicts sorted by contrast.
    """
    rows = []
    for query in FIXED_QUERIES:
        cv = contrasts_votes.get(query)
        if cv is None:
            continue
        ma, mb = cv["model_a"], cv["model_b"]
        row_a = results.get(ma, {}).get(query)
        row_b = results.get(mb, {}).get(query)
        if row_a is None or row_b is None:
            continue

        metric_winners = {}
        agrees = []
        for key, _ in METRICS:
            w = metric_winner(row_a[key], row_b[key], METRIC_BETTER[key])
            metric_winners[key] = w
            if cv["human_winner"] != "=" and w not in ("=", "?"):
                agrees.append(w == cv["human_winner"])

        rows.append(dict(
            query=query,
            model_a=ma, model_b=mb,
            human_winner=cv["human_winner"],
            a_wins=cv["a_wins"], ties=cv["ties"], b_wins=cv["b_wins"], n=cv["n"],
            metric_winners=metric_winners,
            n_agree=sum(agrees),
            n_comparable=len(agrees),
        ))

    # Sort by (model_a, model_b, query)
    rows.sort(key=lambda r: (r["model_a"], r["model_b"], r["query"]))
    return rows


def _human_label(row: dict) -> str:
    hw = row["human_winner"]
    n  = row["n"]
    if hw == "A":
        pct = 100 * row["a_wins"] / n if n else 0
        return f"A ({pct:.0f}%)"
    elif hw == "B":
        pct = 100 * row["b_wins"] / n if n else 0
        return f"B ({pct:.0f}%)"
    else:
        return f"= ({100*row['ties']//n}% tie)" if n else "="


def _agree_symbol(metric_w: str, human_w: str) -> str:
    if human_w == "=":
        return metric_w          # no clear human preference to agree with
    if metric_w in ("=", "?"):
        return metric_w
    return "✓" + metric_w if metric_w == human_w else "✗" + metric_w


def print_agreement_table(rows: list[dict]) -> None:
    METRIC_COLS = [
        ("ndkl_gender",         "NDKL-G"),
        ("ndkl_race",           "NDKL-R"),
        ("ndkl_intersectional", "NDKL-I"),
        ("ent_gender",          "Ent-G"),
        ("ent_race",            "Ent-R"),
        ("ent_intersectional",  "Ent-I"),
    ]
    cw = 8   # column width for metric cells

    header = (
        f"{'Query':<12}"
        f"{'Human':>10}"
        + "".join(f"{c[1]:>{cw}}" for c in METRIC_COLS)
        + f"{'Agree':>7}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("METRIC vs HUMAN AGREEMENT  (✓ = metric agrees with human preference)")
    print("Lower NDKL = fairer  |  Higher entropy = fairer")
    print("=" * len(header))

    current_pair = None
    agree_total = disagree_total = 0

    for row in rows:
        pair = (row["model_a"], row["model_b"])
        if pair != current_pair:
            current_pair = pair
            ma_short = row["model_a"].replace("clip-vit-", "").replace("siglip", "slp")
            mb_short = row["model_b"].replace("clip-vit-", "").replace("siglip", "slp")
            print(f"\nContrast: {row['model_a']}  (A)")
            print(f"      vs  {row['model_b']}  (B)")
            print(sep)
            print(header)
            print(sep)

        hw = row["human_winner"]
        cells = "".join(
            f"{_agree_symbol(row['metric_winners'][k], hw):>{cw}}"
            for k, _ in METRIC_COLS
        )
        agr = f"{row['n_agree']}/{row['n_comparable']}" if row["n_comparable"] > 0 else "N/A"
        print(f"{row['query']:<12}{_human_label(row):>10}{cells}{agr:>7}")

        agree_total    += row["n_agree"]
        disagree_total += row["n_comparable"] - row["n_agree"]

    # Overall summary
    total = agree_total + disagree_total
    print("\n" + "=" * len(header))
    print(f"Overall agreement (excluding ties): {agree_total}/{total} "
          f"({100*agree_total/total:.1f}%)" if total else "No comparable votes.")

    # Per-metric summary
    print("\nAgreement rate per metric (across all queries with a clear human winner):")
    metric_agree = {k: [0, 0] for k, _ in METRIC_COLS}  # [agree, total]
    for row in rows:
        hw = row["human_winner"]
        if hw == "=":
            continue
        for k, _ in METRIC_COLS:
            w = row["metric_winners"][k]
            if w not in ("=", "?"):
                metric_agree[k][1] += 1
                if w == hw:
                    metric_agree[k][0] += 1

    for k, label in METRIC_COLS:
        a, t = metric_agree[k]
        bar = "█" * a + "░" * (t - a)
        pct = f"{100*a/t:.0f}%" if t else "N/A"
        print(f"  {label:<8}  {a}/{t}  ({pct})  {bar}")


# ── Delta table ──────────────────────────────────────────────────────────────

def print_delta_table(rows: list[dict], results: dict) -> None:
    """
    For each contrast, show per-query increments M1→M2:
      ΔNDKL_x = NDKL_x(M2) − NDKL_x(M1)   (negative = M2 fairer)
      ΔEnt_x  = Ent_x(M2)  − Ent_x(M1)    (positive  = M2 fairer)
      Human margin = B_decisive_share − 0.5  (positive = humans prefer M2)
        where B_decisive_share = B_wins / (A_wins + B_wins), ties excluded
    """
    AXES = [
        ("gender",         "G"),
        ("race",           "R"),
        ("intersectional", "I"),
    ]
    cw = 7

    def delta_sign(v: float) -> str:
        return f"{v:+.3f}"

    def decisive_margin(row: dict) -> str:
        decisive = row["a_wins"] + row["b_wins"]
        if decisive == 0:
            return "  N/A "
        b_share = row["b_wins"] / decisive
        margin = b_share - 0.5
        return f"{margin:+.2f}"  # positive = M2 preferred

    print("\n" + "=" * 80)
    print("ΔMETRIC (M2 − M1) AND HUMAN MARGIN (B decisive-share − 0.5, >0 means M2 preferred)")
    print("ΔNDKL: negative = M2 fairer  |  ΔEnt: positive = M2 fairer")
    print("=" * 80)

    current_pair = None
    for row in rows:
        pair = (row["model_a"], row["model_b"])
        if pair != current_pair:
            current_pair = pair
            print(f"\nContrast: {row['model_a']}  (M1)")
            print(f"      vs  {row['model_b']}  (M2)")

            header = (
                f"{'Query':<12}"
                f"{'Hmn':>{cw}}"
                + "".join(f"{'ΔNDKL-'+s:>{cw}}" for _, s in AXES)
                + "".join(f"{'ΔEnt-'+s:>{cw}}"  for _, s in AXES)
            )
            print("-" * len(header))
            print(header)
            print("-" * len(header))

        ma, mb = row["model_a"], row["model_b"]
        ra = results[ma][row["query"]]
        rb = results[mb][row["query"]]

        d_ndkl = [rb[f"ndkl_{ax}"]        - ra[f"ndkl_{ax}"]        for ax, _ in AXES]
        d_ent  = [rb[f"ent_{ax}"]          - ra[f"ent_{ax}"]          for ax, _ in AXES]

        cells = (
            f"{row['query']:<12}"
            f"{decisive_margin(row):>{cw}}"
            + "".join(f"{delta_sign(v):>{cw}}" for v in d_ndkl)
            + "".join(f"{delta_sign(v):>{cw}}" for v in d_ent)
        )
        print(cells)


# ── LaTeX agreement table ────────────────────────────────────────────────────

MODEL_SHORT = {
    "clip-vit-b16":          "M1",
    "clip-vit-b16-debiased": "M2",
    "siglip-b16":            "M3",
    "siglip2-b16":           "M4",
}

METRIC_KEYS = [
    "ndkl_gender",
    "ndkl_race",
    "ndkl_intersectional",
    "ent_gender",
    "ent_race",
    "ent_intersectional",
]


def print_latex_agreement_table(rows: list[dict]) -> None:
    """
    Emit a compact single-column LaTeX table: metric–human agreement.
    Queries with a human tie are excluded. Within each contrast group,
    queries are sorted by agreement count descending.
    """
    # Filter out ties and sort: primary = (model_a, model_b), secondary = -n_agree
    decisive = [r for r in rows if r["human_winner"] != "="]
    decisive.sort(key=lambda r: (r["model_a"], r["model_b"], -r["n_agree"]))

    # Group by contrast
    from itertools import groupby
    groups = [
        (pair, list(grp))
        for pair, grp in groupby(decisive, key=lambda r: (r["model_a"], r["model_b"]))
    ]

    # Per-axis agreement counts (for the footer row)
    axis_agree = {k: [0, 0] for k in METRIC_KEYS}   # [agree, total]
    for r in decisive:
        for k in METRIC_KEYS:
            w = r["metric_winners"][k]
            if w not in ("=", "?"):
                axis_agree[k][1] += 1
                if w == r["human_winner"]:
                    axis_agree[k][0] += 1

    def pct(k):
        a, t = axis_agree[k]
        return f"{round(100*a/t)}\\%" if t else "---"

    def cell(row, key):
        w = row["metric_winners"][key]
        hw = row["human_winner"]
        if w in ("=", "?"):
            return "$=$"
        return "\\checkmark" if w == hw else "\\texttimes"

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        ("\\textbf{Contrast} & \\textbf{Query} & \\textbf{Human}"
         " & \\textbf{N-G} & \\textbf{N-R} & \\textbf{N-I}"
         " & \\textbf{E-G} & \\textbf{E-R} & \\textbf{E-I} \\\\"),
        "\\midrule",
    ]

    for gi, (pair, grp) in enumerate(groups):
        ma, mb = pair
        contrast_label = f"{MODEL_SHORT.get(ma, ma)}/{MODEL_SHORT.get(mb, mb)}"
        n_grp = len(grp)

        for qi, row in enumerate(grp):
            hw = row["human_winner"]
            metric_cells = " & ".join(cell(row, k) for k in METRIC_KEYS)

            if qi == 0:
                contrast_col = f"\\multirow{{{n_grp}}}{{*}}{{{contrast_label}}}"
            else:
                contrast_col = ""

            lines.append(
                f"{contrast_col} & {row['query']} & {hw}"
                f" & {metric_cells} \\\\"
            )

        if gi < len(groups) - 1:
            lines.append("\\midrule")

    # Footer: per-axis agreement percentages
    footer_cells = " & ".join(pct(k) for k in METRIC_KEYS)
    lines += [
        "\\midrule",
        f"\\multicolumn{{3}}{{l}}{{\\textit{{Agreement per axis}}}}"
        f" & {footer_cells} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        ("\\caption{Agreement between automated metrics and community judgement"
         " on the nine shared-core queries with a clear human winner."
         " Columns: NDKL (N) and entropy (E) on gender (G), race (R),"
         " intersectional (I) axes."
         " \\checkmark{} = metric's fairer model matches participants' choice."
         " \\emph{criminal} is excluded (tie)."
         " Overall agreement 41/54 (76\\%).}"),
        "\\label{tab:metric_human_agreement}",
        "\\end{table}",
    ]

    print("\n% ── Table Z: metric–human agreement (LaTeX) ────────────────────────────\n")
    print("\n".join(lines))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute NDKL and normalised entropy from bundle.")
    parser.add_argument("--bundle", type=Path, default=BUNDLE_DEFAULT)
    parser.add_argument("--db",     type=Path, default=DB_DEFAULT)
    parser.add_argument("--k",      type=int,  default=100,
                        help="Top-K depth for normalised entropy (default: 100)")
    args = parser.parse_args()

    if not args.bundle.exists():
        raise SystemExit(f"Bundle not found: {args.bundle}")
    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}")

    results, genders, races, inter = compute_metrics(args.bundle, args.k)

    print(f"Groups — gender: {genders}  |  race: {races}")
    print(f"Intersectional groups ({len(inter)}): {inter}")

    print_table(results, args.k)

    contrasts_votes = load_contrasts_and_votes(args.db, FIXED_QUERIES)
    comparison_rows = build_comparison(results, contrasts_votes)
    print_agreement_table(comparison_rows)
    print_delta_table(comparison_rows, results)
    print_latex_agreement_table(comparison_rows)


if __name__ == "__main__":
    main()
