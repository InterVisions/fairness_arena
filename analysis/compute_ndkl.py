"""
compute_ndkl.py
===============
Compute NDKL@50 (Normalized Discounted Cumulative KL-divergence) for each vote
in the exported analysis CSV, using Chicago Face Database (CFD) metadata.

NDKL implementation follows vl_bias_metrics.py:
  - desired_dist is uniform over known groups (demographic parity baseline)
  - eps-smoothed KL to avoid log(0) without returning inf
  - Z normalises by sum of discounts over the full ranking depth

Inputs
------
  --votes     Path to analysis export CSV  (default: ../data/analysis_export.csv)
  --metadata  Path to face_metadata.csv    (default: ../data/face_metadata.csv)
  --output    Path for output CSV           (default: automated_metrics.csv)
  --k         Ranking depth                 (default: 50)

face_metadata.csv expected columns: image_id, race, gender, age
  image_id must match the integer indices stored in ranking_left / ranking_right.

Output: automated_metrics.csv
  vote_id, ndkl_gender_left, ndkl_gender_right,
  ndkl_race_left, ndkl_race_right,
  ndkl_age_left, ndkl_age_right

Other metrics from vl_bias_metrics.py (SC-WEAT, Markedness, MaxSkew) require
live CLIP embeddings and are not computed here from stored rankings alone.
Run vl_bias_metrics.py directly against the full dataset for those.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import numpy as np
from pathlib import Path
from collections import Counter


# ── NDKL — ported directly from vl_bias_metrics.py ───────────────────────

EPS = 1e-12


def ndkl(ranked_labels: list[str], desired_dist: dict[str, float]) -> float:
    """
    Normalized Discounted KL-Divergence.

    Measures how far ranked exposure diverges from desired_dist,
    with higher ranks weighted more heavily (via log2 discount).

    ranked_labels : group label for each item in rank order
    desired_dist  : target distribution, e.g. {group: 1/n_groups} for parity
    """
    if not ranked_labels:
        return float("nan")

    Z = sum(1.0 / np.log2(i + 2) for i in range(len(ranked_labels)))
    counts = {g: 0 for g in desired_dist}
    acc = 0.0

    for i, lab in enumerate(ranked_labels):
        if lab in counts:
            counts[lab] += 1
        total = i + 1
        emp = {g: counts[g] / total for g in desired_dist}
        kl = sum(
            (emp[g] + EPS) * np.log((emp[g] + EPS) / (desired_dist[g] + EPS))
            for g in desired_dist
        )
        acc += kl / np.log2(i + 2)

    return float(acc / Z) if Z > 0 else float("nan")


def max_skew_at_k(topk_labels: list[str], desired_dist: dict[str, float]) -> float:
    """
    Maximum absolute log-skew in top-K ranking.
    Ported from vl_bias_metrics.py.

    skew(g) = log( p_tau(g) / p_d(g) )
    """
    if not topk_labels:
        return float("nan")
    skews = {}
    for g, p_d in desired_dist.items():
        p_tau = topk_labels.count(g) / len(topk_labels)
        skews[g] = np.log((p_tau + EPS) / (p_d + EPS))
    return float(max(abs(v) for v in skews.values()))


# ── I/O helpers ───────────────────────────────────────────────────────────

def load_metadata(path: str) -> tuple[dict, dict, dict]:
    """Return (gender_map, race_map, age_map) keyed by filename."""
    gender, race, age = {}, {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("filename") or row.get("image_id")
            if key is None:
                continue
            gender[key] = row.get("gender", "unknown")
            race[key]   = row.get("race",   "unknown")
            age[key]    = row.get("age",     "unknown")
    return gender, race, age


def load_filenames_from_bundle(bundle_path: str) -> list[str]:
    """Return ordered list of image basenames from bundle."""
    import json as _json
    data = np.load(bundle_path, allow_pickle=False)
    if "filenames_json" not in data:
        raise KeyError(
            "Bundle does not contain 'filenames_json'. "
            "Rebuild the bundle with the updated precompute.py."
        )
    return _json.loads(str(data["filenames_json"][0]))


def parse_ranking(json_str: str) -> list[int]:
    if not json_str:
        return []
    try:
        return [int(x) for x in json.loads(json_str)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def to_labels(ranking: list[int], attr_map: dict[int, str], k: int) -> list[str]:
    """Convert list of image ids to group-label list, dropping unknown ids, capped at k."""
    return [attr_map[i] for i in ranking if i in attr_map][:k]


def uniform_dist(labels: list[str]) -> dict[str, float]:
    """Uniform distribution over the unique groups seen in labels."""
    groups = sorted(set(labels))
    return {g: 1.0 / len(groups) for g in groups} if groups else {}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--votes",    default="../data/analysis_export.csv")
    p.add_argument("--metadata", default="../data/face_metadata.csv")
    p.add_argument("--bundle",   default=None,
                   help="Path to .npz bundle — used to map integer indices to filenames. "
                        "Required when face_metadata.csv uses filename keys (recommended).")
    p.add_argument("--output",   default="automated_metrics.csv")
    p.add_argument("--k",        type=int, default=50)
    args = p.parse_args()

    if not Path(args.metadata).exists():
        raise FileNotFoundError(
            f"face_metadata.csv not found at {args.metadata}. "
            "Run make_face_metadata.py --bundle <bundle.npz> first."
        )

    gender_map, race_map, age_map = load_metadata(args.metadata)

    # If metadata is filename-keyed, build idx->key via bundle
    first_key = next(iter(gender_map))
    if not isinstance(first_key, int) and not first_key.isdigit():
        # filename-keyed metadata: need bundle to resolve integer indices
        if not args.bundle:
            raise SystemExit(
                "face_metadata.csv uses filename keys. "
                "Pass --bundle path/to/bundle.npz to map indices to filenames."
            )
        filenames = load_filenames_from_bundle(args.bundle)
        gender_map = {i: gender_map.get(fn, "unknown") for i, fn in enumerate(filenames)}
        race_map   = {i: race_map.get(fn,   "unknown") for i, fn in enumerate(filenames)}
        age_map    = {i: age_map.get(fn,    "unknown") for i, fn in enumerate(filenames)}
        print(f"Loaded {len(filenames)} filenames from bundle")

    # Build per-attribute uniform desired distributions over all known groups
    desired = {
        "gender": uniform_dist(list(gender_map.values())),
        "race":   uniform_dist(list(race_map.values())),
        "age":    uniform_dist(list(age_map.values())),
    }
    for attr, dist in desired.items():
        print(f"  {attr} groups ({len(dist)}): {list(dist.keys())}")

    rows_out = []
    attr_maps = {"gender": gender_map, "race": race_map, "age": age_map}

    with open(args.votes, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid   = row["vote_id"]
            left  = parse_ranking(row.get("ranking_left",  ""))
            right = parse_ranking(row.get("ranking_right", ""))

            out = {"vote_id": vid}
            for attr, amap in attr_maps.items():
                dist = desired[attr]
                ll = to_labels(left,  amap, args.k)
                rl = to_labels(right, amap, args.k)
                out[f"ndkl_{attr}_left"]    = round(ndkl(ll, dist),             6)
                out[f"ndkl_{attr}_right"]   = round(ndkl(rl, dist),             6)
                out[f"maxskew_{attr}_left"]  = round(max_skew_at_k(ll, dist),   6)
                out[f"maxskew_{attr}_right"] = round(max_skew_at_k(rl, dist),   6)
            rows_out.append(out)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_out[0].keys()) if rows_out else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Written {len(rows_out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
