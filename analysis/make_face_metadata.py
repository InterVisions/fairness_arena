"""
make_face_metadata.py
=====================
Generate face_metadata.csv from CFD image filenames stored in a bundle.

CFD filename format: CFD-{race}{gender}-{person_id}-{attractiveness}-{expression}.jpg
  Race codes:   A=Asian, B=Black, L=Latino/Hispanic, W=White, M=Multiracial
  Gender codes: F=Female, M=Male

Usage
-----
    # From bundle (recommended — indices are guaranteed to match)
    python make_face_metadata.py --bundle ../data/arena_bundle_cfd.npz

    # From folder (fallback — assumes same sort order as when bundle was built)
    python make_face_metadata.py --folder /data/datasets/CFD/CFD_balanced50/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

RACE_MAP = {
    "A": "Asian",
    "B": "Black",
    "L": "Latino",
    "W": "White",
    "M": "Multiracial",
}

GENDER_MAP = {
    "F": "Female",
    "M": "Male",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_PATTERN = re.compile(r"CFD-([A-Z])([FM])-", re.IGNORECASE)


def parse_filename(name: str) -> tuple[str, str]:
    m = _PATTERN.search(name)
    if not m:
        return "unknown", "unknown"
    race   = RACE_MAP.get(m.group(1).upper(), "unknown")
    gender = GENDER_MAP.get(m.group(2).upper(), "unknown")
    return race, gender


def filenames_from_bundle(bundle_path: str) -> list[str]:
    data = np.load(bundle_path, allow_pickle=False)
    if "filenames_json" not in data:
        raise KeyError(
            "Bundle does not contain 'filenames_json'. "
            "Rebuild the bundle with the updated precompute.py."
        )
    return json.loads(str(data["filenames_json"][0]))


def filenames_from_folder(folder: str, max_images: int | None = None) -> list[str]:
    p = Path(folder)
    paths = sorted([f for f in p.rglob("*") if f.suffix.lower() in EXTENSIONS])
    if max_images:
        paths = paths[:max_images]
    return [f.name for f in paths]


def build_metadata(filenames: list[str]) -> list[dict]:
    rows = []
    for fname in filenames:
        race, gender = parse_filename(fname)
        rows.append({"filename": fname, "race": race, "gender": gender, "age": "unknown"})
    return rows


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", help="Path to .npz bundle (recommended)")
    src.add_argument("--folder", help="Path to image folder (fallback)")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--output", default="face_metadata.csv")
    args = p.parse_args()

    if args.bundle:
        filenames = filenames_from_bundle(args.bundle)
        print(f"Loaded {len(filenames)} filenames from bundle")
    else:
        filenames = filenames_from_folder(args.folder, args.max_images)
        print(f"Loaded {len(filenames)} filenames from folder")

    rows = build_metadata(filenames)
    unknown = sum(1 for r in rows if r["race"] == "unknown")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "race", "gender", "age"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows -> {args.output}")
    if unknown:
        print(f"  Warning: {unknown} filenames did not match expected CFD pattern")

    races   = Counter(r["race"]   for r in rows)
    genders = Counter(r["gender"] for r in rows)
    print(f"  Race distribution:   {dict(sorted(races.items()))}")
    print(f"  Gender distribution: {dict(sorted(genders.items()))}")


if __name__ == "__main__":
    main()
