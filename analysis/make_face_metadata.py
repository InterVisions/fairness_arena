"""
make_face_metadata.py
=====================
Generate face_metadata.csv from CFD image filenames.

CFD filename format: CFD-{race}{gender}-{person_id}-{attractiveness}-{expression}.jpg
  Race codes:   A=Asian, B=Black, L=Latino/Hispanic, W=White
  Gender codes: F=Female, M=Male

The image_id assigned to each file is its 0-based index in the sorted file list,
which matches the order used by load_dataset_from_folder() in retrieval.py.

Usage
-----
    python make_face_metadata.py --folder /data/datasets/CFD/CFD_balanced50/
    python make_face_metadata.py --folder /data/datasets/CFD/CFD_balanced50/ --output ../data/face_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

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

# CFD-{race}{gender}-{person_id}-...-N.jpg
_PATTERN = re.compile(r"CFD-([A-Z])([FM])-", re.IGNORECASE)


def parse_filename(name: str) -> tuple[str, str]:
    m = _PATTERN.search(name)
    if not m:
        return "unknown", "unknown"
    race   = RACE_MAP.get(m.group(1).upper(), "unknown")
    gender = GENDER_MAP.get(m.group(2).upper(), "unknown")
    return race, gender


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, help="Path to CFD image folder")
    p.add_argument("--output", default="face_metadata.csv")
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    folder = Path(args.folder)
    paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS])
    if args.max_images:
        paths = paths[:args.max_images]

    rows = []
    unknown = 0
    for idx, path in enumerate(paths):
        race, gender = parse_filename(path.name)
        if race == "unknown":
            unknown += 1
        rows.append({"image_id": idx, "race": race, "gender": gender, "age": "unknown"})

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "race", "gender", "age"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows -> {args.output}")
    if unknown:
        print(f"  Warning: {unknown} filenames did not match expected CFD pattern")

    # Summary
    from collections import Counter
    races   = Counter(r["race"]   for r in rows)
    genders = Counter(r["gender"] for r in rows)
    print(f"  Race distribution:   {dict(sorted(races.items()))}")
    print(f"  Gender distribution: {dict(sorted(genders.items()))}")


if __name__ == "__main__":
    main()
