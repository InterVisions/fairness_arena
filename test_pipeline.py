"""
test_pipeline.py
================
Integration test for the AIES26 data-collection pipeline.

Creates 4 synthetic workshops, inserts 100 votes per workshop (400 total),
exports analysis CSV, runs all analysis scripts, and validates outputs.

Usage:
    python test_pipeline.py
    python test_pipeline.py --db /tmp/test_arena.db
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import database as db

SYNTHETIC_WORKSHOPS = [
    {"name": "LGBT+ Community Barcelona", "location": "Barcelona", "date": "2026-03-01",
     "community_context": "LGBTQ+", "facilitator": "Ana Garcia"},
    {"name": "Roma Youth Madrid",         "location": "Madrid",    "date": "2026-03-08",
     "community_context": "Roma",    "facilitator": "Pedro Lopez"},
    {"name": "Migrants Solidarity Hub",   "location": "Valencia",  "date": "2026-03-15",
     "community_context": "Migrants","facilitator": "Sara Kim"},
    {"name": "Civil Servants Seminar",    "location": "Seville",   "date": "2026-03-22",
     "community_context": "Civil Servants", "facilitator": "Marta Ruiz"},
]

MODEL_PAIRS = [
    ("openai/clip-vit-base-patch16",           "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"),
    ("openai/clip-vit-base-patch16",           "google/siglip-base-patch16-224"),
    ("google/siglip-base-patch16-224",         "google/siglip2-base-patch16-224"),
    ("laion/CLIP-ViT-B-16-laion2B-s34B-b88K", "M2_SANER"),
    ("laion/CLIP-ViT-B-16-laion2B-s34B-b88K", "M2_NeuralInt"),
    ("M2_SANER",                               "M2_NeuralInt"),
    ("laion/CLIP-ViT-B-16-laion2B-s34B-b88K", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"),
]

SAMPLE_QUERIES = [
    "a nurse",        "a CEO",          "a teacher",       "a scientist",
    "a strong person","an intelligent person","protesting","cooking",
    "leading a team", "working at a computer","a beautiful person","custom scene",
]


async def insert_synthetic_data(tmp_db: str):
    db.DB_PATH = Path(tmp_db)
    await db.init_db()

    print("Creating workshops ...")
    workshop_ids = []
    for w in SYNTHETIC_WORKSHOPS:
        workshop = await db.create_workshop(**w)
        workshop_ids.append(workshop["id"])
        print(f"  Workshop {workshop['id']}: {w['name']} ({w['community_context']})")

    print("Inserting votes ...")
    vote_count = 0
    for wid in workshop_ids:
        for _ in range(100):
            model_a, model_b = random.choice(MODEL_PAIRS)
            query  = random.choice(SAMPLE_QUERIES)
            winner = random.choice(["A", "B", "tie"])
            pos_a  = random.choice(["left", "right"])
            n_imgs = 12
            images = random.sample(range(1000), n_imgs)
            vote = {
                "participant_id": f"p{random.randint(1000, 9999)}",
                "query":          query,
                "model_a":        model_a,
                "model_b":        model_b,
                "winner":         winner,
                "position_a":     pos_a,
                "why_tags":       [],
                "why_freetext":   "",
                "images_a":       images[:n_imgs // 2],
                "images_b":       images[n_imgs // 2:],
                "session_meta":   {},
                "session_id":     None,
                "workshop_id":    wid,
            }
            await db.record_vote(vote)
            vote_count += 1

    print(f"Inserted {vote_count} votes across {len(workshop_ids)} workshops.")
    return workshop_ids


async def export_csv(tmp_db: str, out_path: str):
    db.DB_PATH = Path(tmp_db)
    csv_data = await db.export_for_analysis()
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_data)
    print(f"Exported analysis CSV -> {out_path}")
    return csv_data


def validate_export(csv_path: str, expected_votes: int, expected_workshops: int):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == expected_votes, (
        f"Expected {expected_votes} votes in export, got {len(rows)}"
    )

    workshops_seen = {r["workshop_id"] for r in rows if r["workshop_id"]}
    assert len(workshops_seen) == expected_workshops, (
        f"Expected {expected_workshops} workshops in export, got {len(workshops_seen)}"
    )

    empty_cats = [r for r in rows if not r.get("query_category")]
    assert not empty_cats, f"{len(empty_cats)} votes have empty query_category"

    contexts = {r["community_context"] for r in rows if r.get("community_context")}
    assert len(contexts) >= 2, f"Expected multiple community contexts, got {contexts}"

    print(f"  [OK] {len(rows)} votes exported")
    print(f"  [OK] {len(workshops_seen)} workshops present")
    print(f"  [OK] query_category populated for all votes")
    print(f"  [OK] community_contexts: {sorted(contexts)}")


def run_analysis_scripts(csv_path: str, analysis_dir: str, tmpdir: str):
    orig_dir = os.getcwd()
    os.chdir(tmpdir)

    py = sys.executable
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    scripts = [
        (
            "win_rates.py",
            [py, str(Path(analysis_dir) / "win_rates.py"), "--votes", csv_path],
            ["win_rates_overall.csv", "win_rates_by_workshop.csv", "win_rates_by_query.csv"],
        ),
        (
            "community_analysis.py",
            [py, str(Path(analysis_dir) / "community_analysis.py"), "--votes", csv_path],
            ["community_differences.csv"],
        ),
    ]

    for name, cmd, expected_outputs in scripts:
        print(f"\nRunning {name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode != 0:
            print(f"STDERR: {result.stderr.strip()}", file=sys.stderr)
            os.chdir(orig_dir)
            raise RuntimeError(f"{name} failed with exit code {result.returncode}")

        for fname in expected_outputs:
            fpath = Path(tmpdir) / fname
            if not fpath.exists():
                os.chdir(orig_dir)
                raise FileNotFoundError(f"Expected output missing: {fname}")
            with open(fpath, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0, f"{fname} is empty"
            print(f"  [OK] {fname} - {len(rows)} rows")

    ww_path = Path(tmpdir) / "win_rates_by_workshop.csv"
    with open(ww_path, newline="", encoding="utf-8") as f:
        ww_rows = list(csv.DictReader(f))
    contexts = {r.get("community_context", "") for r in ww_rows}
    assert len(contexts) >= 2, f"win_rates_by_workshop lacks stratification: {contexts}"
    print(f"  [OK] workshop stratification: {sorted(contexts)}")

    os.chdir(orig_dir)


def validate_query_categorization():
    cases = [
        ("a nurse at work",           "occupation"),
        ("doctor",                    "occupation"),
        ("a very strong athlete",     "trait"),
        ("someone dangerous",         "trait"),
        ("protesting in the streets", "action"),
        ("cooking at home",           "action"),
        ("happy birthday scene",      "custom"),
    ]
    print("\nValidating categorize_query ...")
    for query, expected in cases:
        got = db.categorize_query(query)
        assert got == expected, f"categorize_query({query!r}) = {got!r}, expected {expected!r}"
        print(f"  [OK] {query!r} -> {got}")


async def main(tmp_db: str):
    print("=" * 60)
    print("  Fairness Arena - AIES26 Pipeline Integration Test")
    print("=" * 60)

    validate_query_categorization()

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "analysis_export.csv")

        print("\nPhase 1: Synthetic data insertion")
        await insert_synthetic_data(tmp_db)

        print("\nPhase 2: Analysis export")
        await export_csv(tmp_db, csv_path)

        print("\nPhase 3: Validating export")
        validate_export(csv_path, expected_votes=400, expected_workshops=4)

        analysis_dir = str(Path(__file__).parent / "analysis")
        print("\nPhase 4: Running analysis scripts")
        run_analysis_scripts(csv_path, analysis_dir, tmpdir)

    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=None, help="Path for test SQLite database")
    args = p.parse_args()

    if args.db:
        tmp_db = args.db
        cleanup = False
    else:
        fd, tmp_db = tempfile.mkstemp(suffix=".db", prefix="test_arena_")
        os.close(fd)
        cleanup = True

    try:
        asyncio.run(main(tmp_db))
    finally:
        if cleanup and Path(tmp_db).exists():
            os.unlink(tmp_db)
