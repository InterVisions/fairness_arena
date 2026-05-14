"""
test_pipeline.py
================
Integration and unit tests for the Fairness Arena server.

Covers:
  - allowed_pairs filtering in _tally, api_live_results, and api_live_results/full
  - api_match pair selection
  - Workshop creation and vote recording

Usage:
    python test_pipeline.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import random
import sys
import tempfile
from pathlib import Path

import database as db

# ── Test fixtures ─────────────────────────────────────────────────────────────

ALLOWED_PAIRS = [
    ("clip-vit-b16", "clip-vit-b16-debiased"),
    ("siglip-b16",   "siglip2-b16"),
    ("clip-vit-b16", "siglip-b16"),
]

DISALLOWED_PAIRS = [
    ("clip-vit-b16",          "siglip2-b16"),
    ("clip-vit-b16-debiased", "siglip-b16"),
    ("clip-vit-b16-debiased", "siglip2-b16"),
]

QUERIES = ["nurse", "doctor", "terrorist", "caretaker", "activist",
           "attractive", "criminal", "hero", "homeless", "rich"]

WORKSHOPS = [
    {"name": "Group A", "location": "Barcelona", "date": "2026-05-14",
     "community_context": "LGBTQ+", "facilitator": "Ana"},
    {"name": "Group B", "location": "Madrid",    "date": "2026-05-15",
     "community_context": "Roma",   "facilitator": "Pedro"},
]


def make_vote(model_a, model_b, workshop_id=None, query=None):
    return {
        "participant_id": f"p{random.randint(1000, 9999)}",
        "query":          query or random.choice(QUERIES),
        "model_a":        model_a,
        "model_b":        model_b,
        "winner":         random.choice(["A", "B", "tie"]),
        "position_a":     random.choice(["left", "right"]),
        "why_tags":       [],
        "why_freetext":   "",
        "images_a":       list(range(6)),
        "images_b":       list(range(6, 12)),
        "session_meta":   {},
        "session_id":     None,
        "workshop_id":    workshop_id,
    }


# ── Unit tests: pair filtering ────────────────────────────────────────────────

def test_tally_filters_allowed_pairs():
    """_tally must exclude votes whose pair is not in allowed_pairs."""
    import server
    server.CONFIG = {"arena": {"allowed_pairs": [list(p) for p in ALLOWED_PAIRS]}}

    rows = (
        [{"model_a": ma, "model_b": mb, "winner": "A", "workshop_id": "1"}
         for ma, mb in ALLOWED_PAIRS]
        +
        [{"model_a": ma, "model_b": mb, "winner": "A", "workshop_id": "1"}
         for ma, mb in DISALLOWED_PAIRS]
    )

    result = server._tally(rows, lambda v: v.get("workshop_id") or "none")

    for ma, mb in ALLOWED_PAIRS:
        key = f"{ma} vs {mb}"
        assert key in result, f"Allowed pair missing from tally: {key}"

    for ma, mb in DISALLOWED_PAIRS:
        key = f"{ma} vs {mb}"
        assert key not in result, f"Disallowed pair present in tally: {key}"

    print("  [OK] _tally filters disallowed pairs")


def test_tally_empty_allowed_pairs_passes_all():
    """When allowed_pairs is empty, _tally must include every pair."""
    import server
    server.CONFIG = {"arena": {"allowed_pairs": []}}

    all_pairs = ALLOWED_PAIRS + DISALLOWED_PAIRS
    rows = [{"model_a": ma, "model_b": mb, "winner": "A", "workshop_id": "1"}
            for ma, mb in all_pairs]

    result = server._tally(rows, lambda v: "all")
    assert len(result) == len(all_pairs), (
        f"Expected {len(all_pairs)} pairs, got {len(result)}"
    )
    print("  [OK] _tally passes all pairs when allowed_pairs is empty")


def test_live_results_full_norm_pair_set():
    """api_live_results/full must build its norm_pair_set from config, not hardcoded values."""
    import server
    server.CONFIG = {"arena": {"allowed_pairs": [list(p) for p in ALLOWED_PAIRS]}}

    def norm(ma, mb):
        return (ma, mb, False) if ma <= mb else (mb, ma, True)

    allowed_pairs = [tuple(p) for p in server.CONFIG["arena"]["allowed_pairs"]]
    norm_pair_set = {norm(a, b)[:2] for a, b in allowed_pairs}

    norm_allowed    = {norm(a, b)[:2] for a, b in ALLOWED_PAIRS}
    norm_disallowed = {norm(a, b)[:2] for a, b in DISALLOWED_PAIRS}

    for pair in norm_allowed:
        assert pair in norm_pair_set, f"Allowed pair missing from norm set: {pair}"
    for pair in norm_disallowed:
        assert pair not in norm_pair_set, f"Disallowed pair in norm set: {pair}"

    print("  [OK] api_live_results/full norm_pair_set is correct")


def test_api_match_pair_selection():
    """api_match must only offer pairs present in allowed_pairs."""
    import server
    server.CONFIG = {"arena": {"allowed_pairs": [list(p) for p in ALLOWED_PAIRS]}}

    all_models  = list({m for pair in ALLOWED_PAIRS for m in pair})
    enabled_set = set(all_models)
    allowed     = [tuple(p) for p in server.CONFIG["arena"]["allowed_pairs"]]
    valid_pairs = [(a, b) for a, b in allowed if a in enabled_set and b in enabled_set]
    allowed_set = set(allowed)

    assert len(valid_pairs) == len(ALLOWED_PAIRS), (
        f"Expected {len(ALLOWED_PAIRS)} valid pairs, got {len(valid_pairs)}"
    )
    for pair in valid_pairs:
        assert pair in allowed_set, f"Pair outside allowed set returned: {pair}"

    print("  [OK] api_match only selects from allowed_pairs")


# ── Integration test: allowed_pairs filtering end-to-end ─────────────────────

async def test_allowed_pairs_end_to_end(tmp_db: str):
    """
    Insert allowed and disallowed votes into the real DB, then verify that
    _tally (used by all live_results endpoints) only surfaces allowed pairs.
    """
    import server
    db.DB_PATH = Path(tmp_db)
    await db.init_db()
    server.CONFIG = {"arena": {"allowed_pairs": [list(p) for p in ALLOWED_PAIRS]}}

    workshop = await db.create_workshop(name="Test", community_context="Test")
    wid = workshop["id"]

    n_each = 5
    for ma, mb in ALLOWED_PAIRS:
        for _ in range(n_each):
            await db.record_vote(make_vote(ma, mb, workshop_id=wid))
    for ma, mb in DISALLOWED_PAIRS:
        for _ in range(n_each):
            await db.record_vote(make_vote(ma, mb, workshop_id=wid))

    csv_str = await db.export_for_analysis()
    rows = list(csv.DictReader(io.StringIO(csv_str)))

    total = (len(ALLOWED_PAIRS) + len(DISALLOWED_PAIRS)) * n_each
    assert len(rows) == total, f"Expected {total} votes in DB, got {len(rows)}"

    tally = server._tally(rows, lambda v: "all")

    for ma, mb in ALLOWED_PAIRS:
        key = f"{ma} vs {mb}"
        assert key in tally, f"Allowed pair missing from end-to-end tally: {key}"
    for ma, mb in DISALLOWED_PAIRS:
        key = f"{ma} vs {mb}"
        assert key not in tally, f"Disallowed pair present in end-to-end tally: {key}"

    print(f"  [OK] {total} votes inserted; tally exposes only {len(ALLOWED_PAIRS)} allowed pairs")


# ── Integration test: workshops and vote recording ────────────────────────────

async def test_workshop_vote_recording(tmp_db: str):
    """Workshops are created; votes are recorded and export joins metadata correctly."""
    db.DB_PATH = Path(tmp_db)
    await db.init_db()

    workshop_ids = []
    for w in WORKSHOPS:
        workshop = await db.create_workshop(**w)
        workshop_ids.append(workshop["id"])

    n_votes = 10
    for wid in workshop_ids:
        for ma, mb in random.choices(ALLOWED_PAIRS, k=n_votes):
            await db.record_vote(make_vote(ma, mb, workshop_id=wid))

    csv_str = await db.export_for_analysis()
    rows = list(csv.DictReader(io.StringIO(csv_str)))

    expected = len(workshop_ids) * n_votes
    assert len(rows) == expected, f"Expected {expected} votes, got {len(rows)}"

    workshops_seen = {r["workshop_id"] for r in rows if r["workshop_id"]}
    assert len(workshops_seen) == len(WORKSHOPS), (
        f"Expected {len(WORKSHOPS)} workshops, got {len(workshops_seen)}"
    )

    contexts = {r["community_context"] for r in rows if r.get("community_context")}
    assert contexts == {"LGBTQ+", "Roma"}, f"Wrong community contexts: {contexts}"

    print(f"  [OK] {len(rows)} votes across {len(workshops_seen)} workshops")
    print(f"  [OK] community contexts: {sorted(contexts)}")


# ── Runner ────────────────────────────────────────────────────────────────────

async def main(tmp_db_1: str, tmp_db_2: str):
    print("=" * 60)
    print("  Fairness Arena — Pipeline Tests")
    print("=" * 60)

    print("\n[1] allowed_pairs filtering — unit tests")
    test_tally_filters_allowed_pairs()
    test_tally_empty_allowed_pairs_passes_all()
    test_live_results_full_norm_pair_set()
    test_api_match_pair_selection()

    print("\n[2] allowed_pairs filtering — end-to-end with real DB")
    await test_allowed_pairs_end_to_end(tmp_db_1)

    print("\n[3] Workshop creation and vote recording")
    await test_workshop_vote_recording(tmp_db_2)

    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=None, help="Base path for test databases (two files created)")
    args = p.parse_args()

    if args.db:
        tmp_db_1 = args.db + "_1.db"
        tmp_db_2 = args.db + "_2.db"
        cleanup = False
    else:
        fd1, tmp_db_1 = tempfile.mkstemp(suffix=".db", prefix="test_arena_")
        fd2, tmp_db_2 = tempfile.mkstemp(suffix=".db", prefix="test_arena_")
        os.close(fd1)
        os.close(fd2)
        cleanup = True

    try:
        asyncio.run(main(tmp_db_1, tmp_db_2))
    except AssertionError as e:
        print(f"\n  FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if cleanup:
            for f in (tmp_db_1, tmp_db_2):
                if Path(f).exists():
                    os.unlink(f)
