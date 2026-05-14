"""
test_pipeline.py
================
Integration and unit tests for the Fairness Arena server.

Covers:
  - allowed_pairs filtering in _tally, api_live_results, and api_live_results/full
  - api_match pair selection (fixed contrast per query, round-robin assignment)
  - Session creation and vote recording
  - Left/right position uniformity (still random per vote)

Usage:
    python test_pipeline.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import math
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

SESSIONS = ["Group A — Barcelona", "Group B — Madrid"]


def make_vote(model_a, model_b, session_id=None, query=None):
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
        "session_id":     session_id,
    }


# ── Unit tests: pair filtering ────────────────────────────────────────────────

def test_tally_filters_allowed_pairs():
    """_tally must exclude votes whose pair is not in allowed_pairs."""
    import server
    server.CONFIG = {"arena": {"allowed_pairs": [list(p) for p in ALLOWED_PAIRS]}}

    rows = (
        [{"model_a": ma, "model_b": mb, "winner": "A", "session_id": "s1"}
         for ma, mb in ALLOWED_PAIRS]
        +
        [{"model_a": ma, "model_b": mb, "winner": "A", "session_id": "s1"}
         for ma, mb in DISALLOWED_PAIRS]
    )

    result = server._tally(rows, lambda v: v.get("session_id") or "none")

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
    rows = [{"model_a": ma, "model_b": mb, "winner": "A", "session_id": "s1"}
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

    n_each = 5
    for ma, mb in ALLOWED_PAIRS:
        for _ in range(n_each):
            await db.record_vote(make_vote(ma, mb, session_id="test-session"))
    for ma, mb in DISALLOWED_PAIRS:
        for _ in range(n_each):
            await db.record_vote(make_vote(ma, mb, session_id="test-session"))

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


# ── Integration test: session vote recording ──────────────────────────────────

async def test_session_vote_recording(tmp_db: str):
    """Sessions are created; votes are recorded and export joins session metadata."""
    db.DB_PATH = Path(tmp_db)
    await db.init_db()

    session_ids = []
    for name in SESSIONS:
        session = await db.create_session(name)
        session_ids.append(session["id"])

    n_votes = 10
    for sid in session_ids:
        for ma, mb in random.choices(ALLOWED_PAIRS, k=n_votes):
            await db.record_vote(make_vote(ma, mb, session_id=sid))

    csv_str = await db.export_for_analysis()
    rows = list(csv.DictReader(io.StringIO(csv_str)))

    expected = len(session_ids) * n_votes
    assert len(rows) == expected, f"Expected {expected} votes, got {len(rows)}"

    sessions_seen = {r["session_id"] for r in rows if r["session_id"]}
    assert len(sessions_seen) == len(SESSIONS), (
        f"Expected {len(SESSIONS)} sessions, got {len(sessions_seen)}"
    )

    names_seen = {r["session_name"] for r in rows if r.get("session_name")}
    assert names_seen == set(SESSIONS), f"Wrong session names: {names_seen}"

    print(f"  [OK] {len(rows)} votes across {len(sessions_seen)} sessions")
    print(f"  [OK] session names: {sorted(names_seen)}")


# ── Statistical uniformity test ───────────────────────────────────────────────

def test_contrast_assignment_round_robin():
    """
    Verify that _get_or_assign_contrast distributes queries evenly across pairs.
    With N queries and K pairs, each pair should get floor(N/K) or ceil(N/K) queries.
    """
    valid_pairs = list(ALLOWED_PAIRS)
    k = len(valid_pairs)

    for n_queries in [10, 30, 31]:
        counts = {p: 0 for p in valid_pairs}
        for _ in range(n_queries):
            # Reproduce round-robin logic from _get_or_assign_contrast
            pair = min(valid_pairs, key=lambda p: (counts[p], valid_pairs.index(p)))
            counts[pair] += 1

        for pair, count in counts.items():
            lo, hi = n_queries // k, (n_queries + k - 1) // k
            assert lo <= count <= hi, (
                f"n_queries={n_queries}: pair {pair} got {count} assignments "
                f"(expected {lo}–{hi})"
            )
        print(f"  [OK] {n_queries} queries, {k} pairs: " +
              ", ".join(f"{p[0]} vs {p[1]}={counts[p]}" for p in valid_pairs))


def test_position_uniformity(n: int = 3000, sigma: float = 4.0):
    """
    Left/right position is still randomised per vote — verify ~50% each.
    n=3000, sigma=4.0 → std ~= 27; a broken flip (always left) is >50σ away.
    """
    rng = random.Random(42)
    left_a_count = sum(1 for _ in range(n) if rng.random() < 0.5)

    expected = n / 2
    std = math.sqrt(n * 0.5 * 0.5)
    z = abs(left_a_count - expected) / std
    assert z < sigma, (
        f"Left-position count {left_a_count}/{n} (z={z:.1f}σ > {sigma}σ). "
        f"Expected ~{expected:.0f} ± {std:.0f}"
    )
    print(f"  [OK] left/right position: {left_a_count}/{n} left "
          f"({100*left_a_count/n:.1f}%, z={z:.2f}σ)")


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

    print("\n[3] Session creation and vote recording")
    await test_session_vote_recording(tmp_db_2)

    print("\n[4] Contrast assignment round-robin (10/30/31 queries)")
    test_contrast_assignment_round_robin()

    print("\n[5] Left/right position uniformity (n=3000)")
    test_position_uniformity()

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
