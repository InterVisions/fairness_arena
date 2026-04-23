from __future__ import annotations
"""
Database layer — SQLite via aiosqlite for async access.
Tables: participants, votes, elo_ratings, retrieval_cache, sessions
"""

import json
import math
import time
import uuid
import aiosqlite
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "arena.db"


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS participants (
                id TEXT PRIMARY KEY,
                nickname TEXT,
                created_at REAL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                started_at REAL NOT NULL,
                stopped_at REAL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT,
                query TEXT,
                model_a TEXT,
                model_b TEXT,
                winner TEXT,  -- 'A', 'B', 'tie'
                position_a TEXT,  -- 'left' or 'right'
                why_tags TEXT DEFAULT '[]',
                why_freetext TEXT DEFAULT '',
                images_a TEXT,  -- JSON list of image indices shown
                images_b TEXT,
                timestamp REAL,
                session_meta TEXT DEFAULT '{}',
                session_id TEXT
            );

            CREATE TABLE IF NOT EXISTS elo_ratings (
                model_id TEXT PRIMARY KEY,
                rating REAL,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                ties INTEGER DEFAULT 0,
                last_updated REAL
            );

            CREATE TABLE IF NOT EXISTS retrieval_cache (
                cache_key TEXT PRIMARY KEY,
                model_id TEXT,
                query TEXT,
                ranked_indices TEXT,  -- JSON list of image indices in ranked order
                similarities TEXT,   -- JSON list of similarity scores
                computed_at REAL
            );
        """)
        await db.commit()
        # Migration: add session_id to existing votes tables that predate sessions feature
        try:
            await db.execute("ALTER TABLE votes ADD COLUMN session_id TEXT")
            await db.commit()
        except Exception:
            pass  # column already exists


# ═══════════════════════════════════════════════════════════════════════════
#  Elo Rating System
# ═══════════════════════════════════════════════════════════════════════════

def elo_expected(ra: float, rb: float) -> float:
    """Expected score of player A against player B."""
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400.0))


def elo_update(ra: float, rb: float, score_a: float, k: float = 32) -> tuple[float, float]:
    """
    Update Elo ratings after a match.
    score_a: 1.0 = A wins, 0.0 = B wins, 0.5 = tie
    Returns (new_ra, new_rb).
    """
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea
    new_ra = ra + k * (score_a - ea)
    new_rb = rb + k * ((1.0 - score_a) - eb)
    return new_ra, new_rb


async def ensure_model_ratings(model_ids: list[str], initial: float = 1500):
    """Create Elo entries for models that don't exist yet."""
    async with aiosqlite.connect(DB_PATH) as db:
        for mid in model_ids:
            await db.execute(
                "INSERT OR IGNORE INTO elo_ratings (model_id, rating, last_updated) VALUES (?, ?, ?)",
                (mid, initial, time.time())
            )
        await db.commit()


async def get_ratings() -> dict:
    """Return current Elo ratings for all models."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM elo_ratings ORDER BY rating DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def record_vote(vote: dict, k_factor: float = 32, initial_rating: float = 1500):
    """Record a vote and update Elo ratings."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Insert vote
        await db.execute(
            """INSERT INTO votes
               (participant_id, query, model_a, model_b, winner, position_a,
                why_tags, why_freetext, images_a, images_b, timestamp, session_meta, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                vote["participant_id"],
                vote["query"],
                vote["model_a"],
                vote["model_b"],
                vote["winner"],
                vote["position_a"],
                json.dumps(vote.get("why_tags", [])),
                vote.get("why_freetext", ""),
                json.dumps(vote.get("images_a", [])),
                json.dumps(vote.get("images_b", [])),
                time.time(),
                json.dumps(vote.get("session_meta", {})),
                vote.get("session_id"),
            )
        )

        # Get current ratings
        cursor = await db.execute(
            "SELECT model_id, rating FROM elo_ratings WHERE model_id IN (?, ?)",
            (vote["model_a"], vote["model_b"])
        )
        rows = await cursor.fetchall()
        ratings = {r[0]: r[1] for r in rows}

        ra = ratings.get(vote["model_a"], initial_rating)
        rb = ratings.get(vote["model_b"], initial_rating)

        # Compute score
        if vote["winner"] == "A":
            score_a = 1.0
        elif vote["winner"] == "B":
            score_a = 0.0
        else:
            score_a = 0.5

        new_ra, new_rb = elo_update(ra, rb, score_a, k=k_factor)

        # Update ratings
        win_a = 1 if vote["winner"] == "A" else 0
        win_b = 1 if vote["winner"] == "B" else 0
        tie = 1 if vote["winner"] == "tie" else 0

        now = time.time()
        await db.execute(
            """INSERT INTO elo_ratings (model_id, rating, wins, losses, ties, last_updated)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(model_id) DO UPDATE SET
               rating=?, wins=wins+?, losses=losses+?, ties=ties+?, last_updated=?""",
            (vote["model_a"], new_ra, win_a, 1 - win_a - tie, tie, now,
             new_ra, win_a, 1 - win_a - tie, tie, now)
        )
        await db.execute(
            """INSERT INTO elo_ratings (model_id, rating, wins, losses, ties, last_updated)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(model_id) DO UPDATE SET
               rating=?, wins=wins+?, losses=losses+?, ties=ties+?, last_updated=?""",
            (vote["model_b"], new_rb, win_b, 1 - win_b - tie, tie, now,
             new_rb, win_b, 1 - win_b - tie, tie, now)
        )

        await db.commit()
        return {"new_rating_a": new_ra, "new_rating_b": new_rb}


async def get_vote_stats() -> dict:
    """Get aggregate statistics for the admin panel."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Total votes
        cursor = await db.execute("SELECT COUNT(*) FROM votes")
        total_votes = (await cursor.fetchone())[0]

        # Unique participants
        cursor = await db.execute("SELECT COUNT(DISTINCT participant_id) FROM votes")
        unique_participants = (await cursor.fetchone())[0]

        # Votes per query
        cursor = await db.execute(
            "SELECT query, COUNT(*) as cnt FROM votes GROUP BY query ORDER BY cnt DESC"
        )
        votes_per_query = [{"query": r[0], "count": r[1]} for r in await cursor.fetchall()]

        # Winner distribution
        cursor = await db.execute(
            "SELECT winner, COUNT(*) FROM votes GROUP BY winner"
        )
        winner_dist = {r[0]: r[1] for r in await cursor.fetchall()}

        # Position bias check
        cursor = await db.execute("""
            SELECT position_a,
                   SUM(CASE WHEN winner='A' THEN 1 ELSE 0 END) as a_wins,
                   SUM(CASE WHEN winner='B' THEN 1 ELSE 0 END) as b_wins,
                   SUM(CASE WHEN winner='tie' THEN 1 ELSE 0 END) as ties,
                   COUNT(*) as total
            FROM votes GROUP BY position_a
        """)
        position_bias = [dict(zip(["position_a", "a_wins", "b_wins", "ties", "total"], r))
                         for r in await cursor.fetchall()]

        # Why tags frequency
        cursor = await db.execute("SELECT why_tags FROM votes WHERE why_tags != '[]'")
        tag_counts = {}
        for row in await cursor.fetchall():
            for tag in json.loads(row[0]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Votes over time (hourly buckets)
        cursor = await db.execute("""
            SELECT CAST(timestamp / 3600 AS INTEGER) * 3600 as bucket,
                   COUNT(*) as cnt
            FROM votes GROUP BY bucket ORDER BY bucket
        """)
        votes_timeline = [{"timestamp": r[0], "count": r[1]} for r in await cursor.fetchall()]

        # Per-model pair stats
        cursor = await db.execute("""
            SELECT model_a, model_b, winner, COUNT(*) as cnt
            FROM votes GROUP BY model_a, model_b, winner
        """)
        pair_stats = [{"model_a": r[0], "model_b": r[1], "winner": r[2], "count": r[3]}
                      for r in await cursor.fetchall()]

        return {
            "total_votes": total_votes,
            "unique_participants": unique_participants,
            "votes_per_query": votes_per_query,
            "winner_distribution": winner_dist,
            "position_bias": position_bias,
            "why_tag_counts": tag_counts,
            "votes_timeline": votes_timeline,
            "pair_stats": pair_stats,
        }


async def get_recent_votes(limit: int = 50) -> list:
    """Get recent votes for admin live feed."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM votes ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def register_participant(nickname: str = "", metadata: dict = None) -> str:
    """Create a new participant and return their ID."""
    pid = str(uuid.uuid4())[:8]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO participants (id, nickname, created_at, metadata) VALUES (?, ?, ?, ?)",
            (pid, nickname, time.time(), json.dumps(metadata or {}))
        )
        await db.commit()
    return pid


async def cache_retrieval(model_id: str, query: str, indices: list, similarities: list):
    """Cache retrieval results for a (model, query) pair."""
    key = f"{model_id}::{query}"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO retrieval_cache
               (cache_key, model_id, query, ranked_indices, similarities, computed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, model_id, query, json.dumps(indices), json.dumps(similarities), time.time())
        )
        await db.commit()


async def get_cached_query_list() -> list[str]:
    """Return distinct queries accumulated in the retrieval cache (open queries)."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT DISTINCT query FROM retrieval_cache ORDER BY computed_at ASC"
        )
        return [row[0] for row in await cursor.fetchall()]


async def get_cached_retrieval(model_id: str, query: str) -> dict | None:
    """Get cached retrieval results."""
    key = f"{model_id}::{query}"
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT ranked_indices, similarities FROM retrieval_cache WHERE cache_key = ?",
            (key,)
        )
        row = await cursor.fetchone()
        if row:
            return {
                "indices": json.loads(row[0]),
                "similarities": json.loads(row[1]),
            }
        return None


async def reset_elo(initial: float = 1500):
    """Reset all Elo ratings (admin action)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE elo_ratings SET rating=?, wins=0, losses=0, ties=0, last_updated=?",
            (initial, time.time())
        )
        await db.commit()


async def export_votes_csv() -> str:
    """Export all votes as CSV string."""
    import csv
    import io
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM votes ORDER BY timestamp")
        rows = await cursor.fetchall()

        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=dict(rows[0]).keys())
            writer.writeheader()
            for r in rows:
                writer.writerow(dict(r))
        return output.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
#  Sessions
# ═══════════════════════════════════════════════════════════════════════════

async def create_session(name: str, started_at: float | None = None) -> dict:
    """Create a new session and return it."""
    session_id = str(uuid.uuid4())[:12]
    now = time.time()
    started_at = started_at or now
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (id, name, started_at, created_at) VALUES (?, ?, ?, ?)",
            (session_id, name, started_at, now)
        )
        await db.commit()
    return {"id": session_id, "name": name, "started_at": started_at, "stopped_at": None, "created_at": now}


async def stop_session(session_id: str) -> dict:
    """Mark a session as stopped."""
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET stopped_at = ? WHERE id = ? AND stopped_at IS NULL",
            (now, session_id)
        )
        await db.commit()
    return {"id": session_id, "stopped_at": now}


async def get_active_session() -> dict | None:
    """Return the currently running session, or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE stopped_at IS NULL ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_sessions() -> list[dict]:
    """Return all sessions with their vote counts."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT s.*, COUNT(v.id) as vote_count
            FROM sessions s
            LEFT JOIN votes v ON v.session_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
        """)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
