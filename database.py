from __future__ import annotations
"""
Database layer — SQLite via aiosqlite for async access.
Tables: participants, votes, elo_ratings, retrieval_cache, sessions
"""

import csv
import io
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
                session_id TEXT,
                query_category TEXT,
                workshop_id INTEGER
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

            CREATE TABLE IF NOT EXISTS open_query_translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                translation TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                reviewed_at REAL
            );

            CREATE TABLE IF NOT EXISTS query_contrasts (
                query TEXT PRIMARY KEY,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                assigned_at REAL NOT NULL
            );
        """)
        await db.commit()
        # Migrations for columns added after initial schema
        for col, defn in [
            ("session_id", "TEXT"),
        ]:
            try:
                await db.execute(f"ALTER TABLE votes ADD COLUMN {col} {defn}")
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
                why_tags, why_freetext, images_a, images_b, timestamp, session_meta,
                session_id, workshop_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                vote.get("workshop_id"),
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
    """Export all votes as CSV, with session_name and participant nickname joined in."""
    import csv
    import io
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT
                v.id, v.timestamp, v.participant_id,
                p.nickname,
                v.query, v.model_a, v.model_b, v.winner, v.position_a,
                v.why_tags, v.why_freetext,
                v.images_a, v.images_b,
                v.session_id,
                s.name  AS session_name,
                s.started_at AS session_started_at,
                v.session_meta
            FROM votes v
            LEFT JOIN participants p ON p.id = v.participant_id
            LEFT JOIN sessions     s ON s.id = v.session_id
            ORDER BY v.timestamp
        """)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis export
# ═══════════════════════════════════════════════════════════════════════════

async def export_for_analysis() -> str:
    """Return CSV for analysis: votes joined with session info."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT
                v.id            AS vote_id,
                v.participant_id,
                v.query,
                v.model_a,
                v.model_b,
                v.winner,
                v.position_a,
                CASE WHEN v.position_a = 'left'  THEN v.images_a ELSE v.images_b END AS ranking_left,
                CASE WHEN v.position_a = 'left'  THEN v.images_b ELSE v.images_a END AS ranking_right,
                v.session_id,
                s.name          AS session_name,
                s.started_at    AS session_started_at,
                s.stopped_at    AS session_stopped_at,
                v.timestamp
            FROM votes v
            LEFT JOIN sessions s ON s.id = v.session_id
            ORDER BY v.timestamp
        """)
        rows = await cursor.fetchall()

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=dict(rows[0]).keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
    return output.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
#  Open query translations
# ═══════════════════════════════════════════════════════════════════════════

async def get_translation(original_text: str, source_lang: str) -> dict | None:
    """Return an existing translation record (any status) or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM open_query_translations WHERE original_text=? AND source_lang=?",
            (original_text, source_lang),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def save_translation(original_text: str, source_lang: str, translation: str,
                           status: str = "pending") -> dict:
    """Insert a translation record and return it."""
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO open_query_translations
               (original_text, source_lang, translation, status, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (original_text, source_lang, translation, status, now),
        )
        row_id = cur.lastrowid
        await db.commit()
    return {"id": row_id, "original_text": original_text, "source_lang": source_lang,
            "translation": translation, "status": status, "created_at": now}


async def list_translations(status: str | None = None) -> list[dict]:
    """List translations, optionally filtered by status."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if status:
            cur = await db.execute(
                "SELECT * FROM open_query_translations WHERE status=? ORDER BY created_at DESC",
                (status,),
            )
        else:
            cur = await db.execute(
                "SELECT * FROM open_query_translations ORDER BY created_at DESC"
            )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def review_translation(row_id: int, status: str, translation: str | None = None) -> dict | None:
    """Approve or reject a translation, optionally editing the translation text."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if translation:
            await db.execute(
                "UPDATE open_query_translations SET status=?, translation=?, reviewed_at=? WHERE id=?",
                (status, translation, time.time(), row_id),
            )
        else:
            await db.execute(
                "UPDATE open_query_translations SET status=?, reviewed_at=? WHERE id=?",
                (status, time.time(), row_id),
            )
        await db.commit()
        cur = await db.execute("SELECT * FROM open_query_translations WHERE id=?", (row_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_approved_translations() -> list[str]:
    """Return distinct English canonicals of all approved open queries."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT DISTINCT translation FROM open_query_translations WHERE status='approved' ORDER BY created_at"
        )
        rows = await cur.fetchall()
        return [r["translation"] for r in rows]


async def get_pending_translation_texts() -> set[str]:
    """Return EN translations that have at least one non-approved (pending/rejected) record.
    Used to suppress them from the shared query list until approved."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT DISTINCT translation FROM open_query_translations WHERE status != 'approved'"
        )
        rows = await cur.fetchall()
        return {r[0] for r in rows}


async def get_translation_by_en_lang(en_text: str, lang: str) -> dict | None:
    """Check if any record already maps source_lang=lang to the given EN canonical."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM open_query_translations WHERE translation=? AND source_lang=?",
            (en_text, lang),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_multilingual_labels() -> dict[str, dict[str, str]]:
    """Return {en_canonical: {lang: display_text}} for all approved open queries.

    Collects every record whose EN canonical has at least one approved entry, so that
    display labels for ES and CA are available even when introduced by the other language.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT DISTINCT translation FROM open_query_translations WHERE status='approved'"
        )
        approved_en = [r["translation"] for r in await cur.fetchall()]
        if not approved_en:
            return {}
        placeholders = ",".join("?" * len(approved_en))
        cur = await db.execute(
            f"SELECT translation, source_lang, original_text FROM open_query_translations "
            f"WHERE translation IN ({placeholders})",
            approved_en,
        )
        rows = await cur.fetchall()
    result: dict[str, dict[str, str]] = {}
    for r in rows:
        en = r["translation"]
        if en not in result:
            result[en] = {}
        if r["source_lang"] not in result[en]:
            result[en][r["source_lang"]] = r["original_text"]
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Query contrasts (fixed model-pair assignment per query)
# ═══════════════════════════════════════════════════════════════════════════

async def get_query_contrast(query: str) -> tuple[str, str] | None:
    """Return the (model_a, model_b) pair assigned to this query, or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT model_a, model_b FROM query_contrasts WHERE query=?", (query,)
        )
        row = await cur.fetchone()
        return (row[0], row[1]) if row else None


async def set_query_contrast(query: str, model_a: str, model_b: str) -> None:
    """Persist the contrast assignment for a query (upsert)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO query_contrasts (query, model_a, model_b, assigned_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(query) DO UPDATE SET model_a=excluded.model_a,
                   model_b=excluded.model_b, assigned_at=excluded.assigned_at""",
            (query, model_a, model_b, time.time()),
        )
        await db.commit()


async def get_contrast_assignment_counts(valid_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Count how many queries are currently assigned to each pair."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT model_a, model_b FROM query_contrasts")
        rows = await cur.fetchall()
    counts: dict[tuple[str, str], int] = {p: 0 for p in valid_pairs}
    for model_a, model_b in rows:
        key = (model_a, model_b)
        if key in counts:
            counts[key] += 1
    return counts
