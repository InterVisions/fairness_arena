from __future__ import annotations
"""
Fairness Arena — CLIP Retrieval Fairness Evaluation Tool
========================================================
An LMSYS-style arena for comparing fairness of CLIP retrieval models.

Usage:
    python server.py
    python server.py --config config/my_config.json --port 8080
    python server.py --bundles-dir data/
    python server.py --bundle data/arena_bundle_flickr30k.npz  (legacy single-bundle)
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import uvicorn

import database as db
from retrieval import RetrievalEngine

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fairness-arena")

# ── Globals ──────────────────────────────────────────────────────────────
CONFIG = {}
ENGINE = None  # RetrievalEngine instance
ADMIN_TOKEN = "changeme"  # Set via --admin-token
BUNDLE_PATH = None        # Explicit single-bundle path (legacy)
BUNDLES_DIR = None        # Directory containing per-dataset bundles
ACTIVE_SESSION = None     # Currently running workshop session (dict or None)
ACTIVE_WORKSHOP = None    # Currently active workshop for data collection (dict or None)

# ── AIES26 Model Registry ─────────────────────────────────────────────────
MODEL_IDS = {
    "M1": "openai/clip-vit-base-patch16",
    "M2": "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "M3": "google/siglip-base-patch16-224",
    "M4": "google/siglip2-base-patch16-224",
    "M5": "M2_SANER",
    "M6": "M2_NeuralInt",
    "M7": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
}

# Each tuple is (model_a_id, model_b_id) using keys from MODEL_IDS
ALLOWED_PAIRS = [
    ("M1", "M2"),  # data source comparison
    ("M1", "M3"),  # objective comparison
    ("M3", "M4"),  # SigLIP design comparison
    ("M2", "M5"),  # text debiasing
    ("M2", "M6"),  # image debiasing
    ("M5", "M6"),  # debiasing method comparison
    ("M2", "M7"),  # scale comparison
]

app = FastAPI(title="Fairness Arena")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────

def get_datasets(config: dict) -> list[dict]:
    """Return datasets list, handling both old single-dataset and new multi-dataset configs."""
    if "datasets" in config:
        return config["datasets"]
    ds = config.get("dataset", {})
    if not ds:
        return []
    if "id" not in ds:
        if ds.get("source") == "huggingface":
            ds = dict(ds, id=ds["hf_repo"].split("/")[-1].lower())
        else:
            ds = dict(ds, id=Path(ds.get("folder_path", "dataset")).stem.lower())
    if "name" not in ds:
        ds = dict(ds, name=ds["id"])
    return [ds]


def bundle_path_for(dataset_id: str) -> Path | None:
    """Resolve the bundle file path for a given dataset id."""
    if BUNDLES_DIR:
        p = Path(BUNDLES_DIR) / f"arena_bundle_{dataset_id}.npz"
        return p if p.exists() else None
    return None


def active_bundle_path() -> Path | None:
    """Return the bundle path for the currently active dataset."""
    if BUNDLE_PATH:
        return Path(BUNDLE_PATH)
    active_id = CONFIG.get("arena", {}).get("active_dataset_id")
    if active_id:
        return bundle_path_for(active_id)
    # Fall back to first dataset with an available bundle
    for ds in get_datasets(CONFIG):
        p = bundle_path_for(ds["id"])
        if p:
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Page routes
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "arena.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(str(STATIC_DIR / "admin.html"))


@app.get("/leaderboard")
async def leaderboard_page():
    return FileResponse(str(STATIC_DIR / "leaderboard.html"))


# ═══════════════════════════════════════════════════════════════════════════
#  Public API — Arena
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/register")
async def api_register(request: Request):
    """Register a new participant."""
    body = await request.json()
    pid = await db.register_participant(
        nickname=body.get("nickname", ""),
        metadata=body.get("metadata", {})
    )
    return {"participant_id": pid}


@app.get("/api/config")
async def api_config():
    """Return public-facing config (queries, question, etc.)."""
    arena = CONFIG.get("arena", {})
    # Predefined queries from bundle + any open queries accumulated in DB cache
    queries = ENGINE.bundle_queries()
    if arena.get("allow_open_queries", False):
        cached_queries = await db.get_cached_query_list()
        for q in cached_queries:
            if q not in queries:
                queries.append(q)
    return {
        "predefined_queries": queries,
        "allow_open_queries": arena.get("allow_open_queries", False),
        "judge_question": arena.get("judge_question", "Which set of images is fairer?"),
        "search_query_label": arena.get("search_query_label", "Search query"),
        "images_per_model": arena.get("images_per_model", 12),
        "max_scroll_images": arena.get("max_scroll_images", 50),
        "grid_columns": arena.get("grid_columns", 4),
        "enable_why_tags": arena.get("enable_why_tags", True),
        "why_tags": arena.get("why_tags", []),
    }


@app.get("/api/match")
async def api_match(query: str, participant_id: str = ""):
    """
    Get a new match: select two models, retrieve results, return image grids.
    When all loaded models are in MODEL_IDS, restricts selection to ALLOWED_PAIRS.
    Randomises left/right assignment.
    """
    enabled_models = ENGINE.loaded_models()
    if not enabled_models:
        enabled_models = [m["id"] for m in CONFIG.get("models", []) if m.get("enabled")]
    if len(enabled_models) < 2:
        raise HTTPException(400, "Need at least 2 enabled models")

    # Build reverse lookup: model_id_string → mnemonic key (M1..M7)
    id_to_key = {v: k for k, v in MODEL_IDS.items()}
    loaded_keys = {id_to_key[m] for m in enabled_models if m in id_to_key}

    # Use allowed pairs when at least 2 loaded models are in the AIES26 registry
    if len(loaded_keys) >= 2:
        valid_pairs = [
            (MODEL_IDS[a], MODEL_IDS[b])
            for a, b in ALLOWED_PAIRS
            if a in loaded_keys and b in loaded_keys
        ]
        if not valid_pairs:
            raise HTTPException(400, "No valid allowed pairs available for loaded models")
        model_a, model_b = random.choice(valid_pairs)
    else:
        model_a, model_b = random.sample(enabled_models, 2)

    # Randomise left/right position
    if random.random() < 0.5:
        left_model, right_model = model_a, model_b
        position_a = "left"
    else:
        left_model, right_model = model_b, model_a
        position_a = "right"

    n_images = CONFIG["arena"].get("max_scroll_images", 50)

    # Get retrieval results (cached or compute)
    left_result = await get_retrieval(left_model, query, n_images)
    right_result = await get_retrieval(right_model, query, n_images)

    dataset_id = CONFIG.get("arena", {}).get("active_dataset_id", "default")

    return {
        "model_a": model_a,
        "model_b": model_b,
        "position_a": position_a,
        "dataset_id": dataset_id,
        "left": {
            "label": "Model A" if position_a == "left" else "Model B",
            "indices": left_result["indices"],
        },
        "right": {
            "label": "Model B" if position_a == "left" else "Model A",
            "indices": right_result["indices"],
        },
        "query": query,
    }


async def get_retrieval(model_id: str, query: str, n_images: int) -> dict:
    """Get retrieval results: bundle → DB cache → live encode with lazy text encoder."""
    # Try bundle first (pre-computed, fastest)
    bundle_result = ENGINE.bundle_retrieve(model_id, query, n_images)
    if bundle_result:
        return bundle_result

    # Try DB cache
    cached = await db.get_cached_retrieval(model_id, query)
    if cached:
        return {"indices": cached["indices"][:n_images], "similarities": cached["similarities"][:n_images]}

    # Encode live using lazy text encoder + stored image embeddings from bundle
    if not CONFIG.get("arena", {}).get("allow_open_queries", False):
        raise HTTPException(400, f"Query '{query}' not in bundle and open queries are disabled")
    if model_id not in ENGINE.image_embeddings:
        raise HTTPException(400, f"No image embeddings available for model {model_id}")
    query_emb = await ENGINE.encode_query_async(model_id, query)
    img_embs = ENGINE.image_embeddings[model_id]
    import numpy as np
    sims = (query_emb @ img_embs.T).squeeze(0).detach().numpy()
    ranked_idx = np.argsort(sims)[::-1]
    indices = ranked_idx.tolist()
    similarities = [round(float(sims[i]), 4) for i in ranked_idx]
    await db.cache_retrieval(model_id, query, indices, similarities)
    return {"indices": indices[:n_images], "similarities": similarities[:n_images]}


@app.get("/api/image/{index}")
async def api_image(index: int):
    """Serve a dataset image by index."""
    if index < 0 or index >= ENGINE.dataset_size():
        raise HTTPException(404, "Image not found")
    img_bytes = ENGINE.get_image_bytes(index)
    return Response(content=img_bytes, media_type="image/jpeg")


@app.post("/api/vote")
async def api_vote(request: Request):
    """Record a vote."""
    vote = await request.json()
    # Validate
    if vote.get("winner") not in ("A", "B", "tie"):
        raise HTTPException(400, "winner must be 'A', 'B', or 'tie'")

    if ACTIVE_SESSION:
        vote["session_id"] = ACTIVE_SESSION["id"]
    if ACTIVE_WORKSHOP:
        vote["workshop_id"] = ACTIVE_WORKSHOP["id"]

    k = CONFIG["arena"].get("elo_k_factor", 32)
    initial = CONFIG["arena"].get("elo_initial_rating", 1500)
    result = await db.record_vote(vote, k_factor=k, initial_rating=initial)
    return {"status": "ok", **result}


@app.get("/api/leaderboard")
async def api_leaderboard():
    """Public leaderboard — Elo ratings."""
    ratings = await db.get_ratings()
    # Add display names
    model_names = {m["id"]: m["name"] for m in CONFIG.get("models", [])}
    for r in ratings:
        r["display_name"] = model_names.get(r["model_id"], r["model_id"])
    return {"ratings": ratings}


# ═══════════════════════════════════════════════════════════════════════════
#  Admin API
# ═══════════════════════════════════════════════════════════════════════════

def check_admin(request: Request):
    """Simple token-based admin auth."""
    token = request.headers.get("X-Admin-Token", "")
    if token != ADMIN_TOKEN:
        # Also check query param for convenience
        token = request.query_params.get("token", "")
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "Invalid admin token")


@app.get("/api/admin/stats")
async def api_admin_stats(request: Request):
    check_admin(request)
    stats = await db.get_vote_stats()
    ratings = await db.get_ratings()
    model_names = {m["id"]: m["name"] for m in CONFIG.get("models", [])}
    for r in ratings:
        r["display_name"] = model_names.get(r["model_id"], r["model_id"])
    return {"stats": stats, "ratings": ratings}


@app.get("/api/admin/recent")
async def api_admin_recent(request: Request, limit: int = 50):
    check_admin(request)
    votes = await db.get_recent_votes(limit)
    return {"votes": votes}


@app.get("/api/admin/config")
async def api_admin_get_config(request: Request):
    check_admin(request)
    return CONFIG


@app.post("/api/admin/config")
async def api_admin_set_config(request: Request):
    """Update configuration (persisted to disk)."""
    global CONFIG
    check_admin(request)
    new_config = await request.json()
    CONFIG.update(new_config)
    # Save to disk
    config_path = Path(__file__).parent / "config" / "active_config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    return {"status": "ok", "message": "Config updated and saved"}


@app.post("/api/admin/reset_elo")
async def api_admin_reset_elo(request: Request):
    check_admin(request)
    initial = CONFIG["arena"].get("elo_initial_rating", 1500)
    await db.reset_elo(initial)
    return {"status": "ok", "message": f"All Elo ratings reset to {initial}"}


@app.get("/api/admin/export_csv")
async def api_admin_export(request: Request):
    check_admin(request)
    csv_data = await db.export_votes_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fairness_arena_votes.csv"}
    )


# Canonical pairs — (model_a, model_b). winner=="A" always means model_a won,
# regardless of which side it appeared on (position is randomized per match).
_ALLOWED_PAIRS = [
    ("openai/clip-vit-base-patch16",           "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"),
    ("openai/clip-vit-base-patch16",           "google/siglip-base-patch16-224"),
    ("google/siglip-base-patch16-224",         "google/siglip2-base-patch16-224"),
    ("laion/CLIP-ViT-B-16-laion2B-s34B-b88K", "M2_SANER"),
    ("laion/CLIP-ViT-B-16-laion2B-s34B-b88K", "M2_NeuralInt"),
    ("M2_SANER",                               "M2_NeuralInt"),
    ("laion/CLIP-ViT-B-16-laion2B-s34B-b88K", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"),
]
_PAIR_SET = set(_ALLOWED_PAIRS)


async def _load_votes_and_workshops():
    """Fetch all votes joined with workshop info as a list of dicts."""
    import csv, io
    csv_str = await db.export_for_analysis()
    return list(csv.DictReader(io.StringIO(csv_str)))


def _empty_bucket():
    return {"n_a": 0, "n_b": 0, "n_tie": 0}


def _tally(rows, group_fn):
    """
    Key: "model_a vs model_b" (full IDs, matching JS PAIR_META).
    winner=="A" = model_a preferred, winner=="B" = model_b preferred.
    Position randomization is already baked in by how votes are recorded.
    """
    from collections import defaultdict
    out = defaultdict(lambda: defaultdict(_empty_bucket))
    for v in rows:
        ma, mb = v.get("model_a", ""), v.get("model_b", "")
        if (ma, mb) not in _PAIR_SET:
            continue
        pair_key = f"{ma} vs {mb}"
        group    = group_fn(v)
        winner   = v.get("winner", "")
        bucket   = "n_a" if winner == "A" else ("n_b" if winner == "B" else "n_tie")
        out[pair_key][group][bucket] += 1
    return {k: dict(v) for k, v in out.items()}



@app.get("/results")
async def results_page():
    return FileResponse(str(STATIC_DIR / "results.html"))


@app.get("/api/live_results")
async def api_live_results():
    """Win rates per model pair x workshop. Public, no auth."""
    import time
    from collections import defaultdict, OrderedDict

    rows = await _load_votes_and_workshops()

    workshops_seen: dict = OrderedDict()
    for v in rows:
        wid = v.get("workshop_id", "")
        if wid and wid not in workshops_seen:
            workshops_seen[wid] = {
                "id":                wid,
                "name":              v.get("workshop_name") or f"Workshop {wid}",
                "community_context": v.get("community_context") or "",
            }

    workshops  = list(workshops_seen.values())
    cell_data  = _tally(rows, lambda v: v.get("workshop_id") or "none")

    overall: dict = defaultdict(_empty_bucket)
    for v in rows:
        ma, mb = v.get("model_a", ""), v.get("model_b", "")
        if (ma, mb) not in _PAIR_SET:
            continue
        winner = v.get("winner", "")
        bucket = "n_a" if winner == "A" else ("n_b" if winner == "B" else "n_tie")
        overall[f"{ma} vs {mb}"][bucket] += 1

    return {
        "workshops":       workshops,
        "data":            cell_data,
        "overall":         dict(overall),
        "total_votes":     len(rows),
        "last_updated":    time.time(),
        "active_workshop": ACTIVE_WORKSHOP,
    }


@app.get("/api/live_results/by_query")
async def api_live_results_by_query():
    """Win rates per model pair x query_category. Public, no auth."""
    import time

    rows      = await _load_votes_and_workshops()
    cats      = sorted({v.get("query_category") or "unknown" for v in rows})
    cell_data = _tally(rows, lambda v: v.get("query_category") or "unknown")

    return {
        "categories":   cats,
        "data":         cell_data,
        "total_votes":  len(rows),
        "last_updated": time.time(),
    }


@app.get("/api/live_results/full")
async def api_live_results_full():
    """Win rates per pair x query x session. Pairs are normalized; sessions from DB table."""
    import time
    import aiosqlite
    from collections import defaultdict

    async with aiosqlite.connect(str(db.DB_PATH)) as conn:
        conn.row_factory = aiosqlite.Row

        cur = await conn.execute("SELECT id, name FROM sessions ORDER BY started_at")
        sessions = [{"id": r["id"], "name": r["name"]} for r in await cur.fetchall()]

        cur = await conn.execute(
            "SELECT model_a, model_b, winner, query, session_id FROM votes ORDER BY timestamp"
        )
        votes = [dict(r) for r in await cur.fetchall()]

    model_names = {m["id"]: m["name"] for m in CONFIG.get("models", [])}

    def norm(ma, mb):
        """Return canonical (ma, mb, flipped). Alphabetical order."""
        return (ma, mb, False) if ma <= mb else (mb, ma, True)

    queries_set: set = set()
    queries_list: list = []
    tally: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"n_a": 0, "n_b": 0, "n_tie": 0}
    )))

    for v in votes:
        ma_raw, mb_raw = v["model_a"] or "", v["model_b"] or ""
        if not ma_raw or not mb_raw:
            continue
        ma, mb, flipped = norm(ma_raw, mb_raw)
        pair_key = f"{ma} vs {mb}"
        q   = v["query"] or "unknown"
        sid = v["session_id"] or "none"

        winner = v["winner"] or ""
        if flipped and winner in ("A", "B"):
            winner = "B" if winner == "A" else "A"

        bucket = "n_a" if winner == "A" else ("n_b" if winner == "B" else "n_tie")
        tally[pair_key][q][sid][bucket] += 1

        if q not in queries_set:
            queries_list.append(q)
            queries_set.add(q)

    queries_list.sort()
    pairs = sorted(tally.keys())

    return {
        "sessions":     sessions,
        "queries":      queries_list,
        "pairs":        pairs,
        "model_names":  model_names,
        "data":         {pk: {q: dict(ws) for q, ws in qm.items()} for pk, qm in tally.items()},
        "total_votes":  len(votes),
        "last_updated": time.time(),
        "active_session": ACTIVE_SESSION,
    }


@app.get("/api/export_analysis")
async def api_export_analysis(request: Request):
    """Export votes joined with workshop metadata for analysis scripts."""
    check_admin(request)
    csv_data = await db.export_for_analysis()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fairness_arena_analysis.csv"}
    )


# ── Workshop endpoints ─────────────────────────────────────────────────────

@app.post("/api/workshop/create")
async def api_workshop_create(request: Request):
    """Create a new workshop record and return it (including its id)."""
    check_admin(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    workshop = await db.create_workshop(
        name=name,
        location=body.get("location", ""),
        date=body.get("date", ""),
        community_context=body.get("community_context", ""),
        facilitator=body.get("facilitator", ""),
        notes=body.get("notes", ""),
    )
    log.info(f"Workshop created: '{name}' (id={workshop['id']})")
    return workshop


@app.post("/api/workshop/set_active")
async def api_workshop_set_active(request: Request):
    """Set the active workshop. All subsequent votes will link to this workshop_id."""
    global ACTIVE_WORKSHOP
    check_admin(request)
    body = await request.json()
    workshop_id = body.get("workshop_id")
    if workshop_id is None:
        raise HTTPException(400, "workshop_id is required")
    workshop = await db.get_workshop(int(workshop_id))
    if workshop is None:
        raise HTTPException(404, f"Workshop {workshop_id} not found")
    ACTIVE_WORKSHOP = workshop
    log.info(f"Active workshop set: '{workshop['name']}' (id={workshop['id']})")
    return {"status": "ok", "active_workshop": ACTIVE_WORKSHOP}


@app.post("/api/workshop/clear_active")
async def api_workshop_clear_active(request: Request):
    """Unset the active workshop (stop linking votes to a workshop)."""
    global ACTIVE_WORKSHOP
    check_admin(request)
    ACTIVE_WORKSHOP = None
    return {"status": "ok", "active_workshop": None}


@app.get("/api/workshop/list")
async def api_workshop_list(request: Request):
    check_admin(request)
    workshops = await db.get_workshops()
    return {"workshops": workshops, "active_workshop": ACTIVE_WORKSHOP}


@app.get("/api/admin/sessions")
async def api_admin_list_sessions(request: Request):
    check_admin(request)
    sessions = await db.get_sessions()
    return {"sessions": sessions, "active_session": ACTIVE_SESSION}


@app.post("/api/admin/sessions")
async def api_admin_create_session(request: Request):
    """Start a new named session. Stops any currently active session first."""
    global ACTIVE_SESSION
    check_admin(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    started_at = body.get("started_at")  # optional UTC unix timestamp override

    if ACTIVE_SESSION:
        await db.stop_session(ACTIVE_SESSION["id"])

    ACTIVE_SESSION = await db.create_session(name, started_at)
    log.info(f"Session started: '{name}' (id={ACTIVE_SESSION['id']})")
    return ACTIVE_SESSION


@app.post("/api/admin/sessions/stop")
async def api_admin_stop_session(request: Request):
    """Stop the currently active session."""
    global ACTIVE_SESSION
    check_admin(request)
    if not ACTIVE_SESSION:
        raise HTTPException(400, "No active session")
    result = await db.stop_session(ACTIVE_SESSION["id"])
    log.info(f"Session stopped: '{ACTIVE_SESSION['name']}' (id={ACTIVE_SESSION['id']})")
    ACTIVE_SESSION = None
    return result


@app.post("/api/admin/precompute")
async def api_admin_precompute(request: Request):
    """Pre-compute retrieval results for all (model, query) pairs."""
    check_admin(request)
    queries = CONFIG["arena"].get("predefined_queries", [])
    models = [m["id"] for m in CONFIG.get("models", []) if m.get("enabled")]
    n_images = CONFIG["arena"].get("images_per_model", 12)

    count = 0
    for mid in models:
        for q in queries:
            cached = await db.get_cached_retrieval(mid, q)
            if not cached:
                result = ENGINE.retrieve(mid, q, top_k=n_images)
                await db.cache_retrieval(mid, q, result["indices"], result["similarities"])
                count += 1

    return {"status": "ok", "newly_computed": count, "total_pairs": len(models) * len(queries)}


# ── Dataset management ────────────────────────────────────────────────────

@app.get("/api/admin/datasets")
async def api_admin_datasets(request: Request):
    """List all configured datasets and their bundle availability."""
    check_admin(request)
    datasets = get_datasets(CONFIG)
    active_id = CONFIG.get("arena", {}).get("active_dataset_id")
    is_bundle_mode = bool(BUNDLE_PATH or BUNDLES_DIR)

    result = []
    for ds in datasets:
        ds_id = ds["id"]
        bundle = bundle_path_for(ds_id)
        # Also check legacy single-bundle path
        if not bundle and BUNDLE_PATH and active_id == ds_id:
            bundle = Path(BUNDLE_PATH) if Path(BUNDLE_PATH).exists() else None
        result.append({
            "id": ds_id,
            "name": ds.get("name", ds_id),
            "source": ds.get("source", "huggingface"),
            "hf_repo": ds.get("hf_repo"),
            "folder_path": ds.get("folder_path"),
            "max_images": ds.get("max_images"),
            "bundle_available": bundle is not None,
            "bundle_path": str(bundle) if bundle else None,
            "active": ds_id == active_id,
        })

    return {
        "datasets": result,
        "active_dataset_id": active_id,
        "bundle_mode": is_bundle_mode,
    }


@app.post("/api/admin/datasets/switch")
async def api_admin_switch_dataset(request: Request):
    """Switch the active dataset by loading its pre-computed bundle."""
    global CONFIG
    check_admin(request)

    body = await request.json()
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        raise HTTPException(400, "dataset_id is required")

    # Verify the dataset exists in config
    datasets = get_datasets(CONFIG)
    ds_cfg = next((d for d in datasets if d["id"] == dataset_id), None)
    if ds_cfg is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found in config")

    # Require bundle mode for switching
    if not BUNDLES_DIR and not BUNDLE_PATH:
        raise HTTPException(400,
            "Dataset switching is only available in bundle mode. "
            "Start the server with --bundles-dir data/")

    # Find the bundle
    bundle = bundle_path_for(dataset_id)
    if not bundle:
        raise HTTPException(404,
            f"No bundle found for dataset '{dataset_id}'. "
            f"Run: python precompute.py --dataset-id {dataset_id}")

    log.info(f"Switching dataset to '{dataset_id}' (bundle: {bundle})")

    # Reload engine with new bundle
    bundle_config = ENGINE.load_bundle(str(bundle))

    # Update active dataset id in config
    CONFIG.setdefault("arena", {})["active_dataset_id"] = dataset_id

    # Save updated config
    config_path = Path(__file__).parent / "config" / "active_config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Ensure Elo entries exist for loaded models
    initial = CONFIG["arena"].get("elo_initial_rating", 1500)
    await db.ensure_model_ratings(ENGINE.loaded_models(), initial)

    log.info(f"Dataset switched to '{dataset_id}': "
             f"{ENGINE.dataset_size()} images, {ENGINE.loaded_models()} models")

    return {
        "status": "ok",
        "active_dataset_id": dataset_id,
        "n_images": ENGINE.dataset_size(),
        "models": ENGINE.loaded_models(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Startup
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


async def startup():
    global ENGINE, CONFIG, ACTIVE_SESSION, ACTIVE_WORKSHOP

    # Init database
    await db.init_db()

    # Restore any session that was active before the server last stopped
    ACTIVE_SESSION = await db.get_active_session()
    if ACTIVE_SESSION:
        log.info(f"Resuming active session: '{ACTIVE_SESSION['name']}' (id={ACTIVE_SESSION['id']})")

    bundle = active_bundle_path()

    if bundle:
        # ── Bundle mode: load pre-computed data, no GPU needed ───────
        log.info(f"Bundle mode: loading {bundle}")
        bundle_config = ENGINE.load_bundle(str(bundle))
        # Merge bundle config with any overrides from active_config
        for key in ["arena", "models", "datasets", "dataset"]:
            if key not in CONFIG or CONFIG[key] == bundle_config.get(key):
                CONFIG[key] = bundle_config.get(key, CONFIG.get(key))
        # Pre-load text encoders if open queries are enabled
        if CONFIG.get("arena", {}).get("allow_open_queries", False):
            await ENGINE.preload_text_encoders()
    else:
        # ── Live mode: load models and dataset, needs GPU ────────────
        ENGINE.load_models(CONFIG.get("models", []))

        # Resolve the active dataset config
        datasets = get_datasets(CONFIG)
        active_id = CONFIG.get("arena", {}).get("active_dataset_id")
        if active_id:
            ds_cfg = next((d for d in datasets if d["id"] == active_id), None)
        else:
            ds_cfg = datasets[0] if datasets else None

        if ds_cfg is None:
            raise RuntimeError("No dataset configured")

        source = ds_cfg.get("source", "huggingface")
        if source == "huggingface":
            ENGINE.load_dataset_from_huggingface(
                repo=ds_cfg["hf_repo"],
                split=ds_cfg.get("hf_split", "train"),
                image_column=ds_cfg.get("image_column", "image"),
                max_images=ds_cfg.get("max_images", 2000),
                hf_config=ds_cfg.get("hf_config"),
            )
        elif source == "folder":
            ENGINE.load_dataset_from_folder(
                folder=ds_cfg["folder_path"],
                max_images=ds_cfg.get("max_images", 2000),
            )

        ENGINE.embed_all_images()

    # Ensure Elo entries exist
    model_ids = ENGINE.loaded_models()
    initial = CONFIG["arena"].get("elo_initial_rating", 1500)
    await db.ensure_model_ratings(model_ids, initial)

    mode = "bundle" if bundle else "live"
    log.info("=" * 60)
    log.info(f"  Fairness Arena ready! (mode: {mode})")
    log.info(f"  Active dataset: {CONFIG.get('arena', {}).get('active_dataset_id', 'unknown')}")
    log.info(f"  Models: {ENGINE.loaded_models()}")
    log.info(f"  Dataset: {ENGINE.dataset_size()} images")
    log.info(f"  Queries: {len(CONFIG['arena'].get('predefined_queries', []))}")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Fairness Arena")
    p.add_argument("--config", default="config/default_config.json")
    p.add_argument("--bundle", default=None,
                   help="Path to a single pre-computed bundle (.npz). "
                        "Legacy option — prefer --bundles-dir for multi-dataset support.")
    p.add_argument("--bundles-dir", default=None,
                   help="Directory containing per-dataset bundles "
                        "(arena_bundle_{dataset_id}.npz). Enables dataset switching.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--device", default="auto")
    p.add_argument("--admin-token", default="changeme",
                   help="Admin API token (default: changeme)")
    return p.parse_args()


def main():
    global CONFIG, ENGINE, ADMIN_TOKEN, BUNDLE_PATH, BUNDLES_DIR

    args = parse_args()
    ADMIN_TOKEN = args.admin_token
    BUNDLE_PATH = args.bundle
    BUNDLES_DIR = args.bundles_dir

    # Auto-detect bundles in data/ when no explicit flag was given
    if not BUNDLE_PATH and not BUNDLES_DIR:
        data_dir = Path(__file__).parent / "data"
        legacy = data_dir / "arena_bundle.npz"
        named = sorted(data_dir.glob("arena_bundle_*.npz")) if data_dir.exists() else []
        if named:
            BUNDLES_DIR = str(data_dir)
            log.info(f"Auto-detected bundles dir: {data_dir} ({len(named)} bundle(s))")
        elif legacy.exists():
            BUNDLE_PATH = str(legacy)
            log.info(f"Auto-detected bundle: {legacy}")

    # Load config
    active_config = Path(__file__).parent / "config" / "active_config.json"
    if active_config.exists():
        CONFIG = load_config(str(active_config))
        log.info(f"Loaded active config from {active_config}")
    else:
        CONFIG = load_config(args.config)
        log.info(f"Loaded config from {args.config}")

    # Init retrieval engine
    ENGINE = RetrievalEngine(device=args.device)

    # Register startup event
    @app.on_event("startup")
    async def on_startup():
        await startup()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
