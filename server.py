from __future__ import annotations
"""
Fairness Arena — CLIP Retrieval Fairness Evaluation Tool
========================================================
An LMSYS-style arena for comparing fairness of CLIP retrieval models.

Usage:
    python server.py
    python server.py --config config/my_config.json --port 8080
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

app = FastAPI(title="Fairness Arena")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    return {
        "predefined_queries": arena.get("predefined_queries", []),
        "allow_open_queries": arena.get("allow_open_queries", False),
        "judge_question": arena.get("judge_question", "Which set of images is fairer?"),
        "images_per_model": arena.get("images_per_model", 12),
        "max_scroll_images": arena.get("max_scroll_images", 50),
        "grid_columns": arena.get("grid_columns", 4),
        "enable_why_tags": arena.get("enable_why_tags", True),
        "why_tags": arena.get("why_tags", []),
    }


@app.get("/api/match")
async def api_match(query: str, participant_id: str = ""):
    """
    Get a new match: select two random models, retrieve results, return image grids.
    Randomises left/right assignment.
    """
    enabled_models = ENGINE.loaded_models()
    if not enabled_models:
        enabled_models = [m["id"] for m in CONFIG.get("models", []) if m.get("enabled")]
    if len(enabled_models) < 2:
        raise HTTPException(400, "Need at least 2 enabled models")

    # Select two models (uniform random for now)
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

    return {
        "model_a": model_a,
        "model_b": model_b,
        "position_a": position_a,
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
    """Get retrieval results: bundle → cache → compute live."""
    # Try bundle first (pre-computed, fastest)
    bundle_result = ENGINE.bundle_retrieve(model_id, query, n_images)
    if bundle_result:
        return bundle_result

    # Try DB cache
    cached = await db.get_cached_retrieval(model_id, query)
    if cached:
        return {"indices": cached["indices"][:n_images], "similarities": cached["similarities"][:n_images]}

    # Compute fresh (needs models loaded)
    if model_id not in ENGINE.models:
        raise HTTPException(400, f"Model {model_id} not loaded and no pre-computed results available")
    result = ENGINE.retrieve(model_id, query, top_k=n_images)
    await db.cache_retrieval(model_id, query, result["indices"], result["similarities"])
    return result


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


# ═══════════════════════════════════════════════════════════════════════════
#  Startup
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


async def startup():
    global ENGINE, CONFIG

    # Init database
    await db.init_db()

    if BUNDLE_PATH:
        # ── Bundle mode: load pre-computed data, no GPU needed ───────
        bundle_config = ENGINE.load_bundle(BUNDLE_PATH)
        # Merge bundle config with any overrides from active_config
        for key in ["arena", "models", "dataset"]:
            if key not in CONFIG or CONFIG[key] == bundle_config.get(key):
                CONFIG[key] = bundle_config.get(key, CONFIG.get(key))
    else:
        # ── Live mode: load models and dataset, needs GPU ────────────
        ENGINE.load_models(CONFIG.get("models", []))

        ds_cfg = CONFIG.get("dataset", {})
        source = ds_cfg.get("source", "huggingface")

        if source == "huggingface":
            ENGINE.load_dataset_from_huggingface(
                repo=ds_cfg["hf_repo"],
                split=ds_cfg.get("hf_split", "train"),
                image_column=ds_cfg.get("image_column", "image"),
                max_images=ds_cfg.get("max_images", 2000),
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

    mode = "bundle" if BUNDLE_PATH else "live"
    log.info("=" * 60)
    log.info(f"  Fairness Arena ready! (mode: {mode})")
    log.info(f"  Models: {ENGINE.loaded_models()}")
    log.info(f"  Dataset: {ENGINE.dataset_size()} images")
    log.info(f"  Queries: {len(CONFIG['arena'].get('predefined_queries', []))}")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

BUNDLE_PATH = None

def parse_args():
    p = argparse.ArgumentParser(description="Fairness Arena")
    p.add_argument("--config", default="config/default_config.json")
    p.add_argument("--bundle", default=None,
                   help="Path to pre-computed bundle (.npz) from precompute.py. "
                        "If provided, no GPU or model loading is needed.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--device", default="auto")
    p.add_argument("--admin-token", default="changeme",
                   help="Admin API token (default: changeme)")
    return p.parse_args()


def main():
    global CONFIG, ENGINE, ADMIN_TOKEN, BUNDLE_PATH

    args = parse_args()
    ADMIN_TOKEN = args.admin_token
    BUNDLE_PATH = args.bundle

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
