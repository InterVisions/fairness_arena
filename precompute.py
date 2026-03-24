#!/usr/bin/env python3
from __future__ import annotations
"""
Fairness Arena — Pre-compute embeddings and retrieval results.
==============================================================
Run this on a GPU machine. It produces one portable bundle file per dataset
that the server can load on any CPU-only machine.

Usage:
    # Precompute the active dataset (default)
    python precompute.py --config config/default_config.json

    # Precompute a specific dataset by id
    python precompute.py --dataset-id fairface

    # Precompute all datasets defined in config
    python precompute.py --all-datasets

    # Explicit output path (single dataset only)
    python precompute.py --dataset-id flickr30k --output data/arena_bundle_flickr30k.npz

What it does:
    1. Loads all enabled CLIP models
    2. Loads the dataset (from HuggingFace or local folder)
    3. Embeds all images with every model
    4. Embeds all predefined query prompts with every model
    5. Computes ranked retrieval results for every (model, query) pair
    6. Saves thumbnail JPEGs of all images for web serving
    7. Packs everything into a single .npz file per dataset

The server then loads a bundle at startup — no GPU, no model loading,
no HuggingFace download needed.
"""

import argparse
import json
import logging
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("precompute")


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_datasets(config: dict) -> list[dict]:
    """
    Return the list of dataset configs, handling both old and new formats.
    Old format: single "dataset" key.
    New format: "datasets" list with per-entry "id" and "name".
    """
    if "datasets" in config:
        return config["datasets"]
    # Backward compat: wrap the single dataset entry
    ds = config.get("dataset", {})
    if not ds:
        raise ValueError("No dataset(s) defined in config")
    if "id" not in ds:
        # Derive an id from the repo name or folder
        if ds.get("source") == "huggingface":
            ds = dict(ds, id=ds["hf_repo"].split("/")[-1].lower())
        else:
            ds = dict(ds, id=Path(ds.get("folder_path", "dataset")).stem.lower())
    if "name" not in ds:
        ds = dict(ds, name=ds["id"])
    return [ds]


def load_dataset(ds_cfg: dict) -> list[Image.Image]:
    """Load images from HuggingFace or local folder."""
    source = ds_cfg.get("source", "huggingface")
    max_images = ds_cfg.get("max_images", 2000)

    if source == "huggingface":
        from datasets import load_dataset as hf_load
        repo = ds_cfg["hf_repo"]
        split = ds_cfg.get("hf_split", "train")
        image_col = ds_cfg.get("image_column", "image")

        hf_config = ds_cfg.get("hf_config")
        log.info(f"Loading dataset {repo} (config={hf_config}, split={split}) …")
        load_kwargs = {"split": split}
        if hf_config:
            load_kwargs["name"] = hf_config
        ds = hf_load(repo, **load_kwargs)
        if max_images and len(ds) > max_images:
            ds = ds.select(range(max_images))

        images = []
        for item in ds:
            img = item[image_col]
            if isinstance(img, Image.Image):
                images.append(img.convert("RGB"))
            else:
                images.append(Image.open(img).convert("RGB"))
        return images

    elif source == "folder":
        folder = Path(ds_cfg["folder_path"])
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in extensions])
        if max_images:
            paths = paths[:max_images]
        log.info(f"Loading {len(paths)} images from {folder} …")
        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                log.warning(f"Failed to load {p}: {e}")
        return images

    else:
        raise ValueError(f"Unknown dataset source: {source}")


def make_thumbnails(images: list[Image.Image], max_size: int = 400) -> list[bytes]:
    """Create JPEG thumbnails for web serving."""
    log.info(f"Creating {len(images)} thumbnails (max {max_size}px) …")
    thumbs = []
    for img in images:
        thumb = img.copy()
        thumb.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        thumbs.append(buf.getvalue())
    return thumbs


@torch.no_grad()
def embed_images(model, preprocess, images, device, batch_size=64) -> np.ndarray:
    """Embed all images → (N, D) float32 numpy array."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        embs = model.encode_image(tensors)
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu())
        if (i // batch_size) % 5 == 0:
            log.info(f"  images: {i + len(batch)}/{len(images)}")
    return torch.cat(all_embs, dim=0).numpy()


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device) -> np.ndarray:
    """Embed a list of text prompts → (N, D) float32 numpy array."""
    tokens = tokenizer(texts).to(device)
    embs = model.encode_text(tokens)
    embs = F.normalize(embs, dim=-1)
    return embs.cpu().numpy()


def compute_retrievals(image_embs: np.ndarray, query_embs: np.ndarray,
                       queries: list[str], top_k: int) -> dict:
    """
    Compute ranked retrieval for all queries.
    Returns dict: query -> {indices: [...], similarities: [...]}
    """
    results = {}
    for i, q in enumerate(queries):
        sims = image_embs @ query_embs[i]  # (N,)
        ranked = np.argsort(sims)[::-1][:top_k]
        results[q] = {
            "indices": ranked.tolist(),
            "similarities": [round(float(sims[j]), 4) for j in ranked],
        }
    return results


def precompute_dataset(ds_cfg: dict, config: dict, output_path: Path,
                       device: str, thumbnail_size: int, batch_size: int):
    """Precompute and save a bundle for a single dataset."""
    import open_clip

    ds_id = ds_cfg.get("id", "dataset")
    log.info(f"\n{'#'*60}")
    log.info(f"  Dataset: {ds_cfg.get('name', ds_id)} (id={ds_id})")
    log.info(f"  Output:  {output_path}")
    log.info(f"{'#'*60}\n")

    # ── Load dataset ─────────────────────────────────────────────────
    images = load_dataset(ds_cfg)
    log.info(f"Dataset: {len(images)} images")

    # ── Create thumbnails ────────────────────────────────────────────
    thumbnails = make_thumbnails(images, max_size=thumbnail_size)

    # ── Queries ──────────────────────────────────────────────────────
    queries = config["arena"].get("predefined_queries", [])
    top_k = config["arena"].get("max_scroll_images", 50)
    log.info(f"Queries: {len(queries)}, top_k: {top_k}")

    # ── Process each model ───────────────────────────────────────────
    all_image_embs = {}     # model_id -> (N, D) array
    all_retrievals = {}     # model_id -> {query -> {indices, similarities}}
    model_ids = []

    for mcfg in config.get("models", []):
        if not mcfg.get("enabled", True):
            continue

        mid = mcfg["id"]
        model_ids.append(mid)
        model_name = mcfg["model_name"]
        pretrained = mcfg.get("pretrained", "openai")

        log.info(f"\n{'='*60}")
        log.info(f"Model: {mid} ({model_name}, {pretrained})")
        log.info(f"{'='*60}")

        # Load model
        t0 = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        log.info(f"Model loaded in {time.time() - t0:.1f}s")

        # Embed images
        log.info("Embedding images …")
        t0 = time.time()
        img_embs = embed_images(model, preprocess, images, device, batch_size=batch_size)
        log.info(f"Image embeddings: {img_embs.shape} in {time.time() - t0:.1f}s")
        all_image_embs[mid] = img_embs

        # Embed queries
        log.info("Embedding queries …")
        query_embs = embed_texts(model, tokenizer, queries, device)
        log.info(f"Query embeddings: {query_embs.shape}")

        # Compute retrievals
        log.info("Computing retrievals …")
        retrievals = compute_retrievals(img_embs, query_embs, queries, top_k)
        all_retrievals[mid] = retrievals
        log.info(f"Retrievals computed for {len(queries)} queries")

        # Free GPU memory
        del model, preprocess, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Pack into bundle ─────────────────────────────────────────────
    log.info(f"\nPacking bundle → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        # Metadata
        "config_json": np.array([json.dumps(config)]),
        "queries_json": np.array([json.dumps(queries)]),
        "model_ids_json": np.array([json.dumps(model_ids)]),
        "dataset_id": np.array([ds_id]),
        "n_images": np.array([len(images)]),

        # Thumbnails as variable-length byte arrays
        "thumbnail_offsets": np.array([0] + [len(t) for t in thumbnails]).cumsum(),
        "thumbnail_bytes": np.frombuffer(b"".join(thumbnails), dtype=np.uint8),

        # Retrieval results
        "retrievals_json": np.array([json.dumps(all_retrievals)]),
    }

    # Image embeddings per model
    for mid in model_ids:
        bundle[f"image_embs__{mid}"] = all_image_embs[mid].astype(np.float16)

    np.savez_compressed(str(output_path), **bundle)
    file_size = output_path.stat().st_size / (1024 * 1024)
    log.info(f"✓ Bundle saved: {file_size:.1f} MB")
    log.info(f"  - {len(images)} images, {len(model_ids)} models, {len(queries)} queries")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Fairness Arena bundle")
    parser.add_argument("--config", default="config/default_config.json")
    parser.add_argument("--output", default=None,
                        help="Output path for the bundle. Defaults to "
                             "data/arena_bundle_{dataset_id}.npz. "
                             "Only used when processing a single dataset.")
    parser.add_argument("--dataset-id", default=None,
                        help="ID of a specific dataset to precompute (must match an entry "
                             "in config 'datasets'). Defaults to the active dataset.")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Precompute bundles for all datasets defined in config.")
    parser.add_argument("--bundles-dir", default="data",
                        help="Directory where bundle files are written (default: data/).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thumbnail-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    log.info(f"Device: {device}")

    all_datasets = get_datasets(config)
    active_id = config.get("arena", {}).get("active_dataset_id")

    # Determine which datasets to process
    if args.all_datasets:
        target_datasets = all_datasets
    elif args.dataset_id:
        target_datasets = [d for d in all_datasets if d["id"] == args.dataset_id]
        if not target_datasets:
            ids = [d["id"] for d in all_datasets]
            raise SystemExit(f"Dataset '{args.dataset_id}' not found. Available: {ids}")
    else:
        # Default: active dataset, or first dataset if no active_id set
        if active_id:
            target_datasets = [d for d in all_datasets if d["id"] == active_id]
            if not target_datasets:
                raise SystemExit(f"Active dataset '{active_id}' not found in config datasets.")
        else:
            target_datasets = all_datasets[:1]

    if not target_datasets:
        raise SystemExit("No datasets to process.")

    bundles_dir = Path(args.bundles_dir)

    for ds_cfg in target_datasets:
        ds_id = ds_cfg["id"]
        if args.output and len(target_datasets) == 1:
            output_path = Path(args.output)
        else:
            output_path = bundles_dir / f"arena_bundle_{ds_id}.npz"

        precompute_dataset(
            ds_cfg=ds_cfg,
            config=config,
            output_path=output_path,
            device=device,
            thumbnail_size=args.thumbnail_size,
            batch_size=args.batch_size,
        )

    log.info("\nAll done. Copy the bundle(s) to your server and run:")
    for ds_cfg in target_datasets:
        ds_id = ds_cfg["id"]
        bundle = bundles_dir / f"arena_bundle_{ds_id}.npz"
        log.info(f"  python server.py --bundles-dir data/")


if __name__ == "__main__":
    main()
