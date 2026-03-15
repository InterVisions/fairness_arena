from __future__ import annotations
"""
Retrieval engine — loads CLIP models and dataset, computes ranked results.
All results are pre-computed and cached in SQLite for fast serving.
"""

import logging
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger("fairness-arena.retrieval")


class RetrievalEngine:
    """Manages CLIP models and dataset for retrieval."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.models = {}          # model_id -> {model, preprocess, tokenizer}
        self.dataset = None       # list of PIL images
        self.image_embeddings = {}  # model_id -> (N, D) tensor
        self.image_paths = []     # list of paths or indices for serving
        self.dataset_loaded = False
        self.thumbnails = []      # pre-computed JPEG bytes (from bundle)
        self._bundle_model_ids = []
        self._bundle_queries = []

    # ── Model loading ────────────────────────────────────────────────────

    def load_model(self, model_config: dict):
        """Load a single CLIP model from config."""
        mid = model_config["id"]
        if mid in self.models:
            log.info(f"Model {mid} already loaded, skipping")
            return

        backend = model_config.get("backend", "open_clip")
        model_name = model_config["model_name"]
        pretrained = model_config.get("pretrained", "openai")

        log.info(f"Loading model {mid}: {model_name} ({pretrained}) via {backend} …")
        t0 = time.time()

        if backend == "open_clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        model.eval()
        self.models[mid] = {
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "config": model_config,
        }
        log.info(f"✓ Model {mid} loaded in {time.time() - t0:.1f}s")

    def load_models(self, model_configs: list[dict]):
        """Load all enabled models."""
        for cfg in model_configs:
            if cfg.get("enabled", True):
                self.load_model(cfg)

    # ── Dataset loading ──────────────────────────────────────────────────

    def load_dataset_from_huggingface(self, repo: str, split: str = "train",
                                       image_column: str = "image",
                                       max_images: int = 2000):
        """Load an image dataset from HuggingFace."""
        from datasets import load_dataset

        log.info(f"Loading dataset {repo} (split={split}, max={max_images}) …")
        t0 = time.time()

        ds = load_dataset(repo, split=split)
        if max_images and len(ds) > max_images:
            ds = ds.select(range(max_images))

        self.dataset = []
        for i, item in enumerate(ds):
            img = item[image_column]
            if isinstance(img, Image.Image):
                self.dataset.append(img.convert("RGB"))
            else:
                # Might be a path or bytes
                self.dataset.append(Image.open(img).convert("RGB"))

        self.dataset_loaded = True
        log.info(f"✓ Dataset loaded: {len(self.dataset)} images in {time.time() - t0:.1f}s")

    def load_dataset_from_folder(self, folder: str, max_images: int = 2000):
        """Load images from a local folder."""
        folder = Path(folder)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in extensions])
        if max_images:
            paths = paths[:max_images]

        log.info(f"Loading {len(paths)} images from {folder} …")
        self.dataset = []
        self.image_paths = []
        for p in paths:
            try:
                self.dataset.append(Image.open(p).convert("RGB"))
                self.image_paths.append(str(p))
            except Exception as e:
                log.warning(f"Failed to load {p}: {e}")

        self.dataset_loaded = True
        log.info(f"✓ Loaded {len(self.dataset)} images from folder")

    # ── Image embedding ──────────────────────────────────────────────────

    @torch.no_grad()
    def embed_images(self, model_id: str, batch_size: int = 64):
        """Embed all dataset images with a specific model. Results are cached in memory."""
        if model_id in self.image_embeddings:
            log.info(f"Image embeddings for {model_id} already computed")
            return

        if not self.dataset_loaded:
            raise RuntimeError("No dataset loaded")

        m = self.models[model_id]
        model = m["model"]
        preprocess = m["preprocess"]

        log.info(f"Embedding {len(self.dataset)} images with {model_id} …")
        t0 = time.time()

        all_embs = []
        for i in range(0, len(self.dataset), batch_size):
            batch_imgs = self.dataset[i:i + batch_size]
            tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(self.device)
            embs = model.encode_image(tensors)
            embs = F.normalize(embs, dim=-1)
            all_embs.append(embs.cpu())

            if (i // batch_size) % 10 == 0:
                log.info(f"  {model_id}: {i + len(batch_imgs)}/{len(self.dataset)} images")

        self.image_embeddings[model_id] = torch.cat(all_embs, dim=0)
        log.info(f"✓ Image embeddings for {model_id}: shape {self.image_embeddings[model_id].shape} "
                 f"in {time.time() - t0:.1f}s")

    def embed_all_images(self):
        """Embed dataset with all loaded models."""
        for mid in self.models:
            self.embed_images(mid)

    # ── Text encoding ────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_query(self, model_id: str, query: str) -> torch.Tensor:
        """Encode a text query → (1, D) normalised embedding."""
        m = self.models[model_id]
        tokens = m["tokenizer"]([query]).to(self.device)
        emb = m["model"].encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu()

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(self, model_id: str, query: str, top_k: int = None) -> dict:
        """
        Retrieve ranked images for a query using a specific model.
        Returns {indices: [...], similarities: [...]}.
        """
        query_emb = self.encode_query(model_id, query)  # (1, D)
        img_embs = self.image_embeddings[model_id]       # (N, D)

        sims = (query_emb @ img_embs.T).squeeze(0).numpy()  # (N,)
        ranked_idx = np.argsort(sims)[::-1]

        if top_k:
            ranked_idx = ranked_idx[:top_k]

        return {
            "indices": ranked_idx.tolist(),
            "similarities": [round(float(sims[i]), 4) for i in ranked_idx],
        }

    # ── Image serving ────────────────────────────────────────────────────

    def get_image_bytes(self, index: int, max_size: int = 400) -> bytes:
        """Get a JPEG-encoded image by dataset index."""
        # If loaded from bundle, thumbnails are already pre-computed
        if self.thumbnails:
            if 0 <= index < len(self.thumbnails):
                return self.thumbnails[index]
            return b""

        img = self.dataset[index]
        # Resize maintaining aspect ratio
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def dataset_size(self) -> int:
        if self.thumbnails:
            return len(self.thumbnails)
        return len(self.dataset) if self.dataset else 0

    def loaded_models(self) -> list[str]:
        if self._bundle_model_ids:
            return self._bundle_model_ids
        return list(self.models.keys())

    # ── Bundle loading (no GPU needed) ───────────────────────────────

    def load_bundle(self, bundle_path: str) -> dict:
        """
        Load a pre-computed bundle from precompute.py.
        Returns the config dict stored in the bundle.
        No GPU, no CLIP models, no dataset download needed.
        """
        import json

        log.info(f"Loading pre-computed bundle: {bundle_path} …")
        t0 = time.time()
        data = np.load(bundle_path, allow_pickle=False)

        # Metadata
        config = json.loads(str(data["config_json"][0]))
        queries = json.loads(str(data["queries_json"][0]))
        model_ids = json.loads(str(data["model_ids_json"][0]))
        n_images = int(data["n_images"][0])

        self._bundle_model_ids = model_ids
        self._bundle_queries = queries

        # Thumbnails
        offsets = data["thumbnail_offsets"]
        raw_bytes = data["thumbnail_bytes"].tobytes()
        self.thumbnails = []
        for i in range(n_images):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            self.thumbnails.append(raw_bytes[start:end])
        log.info(f"  Loaded {len(self.thumbnails)} thumbnails")

        # Retrieval results
        self._bundle_retrievals = json.loads(str(data["retrievals_json"][0]))
        log.info(f"  Loaded retrievals for {len(model_ids)} models × {len(queries)} queries")

        # Image embeddings (for open queries if enabled)
        for mid in model_ids:
            key = f"image_embs__{mid}"
            if key in data:
                self.image_embeddings[mid] = torch.from_numpy(
                    data[key].astype(np.float32)
                )
                log.info(f"  Image embeddings for {mid}: {self.image_embeddings[mid].shape}")

        log.info(f"✓ Bundle loaded in {time.time() - t0:.1f}s")
        return config

    def bundle_retrieve(self, model_id: str, query: str, top_k: int = None) -> dict | None:
        """Get pre-computed retrieval results from the bundle."""
        if not hasattr(self, '_bundle_retrievals'):
            return None
        model_results = self._bundle_retrievals.get(model_id, {})
        result = model_results.get(query)
        if result and top_k:
            return {
                "indices": result["indices"][:top_k],
                "similarities": result["similarities"][:top_k],
            }
        return result
