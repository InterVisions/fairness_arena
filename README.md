# Fairness Arena — CLIP Retrieval Fairness Evaluation

An LMSYS Chatbot Arena-style tool for evaluating the **fairness** of CLIP-based image retrieval models through human preference voting.

Participants see side-by-side image search results from two anonymous models, and vote for which set better represents the diversity of their community. Votes are aggregated using the Elo rating system to produce a fairness leaderboard.

## Quick Start

There are two ways to run the server: **live mode** (GPU machine does everything) or **bundle mode** (pre-compute on GPU, serve from any CPU machine). Bundle mode is recommended for workshops and cloud deployment.

### Option A: Bundle mode (recommended)

**Step 1 — On a GPU machine**, pre-compute all embeddings and retrieval results:

```bash
pip install -r requirements.txt
python precompute.py --config config/default_config.json --output data/arena_bundle.npz
```

This loads every CLIP model, embeds all dataset images, computes retrieval rankings for all (model × query) pairs, creates web-ready thumbnails, and packs everything into a single portable `.npz` file. For 2000 images × 4 models, expect ~5-15 minutes.

**Step 2 — Copy the bundle** to your server (any machine, no GPU needed):

```bash
scp data/arena_bundle.npz yourserver:/path/to/fairness-arena/data/
```

**Step 3 — Run the server** (CPU-only, no PyTorch needed at runtime):

```bash
python server.py --bundle data/arena_bundle.npz --admin-token my_secret
```

The bundle contains thumbnails, all retrieval rankings, image embeddings (for open queries), and the config snapshot. Startup takes a few seconds.

### Option B: Live mode (single GPU machine)

```bash
pip install -r requirements.txt
python server.py --admin-token my_secret
```

This loads models, downloads the dataset, and embeds everything at startup. Requires GPU and takes several minutes to start.

---

Open `http://localhost:8080` for the arena, `/admin` for the dashboard, `/leaderboard` for rankings.

## Architecture

```
                    ┌─────────────────────────────┐
                    │   GPU machine (one-time)     │
                    │                              │
                    │   precompute.py              │
                    │   ├── Load CLIP models       │
                    │   ├── Load dataset (HF/local)│
                    │   ├── Embed all images       │
                    │   ├── Compute all retrievals │
                    │   └── Save arena_bundle.npz  │
                    └──────────────┬───────────────┘
                                   │ scp
                    ┌──────────────▼───────────────┐
                    │   Server (CPU, AWS, etc.)     │
                    │                              │
Browser ──────────► │   server.py --bundle ...      │
(participant)       │   ├── Load bundle (fast)     │
                    │   ├── Serve image thumbnails  │
Browser ──────────► │   ├── Serve retrieval results │
(admin)             │   ├── Record votes (SQLite)  │
                    │   └── Compute Elo ratings    │
                    └──────────────────────────────┘
```

## CLI Options

### `server.py`

| Flag | Default | Description |
|---|---|---|
| `--bundle` | `None` | Path to pre-computed `.npz` bundle. If provided, no GPU or model loading needed |
| `--config` | `config/default_config.json` | Configuration file (used if no bundle or as overrides) |
| `--port` | `8080` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--device` | `auto` | PyTorch device (only relevant in live mode) |
| `--admin-token` | `changeme` | Token for admin API |

### `precompute.py`

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/default_config.json` | Configuration file (defines models, dataset, queries) |
| `--output` | `data/arena_bundle.npz` | Output bundle path |
| `--device` | `auto` | PyTorch device |
| `--thumbnail-size` | `400` | Max thumbnail dimension in pixels |
| `--batch-size` | `64` | Batch size for image embedding |

## Configuration

All settings are in `config/default_config.json` and can be changed live via the admin panel:

- **Elo parameters:** K-factor, initial rating
- **Arena settings:** images per model, grid layout, predefined queries, open queries toggle
- **Judge question:** the prompt shown to participants
- **Why tags:** optional tags for qualitative feedback
- **Models:** list of CLIP models (open_clip backend)
- **Dataset:** HuggingFace repo or local folder

## What's Inside the Bundle

The `.npz` file produced by `precompute.py` contains:

- **JPEG thumbnails** of all dataset images (web-ready, no need to ship the original dataset)
- **Retrieval rankings** for every (model × query) pair (pre-computed, served instantly)
- **Image embeddings** per model in float16 (enables open queries without GPU — just NumPy matrix multiplication)
- **Config snapshot** (models, queries, dataset metadata)

Typical bundle size: ~50-200 MB depending on dataset size and number of models.

## Key Design Decisions

- **Side-by-side layout** with randomised left/right assignment and position logging for bias detection
- **Pre-computed retrieval results** via portable bundle for GPU-free serving
- **Optional "why" tags** for qualitative signal alongside the quantitative vote
- **Bradley-Terry analysis** can be run post-hoc on the exported CSV for publishable confidence intervals
- **Admin dashboard** with real-time stats, position bias monitoring, and data export

## For the Workshops

1. Run `precompute.py` on any GPU machine (local workstation, Colab, etc.)
2. Copy `arena_bundle.npz` to your AWS/cloud server
3. Run `server.py --bundle arena_bundle.npz` — no GPU, fast startup
4. Share the URL with participants
5. Monitor live via `/admin`
6. Export data via CSV for post-hoc Bradley-Terry analysis

## Project Structure

```
fairness-arena/
├── server.py              # FastAPI server (live or bundle mode)
├── precompute.py          # Offline: embed + retrieve + pack bundle
├── database.py            # SQLite + Elo logic
├── retrieval.py           # CLIP model loading + retrieval + bundle loading
├── requirements.txt
├── config/
│   └── default_config.json
├── data/
│   ├── arena.db           # Created at runtime (votes, ratings)
│   └── arena_bundle.npz   # Created by precompute.py
└── static/
    ├── arena.html          # Public voting interface
    ├── admin.html          # Admin dashboard
    └── leaderboard.html    # Public leaderboard
```
