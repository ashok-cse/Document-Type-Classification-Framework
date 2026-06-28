# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DTCF — German Document Type Classification Framework. A Pattern Recognition course project (Phase 2) that classifies scanned German document pages into six functional types from DocLayNet using only visual layout (no OCR text features):

`financial_reports` · `scientific_articles` · `laws_and_regulations` · `government_tenders` · `manuals` · `patents`

Two deliverable halves live in one repo:
1. **Training/eval** — a Kaggle notebook (TensorFlow/Keras) that builds a German subset, trains a custom CNN and a ResNet50 transfer-learning model, and emits figures + metrics CSVs.
2. **Inference** — a FastAPI service (`app/`) that serves the trained ResNet50 over HTTP, deployable as a CPU-only Docker image.

## The notebook is generated — never hand-edit the .ipynb

`notebooks/german_doc_classification.ipynb` is **generated** from `notebooks/build_notebook.py`. The builder defines every cell via `md(...)` / `code(...)` calls and writes the `.ipynb` using only the stdlib. To change the notebook, edit `build_notebook.py` then regenerate:

```bash
cd notebooks && python build_notebook.py
```

Editing the `.ipynb` directly will be silently overwritten on the next build.

## Data pipeline gotcha (read before touching data loading)

The data source has been pivoted twice and the README is stale on this point. Current reality (see git log and `build_notebook.py` section 1):

- Source is **`pierreguillou/DocLayNet-base`** (Parquet, ~8,000 pages), **not** the `DocLayNet-large` streaming variant the README still describes.
- `datasets.load_dataset` is **bypassed entirely** — both DocLayNet loader scripts are broken (script-based loaders dropped in `datasets` 4.x; large variant's loader fails on `zip://...` URLs). The notebook downloads data files directly via `huggingface_hub.hf_hub_download` and parses them itself.
- German pages are selected with a layered heuristic (collection/filename) plus a `langdetect` fallback (~2k pages).

If you change data loading, do not reintroduce `load_dataset`.

## Commands

```bash
# Regenerate the notebook after editing build_notebook.py
cd notebooks && python build_notebook.py

# Run the notebook locally (GPU strongly recommended; CPU runs take hours)
pip install -r requirements.txt
jupyter lab notebooks/german_doc_classification.ipynb

# Run the inference service locally (models/resnet50.keras is committed in the repo)
pip install -r app/requirements.txt
LOCAL_MODEL_PATH=./models/resnet50.keras uvicorn app.main:app --reload

# Build + run the inference Docker image (model is baked in — self-contained)
docker build -t dtcf .
docker run -p 8000:8000 dtcf

# Validate a running service over HTTP (stdlib only — no TF needed locally)
python scripts/validate_endpoint.py --url http://localhost:8000 page.jpg          # single doc
python scripts/validate_endpoint.py --url http://localhost:8000 ./labelled_docs   # batch accuracy (one sub-folder per class)

# Regenerate the REAL evaluation figures from the committed models + local dataset
MPLCONFIGDIR=/private/tmp/dtcf-mpl python scripts/evaluate_actual_figures.py

# Rebuild submission artifacts
python docs/build_submission_docx.py  # regenerates the Phase 2 .docx
```

## Deployment

The image is self-contained (model baked in) and deploys as a container. **Two Docker images, by RAM budget:**
- **`Dockerfile` (full TensorFlow)** — serves `models/resnet50.keras`. Needs ~1 GB RAM. For Easypanel / Render Standard (≥2 GB).
- **`Dockerfile.lite` (slim)** — serves `models/resnet50.tflite` via a LiteRT runtime (`ai-edge-litert`, no TensorFlow). ~250–350 MB RAM, smaller image. **This is what fits Render's free 512 MB plan**, and what `render.yaml` points at.

Both bake the model in and bind to `$PORT` (Render injects it; ENV default 8000 for local/Easypanel). `app/inference.py` auto-selects the backend by file extension (`.tflite` → LiteRT, else Keras) and uses **pure-NumPy ResNet50 preprocessing** (so the lite path needs no TF). The TFLite model is regenerated with `python scripts/convert_to_tflite.py` (needs full TF; uses a SavedModel-export workaround because direct `from_keras_model` conversion hits a Keras 3 MLIR bug). Slim deps live in `app/requirements-lite.txt`.

Deploy targets: **Render** — `render.yaml` Blueprint + `docs/deploy_render.md` (free tier via the slim image). **Easypanel** — `docs/deploy_easypanel.md`.

Serverless platforms (Vercel, Cloudflare Workers/Pages) **cannot** host the backend — even the slim runtime exceeds their function size/runtime limits. They can host a static front-end only.

## Trained models are committed; figures have two generators

Both trained models are checked into the repo (unusual, but intentional for this submission): `models/resnet50.keras` (the inference model) and `notebooks/models/custom_cnn.keras` (the baseline). Local inference and figure regeneration therefore work with no Kaggle run.

There are **two** figure generators and they are not interchangeable:
- `scripts/evaluate_actual_figures.py` — the **real** one. Loads the committed models and the local dataset manifest at `notebooks/german_docs/index.csv`, evaluates the stratified test split, and writes the committed `figures/*.png` + CSVs. Prefer this.
- `docs/build_figures.py` — older/placeholder generator (e.g. synthetic training curves). Do not use it to refresh the committed real figures.

`scripts/train_local_resnet50_quick.py` does a fast, subsampled local ResNet50 train (env knobs `DTCF_QUICK_EPOCHS`, `DTCF_MAX_PER_CLASS`) to produce `models/resnet50.keras` without Kaggle — a convenience, not the canonical training path (that is the notebook).

There is no test suite, linter, or build step configured.

## Architecture notes

**Two dependency sets, intentionally separate.** `requirements.txt` is the heavy training side (full `tensorflow`, `datasets`, `langdetect`, jupyter). `app/requirements.txt` is the inference side — `tensorflow-cpu`, FastAPI, `huggingface_hub`, no training deps. Keep them divergent; don't merge.

**Inference model resolution** (`app/inference.py`, `_model_path`): precedence is `LOCAL_MODEL_PATH` (explicit file / mounted volume) → else download from HF Hub via `HF_MODEL_REPO` + `HF_MODEL_FILE` (default `resnet50.keras`), cached under `HF_HOME`. The **Docker image bakes the model in** (`COPY models/resnet50.keras /models/...`) and sets `LOCAL_MODEL_PATH=/models/resnet50.keras` by default, so the container is self-contained; to use the HF path instead, override with an **empty** `LOCAL_MODEL_PATH` (empty = falsy → falls through to HF). TensorFlow and the model are loaded **lazily** on first request (TF import is ~3s); `app/main.py` startup optionally warms the model (`EAGER_LOAD=1`) but swallows failures so the worker stays up and surfaces the problem via `/health` instead of crashing.

**Preprocessing must match training exactly.** Inference resizes to 224×224 bilinear and applies `tensorflow.keras.applications.resnet50.preprocess_input`. The `CLASS_NAMES` order in `inference.py` is the label index order — it must stay in sync with the notebook's training label order, or predictions map to wrong classes.

**ResNet50 training is two-stage:** train the head with the base frozen, then unfreeze and fine-tune `conv5_*`. This is the project's chosen transfer-learning recipe (per the course tooling preference: TF/Keras + ResNet50 on Kaggle).

**Service endpoints:** `GET /` (upload UI), `POST /predict` (multipart image → `{predicted_class, confidence, uncertainty, recommendation, top_k, all_probabilities, gradcam?}`; `gradcam` is a base64 PNG overlay returned only on the Keras backend — TFLite/lite build omits it since Grad-CAM needs gradients), `GET /health`, `GET /classes`.

## Conventions

- Target environment for the notebook is a **Kaggle GPU notebook**; `/kaggle/working/...` paths and `!pip install` cells are expected and fine.
- No DocLayNet image data is committed; `data/` and `figures/` are reproduced at runtime (only `.gitkeep` is tracked).
- Default to **TensorFlow 2.x / `tf.keras`** for this project — not PyTorch, not Colab-specific paths.
