# Deploying DTCF on Render

Render builds this repo's `Dockerfile` directly and runs it as a container. The
trained model (`models/resnet50.keras`, ~97 MB) is **committed to the repo and
baked into the image**, so there is no HuggingFace setup, external storage, or
extra configuration — push and deploy.

---

## 0. Prerequisites

- A **Render** account (<https://render.com>) — free to create.
- This project pushed to a **GitHub** (or GitLab) repo Render can access.
- That's it. No model hosting, no env vars to invent.

> **Plan / RAM note (important).** TensorFlow + ResNet50 needs **> 512 MB** of
> RAM at load time. The Free and Starter plans (512 MB) will **OOM-kill** the
> container during model load. Use **Standard (2 GB)** or larger. The committed
> `render.yaml` already pins `plan: standard`.

---

## Option A — Blueprint (recommended, one click)

The repo ships a `render.yaml` Blueprint, so Render configures everything for you.

1. In Render: **New +  →  Blueprint**.
2. Connect the GitHub repo and select it.
3. Render reads `render.yaml` and proposes a web service named `dtcf` on the
   `standard` plan with `/health` as the health check. Confirm and **Apply**.
4. First build takes ~6–10 min (TensorFlow is a large pip install; the ~97 MB
   model is copied into the image). Subsequent deploys reuse Docker layer cache.

That's the whole deploy. Skip to **Verify**.

---

## Option B — Manual web service (no Blueprint)

1. **New +  →  Web Service**, connect the repo.
2. **Runtime:** Docker (Render autodetects the `Dockerfile`).
3. **Plan:** **Standard** (2 GB) or larger — *not* Free/Starter (see RAM note).
4. **Health Check Path:** `/health`.
5. **Environment variables:**
   - `EAGER_LOAD = 1` (pre-load the model on startup)
   - `TF_CPP_MIN_LOG_LEVEL = 3` (quieten TF logs)
   - No model variables needed — the model is baked into the image.
6. **Create Web Service.**

> Render injects a `PORT` env var (default 10000). The Dockerfile's start
> command binds to `$PORT` automatically, so no port config is required.

---

## Verify

When the service is **Live**, open the Render-issued URL (`https://dtcf-XXXX.onrender.com`):

- `/`            → the upload UI.
- `/health`      → `{"status":"ok","model_loaded":true,...}`.
- `/docs`        → FastAPI Swagger UI.

Batch / scripted validation (stdlib only — no TensorFlow needed locally):

```bash
python scripts/validate_endpoint.py --url https://dtcf-XXXX.onrender.com page.jpg
python scripts/validate_endpoint.py --url https://dtcf-XXXX.onrender.com ./labelled_docs
```

---

## Notes for a showcase

- **Cold starts.** On plans that sleep when idle, the first request after a
  sleep pays the container start + TF load (~30–60 s). For a live demo, hit the
  URL a minute before presenting, or use an always-on plan.
- **Throughput.** Single worker, CPU inference ≈ 1–3 s/image; requests queue
  under concurrency. Fine for a demo; scale the plan or instances for more.
- **Updating the model.** Commit a new `models/resnet50.keras` and push —
  `autoDeploy` rebuilds the image with the new weights.
- **Inputs.** The model classifies **page images** (PNG/JPG) of six German
  DocLayNet types using layout only; PDFs / off-distribution images yield
  confident but wrong labels.
