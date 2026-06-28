# Deploying DTCF on Render (free tier)

Render builds this repo's Docker image and runs it as a container. The default
Render deploy uses the **slim image** (`Dockerfile.lite`): it serves a **TFLite**
model with a lightweight LiteRT runtime instead of full TensorFlow, so it fits
Render's **free 512 MB plan**. The model is committed to the repo and **baked into
the image**, so there's no HuggingFace setup, storage, or env vars to invent —
push and deploy.

> **Why slim?** Full TensorFlow + ResNet50 needs ~1 GB RAM and OOM-kills on a
> 512 MB plan. The `.tflite` model + LiteRT runtime use ~250–350 MB and a much
> smaller image. Predictions are identical to the Keras model (verified to 1e-6).

---

## 0. Prerequisites

- A **Render** account (<https://render.com>) — free.
- This project pushed to **GitHub** (or GitLab) so Render can build it.
- `models/resnet50.tflite` committed in the repo (it is). Regenerate it with
  `python scripts/convert_to_tflite.py` if you retrain.

---

## Option A — Blueprint (recommended, one click)

The repo ships `render.yaml`, which pins the slim Dockerfile and the free plan.

1. In Render: **New +  →  Blueprint**.
2. Connect the GitHub repo and select it.
3. Render reads `render.yaml` and proposes a web service `dtcf` on the **free**
   plan with `/health` as the health check. Confirm and **Apply**.
4. First build ~3–5 min (slim image; no TensorFlow to install). The ~92 MB model
   is copied into the image.

Skip to **Verify**.

---

## Option B — Manual web service (no Blueprint)

1. **New +  →  Web Service**, connect the repo.
2. **Runtime:** Docker. Set **Dockerfile Path** to `Dockerfile.lite`.
3. **Plan:** **Free**.
4. **Health Check Path:** `/health`.
5. **Environment variables:** `EAGER_LOAD = 1`. (No model vars — it's baked in.)
6. **Create Web Service.**

> Render injects a `PORT` env var (default 10000). The image's start command
> binds to `$PORT` automatically — no port config needed.

---

## Verify

When the service is **Live**, open the Render URL (`https://dtcf-XXXX.onrender.com`):

- `/`        → upload UI.
- `/health`  → `{"status":"ok","model_loaded":true,...}`.
- `/docs`    → FastAPI Swagger UI.

Scripted validation (stdlib only — no TensorFlow needed locally):

```bash
python scripts/validate_endpoint.py --url https://dtcf-XXXX.onrender.com page.jpg
python scripts/validate_endpoint.py --url https://dtcf-XXXX.onrender.com ./labelled_docs
```

---

## Free-tier limits to know (for a showcase)

- **Spins down when idle.** After ~15 min of no traffic the free instance
  sleeps; the next request pays a cold start (container boot + model load,
  ~20–40 s). Hit the URL a minute before presenting.
- **Monthly hours.** Free web services share a monthly instance-hour budget;
  fine for a demo, not for always-on production.
- **Throughput.** Single worker, CPU inference ≈ 1–3 s/image; requests queue
  under concurrency.

### Need more (always-on / higher throughput)?

Switch to the full-TensorFlow image on a paid plan: in `render.yaml` set
`dockerfilePath: ./Dockerfile` and `plan: standard` (2 GB). Same app, same API.

---

## Updating the model

1. Retrain → produce a new `models/resnet50.keras`.
2. `python scripts/convert_to_tflite.py` → refreshes `models/resnet50.tflite`.
3. Commit both and push — `autoDeploy` rebuilds the image.

## Inputs

The model classifies **page images** (PNG/JPG) of six German DocLayNet types
using layout only; PDFs / off-distribution images yield confident but wrong
labels.
