# Deploying DTCF on Easypanel

This guide deploys the **whole DTCF project** on an Easypanel VPS. The project is
a single FastAPI app that serves **both the web UI (front-end) and the prediction
API (back-end)** from one container — so this one app is the complete deployment;
there is no separate front-end to host.

> **Recommended build for Easypanel: the root `Dockerfile` (full TensorFlow).**
> Unlike the free-tier TFLite image, this build runs the full Keras model, so the
> result panel includes **live Grad-CAM explanations, an uncertainty estimate,
> and a reliability recommendation** — the project's headline advantage. This
> needs more RAM (see §0).

> **The trained model is baked into the Docker image** (`models/resnet50.keras`,
> committed in the repo). The deployment is self-contained — no HuggingFace
> upload and no model env vars are required. Sections 1–2 and the HF env vars in
> §5 are an **optional** alternative for serving a model from HuggingFace Hub.

---

## 0. Prerequisites

- An Easypanel instance (free Easypanel install on any VPS — Hetzner, Contabo, DigitalOcean, etc.).
- A VPS with **≥ 2 GB RAM** (≥ 3 GB recommended) and **≥ 10 GB disk**. Full
  TensorFlow + ResNet50 use ~1–1.5 GB at load, and Grad-CAM (gradient pass) adds
  headroom — do **not** use a 512 MB box for this build.
- A public **GitHub** repository containing this project (already at
  `github.com/ashok-cse/PR_SE26_German_Document_Type_Classification_with_CNNs`).
- *(Optional)* a HuggingFace account — only if you choose the HF model path (§2).

---

## 1. Train the model on Kaggle *(optional — only to retrain)*

The repo ships with a trained `models/resnet50.keras` already baked into the
image, so you can skip to §3. Follow this section only if you want to retrain.

1. Upload `notebooks/german_doc_classification.ipynb` to a new Kaggle Notebook with GPU T4 runtime and Internet enabled.
2. Run all cells. Expect 45–75 minutes.
3. When the notebook finishes, the trained ResNet50 is at `/kaggle/working/models/resnet50.keras` (~95 MB).
4. Download `resnet50.keras` to your laptop.

---

## 2. Upload the model to HuggingFace Hub *(optional)*

> Skip this for the default self-contained deployment. Do it only if you want to
> serve a model from HF Hub instead of the baked-in one (set the HF env vars in
> §5 **and** an empty `LOCAL_MODEL_PATH` to disable the baked model).

1. Create a new model repo at <https://huggingface.co/new>. Name it something like `ashokmarmath/dtcf-resnet50`. Public is fine; private requires an `HF_TOKEN` env var on Easypanel.
2. From your laptop:

   ```bash
   pip install huggingface_hub
   huggingface-cli login   # paste a write token from https://huggingface.co/settings/tokens
   huggingface-cli upload ashokmarmath/dtcf-resnet50 resnet50.keras
   ```

3. Confirm the file appears at `https://huggingface.co/ashokmarmath/dtcf-resnet50/blob/main/resnet50.keras`.

---

## 3. Push the project to GitHub

If you haven't already:

```bash
cd /path/to/dtcf
git init
git add .
git commit -m "DTCF Phase 2 + inference service"
git branch -M main
git remote add origin https://github.com/<you>/dtcf.git
git push -u origin main
```

The repo's `Dockerfile` is at the root — Easypanel will detect and use it.

---

## 4. Create the Easypanel app

1. In Easypanel, **Create Project → App**.
2. **Source:** GitHub. Connect your account, select the `dtcf` repository, branch `main`.
3. **Build:** Dockerfile (Easypanel autodetects). No build arguments needed.
4. **Domain:** Easypanel auto-generates one, or attach a custom domain. Enable HTTPS (Let's Encrypt).
5. **Port:** internal `8000` (already exposed by the Dockerfile).

---

## 5. Configure environment variables

**For the default (baked-in model) deployment, you need _no_ env vars** — the
image already sets `LOCAL_MODEL_PATH=/models/resnet50.keras` and `EAGER_LOAD=1`.
Optionally set `EAGER_LOAD=1` explicitly (it is the default) to pre-warm.

Add the variables below **only** if you chose the HuggingFace path in §2:

| Variable           | Value                            | Notes                                                                 |
| ------------------ | -------------------------------- | --------------------------------------------------------------------- |
| `LOCAL_MODEL_PATH` | *(empty string)*                 | Required to **disable** the baked-in model so HF is used.             |
| `HF_MODEL_REPO`    | `ashokmarmath/dtcf-resnet50`     | The HF repo you created in step 2.                                    |
| `HF_MODEL_FILE`    | `resnet50.keras`                 | Filename in that repo.                                                 |
| `HF_TOKEN`         | (only if the HF repo is private) | A *read* token from huggingface.co/settings/tokens.                   |
| `EAGER_LOAD`       | `1`                              | Pre-load the model on container start (already the image default).   |

---

## 6. Persistent volume for the model cache *(HF path only)*

> Not needed for the default baked-in model — it's already inside the image.
> This applies only if you switched to the HuggingFace path in §2.

The model is ~95 MB. Without a persistent volume it re-downloads from
HuggingFace on every container restart.

1. **App → Mounts → Add Volume Mount**.
2. **Container path:** `/app/.cache/huggingface`
3. **Size:** 1 GB is plenty.

---

## 7. Deploy

Hit **Deploy**. The first build takes 5–8 minutes (TF is a large pip install).
Subsequent deploys are faster thanks to Docker layer caching.

When the container is up:

- Visit your Easypanel-issued URL → the upload UI (front-end).
- Upload a page or click a sample → the result panel shows the predicted type,
  confidence, **uncertainty**, a **recommendation**, and a **Grad-CAM overlay**
  (this full build serves Grad-CAM live).
- Visit `<url>/health` → should return `{"status":"ok","model_loaded":true,...}`.
- Visit `<url>/docs` → FastAPI's interactive Swagger UI.

### Validate the deployment with real documents

`scripts/validate_endpoint.py` exercises the live service over HTTP (stdlib only —
no TensorFlow needed on your laptop):

```bash
# Liveness + a single page
python scripts/validate_endpoint.py --url https://<your-url> page.jpg

# Batch accuracy: one sub-folder per class, named exactly as the labels
#   docs/financial_reports/*.jpg  docs/patents/*.jpg  ...
python scripts/validate_endpoint.py --url https://<your-url> ./labelled_docs
```

The batch run prints overall accuracy, per-class recall, and a confusion matrix.
Note the model classifies **page images** (PNG/JPG) of the six German DocLayNet
types using layout only — feeding it PDFs, English docs, or other document types
will produce confidently wrong labels (expected, not a bug).

---

## 8. Resource sizing

| Workload                      | Suggested resources                         |
| ----------------------------- | ------------------------------------------- |
| Demo / portfolio              | 1 vCPU, 2 GB RAM, 10 GB disk                |
| Class presentation (5–20 RPS) | 2 vCPU, 4 GB RAM                            |
| Cold-start sensitivity        | Set `EAGER_LOAD=1`, use a persistent volume |

CPU inference latency on a typical Easypanel VPS: **1–3 seconds per image**.

---

## 9. Updating the model

1. Re-train on Kaggle → download new `resnet50.keras`.
2. `huggingface-cli upload ashokmarmath/dtcf-resnet50 resnet50.keras`.
3. In Easypanel, **App → Restart** to drop the in-memory model. The container
   will re-download the latest version from HF on next request.

If you want versioning, upload to a versioned filename (`resnet50-v2.keras`) and
bump the `HF_MODEL_FILE` env var so rollbacks are trivial.

---

## 10. Troubleshooting

| Symptom                                                | Likely cause / fix                                                                                                |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Container restarts continuously                        | Out of memory. Bump VPS RAM to 2 GB+, or switch to `tensorflow-lite-runtime` (see future-work in [proposal.md](../proposal.md)). |
| `/health` shows `"model_loaded": false`                | Baked-in model missing/corrupt — rebuild the image (ensure `models/resnet50.keras` is committed and not blocked by `.dockerignore`). On the HF path: check `HF_MODEL_REPO`/`HF_MODEL_FILE` and `HF_TOKEN` for private repos. |
| First request very slow (~30 s)                        | Model is loading on first call. Set `EAGER_LOAD=1` and a persistent HF cache volume to hide this from users.      |
| 413 Request Entity Too Large on big TIFF / PDF uploads | Easypanel default Nginx body size. Either resize before upload or raise the body-size limit in the proxy config.  |
