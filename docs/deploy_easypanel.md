# Deploying DTCF on Easypanel

This guide walks through the end-to-end deployment of the DTCF inference service
on an Easypanel VPS, including model hosting.

---

## 0. Prerequisites

- An Easypanel instance (free Easypanel install on any VPS — Hetzner, Contabo, DigitalOcean, etc.).
- A VPS with **≥ 2 GB RAM** and **≥ 10 GB disk** (TF + ResNet50 are the dominant footprint).
- A public **GitHub** repository containing this project.
- A **HuggingFace** account (free) for hosting the trained model file.

---

## 1. Train the model on Kaggle

The inference service has nothing to serve until you've trained a model.

1. Upload `notebooks/german_doc_classification.ipynb` to a new Kaggle Notebook with GPU T4 runtime and Internet enabled.
2. Run all cells. Expect 45–75 minutes.
3. When the notebook finishes, the trained ResNet50 is at `/kaggle/working/models/resnet50.keras` (~95 MB).
4. Download `resnet50.keras` to your laptop.

---

## 2. Upload the model to HuggingFace Hub

We do **not** bake the model into the Docker image — it would bloat the image
to ~1.5 GB and force a full rebuild every time you retrain.

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

In **App → Environment** add:

| Variable        | Value                            | Notes                                            |
| --------------- | -------------------------------- | ------------------------------------------------ |
| `HF_MODEL_REPO` | `ashokmarmath/dtcf-resnet50`     | The HF repo you created in step 2.               |
| `HF_MODEL_FILE` | `resnet50.keras`                 | Filename in that repo.                           |
| `HF_TOKEN`      | (only if the HF repo is private) | A *read* token from huggingface.co/settings/tokens. |
| `EAGER_LOAD`    | `1`                              | Pre-load the model on container start.           |

---

## 6. (Optional but recommended) Persistent volume for the model cache

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

- Visit your Easypanel-issued URL → the upload UI.
- Visit `<url>/health` → should return `{"status":"ok","model_loaded":true,...}`.
- Visit `<url>/docs` → FastAPI's interactive Swagger UI.

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
| `/health` shows `"model_loaded": false`                | HF download failed. Check `HF_MODEL_REPO`/`HF_MODEL_FILE`. If private repo, ensure `HF_TOKEN` is set.              |
| First request very slow (~30 s)                        | Model is loading on first call. Set `EAGER_LOAD=1` and a persistent HF cache volume to hide this from users.      |
| 413 Request Entity Too Large on big TIFF / PDF uploads | Easypanel default Nginx body size. Either resize before upload or raise the body-size limit in the proxy config.  |
