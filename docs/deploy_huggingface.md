# Deploying DTCF on Hugging Face Spaces (free, with Grad-CAM)

Hugging Face Spaces' **free CPU tier has ~16 GB RAM**, enough to run the **full
TensorFlow** build — so the live result panel includes **Grad-CAM, uncertainty,
and a recommendation** (the 512 MB free tiers can only run the TFLite build,
which can't compute Grad-CAM). The Space sleeps after inactivity and wakes on the
next visit. No credit card required.

The ready-to-use Space files are in **`hf_space/`**:
- `README.md` — Space metadata (`sdk: docker`, `app_port: 8000`).
- `Dockerfile` — clones this GitHub repo at build time (app + model), so the
  Space repo needs **no large files and no git-LFS**.

---

## Option A — Web UI (no local git, easiest)

1. Go to <https://huggingface.co/new-space>.
2. **Owner/Name:** e.g. `your-username/dtcf`. **License:** MIT.
3. **SDK:** choose **Docker** → **Blank**. **Hardware:** **CPU basic (free)**.
4. Create the Space, then open the **Files** tab → **Add file → Create new file**:
   - Create **`README.md`** and paste the contents of `hf_space/README.md`.
   - Create **`Dockerfile`** and paste the contents of `hf_space/Dockerfile`.
5. The Space builds automatically (first build ~6–10 min: TF install + clone).
6. When it shows **Running**, open the Space URL — upload a page or click a
   sample and you'll see the prediction **plus the Grad-CAM overlay**.

---

## Option B — Git push

```bash
# 1) Create the Space (Docker SDK) at https://huggingface.co/new-space, then:
git clone https://huggingface.co/spaces/<your-username>/dtcf hf-dtcf-space
cp hf_space/README.md hf_space/Dockerfile hf-dtcf-space/
cd hf-dtcf-space
git add README.md Dockerfile
git commit -m "DTCF full-TF Space with Grad-CAM"
git push                      # paste an HF access token if prompted
```

The Space builds the Dockerfile and goes live at
`https://<your-username>-dtcf.hf.space`.

---

## Verify

```bash
python scripts/validate_endpoint.py --url https://<your-username>-dtcf.hf.space page.jpg
```
- `GET /health` → `{"status":"ok","model_loaded":true,...}`
- `POST /predict` response includes a non-empty **`gradcam`** field (base64 PNG)
  and an **`uncertainty`** value — confirming the full build with live Grad-CAM.

---

## Notes

- **Updating:** the Dockerfile clones GitHub at build time. After you push new
  commits to GitHub, trigger **Settings → Factory rebuild** on the Space (or bump
  `CACHE_BUST` in the Dockerfile) to pick them up.
- **Cold start:** a sleeping Space takes ~30–60 s to wake and load TF on the first
  request — open it a minute before a live demo.
- **Why not just push this whole repo to the Space?** The repo commits the 96 MB
  `resnet50.keras` as a normal git blob; Hugging Face rejects large non-LFS files
  on push. The clone-at-build approach sidesteps that entirely.
- **Private/alternative model source:** if you prefer, upload the model to an HF
  model repo and set Space secrets `LOCAL_MODEL_PATH=` (empty) + `HF_MODEL_REPO`,
  `HF_MODEL_FILE` instead of cloning it.
