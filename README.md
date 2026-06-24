# DTCF — German Document Type Classification Framework

A CNN-based image-classification project that distinguishes **six functional types
of German scanned documents** using the German subset of the **DocLayNet** dataset.

**Course:** Pattern Recognition — Phase 2 (Proposal + Code)
**Author:** Ashok Kumar Meena

---

## Problem

Given a scanned page of a German document, predict its functional type:

`financial_reports` · `scientific_articles` · `laws_and_regulations` · `government_tenders` · `manuals` · `patents`

Only the **visual layout** is used — no OCR text features.

## Approach

- **Dataset:** [DocLayNet (large)](https://huggingface.co/datasets/pierreguillou/DocLayNet-large), filtered for German pages (~2,000) via collection / filename heuristics with a `langdetect` fallback.
- **Custom CNN baseline:** 4 conv blocks, ~1M parameters, trained from scratch.
- **Transfer learning:** ResNet50 (ImageNet pretrained) — two-stage training (head only → fine-tune `conv5_*`).
- **Evaluation:** macro-F1, accuracy, confusion matrices, ROC + per-class AUC, Grad-CAM, robustness under simulated phone-camera degradations.

Full details are in [`proposal.md`](proposal.md).

## Repository layout

```
dtcf/
├── README.md                              ← this file
├── proposal.md                            ← Phase 2 proposal document
├── requirements.txt                       ← training-side Python dependencies
├── Dockerfile                             ← builds the inference service
├── .dockerignore
├── .gitignore
├── notebooks/
│   ├── german_doc_classification.ipynb    ← main Kaggle notebook (training + eval)
│   └── build_notebook.py                  ← regenerates the .ipynb from source
├── app/                                   ← FastAPI inference service
│   ├── main.py                            ← HTTP routes (/, /predict, /health, /classes)
│   ├── inference.py                       ← model load + preprocessing + predict
│   ├── templates/index.html               ← upload UI
│   └── requirements.txt                   ← inference-only Python dependencies
├── figures/                               ← generated figures (produced at runtime)
├── data/                                  ← persisted German subset (produced at runtime)
└── docs/
    └── deploy_easypanel.md                ← step-by-step Easypanel deployment guide
```

## Running on Kaggle (recommended)

1. Create a new Kaggle Notebook with **GPU T4 x1** runtime and **Internet on**.
2. Upload `notebooks/german_doc_classification.ipynb` (or paste its cells into a new notebook).
3. **Run all cells.** The notebook will:
   - install `datasets` and `langdetect`,
   - stream DocLayNet from HuggingFace,
   - persist only the German subset (~2k pages) to `/kaggle/working/german_docs/`,
   - train both models,
   - write all figures and metrics CSVs to `/kaggle/working/figures/`.

Typical end-to-end run time on a T4 GPU: **~45–75 minutes**, dominated by streaming and ResNet50 fine-tuning.

When complete, download the `figures/` and `models/` directories from
`/kaggle/working/` as the run artefacts to attach to the Phase 2 submission.

## Running locally

Tested on Python 3.10+, TensorFlow 2.15+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/german_doc_classification.ipynb
```

GPU strongly recommended; CPU-only runs are technically possible but slow
(several hours).

## Regenerating the notebook

The notebook is generated from `notebooks/build_notebook.py`. If you edit the
builder, regenerate the `.ipynb` with:

```bash
cd notebooks && python build_notebook.py
```

## Dataset license and citation

DocLayNet is released under the **CDLA-Permissive-1.0** license.

> Pfitzmann, B., Auer, C., Dolfi, M., Nassar, A. S., & Staar, P. (2022).
> DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis.
> *KDD 2022.* arXiv:2206.01062.

The German subset used here is derived from DocLayNet under the same license. No
DocLayNet image data is committed to this repository; the notebook reproduces the
subset by streaming from HuggingFace.

## Results

Final numbers will be populated after the Phase 2 run; see `figures/t1_model_comparison.csv`
and `figures/f8_robustness.csv` after the notebook completes.

## Deploying the inference service

Once the model is trained on Kaggle, the FastAPI service in `app/` can be deployed
on any Docker host. The included `Dockerfile` ships a CPU-only TensorFlow image
that loads the trained model from HuggingFace Hub at startup.

Full Easypanel walkthrough: [`docs/deploy_easypanel.md`](docs/deploy_easypanel.md).

Local smoke test (after you have a `resnet50.keras` on disk):

```bash
docker build -t dtcf .
docker run -p 8000:8000 -e LOCAL_MODEL_PATH=/models/resnet50.keras \
  -v $(pwd)/models:/models dtcf
# then open http://localhost:8000
```

Endpoints:
- `GET  /` — upload UI
- `POST /predict` — multipart image upload, returns `{predicted_class, confidence, top_k, all_probabilities}`
- `GET  /health` — liveness + model status
- `GET  /classes` — class list

## Submission checklist

- [x] Proposal document — `proposal.md`
- [x] Code — `notebooks/german_doc_classification.ipynb`
- [x] README with run instructions — this file
- [x] Public GitHub repository — https://github.com/ashok-cse/Document-Type-Classification-Framework
- [ ] Kaggle Notebook link (add once notebook is uploaded and made public)
- [ ] Generated figures attached to the Teams submission
