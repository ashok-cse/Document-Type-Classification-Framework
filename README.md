# DTCF — German Document Type Classification Framework

A CNN-based image-classification project that distinguishes **six functional
document types** using the DocLayNet-base page-image dataset.

**Course:** Pattern Recognition — Phase 2 (Proposal + Code)
**Author:** Ashok Kumar Meena
**Live demo:** <https://ashokcse-document-type-classification-framework.hf.space> — upload a German page and get the predicted type, confidence, uncertainty, and a **live Grad-CAM** explanation (full-TensorFlow build on a free Hugging Face Space).

---

## Problem

Given a scanned document page, predict its functional type:

`financial_reports` · `scientific_articles` · `laws_and_regulations` · `government_tenders` · `manuals` · `patents`

Only the **visual layout** is used — no OCR text features.

## Approach

- **Dataset:** [DocLayNet-base](https://huggingface.co/datasets/pierreguillou/DocLayNet-base), six document categories, **8,057 page images** persisted locally under `notebooks/german_docs/`.
- **Custom CNN baseline:** 4 conv blocks, ~1M parameters, trained from scratch.
- **Transfer learning:** ResNet50 (ImageNet pretrained) — two-stage training (head only → fine-tune `conv5_*`).
- **Evaluation:** macro-F1, accuracy, confusion matrices, ROC + per-class AUC, Grad-CAM, robustness under simulated phone-camera degradations.

Full details are in [`proposal.md`](proposal.md).

## Dataset

The current local dataset contains **8,057 document images** across six classes:

| Class | Images |
|---|---:|
| `financial_reports` | 2,627 |
| `manuals` | 1,679 |
| `scientific_articles` | 1,396 |
| `laws_and_regulations` | 1,287 |
| `patents` | 647 |
| `government_tenders` | 421 |

The notebook uses a stratified **70 / 15 / 15** train/validation/test split over
these images.

### German-language filtering

DocLayNet is ~95% English and only ~2.5% German. Because this project targets
German documents, the notebook applies a **strict German-language filter** in
section 1: each page's annotation text is run through `langdetect`, and a page is
kept only if it is **confidently German** (`p(de) ≥ GERMAN_MIN_PROB`, default
0.90, with at least `GERMAN_MIN_TEXT` characters). Knobs: `STRICT_GERMAN`,
`GERMAN_MIN_PROB`, `GERMAN_MIN_TEXT` (set `STRICT_GERMAN=False` for the
all-languages baseline). The notebook emits a **per-class yield report** — pages
scanned (all languages) vs. German pages kept — to `figures/t0_german_yield.csv`
and a grouped-bar figure `figures/f0_german_yield.png`, making the size of the
genuinely-German subset explicit and surfacing under-represented classes.

> The **8,057-image counts above are the pre-filter** DocLayNet-base
> distribution; the committed model and figures predate the filter. Re-running
> the notebook with `STRICT_GERMAN=True` trains on the (smaller) genuinely-German
> subset. Layout is largely language-independent, so the method is unchanged —
> only which pages train the model.

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
├── figures/                               ← actual figures generated from saved models
├── data/                                  ← optional runtime data directory
└── docs/
    └── deploy_easypanel.md                ← step-by-step Easypanel deployment guide
```

## Running on Kaggle (recommended)

1. Create a new Kaggle Notebook with **GPU T4 x1** runtime and **Internet on**.
2. Upload `notebooks/german_doc_classification.ipynb` (or paste its cells into a new notebook).
3. **Run all cells.** The notebook will:
   - download `dataset_base.zip` from HuggingFace,
   - extract and persist the six-class image dataset to `/kaggle/working/german_docs/`,
   - train both models,
   - write all figures and metrics CSVs to `/kaggle/working/figures/`.

Typical end-to-end run time on a T4 GPU: **~45–75 minutes**, dominated by dataset
download/extraction and ResNet50 fine-tuning.

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

## Regenerating actual figures

The committed figures are generated from the local 8,057-image dataset and saved
models:

```bash
MPLCONFIGDIR=/private/tmp/dtcf-mpl .venv311/bin/python scripts/evaluate_actual_figures.py
```

This writes actual test-set artifacts to `figures/`, including confusion
matrices, ROC curves, model comparison, Grad-CAM overlays, and robustness
results. The old synthetic training-curve placeholders are kept separately under
`figures/placeholders/`; real F3/F4 training curves require rerunning training
because the original Keras history objects are not saved in the model files.

## Dataset license and citation

DocLayNet is released under the **CDLA-Permissive-1.0** license.

> Pfitzmann, B., Auer, C., Dolfi, M., Nassar, A. S., & Staar, P. (2022).
> DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis.
> *KDD 2022.* arXiv:2206.01062.

The image dataset used here is derived from DocLayNet under the same license. No
DocLayNet image data is committed to this repository; the notebook reproduces the
local dataset by downloading and extracting DocLayNet-base from HuggingFace.

## Results

The root `figures/` directory now contains actual figures generated from the
local dataset and saved models. The evaluation uses the stratified test split of
**1,209 images** from the full **8,057-image** dataset.

| Model | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|
| Custom CNN | 0.326 | 0.082 | 0.479 |
| ResNet50 | 0.761 | 0.724 | 0.956 |

Actual generated artifacts:

- `figures/f1_class_distribution.png`
- `figures/f2_sample_per_class.png`
- `figures/f5_custom_cnn_confusion.png`
- `figures/f6_custom_cnn_roc.png`
- `figures/f7_resnet50_confusion.png`
- `figures/f8_resnet50_roc.png`
- `figures/f9_model_comparison.png`
- `figures/f10_gradcam.png`
- `figures/f11_robustness.png`
- `figures/t1_model_comparison.csv`
- `figures/f11_robustness.csv`

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
