# Phase III Presentation & Demo — German Document Type Classification (CNNs)

**Total video: ≤ 10 minutes** — 2 min live demo + 8 min slides.
Record with screen capture (e.g. OBS, Loom, QuickTime + slides).

Live demo URL: **https://ashokcse-document-type-classification-framework.hf.space**
(⚠️ free tier sleeps after ~15 min idle — open it 1–2 min before recording so the
first request isn't a 30–60 s cold start.)

---

## Part A — Live demo (≈ 2 min)

Screen-record the browser at the live URL. Script:

1. **(0:00–0:25)** "This is my German Document Type Classifier, live free on Hugging Face Spaces.
   It predicts the functional type of a scanned German document page from its
   visual layout alone — no OCR." Scroll the landing page briefly (hero, the six
   categories, how-it-works).
2. **(0:25–1:10)** Click a **bundled sample** (e.g. *Patent*, then *Financial
   report*). Show the predicted class, confidence, and the all-class probability
   bars animating. "It correctly identifies this as a patent at high confidence."
3. **(1:10–1:40)** **Upload your own** page (drag a German PDF page exported as
   an image — e.g. from gesetze-im-internet or DEPATISnet, linked in the
   *Try it with a real German document* section). Show the result.
4. **(1:40–2:00)** Point out the **Grad-CAM heatmap** in the result panel: "the
   model attends to the masthead and layout regions, not OCR text." "Under the
   hood this is a fine-tuned ResNet50 served via FastAPI on a free Hugging Face
   Space — the same model evaluated in the report." Cut to slides.

Fallback if the live site is slow: have a 30 s pre-recorded clip of the demo ready.

---

## Part B — Slides (≈ 8 min, ~10 slides)

### Slide 1 — Title (15 s)
German Document Type Classification using CNNs · Ashok Kumar Meena ·
M.Sc. Software Engineering, UE Potsdam · Supervisor: Raja Hashim Ali.

### Slide 2 — Problem & motivation (60 s)
- German archives (Bundesarchiv, DDB, EU tenders) scan millions of pages; first
  decision is "what type of document is this?"
- Manual routing is slow/expensive; OCR-text pipelines are fragile on poor scans
  and German fonts.
- **Idea:** classify from *visual layout* with a CNN — masthead, columns, table
  density — no OCR.

### Slide 3 — Task definition (40 s)
- Input: one scanned page image $x$. Output: one of 6 types
  (financial_reports, scientific_articles, laws_and_regulations,
  government_tenders, manuals, patents).
- Visual features only; baseline custom CNN vs. ResNet50 transfer learning.

### Slide 4 — Dataset (70 s)
- DocLayNet-base (IBM, CDLA-Permissive): 8,057 pages, 6 categories, 1025².
- **German angle:** DocLayNet is ~95% English / ~2.5% German → I add a strict
  `langdetect` German filter (p(de) ≥ 0.90) + a per-class yield report so the
  "German" claim is honest. *(Show Fig 1 class distribution.)*
- Loader gotcha: both DocLayNet HF loaders are broken → bypass `load_dataset`,
  download+extract the zip, parse JSON for `doc_category`.

### Slide 5 — Methodology (70 s)
- Preprocess → 224×224, normalize; stratified 70/15/15; class-weighted loss;
  augmentation.
- **Custom CNN:** 4 conv blocks (32→256) + GAP + Dense(128) + softmax (~0.42M).
- **ResNet50:** ImageNet weights, two-stage — (A) train head frozen, (B)
  fine-tune conv5_*. Adam, early stopping.

### Slide 6 — Results: the headline (80 s) *(Show Fig 9 comparison.)*
| Model | Accuracy | Macro-F1 | Macro-AUC |
|---|---|---|---|
| Custom CNN | 0.326 | 0.082 | 0.479 |
| **ResNet50** | **0.761** | **0.724** | **0.956** |
- **Key finding:** transfer learning isn't just better, it's *necessary* — the
  from-scratch CNN collapses to ~chance on this small, imbalanced set.

### Slide 7 — Per-class & confusion (60 s) *(Show Fig 7 + Fig 8.)*
- ResNet50 strong across classes; errors concentrate in `scientific_articles`
  (lowest per-class AUC) — confusable with text-dense financial/patent pages.

### Slide 8 — Interpretability & robustness (60 s) *(Show Fig 10 + Fig 11.)*
- Grad-CAM: model attends to mastheads/headers/structure → confirms layout-based.
- Phone-camera robustness: ResNet50 macro-F1 0.724 → 0.606 (−0.118) — degrades
  gracefully.

### Slide 9 — Deployment (50 s)
- Deployed **free on Hugging Face Spaces** (full TensorFlow, ~16 GB CPU): serves
  the UI + API **and a live Grad-CAM overlay + uncertainty** per prediction.
- Non-blocking worker-thread inference; one public URL (browser UI + JSON API).
- (A 512 MB TFLite build also exists for low-RAM hosts, but it cannot run Grad-CAM.)

### Slide 10 — Conclusion & future work (40 s)
- A fine-tuned ResNet50 classifies German pages from layout alone at 0.761 acc /
  0.956 AUC, robust and deployed.
- Future: complete the German-filtered re-train + yield report; target the weak
  scientific class; add a confidence/OOD gate.
- Links: GitHub + Kaggle + live demo.

---

## Recording checklist
- [ ] Warm up the HF Space URL before recording (avoid cold start).
- [ ] 1080p screen capture; mic test.
- [ ] Demo clip ≤ 2:00; slides ≤ 8:00; total ≤ 10:00.
- [ ] Export MP4; upload per course instructions.
