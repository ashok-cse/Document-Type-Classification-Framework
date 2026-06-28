# Phase III Submission Package

Phase III = **Final report + presentation** (40 pts). Two things are due **28 June 2026**:

1. A **recorded presentation video** (≤ 10 min): 2 min frontend demo + 8 min slides.
2. A **text file containing the EDITABLE OVERLEAF LINK** to the report (in the
   provided template). The link must be submitted by the deadline; you can keep
   editing the same Overleaf project afterward.

Grading timeline: initial (ungraded) feedback **04 July**; final grading on the
**same Overleaf link** on **11 July**.

---

## 1. The report → Overleaf (editable link)

The report is written and ready in **`docs/report/`**:
- `main.tex` — the full report (title page, all sections, real results, figures, refs).
- `figures/` — the 9 figures referenced by `main.tex`.

### Create the editable Overleaf link
1. Zip the report folder so it's self-contained:
   ```bash
   cd docs/report && zip -r ../dtcf_report_overleaf.zip main.tex figures
   ```
2. Go to Overleaf → **New Project → Upload Project** → upload `dtcf_report_overleaf.zip`.
3. Open the project; **Menu → Compiler: pdfLaTeX**; click **Recompile** (it
   compiles with stock Overleaf packages — no extra setup).
4. **Share → Turn on link sharing → "Anyone with this link can edit"** → copy the
   **edit** link (not the read-only one).
5. Put that URL in a plain text file (e.g. `overleaf_link.txt`) and submit it.

> If the instructor provided a *specific* Overleaf template, create the project
> from that template instead and paste the section bodies from `main.tex` into it
> (the content maps 1:1 to the template's Contents page).

### ⚠️ Accuracy note before you submit
The results in the report are the **real committed metrics** from the
DocLayNet-base run (ResNet50: 0.761 acc / 0.724 macro-F1 / 0.956 AUC; the custom
CNN collapses). These are honest and defensible. The **strict German-language
filter is implemented but not yet re-run on Kaggle** — the report says so
explicitly. After you run the filtered notebook, regenerate the figures and
update Table 1 / the numbers in the same Overleaf project (allowed until 11 July).

---

## 2. The video

Use **`docs/presentation_outline.md`** — it has the 2-min demo script and the
~10-slide, 8-min deck with per-slide speaker notes and which figure to show.

- **Demo:** record the live site **https://document-type-classification-framework.onrender.com**
  (warm it up first — free tier cold-starts in 30–60 s).
- **Slides:** build ~10 slides from the outline (PowerPoint/Google Slides/Beamer).
- Export MP4 ≤ 10 min and submit per the course instructions.

---

## 3. Links to include in the submission

| Item | URL |
|---|---|
| Live demo (frontend) | https://document-type-classification-framework.onrender.com |
| GitHub repository | https://github.com/ashok-cse/Document-Type-Classification-Framework |
| Kaggle notebook | https://www.kaggle.com/code/ashokkrcse/german-document-type-classification-with-cnns |
| Overleaf report (edit link) | *(create per step 1, then paste here)* |

---

## 4. Checklist

- [ ] `main.tex` compiles on Overleaf (pdfLaTeX) with all figures.
- [ ] Overleaf link set to **edit** sharing; URL saved in a text file.
- [ ] Video recorded (≤ 10 min): demo + slides; exported MP4.
- [ ] Overleaf link + video submitted by **28 June 2026**.
- [ ] (After Kaggle filtered run) update results/figures in the same Overleaf project before **11 July**.

---

## Deliverable status (what's done in this repo)

- [x] **Frontend demo** — public, with bundled one-click samples + example sources.
- [x] **Non-blocking inference worker** — `run_in_threadpool` in `app/main.py`.
- [x] **Free-tier deployment** — TFLite + LiteRT, `Dockerfile.lite`, `render.yaml`.
- [x] **Report (LaTeX)** — `docs/report/main.tex` + figures, real metrics.
- [x] **Presentation + demo script** — `docs/presentation_outline.md`.
- [ ] **Overleaf edit link** — *you* must create it (needs your Overleaf account).
- [ ] **Recorded video** — *you* must record it.
- [ ] **German-filtered Kaggle re-run** — optional before 11 July to finalize numbers.
