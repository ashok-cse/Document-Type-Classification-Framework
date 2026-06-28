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

The report follows the **course Elsevier `elsarticle` template**
(SS26_Machine_Learning_Report_Template) and is ready in **`docs/report/`**:
- `main.tex` — the full report in `elsarticle` format (frontmatter with
  highlights/keywords, Introduction → Literature Review → Methodology →
  Results-by-RQ → Discussion → Conclusion → Declarations), with our real results,
  figures, equations, hyperparameter table, and a literature-review matrix.
- `Bibliography.bib` — real starter references (cited via `\cite`, `elsarticle-num`).
- `elsarticle-num.bst` — bibliography style (Overleaf also has it built in).
- `figures/` — the 9 result figures referenced by `main.tex`.

### Create the editable Overleaf link
1. Zip the report folder so it's self-contained:
   ```bash
   cd docs/report && zip -r ../dtcf_report_overleaf.zip main.tex Bibliography.bib elsarticle-num.bst figures
   ```
2. Go to Overleaf → **New Project → Upload Project** → upload `dtcf_report_overleaf.zip`.
3. **Menu → Compiler: pdfLaTeX**; **Recompile** (run twice so BibTeX resolves
   citations). `elsarticle` is built into Overleaf — no extra setup.
4. **Share → Turn on link sharing → "Anyone with this link can edit"** → copy the
   **edit** link (not read-only).
5. Put that URL in a plain text file (e.g. `overleaf_link.txt`) and submit it.

### ⚠️ Student TODOs before the 11 July grading (the template demands more than is auto-fillable)
The structure and our content are in place, but the template has hard
requirements only you can finish:
- [ ] **Length:** the template asks for **≥ 30 pages** — expand each section's
      paragraphs to the required 7–10 sentences.
- [ ] **References:** grow `Bibliography.bib` to **25–50** entries with **≥ 80 %
      after 2021**, journals/conferences preferred (IEEE/Springer/ACM/Elsevier/MDPI),
      BibTeX from Google Scholar; avoid arXiv/websites.
- [ ] **Literature matrix:** expand `tab:literature_matrix` to **12–20** real studies.
- [ ] **SHAP:** the template's Explainable-AI section requires **Grad-CAM *and*
      SHAP**. We have Grad-CAM only — either add a SHAP analysis or keep it
      explicitly scoped as future work (already flagged in the text).
- [ ] **Workflow + graphical-abstract figures:** make these in PowerPoint/Canva,
      export as **PDF**, and drop into `figures/` (placeholders are commented in `main.tex`).
- [ ] **Figures as PDF:** the template prefers vector PDFs at ≥300 ppi; our result
      charts are PNG (fine to start, regenerate as PDF if the grader insists).

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

- **Demo:** record the live site **https://ashokcse-document-type-classification-framework.hf.space**
  (warm it up first — free tier cold-starts in 30–60 s).
- **Slides:** build ~10 slides from the outline (PowerPoint/Google Slides/Beamer).
- Export MP4 ≤ 10 min and submit per the course instructions.

---

## 3. Links to include in the submission

| Item | URL |
|---|---|
| Live demo (frontend) | https://ashokcse-document-type-classification-framework.hf.space |
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
- [x] **Free deployment (with Grad-CAM)** — Hugging Face Space (`hf_space/`, full TensorFlow).
- [x] **Report (LaTeX, elsarticle template)** — `docs/report/` (main.tex + Bibliography.bib + bst + figures), real metrics; needs student expansion (see TODOs above).
- [x] **Presentation + demo script** — `docs/presentation_outline.md`.
- [ ] **Overleaf edit link** — *you* must create it (needs your Overleaf account).
- [ ] **Recorded video** — *you* must record it.
- [ ] **German-filtered Kaggle re-run** — optional before 11 July to finalize numbers.
