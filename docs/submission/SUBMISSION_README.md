# Final Submission - German Document Type Classification using CNNs

Author: Ashok Kumar Meena - M.Sc. Software Engineering, UE Potsdam.
Course: Pattern Recognition (SS26), Phase III.

## What is here
This package follows the course Elsevier elsarticle template structure.
- `elsarticle-template-num.tex` : the report (our content) - MAIN file to compile.
- `Bibliography.bib`            : references (numbered, elsarticle-num).
- `Figures/`                    : result figures as PDF (f1, f2, f5-f11).
- `Codes/`                      : GitHub / Kaggle / live-demo links.
- `Datasets/`                   : DocLayNet dataset link + access note.
- `Presentations/`              : presentation outline (build slides + record video).
- `Figures and Tables Source Files/` : where figure/table sources belong.
- elsarticle .bst/.dtx/.ins/doc : Elsevier class machinery (kept; Overleaf has it).

## Compile (Overleaf)
New Project -> Upload Project -> this zip. Compiler: pdfLaTeX. Recompile twice
(so BibTeX resolves citations). Set link sharing to "can edit" and submit the link.

## Remaining student TODOs (template requirements not auto-completable)
- Expand prose to >= 30 pages (7-10 sentences/paragraph).
- Reference list: reach 25-50 with >= 80% after 2021 (verify BibTeX on Google Scholar).
- Literature-review matrix: extend to 12-20 rows.
- Add a SECOND transfer-learning model (e.g. MobileNetV2/EfficientNet) - template
  expects two TL models compared.
- Add SHAP analysis (template requires Grad-CAM AND SHAP).
- Make the Workflow figure and Graphical Abstract in PowerPoint/Canva, export PDF.
- Note: results are the committed all-languages DocLayNet-base run; the strict
  German-filtered re-run is pending.
