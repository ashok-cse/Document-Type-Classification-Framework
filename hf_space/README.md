---
title: DTCF German Document Classifier
emoji: 📄
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Classify German document pages from layout, with Grad-CAM.
---

# DTCF — German Document Type Classifier (Hugging Face Space)

This Space runs the **full TensorFlow** build of the DTCF inference service, so the
result panel includes **live Grad-CAM explanations, an uncertainty estimate, and a
reliability recommendation** — the project's headline advantage (the free 512 MB
hosts can only run the TFLite build, which cannot compute Grad-CAM).

It serves both the web UI and the prediction API from one container:

- `GET /` — upload UI
- `POST /predict` — image → `{predicted_class, confidence, uncertainty, recommendation, top_k, all_probabilities, gradcam}`
- `GET /health`, `GET /classes`, `GET /docs`

The `Dockerfile` clones the project from GitHub at build time (app code + the
trained `resnet50.keras`), so this Space repo stays tiny and needs no git-LFS.

- Source: https://github.com/ashok-cse/PR_SE26_German_Document_Type_Classification_with_CNNs
