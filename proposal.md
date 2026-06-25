# A CNN-Based Framework for German Document Type Classification

**Course:** Pattern Recognition — Phase 2 Proposal
**Student:** Ashok Kumar Meena
**Date:** June 2026

---

## 1. Project Title

**Deutsche Dokumenten-Typ-Klassifikation (DTCF):** A Convolutional Neural Network Framework for Classifying German Scanned Documents into Functional Categories.

## 2. Background and Motivation

Digitisation of administrative, legal, scientific, and financial archives is a major workflow in German-speaking institutions (e.g. Deutsche Digitale Bibliothek, Bundesarchiv, EU tender portals, German regulatory bodies). Once a document is scanned, the very first downstream decision is almost always *"what kind of document is this?"* — routing an invoice differs from routing a patent, which differs from routing a regulatory notice. Performing this routing manually is slow and error-prone.

Most published research on document image classification has focused on English-language benchmarks such as RVL-CDIP and Tobacco-3482. German documents pose a *visually* distinct problem: longer compound words alter line-wrapping patterns, different masthead conventions in official documents (DIN-A4 layouts, Deutsche-Behörden-Stil headers), and different paragraph density in legal *Verordnungen* versus scientific *Fachartikel*. A model trained purely on English documents may not transfer cleanly to these layouts.

This project applies the pattern-recognition concepts of supervised learning, convolutional neural networks, transfer learning, and model evaluation to build and rigorously assess a CNN-based classifier specifically on German document images.

## 3. Problem Statement

Given a single-page scanned German document image **x**, predict its **document type** **y ∈ {financial_reports, scientific_articles, laws_and_regulations, government_tenders, manuals, patents}**.

The classifier is constrained to use only the **visual features** of the page image — no OCR text features are used. The challenge is therefore to learn the *layout signatures* (heading styles, column structures, figure placement, whitespace patterns, table density) that distinguish these document categories.

## 4. Selected Dataset

**Name:** DocLayNet-base six-class page-image dataset
**Source:** IBM Research, released under CDLA-Permissive-1.0
**HuggingFace link:** <https://huggingface.co/datasets/pierreguillou/DocLayNet-base>
**Original paper:** Pfitzmann et al., *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis*, KDD 2022. arXiv:2206.01062.

### 4.1 Why DocLayNet

DocLayNet is a large-scale, openly licensed document image dataset that (a) provides high-quality scanned page images at 1025×1025 resolution, (b) includes document-source labels (`doc_category`) covering six functional categories, and (c) contains real-world documents from financial reports, scientific publications, legal sources, tenders, manuals, and patents.

### 4.2 Dataset Description

| Property                  | Value                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| Local dataset size        | 8,057 page images                                                                                         |
| Image resolution          | 1025 × 1025 px source pages, resized to 224 × 224 for model input                                          |
| Image type                | RGB rasterised PDF pages of real-world scanned/born-digital documents                                     |
| Number of classes         | 6 functional document types                                                                               |
| Class labels              | `financial_reports`, `scientific_articles`, `laws_and_regulations`, `government_tenders`, `manuals`, `patents` |
| Split                     | Stratified 70 / 15 / 15 train/validation/test split                                                       |
| License                   | CDLA-Permissive-1.0 (research and educational use)                                                        |

### 4.3 Class Distribution

The local extracted dataset contains **8,057 pages**. The class distribution is:

| Class                  | Pages |
| ---------------------- | ----: |
| `financial_reports`    | 2,627 |
| `manuals`              | 1,679 |
| `scientific_articles`  | 1,396 |
| `laws_and_regulations` | 1,287 |
| `patents`              |   647 |
| `government_tenders`   |   421 |

The distribution is **imbalanced**; mitigation will use class-weighted loss and strong augmentation for minority classes.

### 4.4 Limitations

- The local six-class dataset contains 8,057 images and remains imbalanced. Class weighting, augmentation, and transfer learning are required.
- Handwritten German documents and real phone-camera captures are **not available** in any public German document-type-classification dataset; this is acknowledged as a limitation. Robustness to phone-camera-like conditions is evaluated using *synthesised* augmentations (perspective warp, brightness shifts, JPEG compression, motion blur). Real handwritten and phone-camera evaluation is left as future work.

## 5. Research Questions

**RQ1.** Can a CNN classifier trained only on visual page layout distinguish the six functional document categories of DocLayNet with strong macro-F1?

**RQ2.** How large is the gap between a custom-built CNN trained from scratch and a ResNet50 transfer-learning model (ImageNet-pretrained, fine-tuned)? Does pretraining on natural images transfer usefully to document layout features?

**RQ3.** How robust is the best model to phone-camera-style degradations (perspective warp, JPEG compression, blur, brightness shift) that simulate the visual gap between flatbed scans and mobile-phone captures?

**RQ4.** Which document classes are most often confused with one another, and what visual evidence (via Grad-CAM) does the model use to make its decisions?

## 6. Expected Results

- **Custom CNN** (≈ 1 M parameters, 4 convolutional blocks): expected test macro-F1 in **0.65 – 0.75** range. Strong on visually distinctive classes (`patents` headers, `government_tenders` form layouts) but weaker on confusable pairs (`financial_reports` vs. `scientific_articles`).
- **ResNet50 transfer learning** (ImageNet-pretrained, last block fine-tuned): expected test macro-F1 in **0.82 – 0.90**. Better representations of low-level page structure (column boundaries, table edges) inherited from ImageNet.
- **Robustness eval:** macro-F1 drop of **5 – 12 percentage points** under the phone-camera simulation, with `manuals` and `scientific_articles` most affected.
- **Grad-CAM** is expected to highlight masthead/header regions for `laws_and_regulations` and `government_tenders`, and the tabular financial-disclosure regions for `financial_reports`.

## 7. Proposed Methodology

### 7.1 Pipeline Overview

```
Raw German pages ──► resize (224×224) ──► normalize ──► augment ──► CNN ──► softmax(6)
                                                          │
                                                          ├── train/val/test = 70/15/15 (stratified)
                                                          └── class-weighted cross-entropy
```

### 7.2 Pre-processing

- Resize the native 1025×1025 page images to **224×224** (compatible with ResNet50 input and tractable on a Kaggle GPU). A side experiment will compare 224 vs. 384 to test whether higher resolution helps preserve layout fidelity.
- Per-channel normalisation: zero-mean / unit-variance using ImageNet statistics for the transfer model, and [0,1] scaling for the custom CNN.
- Stratified train/val/test split: **70 / 15 / 15**, seeded for reproducibility.

### 7.3 Data Augmentation

Two augmentation regimes are used:

| Regime               | Transformations                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Standard (train)** | Random rotation ±5°, horizontal shift ±5 %, zoom ±10 %, brightness ±15 %                                              |
| **Phone-camera sim** | Random perspective warp (±15 % corners), JPEG compression Q ∈ [40, 80], Gaussian blur σ ∈ [0, 1.5], brightness ±25 % |

The phone-camera regime is applied only at evaluation time to build a robustness test set.

### 7.4 Custom CNN Architecture

```
Input (224×224×3)
 → Conv(32,3×3) + BN + ReLU → MaxPool(2)
 → Conv(64,3×3) + BN + ReLU → MaxPool(2)
 → Conv(128,3×3) + BN + ReLU → MaxPool(2)
 → Conv(256,3×3) + BN + ReLU → MaxPool(2)
 → GlobalAveragePool
 → Dense(128) + Dropout(0.5)
 → Dense(6) + softmax
```

≈ 1 M trainable parameters. Optimiser: Adam, lr = 1e-3, ReduceLROnPlateau. Loss: categorical cross-entropy with class weights = `n_total / (n_classes · n_c)`.

### 7.5 Transfer Learning Model

- Base: `tf.keras.applications.ResNet50(weights='imagenet', include_top=False)`
- Head: GlobalAveragePool → Dense(256) + Dropout(0.4) → Dense(6) + softmax
- **Stage A:** freeze base, train head, 10 epochs, lr = 1e-3
- **Stage B:** unfreeze last residual block (`conv5_*`), fine-tune 20 epochs, lr = 1e-5

### 7.6 Evaluation Metrics

- **Primary:** macro-averaged precision, recall, F1 (handles class imbalance correctly).
- **Secondary:** overall accuracy, per-class F1, ROC curves with macro and per-class AUC (one-vs-rest).
- **Diagnostic:** confusion matrix (counts and row-normalised), training/validation accuracy and loss curves.
- **Interpretability:** Grad-CAM heatmaps on the ResNet50 final convolutional layer.
- **Robustness:** the above metrics re-computed on the phone-camera-simulated test set; report deltas.

## 8. Expected Figures and Tables

| # | Figure / Table                                                              | Purpose                                          |
| - | --------------------------------------------------------------------------- | ------------------------------------------------ |
| F1 | Class-distribution bar chart of the local dataset                          | Show dataset imbalance                           |
| F2 | Grid of sample images (one per class)                                      | Qualitative description of the data              |
| F3 | Training/validation accuracy and loss curves — custom CNN                  | Convergence diagnosis                            |
| F4 | Training/validation accuracy and loss curves — ResNet50                    | Convergence diagnosis                            |
| F5 | Confusion matrices (counts + row-normalised) for both models               | Per-class error analysis                         |
| F6 | ROC curves with macro/per-class AUC                                        | Threshold-independent performance                |
| F7 | Grad-CAM overlays on 6 example pages                                       | Interpretability — visual evidence used by model |
| F8 | Robustness comparison: clean vs. phone-camera-simulated macro-F1           | RQ3 answer                                       |
| T1 | Model-comparison table: params, train time, accuracy, macro-F1, macro-AUC | Quantitative head-to-head                        |
| T2 | Per-class classification report for the best model                         | Reporting standard                               |

## 9. Risks and Mitigations

| Risk                                                                                        | Mitigation                                                                                                                   |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| The 8,057-image dataset is imbalanced and challenging for a custom CNN from scratch          | Heavy augmentation, class weighting, early stopping, and rely on the ResNet50 transfer model as the primary reported result |
| HuggingFace `DocLayNet-large` is 37.6 GB — exceeds Kaggle disk quota if downloaded in full | Use DocLayNet-base via `dataset_base.zip`, then persist only the extracted six-class page-image dataset                     |
| Class imbalance harms minority classes (`patents`, `scientific_articles`)                    | Class-weighted loss; oversample minority classes via augmentation                                                            |

## 10. Reproducibility

- Public GitHub repository with the full notebook, this proposal, a README with step-by-step run instructions, and `requirements.txt` pinning library versions.
- All random seeds fixed (`numpy`, `tensorflow`, `random`).
- Kaggle Notebook link will be provided in the README for one-click reproduction with a GPU runtime.

## 11. References

1. Pfitzmann, B., et al. "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis." *KDD 2022*. arXiv:2206.01062.
2. Harley, A. W., Ufkes, A., Derpanis, K. G. "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval." *ICDAR 2015*. (RVL-CDIP)
3. He, K., Zhang, X., Ren, S., Sun, J. "Deep Residual Learning for Image Recognition." *CVPR 2016*. (ResNet)
4. Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *ICCV 2017*.
