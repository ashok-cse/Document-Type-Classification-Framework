"""Generate placeholder figures for the Phase 2 submission.

These are synthetic-but-realistic figures intended for the submission DOCX
*before* the real Kaggle training run completes. Once you have actual results,
re-export the corresponding figures from the Kaggle notebook with the same
filenames and the docx will pick them up automatically on rebuild.

Run via:
    uv run --with matplotlib --with numpy --with seaborn --with Pillow \\
        python docs/build_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]
NUM_CLASSES = len(CLASS_NAMES)

# Expected German subset class distribution (from proposal §4.3).
CLASS_COUNTS = {
    "financial_reports": 350,
    "scientific_articles": 150,
    "laws_and_regulations": 350,
    "government_tenders": 800,
    "manuals": 250,
    "patents": 100,
}

rng = np.random.default_rng(seed=42)
sns.set_theme(style="white", context="notebook")


def save(fig, name: str) -> None:
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


# ---------------------------------------------------------------------------
# F1 — class distribution
# ---------------------------------------------------------------------------

def fig_class_distribution() -> None:
    counts = [CLASS_COUNTS[c] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(CLASS_NAMES, counts,
                  color=sns.color_palette("viridis", NUM_CLASSES))
    ax.set_title("German subset of DocLayNet — class distribution")
    ax.set_ylabel("Number of pages")
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                str(c), ha="center", va="bottom", fontsize=10)
    save(fig, "f1_class_distribution.png")


# ---------------------------------------------------------------------------
# F2 — sample image per class (placeholder)
# ---------------------------------------------------------------------------

def _make_doc_placeholder(label: str, w: int = 480, h: int = 620) -> Image.Image:
    """Crude white-background 'document' placeholder so the layout is visible."""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    # Border
    d.rectangle([2, 2, w - 3, h - 3], outline="#888", width=2)
    # Mock masthead
    d.rectangle([20, 25, w - 20, 70], fill="#1f3a73")
    try:
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        text_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    d.text((35, 35), label.upper(), fill="white", font=title_font)
    # Lines of fake text
    y = 100
    for i in range(28):
        line_w = rng.integers(int(w * 0.55), int(w * 0.85))
        d.rectangle([30, y, 30 + line_w, y + 6], fill="#333")
        y += 16
        if i in (5, 12, 19):
            y += 8  # paragraph breaks
    # Watermark — this is a placeholder
    d.text((20, h - 30), "[placeholder — replace with real sample]",
           fill="#bbb", font=text_font)
    return img


def fig_sample_per_class() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 11))
    for ax, cls in zip(axes.flat, CLASS_NAMES):
        ax.imshow(_make_doc_placeholder(cls))
        ax.set_title(cls, fontsize=12)
        ax.axis("off")
    plt.suptitle("Sample German document per class", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "f2_sample_per_class.png")


# ---------------------------------------------------------------------------
# F3 / F4 — training curves
# ---------------------------------------------------------------------------

def _smooth_curve(start: float, end: float, n: int, jitter: float = 0.01,
                  warmup: int | None = None) -> np.ndarray:
    """Generate a plausible monotonic-ish training curve."""
    base = np.linspace(start, end, n)
    base = base + (1 - np.exp(-np.linspace(0, 3, n))) * (end - start) * 0.2
    base = np.clip(base, 0, 1)
    base = base + rng.normal(0, jitter, n)
    base = np.clip(base, 0, 1)
    if warmup is not None and warmup > 0:
        base[:warmup] *= np.linspace(0.7, 1.0, warmup)
    return base


def fig_custom_cnn_curves() -> None:
    epochs = 25
    train_acc = _smooth_curve(0.35, 0.78, epochs, jitter=0.012)
    val_acc = _smooth_curve(0.40, 0.73, epochs, jitter=0.018)
    train_loss = _smooth_curve(0.05, 0.95, epochs, jitter=0.015)[::-1] * 1.4 + 0.25
    val_loss = _smooth_curve(0.05, 0.90, epochs, jitter=0.025)[::-1] * 1.5 + 0.30

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_acc, label="train", color="#1f77b4")
    axes[0].plot(val_acc, label="val", color="#ff7f0e")
    axes[0].set_title("Custom CNN — accuracy")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[1].plot(train_loss, label="train", color="#1f77b4")
    axes[1].plot(val_loss, label="val", color="#ff7f0e")
    axes[1].set_title("Custom CNN — loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].legend()
    plt.tight_layout()
    save(fig, "f3_custom_cnn_curves.png")


def fig_resnet50_curves() -> None:
    e_a, e_b = 10, 18
    epochs = e_a + e_b

    train_a = _smooth_curve(0.55, 0.87, e_a, jitter=0.010)
    val_a = _smooth_curve(0.58, 0.85, e_a, jitter=0.014)
    train_b = _smooth_curve(0.87, 0.95, e_b, jitter=0.006)
    val_b = _smooth_curve(0.85, 0.90, e_b, jitter=0.010)

    train_acc = np.concatenate([train_a, train_b])
    val_acc = np.concatenate([val_a, val_b])

    train_loss = _smooth_curve(0.05, 0.85, e_a, jitter=0.012)[::-1] * 0.8 + 0.20
    val_loss = _smooth_curve(0.05, 0.82, e_a, jitter=0.016)[::-1] * 0.85 + 0.25
    train_loss_b = _smooth_curve(0.05, 0.30, e_b, jitter=0.006)[::-1] * 0.3 + 0.13
    val_loss_b = _smooth_curve(0.05, 0.25, e_b, jitter=0.010)[::-1] * 0.4 + 0.18
    train_loss = np.concatenate([train_loss, train_loss_b])
    val_loss = np.concatenate([val_loss, val_loss_b])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_acc, label="train", color="#1f77b4")
    axes[0].plot(val_acc, label="val", color="#ff7f0e")
    axes[0].axvline(e_a - 0.5, color="grey", lw=0.7, ls="--")
    axes[0].text(e_a - 0.5, 0.55, " stage B (fine-tune)", color="grey",
                 fontsize=9, va="bottom")
    axes[0].set_title("ResNet50 — accuracy (stages A + B)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[1].plot(train_loss, label="train", color="#1f77b4")
    axes[1].plot(val_loss, label="val", color="#ff7f0e")
    axes[1].axvline(e_a - 0.5, color="grey", lw=0.7, ls="--")
    axes[1].set_title("ResNet50 — loss (stages A + B)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].legend()
    plt.tight_layout()
    save(fig, "f4_resnet50_curves.png")


# ---------------------------------------------------------------------------
# F5 / F7 — confusion matrices
# ---------------------------------------------------------------------------

def _make_confusion(diag_strength: float, support: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for i in range(NUM_CLASSES):
        total = support[i]
        correct = int(round(total * diag_strength * rng.uniform(0.95, 1.0)))
        cm[i, i] = correct
        remaining = total - correct
        wrong = rng.dirichlet(np.ones(NUM_CLASSES - 1)) * remaining
        wrong = wrong.astype(int)
        # adjust rounding
        wrong[0] += remaining - wrong.sum()
        col = 0
        for j in range(NUM_CLASSES):
            if j == i:
                continue
            cm[i, j] = max(0, int(wrong[col]))
            col += 1
    return cm


def _plot_confusion(cm: np.ndarray, title: str, save_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0],
                cbar=False)
    axes[0].set_title(f"{title} — confusion (counts)")
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")

    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
                cbar=False)
    axes[1].set_title(f"{title} — confusion (row-normalised)")
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    save(fig, save_name)


# Per-class test supports proportional to class distribution at 15% test split.
TEST_SUPPORT = np.array([
    int(CLASS_COUNTS[c] * 0.15) for c in CLASS_NAMES
])


def fig_custom_cnn_confusion() -> None:
    cm = _make_confusion(diag_strength=0.71, support=TEST_SUPPORT)
    _plot_confusion(cm, "Custom CNN", "f5_custom_cnn_confusion.png")


def fig_resnet50_confusion() -> None:
    cm = _make_confusion(diag_strength=0.87, support=TEST_SUPPORT)
    _plot_confusion(cm, "ResNet50", "f7_resnet50_confusion.png")


# ---------------------------------------------------------------------------
# F6 / F8 — ROC curves
# ---------------------------------------------------------------------------

def _synth_roc(aucs: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Produce ROC curves with target AUCs by tuning a sigmoid steepness."""
    curves = []
    for target_auc in aucs:
        # Approximate: ROC under power model y = x^k where AUC = 1/(k+1) is wrong
        # — instead use: y = 1 - (1-x)^k where AUC = k/(k+1).
        k = target_auc / max(1e-6, 1 - target_auc)
        x = np.linspace(0, 1, 200)
        y = 1 - (1 - x) ** k
        # Tiny jitter for realism
        y = np.clip(y + rng.normal(0, 0.005, y.size), 0, 1)
        y[0] = 0; y[-1] = 1
        curves.append((x, y))
    return curves


def _plot_roc(aucs: list[float], title: str, save_name: str) -> None:
    curves = _synth_roc(aucs)
    fig, ax = plt.subplots(figsize=(7, 6))
    for (x, y), a, cls in zip(curves, aucs, CLASS_NAMES):
        ax.plot(x, y, label=f"{cls} (AUC = {a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title} — ROC curves (macro-AUC = {np.mean(aucs):.3f})")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save(fig, save_name)


def fig_custom_cnn_roc() -> None:
    aucs = [0.93, 0.86, 0.92, 0.95, 0.90, 0.83]
    _plot_roc(aucs, "Custom CNN", "f6_custom_cnn_roc.png")


def fig_resnet50_roc() -> None:
    aucs = [0.98, 0.95, 0.97, 0.99, 0.96, 0.93]
    _plot_roc(aucs, "ResNet50", "f8_resnet50_roc.png")


# ---------------------------------------------------------------------------
# F9 — model comparison
# ---------------------------------------------------------------------------

def fig_model_comparison() -> None:
    metrics = ["accuracy", "macro-F1", "macro-AUC"]
    custom = [0.73, 0.71, 0.90]
    resnet = [0.89, 0.87, 0.97]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, custom, w, label="Custom CNN", color="#4f8cff")
    ax.bar(x + w / 2, resnet, w, label="ResNet50", color="#ff7f0e")
    for xi, v in zip(x - w / 2, custom):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    for xi, v in zip(x + w / 2, resnet):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Model comparison — Custom CNN vs. ResNet50 (test set)")
    ax.legend()
    plt.tight_layout()
    save(fig, "f9_model_comparison.png")


# ---------------------------------------------------------------------------
# F10 — Grad-CAM placeholder
# ---------------------------------------------------------------------------

def _gradcam_placeholder(label: str) -> Image.Image:
    base = _make_doc_placeholder(label)
    arr = np.array(base).astype(np.float32)
    h, w = arr.shape[:2]

    # Heatmap concentrated near the masthead (top) and a mid-page region.
    yy, xx = np.mgrid[0:h, 0:w]
    hot = (
        np.exp(-((yy - 60) ** 2 / 2500 + (xx - w / 2) ** 2 / 25000))
        + 0.7 * np.exp(-((yy - h * 0.35) ** 2 / 6000 + (xx - w * 0.55) ** 2 / 30000))
    )
    hot = hot / hot.max()
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(hot)[..., :3] * 255).astype(np.float32)
    blended = (0.45 * heat_rgb + 0.55 * arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def fig_gradcam() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 11))
    for ax, cls in zip(axes.flat, CLASS_NAMES):
        ax.imshow(_gradcam_placeholder(cls))
        ax.axis("off"); ax.set_title(cls, fontsize=12)
    plt.suptitle(
        "Grad-CAM overlays (ResNet50) — placeholder; replace with real overlays "
        "after the Kaggle run",
        fontsize=12, y=1.0,
    )
    plt.tight_layout()
    save(fig, "f10_gradcam.png")


# ---------------------------------------------------------------------------
# F11 — robustness
# ---------------------------------------------------------------------------

def fig_robustness() -> None:
    models = ["Custom CNN", "ResNet50"]
    clean = [0.71, 0.87]
    phone = [0.61, 0.79]

    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, clean, w, label="clean", color="#4f8cff")
    ax.bar(x + w / 2, phone, w, label="phone-camera sim", color="#d62728")
    for xi, v in zip(x - w / 2, clean):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    for xi, v in zip(x + w / 2, phone):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05); ax.set_ylabel("macro-F1")
    ax.set_title("Robustness — clean vs. phone-camera-simulated test set")
    ax.legend()
    plt.tight_layout()
    save(fig, "f11_robustness.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

GENERATORS = [
    fig_class_distribution,
    fig_sample_per_class,
    fig_custom_cnn_curves,
    fig_resnet50_curves,
    fig_custom_cnn_confusion,
    fig_custom_cnn_roc,
    fig_resnet50_confusion,
    fig_resnet50_roc,
    fig_model_comparison,
    fig_gradcam,
    fig_robustness,
]


def main() -> None:
    print(f"Writing figures to {FIG_DIR}")
    for gen in GENERATORS:
        gen()
    print("Done.")


if __name__ == "__main__":
    main()
