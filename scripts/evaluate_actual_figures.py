"""Regenerate evaluation figures from saved models and the local dataset.

This script uses the 8,057-image dataset listed in notebooks/german_docs/index.csv
and the saved Keras models to produce actual test-set artifacts in figures/.
"""

from __future__ import annotations

import io
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image, ImageFilter
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tensorflow.keras import applications


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "notebooks" / "german_docs"
INDEX_CSV = DATA_DIR / "index.csv"
FIG_DIR = ROOT / "figures"
CUSTOM_MODEL = ROOT / "notebooks" / "models" / "custom_cnn.keras"
RESNET_MODEL = ROOT / "models" / "resnet50.keras"

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 8
CLASS_NAMES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]
NUM_CLASSES = len(CLASS_NAMES)


def prepare_paths(index_df: pd.DataFrame) -> pd.DataFrame:
    df = index_df.copy()
    df["path"] = df["path"].map(lambda p: str((ROOT / "notebooks" / p).resolve()))
    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset image: {missing[0]}")
    return df


def split_test(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = index_df["label"].values
    paths = index_df["path"].values
    paths_tv, paths_test, y_tv, y_test = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    train_test_split(
        paths_tv,
        y_tv,
        test_size=0.1765,
        stratify=y_tv,
        random_state=SEED,
    )
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    return paths_test, np.array([label_to_idx[y] for y in y_test])


def decode_custom(path: str, label: int) -> tuple[tf.Tensor, int]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def decode_resnet(path: str, label: int) -> tuple[tf.Tensor, int]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = applications.resnet50.preprocess_input(img)
    return img, label


def make_ds(
    paths: np.ndarray,
    labels: np.ndarray,
    decoder,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decoder, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def plot_confusion(cm: np.ndarray, title: str, save_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title(f"{title} - confusion (counts)")
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")

    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title(f"{title} - confusion (row-normalised)")
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / save_name, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / save_name.replace(".png", ".pdf"), bbox_inches="tight")  # vector copy
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_name: str) -> float:
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        score = auc(fpr, tpr)
        aucs.append(score)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={score:.2f})")
    macro_auc = float(np.mean(aucs))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title} - ROC, macro-AUC={macro_auc:.3f}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / save_name, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / save_name.replace(".png", ".pdf"), bbox_inches="tight")  # vector copy
    plt.close(fig)
    return macro_auc


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    y_true: np.ndarray,
    name: str,
    confusion_name: str,
    roc_name: str,
) -> dict[str, float | int | str]:
    y_prob = model.predict(dataset, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, name, confusion_name)
    macro_auc = plot_roc(y_true, y_prob, name, roc_name)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0
    )
    print(f"\n=== {name} ===")
    print(report)
    return {
        "name": name,
        "params": int(model.count_params()),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_auc": macro_auc,
    }


def plot_model_comparison(metrics: list[dict[str, float | int | str]]) -> None:
    df = pd.DataFrame(metrics)
    df = df[
        [
            "name",
            "params",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_auc",
        ]
    ]
    df.to_csv(FIG_DIR / "t1_model_comparison.csv", index=False)

    metric_names = ["accuracy", "macro_f1", "macro_auc"]
    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, row in zip([-width / 2, width / 2], metrics):
        values = [float(row[m]) for m in metric_names]
        ax.bar(x + offset, values, width, label=str(row["name"]))
        for xi, value in zip(x + offset, values):
            ax.text(xi, value + 0.01, f"{value:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["accuracy", "macro-F1", "macro-AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Model comparison - actual test-set scores")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "f9_model_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "f9_model_comparison.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


def gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "conv5_block3_out",
) -> np.ndarray:
    resnet_layer = model.get_layer("resnet50")
    conv_model = tf.keras.models.Model(
        resnet_layer.input,
        resnet_layer.get_layer(last_conv_layer_name).output,
    )
    with tf.GradientTape() as tape:
        conv_out = conv_model(img_array, training=False)
        tape.watch(conv_out)
        preds = conv_out
        for layer in model.layers[2:]:
            try:
                preds = layer(preds, training=False)
            except TypeError:
                preds = layer(preds)
        class_channel = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    return np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    import matplotlib.cm as cm

    heat = np.uint8(255 * heatmap)
    heat = np.array(Image.fromarray(heat).resize(pil_img.size, Image.BILINEAR))
    cmap = (cm.jet(heat / 255.0)[..., :3] * 255).astype(np.uint8)
    base = np.array(pil_img.convert("RGB"))
    blended = (alpha * cmap + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)


def plot_gradcam(index_df: pd.DataFrame, resnet_model: tf.keras.Model) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, class_name in zip(axes.flat, CLASS_NAMES):
        path = (
            index_df[index_df["label"] == class_name]
            .sample(1, random_state=SEED)
            .iloc[0]["path"]
        )
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        arr = applications.resnet50.preprocess_input(arr)[None, ...]
        heat = gradcam_heatmap(arr, resnet_model)
        ax.imshow(overlay_heatmap(pil, heat))
        ax.set_title(class_name)
        ax.axis("off")
    plt.suptitle("Grad-CAM overlays (ResNet50 actual model)", y=1.02, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "f10_gradcam.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "f10_gradcam.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


def phone_camera_augment(pil_img: Image.Image) -> Image.Image:
    arr = np.array(pil_img)
    if np.random.rand() < 0.7:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.3, 1.5)))
        arr = np.array(pil_img)
    if np.random.rand() < 0.8:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=int(np.random.uniform(40, 80)))
        buf.seek(0)
        pil_img = Image.open(buf)
        arr = np.array(pil_img)
    if np.random.rand() < 0.7:
        arr = np.clip(arr.astype(np.float32) * np.random.uniform(0.75, 1.25), 0, 255)
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def build_phone_dataset(paths_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phone_dir = DATA_DIR / "phone_sim_test"
    phone_dir.mkdir(exist_ok=True)
    phone_paths = []
    for path in paths_test:
        out_path = phone_dir / Path(str(path)).name
        if not out_path.exists():
            phone_camera_augment(Image.open(path).convert("RGB")).save(
                out_path, "JPEG", quality=92
            )
        phone_paths.append(str(out_path))
    return np.array(phone_paths), y_test


def macro_f1(model: tf.keras.Model, dataset: tf.data.Dataset, y_true: np.ndarray) -> float:
    y_prob = model.predict(dataset, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(f1)


def plot_robustness(
    custom_model: tf.keras.Model,
    resnet_model: tf.keras.Model,
    paths_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[dict[str, float | int | str]],
) -> None:
    phone_paths, phone_labels = build_phone_dataset(paths_test, y_test)
    custom_phone = make_ds(phone_paths, phone_labels, decode_custom)
    resnet_phone = make_ds(phone_paths, phone_labels, decode_resnet)
    rows = [
        {
            "model": "Custom CNN",
            "clean_macro_f1": float(metrics[0]["macro_f1"]),
            "phone_macro_f1": macro_f1(custom_model, custom_phone, y_test),
        },
        {
            "model": "ResNet50",
            "clean_macro_f1": float(metrics[1]["macro_f1"]),
            "phone_macro_f1": macro_f1(resnet_model, resnet_phone, y_test),
        },
    ]
    df = pd.DataFrame(rows)
    df["delta"] = df["phone_macro_f1"] - df["clean_macro_f1"]
    df.to_csv(FIG_DIR / "f11_robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["clean_macro_f1"], width, label="clean")
    ax.bar(x + width / 2, df["phone_macro_f1"], width, label="phone-camera sim")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("macro-F1")
    ax.set_title("Robustness - clean vs. phone-camera simulation")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "f11_robustness.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "f11_robustness.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


def plot_eda_figures(index_df: pd.DataFrame) -> None:
    counts = index_df["label"].value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(CLASS_NAMES, counts.values, color=sns.color_palette("viridis", NUM_CLASSES))
    ax.set_title("DocLayNet-base - class distribution")
    ax.set_ylabel("Number of pages")
    ax.tick_params(axis="x", rotation=25)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 35,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "f1_class_distribution.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "f1_class_distribution.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, class_name in zip(axes.flat, CLASS_NAMES):
        path = (
            index_df[index_df["label"] == class_name]
            .sample(1, random_state=SEED)
            .iloc[0]["path"]
        )
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(class_name)
        ax.axis("off")
    plt.suptitle("Sample document page per class", y=1.02, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "f2_sample_per_class.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "f2_sample_per_class.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    sns.set_theme(style="white", context="notebook")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    index_df = prepare_paths(pd.read_csv(INDEX_CSV))
    paths_test, y_test = split_test(index_df)
    print(f"Dataset images: {len(index_df):,}")
    print(f"Test images: {len(paths_test):,}")

    plot_eda_figures(index_df)

    custom_model = tf.keras.models.load_model(CUSTOM_MODEL)
    resnet_model = tf.keras.models.load_model(RESNET_MODEL)

    custom_ds = make_ds(paths_test, y_test, decode_custom)
    resnet_ds = make_ds(paths_test, y_test, decode_resnet)
    metrics = [
        evaluate_model(
            custom_model,
            custom_ds,
            y_test,
            "Custom CNN",
            "f5_custom_cnn_confusion.png",
            "f6_custom_cnn_roc.png",
        ),
        evaluate_model(
            resnet_model,
            resnet_ds,
            y_test,
            "ResNet50",
            "f7_resnet50_confusion.png",
            "f8_resnet50_roc.png",
        ),
    ]
    plot_model_comparison(metrics)
    plot_gradcam(index_df, resnet_model)
    plot_robustness(custom_model, resnet_model, paths_test, y_test, metrics)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
