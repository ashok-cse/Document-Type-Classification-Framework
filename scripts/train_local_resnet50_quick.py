from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import applications, callbacks, layers, models


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = int(os.getenv("DTCF_QUICK_EPOCHS", "3"))
MAX_PER_CLASS = int(os.getenv("DTCF_MAX_PER_CLASS", "300"))

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
INDEX_CSV = NOTEBOOK_DIR / "german_docs" / "index.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "resnet50.keras"

CLASS_NAMES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = applications.resnet50.preprocess_input(img)
    return img, label


def make_dataset(paths: np.ndarray, labels: np.ndarray, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return (
        ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not INDEX_CSV.exists():
        raise FileNotFoundError(
            f"{INDEX_CSV} not found. Run the notebook preprocessing first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INDEX_CSV)
    df = df[df["label"].isin(CLASS_NAMES)].copy()
    df["path"] = df["path"].map(lambda p: str(NOTEBOOK_DIR / p))

    balanced = []
    for label in CLASS_NAMES:
        part = df[df["label"] == label].sample(
            n=min(MAX_PER_CLASS, int((df["label"] == label).sum())),
            random_state=SEED,
        )
        balanced.append(part)
    df = pd.concat(balanced).sample(frac=1, random_state=SEED).reset_index(drop=True)

    label_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    labels = df["label"].map(label_to_id).to_numpy(dtype=np.int64)
    paths = df["path"].to_numpy(dtype=str)

    train_paths, val_paths, train_y, val_y = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    print(f"Training examples: {len(train_paths)}")
    print(f"Validation examples: {len(val_paths)}")
    print(f"Max per class: {MAX_PER_CLASS}")
    print(f"Epochs: {EPOCHS}")
    print("TensorFlow:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())

    train_ds = make_dataset(train_paths, train_y, shuffle=True)
    val_ds = make_dataset(val_paths, val_y, shuffle=False)

    base = applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = models.Model(inp, out, name="resnet50_transfer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            callbacks.ModelCheckpoint(
                str(MODEL_PATH), save_best_only=True, monitor="val_accuracy"
            )
        ],
        verbose=2,
    )

    if not MODEL_PATH.exists():
        model.save(MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
