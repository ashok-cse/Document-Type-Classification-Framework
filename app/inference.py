"""Model loading and inference for the DTCF German document classifier."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]

IMG_SIZE = 224

# Lazy globals — populated on first request so container startup is fast and
# import-time failures don't crash the worker before health checks attach.
_model: Any | None = None
_preprocess: Any | None = None


def _model_path() -> Path:
    """Locate the model file.

    Precedence:
    1. `LOCAL_MODEL_PATH` env var — explicit path on disk (or mounted volume).
    2. Download from HuggingFace Hub using `HF_MODEL_REPO` and `HF_MODEL_FILE`.
    """
    local = os.getenv("LOCAL_MODEL_PATH")
    if local:
        p = Path(local)
        if not p.exists():
            raise FileNotFoundError(f"LOCAL_MODEL_PATH={local} does not exist")
        return p

    repo = os.getenv("HF_MODEL_REPO")
    filename = os.getenv("HF_MODEL_FILE", "resnet50.keras")
    if not repo:
        raise RuntimeError(
            "No model source configured. Set LOCAL_MODEL_PATH or HF_MODEL_REPO."
        )

    from huggingface_hub import hf_hub_download

    cache_dir = os.getenv("HF_HOME", "/app/.cache/huggingface")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            cache_dir=cache_dir,
            token=os.getenv("HF_TOKEN"),
        )
    )


def load_model() -> None:
    """Load the Keras model and the matching ResNet50 preprocessing fn."""
    global _model, _preprocess
    if _model is not None:
        return

    # Importing TF is expensive (~3 s) — defer until first call.
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input

    path = _model_path()
    _model = tf.keras.models.load_model(path)
    _preprocess = preprocess_input


def _prepare(image_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32)
    return _preprocess(arr)[None, ...]


def predict(image_bytes: bytes, top_k: int = 3) -> dict[str, Any]:
    """Run inference on a single image."""
    if _model is None:
        load_model()

    x = _prepare(image_bytes)
    probs = _model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    top_class = CLASS_NAMES[int(order[0])]
    top_conf = float(probs[order[0]])
    top_k_list = [
        {"class": CLASS_NAMES[int(i)], "confidence": float(probs[int(i)])}
        for i in order[:top_k]
    ]
    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "predicted_class": top_class,
        "confidence": top_conf,
        "top_k": top_k_list,
        "all_probabilities": all_probs,
    }


def model_loaded() -> bool:
    return _model is not None
