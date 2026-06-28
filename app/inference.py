"""Model loading and inference for the DTCF German document classifier.

Two model formats are supported, chosen by file extension:

* ``.keras``  — loaded with TensorFlow/Keras. Used locally and on memory-generous
  hosts (Easypanel ≥2 GB, the training environment).
* ``.tflite`` — loaded with a lightweight LiteRT interpreter (``ai-edge-litert``,
  falling back to ``tflite_runtime`` or ``tensorflow.lite``). Needs far less RAM
  and a much smaller image, so this is what the free-tier deployment ships — no
  full TensorFlow at runtime.

Preprocessing is a pure-NumPy reimplementation of
``tensorflow.keras.applications.resnet50.preprocess_input`` (caffe mode: RGB→BGR
plus per-channel mean subtraction), verified to match within 1e-6. This keeps the
``.tflite`` path free of any TensorFlow dependency.
"""

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

# ImageNet BGR channel means used by ResNet50 'caffe' preprocessing.
_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)

# Lazy globals — populated on first request so container startup is fast and
# import-time failures don't crash the worker before health checks attach.
# Exactly one of _model / _interpreter is set once loaded.
_model: Any | None = None          # Keras model (.keras path)
_interpreter: Any | None = None    # LiteRT interpreter (.tflite path)
_tfl_in: Any | None = None
_tfl_out: Any | None = None


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


def _litert_interpreter_cls():
    """Return a LiteRT Interpreter class, preferring the lightweight runtimes."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # current LiteRT pkg
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # older standalone pkg
        return Interpreter
    except ImportError:
        pass
    import tensorflow as tf  # full-TF fallback (local/dev)
    return tf.lite.Interpreter


def load_model() -> None:
    """Load the model — Keras for `.keras`, a LiteRT interpreter for `.tflite`."""
    global _model, _interpreter, _tfl_in, _tfl_out
    if _model is not None or _interpreter is not None:
        return

    path = _model_path()
    if str(path).endswith(".tflite"):
        Interpreter = _litert_interpreter_cls()
        interp = Interpreter(model_path=str(path))
        interp.allocate_tensors()
        _tfl_in = interp.get_input_details()[0]
        _tfl_out = interp.get_output_details()[0]
        _interpreter = interp
    else:
        # Importing TF is expensive (~3 s) — only happens for the .keras path.
        import tensorflow as tf

        _model = tf.keras.models.load_model(path)


def _prepare(image_bytes: bytes) -> np.ndarray:
    """Resize to 224×224 and apply ResNet50 'caffe' preprocessing (NumPy only)."""
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32)
    bgr = arr[..., ::-1].copy()  # RGB -> BGR
    bgr -= _BGR_MEAN             # per-channel mean subtraction
    return bgr[None, ...]


def _infer(x: np.ndarray) -> np.ndarray:
    if _interpreter is not None:
        _interpreter.set_tensor(_tfl_in["index"], x.astype(_tfl_in["dtype"]))
        _interpreter.invoke()
        return np.asarray(_interpreter.get_tensor(_tfl_out["index"])[0])
    return np.asarray(_model.predict(x, verbose=0)[0])


def predict(image_bytes: bytes, top_k: int = 3) -> dict[str, Any]:
    """Run inference on a single image."""
    if _model is None and _interpreter is None:
        load_model()

    probs = _infer(_prepare(image_bytes))
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
    return _model is not None or _interpreter is not None
