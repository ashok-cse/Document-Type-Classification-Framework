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

    # Uncertainty = normalised Shannon entropy of the class distribution (0..1).
    p = np.clip(probs.astype(np.float64), 1e-9, 1.0)
    uncertainty = float(-np.sum(p * np.log(p)) / np.log(len(CLASS_NAMES)))

    if top_conf >= 0.70 and uncertainty <= 0.50:
        rec = ("This prediction looks reliable based on the current confidence "
               "and uncertainty signals.")
    elif top_conf >= 0.50:
        rec = ("Moderately confident — review is advisable for borderline cases.")
    else:
        rec = ("Low confidence / high uncertainty — manual review recommended; "
               "the page may fall outside the six trained German document types.")

    return {
        "predicted_class": top_class,
        "confidence": top_conf,
        "uncertainty": uncertainty,
        "recommendation": rec,
        "top_k": top_k_list,
        "all_probabilities": all_probs,
    }


# --- Grad-CAM (Keras backend only; needs gradients, so not available on TFLite) ---
_gradcam_cache: Any | None = None  # (conv_model, head_layers) or (None, None)


def _colorize(h: np.ndarray) -> np.ndarray:
    """Map a [0,1] heatmap to an RGB uint8 'jet' colour image."""
    r = np.clip(1.5 - np.abs(4 * h - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _build_gradcam():
    global _gradcam_cache
    if _gradcam_cache is not None:
        return _gradcam_cache
    import tensorflow as tf

    base = next((l for l in _model.layers if "resnet" in l.name.lower()), None)
    if base is None:
        _gradcam_cache = (None, None)
        return _gradcam_cache
    last_conv = base.get_layer("conv5_block3_out")
    conv_model = tf.keras.models.Model(base.input, last_conv.output)
    head = _model.layers[_model.layers.index(base) + 1:]
    _gradcam_cache = (conv_model, head)
    return _gradcam_cache


def gradcam_png(image_bytes: bytes, alpha: float = 0.45):
    """Return a base64 PNG data-URL Grad-CAM overlay, or None if unavailable.

    Requires the Keras model (gradients); returns None on the TFLite backend.
    """
    if _model is None:  # TFLite or not loaded -> no gradients available
        return None
    import base64
    import tensorflow as tf

    conv_model, head = _build_gradcam()
    if conv_model is None:
        return None

    pil = (Image.open(io.BytesIO(image_bytes)).convert("RGB")
           .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    x = _prepare(image_bytes)
    with tf.GradientTape() as tape:
        conv_out = conv_model(x, training=False)
        tape.watch(conv_out)
        preds = conv_out
        for layer in head:
            try:
                preds = layer(preds, training=False)
            except TypeError:
                preds = layer(preds)
        class_channel = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]).numpy()
    heat = np.maximum(heat, 0) / (heat.max() + 1e-8)

    heat = np.array(Image.fromarray(np.uint8(255 * heat))
                    .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
    overlay = np.clip(np.array(pil) * (1 - alpha) + _colorize(heat) * alpha, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(overlay.astype(np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gradcam_available() -> bool:
    return _model is not None


def model_loaded() -> bool:
    return _model is not None or _interpreter is not None
