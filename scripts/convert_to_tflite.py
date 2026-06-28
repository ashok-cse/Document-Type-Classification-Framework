#!/usr/bin/env python3
"""Convert the trained Keras ResNet50 to a TFLite model for the free-tier image.

Why: full TensorFlow + ResNet50 needs ~1 GB RAM, which OOMs on small/free hosts
(e.g. Render's 512 MB free tier). A `.tflite` model served with a lightweight
LiteRT runtime fits comfortably and shrinks the image.

This requires full TensorFlow (the training-side `requirements.txt`), so run it
locally/on Kaggle — NOT in the slim inference image.

    python scripts/convert_to_tflite.py
    # -> models/resnet50.tflite

Note: direct `from_keras_model` conversion fails on this Keras 3 model (an MLIR
"missing attribute 'value'" error on a BatchNorm cast). Exporting a TF
SavedModel first and converting from that is the working path.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KERAS_PATH = ROOT / "models" / "resnet50.keras"
TFLITE_PATH = ROOT / "models" / "resnet50.tflite"
_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def convert() -> bytes:
    model = tf.keras.models.load_model(KERAS_PATH)
    tmp = Path(tempfile.mkdtemp(prefix="dtcf_sm_"))
    try:
        # Export an inference SavedModel, then convert from it (avoids the
        # from_keras_model MLIR converter bug on this model).
        model.export(tmp)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tmp))
        data = converter.convert()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    TFLITE_PATH.write_bytes(data)
    print(f"wrote {TFLITE_PATH} ({len(data) / 1048576:.1f} MB)")
    return data, model


def _prep(path: Path) -> np.ndarray:
    from PIL import Image

    pil = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32)
    bgr = arr[..., ::-1].copy()
    bgr -= _BGR_MEAN
    return bgr[None, ...]


def verify(data: bytes, model) -> None:
    """Sanity-check: TFLite output should match Keras on a sample image."""
    samples = sorted((ROOT / "figures").glob("*.png"))
    if not samples:
        print("no sample image found — skipping verification")
        return
    x = _prep(samples[0])
    keras_out = model.predict(x, verbose=0)[0]

    interp = tf.lite.Interpreter(model_content=data)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    tfl_out = interp.get_tensor(out["index"])[0]

    diff = float(np.abs(keras_out - tfl_out).max())
    print(f"keras argmax={int(np.argmax(keras_out))} "
          f"tflite argmax={int(np.argmax(tfl_out))} max_abs_diff={diff:.2e}")
    assert int(np.argmax(keras_out)) == int(np.argmax(tfl_out)), "argmax mismatch!"


if __name__ == "__main__":
    data, model = convert()
    verify(data, model)
