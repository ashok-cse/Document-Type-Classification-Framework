#!/usr/bin/env python3
"""Validate documents against a running DTCF inference service.

Talks to the service purely over HTTP (`POST /predict`), so it needs **no**
TensorFlow / heavy deps locally — only the Python standard library. Use it to
smoke-test a fresh Easypanel deploy and to measure accuracy on a labelled set.

Usage
-----
Liveness + a single document:

    python scripts/validate_endpoint.py --url https://<host> page.jpg

Batch accuracy over a labelled folder. Lay the images out one sub-folder per
class (sub-folder name must match a class label); any other layout is treated as
"unlabelled" and only predictions are printed:

    docs/
      financial_reports/  a.jpg b.png ...
      patents/            c.jpg ...
      ...
    python scripts/validate_endpoint.py --url https://<host> docs/

Exit code is non-zero if /health is unhealthy or any request errors, so it is
CI / shell friendly.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Kept in sync with app/inference.py CLASS_NAMES (training label-index order).
CLASS_NAMES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def _http_get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post_image(url: str, path: Path, timeout: float) -> dict:
    """POST one image as multipart/form-data using only the stdlib."""
    boundary = "----dtcfvalidate7e3b1f"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()

    pre = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    post = f"\r\n--{boundary}--\r\n".encode()
    body = pre + data + post

    req = urllib.request.Request(
        url.rstrip("/") + "/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def check_health(url: str, timeout: float) -> bool:
    try:
        h = _http_get_json(url.rstrip("/") + "/health", timeout)
    except (urllib.error.URLError, OSError) as exc:
        print(f"✗ /health unreachable: {exc}")
        return False
    loaded = bool(h.get("model_loaded"))
    mark = "✓" if loaded and h.get("status") == "ok" else "✗"
    print(f"{mark} /health  status={h.get('status')!r}  model_loaded={loaded}")
    if h.get("classes") and list(h["classes"]) != CLASS_NAMES:
        print(f"  ⚠ server classes differ from expected order: {h['classes']}")
    return loaded


def predict_one(url: str, path: Path, timeout: float) -> dict | None:
    try:
        return _post_image(url, path, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        print(f"  ✗ {path.name}: HTTP {exc.code} {detail}")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"  ✗ {path.name}: {exc}")
    return None


def _print_confusion(matrix: dict[str, dict[str, int]]) -> None:
    """Text confusion matrix: rows = true label, cols = predicted."""
    width = max(len(c) for c in CLASS_NAMES)
    short = [c[:7] for c in CLASS_NAMES]
    header = " " * (width + 2) + " ".join(f"{s:>7}" for s in short)
    print(header)
    for true in CLASS_NAMES:
        row = matrix.get(true, {})
        cells = " ".join(f"{row.get(pred, 0):>7}" for pred in CLASS_NAMES)
        print(f"{true:<{width}}  {cells}")
    print("  (rows = true, cols = predicted; col labels truncated to 7 chars)")


def run_batch(url: str, root: Path, timeout: float) -> int:
    # Labelled if every immediate sub-dir holding images is a known class name.
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    labelled = bool(subdirs) and all(d.name in CLASS_NAMES for d in subdirs)

    if not labelled:
        print(f"\nUnlabelled folder — printing predictions only ({root}):")
        errors = 0
        for img in _iter_images(root):
            res = predict_one(url, img, timeout)
            if res is None:
                errors += 1
                continue
            print(f"  {img.name:<40} -> {res['predicted_class']:<20} "
                  f"({res['confidence']:.3f})")
        return 1 if errors else 0

    print(f"\nLabelled validation over {root} "
          f"({sum(1 for _ in _iter_images(root))} images):")
    confusion = {t: {p: 0 for p in CLASS_NAMES} for t in CLASS_NAMES}
    per_class_total = dict.fromkeys(CLASS_NAMES, 0)
    correct = total = errors = 0

    for sub in subdirs:
        true_label = sub.name
        for img in _iter_images(sub):
            res = predict_one(url, img, timeout)
            if res is None:
                errors += 1
                continue
            pred = res["predicted_class"]
            confusion[true_label][pred] += 1
            per_class_total[true_label] += 1
            total += 1
            correct += int(pred == true_label)

    if total == 0:
        print("  no images classified.")
        return 1

    print(f"\nOverall accuracy: {correct}/{total} = {correct / total:.1%}"
          + (f"   ({errors} request errors)" if errors else ""))

    print("\nPer-class recall:")
    for c in CLASS_NAMES:
        n = per_class_total[c]
        rec = confusion[c][c] / n if n else 0.0
        print(f"  {c:<22} {confusion[c][c]:>3}/{n:<3} = "
              + (f"{rec:.1%}" if n else "  n/a (no samples)"))

    print("\nConfusion matrix:")
    _print_confusion(confusion)
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="image file, or a folder (optionally one sub-folder per class)")
    ap.add_argument("--url", default="http://localhost:8000",
                    help="base URL of the running service (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="per-request timeout in seconds (default: %(default)s)")
    ap.add_argument("--skip-health", action="store_true",
                    help="don't probe /health before classifying")
    args = ap.parse_args()

    healthy = True
    if not args.skip_health:
        healthy = check_health(args.url, args.timeout)

    target = Path(args.path)
    if not target.exists():
        print(f"✗ path not found: {target}")
        return 2

    if target.is_file():
        res = predict_one(args.url, target, args.timeout)
        if res is None:
            return 1
        print(json.dumps(res, indent=2))
        return 0 if healthy else 1

    rc = run_batch(args.url, target, args.timeout)
    return rc or (0 if healthy else 1)


if __name__ == "__main__":
    sys.exit(main())
