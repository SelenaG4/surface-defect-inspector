"""Generate a SYNTHETIC stand-in for the NEU surface-defect dataset.

This is NOT the real data and is never used for the numbers quoted in the
README -- those come from the real NEU benchmark (see scripts/fetch_data.py and
the Colab notebook). Its only jobs are:

  1. Let the whole pipeline (baseline training, the API, the ONNX inference
     path, the tests) run and be verified end to end *without* the multi-hundred-
     MB real download -- so CI is fast and a fresh checkout works offline.
  2. Provide the handful of committed demo images under data/sample/ that power
     the landing page's "try it" button and the test suite.

Each class is a crude procedural caricature of the real defect's texture
(cracks, blobs, pits, streaks, scratches) -- deliberately distinguishable so the
baseline learns *something* and the plumbing is exercised, but no claim is made
that these look like real steel. Deterministic given --seed.

Usage:
    python scripts/make_synthetic_sample.py --out data/synthetic --per-class 60
    python scripts/make_synthetic_sample.py --out data/sample --per-class 3 --seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.labels import BASELINE_IMAGE_SIZE, CLASS_NAMES  # noqa: E402

S = BASELINE_IMAGE_SIZE


def _base(rng: np.random.Generator) -> np.ndarray:
    """Mid-gray field with mild sensor-like noise -- the 'clean metal' backdrop."""
    img = np.full((S, S), 128, dtype=np.float32)
    img += rng.normal(0, 8, (S, S))
    return img


def _crazing(rng):  # dense web of short cracks
    img = _base(rng)
    for _ in range(rng.integers(40, 70)):
        x, y = rng.integers(0, S, 2)
        ang = rng.uniform(0, np.pi)
        ln = rng.integers(8, 22)
        x2 = int(x + ln * np.cos(ang)); y2 = int(y + ln * np.sin(ang))
        cv2.line(img, (x, y), (x2, y2), float(rng.integers(60, 100)), 1)
    return img


def _inclusion(rng):  # a few dark embedded blobs
    img = _base(rng)
    for _ in range(rng.integers(2, 5)):
        c = (int(rng.integers(30, S - 30)), int(rng.integers(30, S - 30)))
        axes = (int(rng.integers(6, 18)), int(rng.integers(6, 18)))
        cv2.ellipse(img, c, axes, float(rng.integers(0, 180)), 0, 360, float(rng.integers(40, 80)), -1)
    return img


def _patches(rng):  # large irregular brighter/darker regions
    img = _base(rng)
    for _ in range(rng.integers(1, 3)):
        c = (int(rng.integers(40, S - 40)), int(rng.integers(40, S - 40)))
        axes = (int(rng.integers(30, 60)), int(rng.integers(30, 60)))
        val = float(rng.choice([70, 185]))
        cv2.ellipse(img, c, axes, float(rng.integers(0, 180)), 0, 360, val, -1)
    return cv2.GaussianBlur(img, (9, 9), 0)


def _pitted(rng):  # many tiny dark pits
    img = _base(rng)
    for _ in range(rng.integers(120, 220)):
        x, y = rng.integers(0, S, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), float(rng.integers(40, 90)), -1)
    return img


def _scale(rng):  # horizontal wavy rolled-in streaks
    img = _base(rng)
    for _ in range(rng.integers(6, 12)):
        y0 = int(rng.integers(0, S))
        pts = [(x, int(y0 + 6 * np.sin(x / 18.0 + rng.uniform(0, 6)))) for x in range(0, S, 4)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, float(rng.integers(150, 195)), int(rng.integers(1, 3)))
    return img


def _scratches(rng):  # a few long straight scratches
    img = _base(rng)
    for _ in range(rng.integers(2, 5)):
        x, y = rng.integers(0, S, 2)
        ang = rng.uniform(0, np.pi)
        ln = rng.integers(80, 160)
        x2 = int(x + ln * np.cos(ang)); y2 = int(y + ln * np.sin(ang))
        cv2.line(img, (x, y), (x2, y2), float(rng.integers(170, 210)), int(rng.integers(1, 3)))
    return img


_GEN = {
    "crazing": _crazing,
    "inclusion": _inclusion,
    "patches": _patches,
    "pitted_surface": _pitted,
    "rolled-in_scale": _scale,
    "scratches": _scratches,
}


def generate(out_dir: Path, per_class: int, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    n = 0
    for cls in CLASS_NAMES:
        d = out_dir / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = np.clip(_GEN[cls](rng), 0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"{cls}_{i:03d}.png"), img)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    total = generate(out, args.per_class, args.seed)
    print(f"Wrote {total} synthetic images ({args.per_class}/class x {len(CLASS_NAMES)} classes) to {out}")


if __name__ == "__main__":
    main()
