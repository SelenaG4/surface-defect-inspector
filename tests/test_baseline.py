"""Tests for the classical LBP+SVM baseline -- real training on real (synthetic)
images, no mocks. The synthetic images are generated on the fly so the test is
self-contained and doesn't need the big real NEU download."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app import baseline
from app.labels import BASELINE_IMAGE_SIZE, CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent


def _make_synth(out: Path, per_class: int = 14) -> None:
    subprocess.run(
        [sys.executable, "scripts/make_synthetic_sample.py",
         "--out", str(out), "--per-class", str(per_class), "--seed", "1"],
        cwd=ROOT, check=True,
    )


def test_lbp_features_shape():
    img = np.random.default_rng(0).integers(0, 256, (BASELINE_IMAGE_SIZE, BASELINE_IMAGE_SIZE), dtype=np.uint8)
    feats = baseline.lbp_features(img)
    # two uniform-LBP histograms: (8+2) + (24+2) = 36 dims, finite, ~L1-normalized (sum ~2).
    assert feats.shape == (36,)
    assert np.isfinite(feats).all()


def test_train_predict_and_metrics(tmp_path):
    data = tmp_path / "synth"
    _make_synth(data)
    result = baseline.train_and_evaluate(data)
    # synthetic classes are deliberately separable, so a working pipeline scores high.
    assert result.metrics.accuracy > 0.7
    assert set(result.metrics.per_class_f1) == set(CLASS_NAMES)

    import cv2
    sample = next((data / "scratches").glob("*.png"))
    pred = result.predict(cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE))
    assert pred["label"] in CLASS_NAMES
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-6
    assert 0.0 <= pred["confidence"] <= 1.0


def test_load_dataset_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        baseline.load_dataset(tmp_path / "does_not_exist")
