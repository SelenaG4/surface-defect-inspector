"""Tests for the ONNX deep-model wrapper. Uses a tiny committed fixture ONNX
model (6-class, random weights) so the inference path is exercised in CI without
torch and without the real ~20 MB EfficientNet."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.deep_model import DeepDefectClassifier, preprocess
from app.labels import CLASS_NAMES, DEEP_IMAGE_SIZE

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_model.onnx"


def test_preprocess_shape():
    gray = np.random.default_rng(0).integers(0, 256, (200, 200), dtype=np.uint8)
    t = preprocess(gray)
    assert t.shape == (1, 3, DEEP_IMAGE_SIZE, DEEP_IMAGE_SIZE)
    assert t.dtype == np.float32


def test_fixture_loads_and_predicts():
    clf = DeepDefectClassifier(FIXTURE)
    assert clf.loaded, clf.load_error
    gray = np.random.default_rng(1).integers(0, 256, (200, 200), dtype=np.uint8)
    pred = clf.predict(gray)
    assert pred.label in CLASS_NAMES
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5
    assert len(pred.probabilities) == len(CLASS_NAMES)
    assert 0.0 <= pred.confidence <= 1.0


def test_graceful_degradation_when_absent(tmp_path):
    clf = DeepDefectClassifier(tmp_path / "nope.onnx")
    assert clf.loaded is False
    assert "No ONNX model found" in (clf.load_error or "")
