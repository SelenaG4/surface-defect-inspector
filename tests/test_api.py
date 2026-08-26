"""Tests for the FastAPI service: the loaded path (baseline + deep both serving)
and the graceful-degradation path (no artifacts present). Model paths are
monkeypatched so the tests fully control state and don't depend on whether the
real trained artifacts happen to be in models/."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app import baseline, deep_model, main
from app.labels import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_model.onnx"


def _make_synth(out: Path, per_class: int = 14) -> None:
    subprocess.run(
        [sys.executable, "scripts/make_synthetic_sample.py",
         "--out", str(out), "--per-class", str(per_class), "--seed", "2"],
        cwd=ROOT, check=True,
    )


@pytest.fixture
def loaded_client(tmp_path, monkeypatch):
    data = tmp_path / "synth"
    _make_synth(data)
    res = baseline.train_and_evaluate(data)
    bpath = tmp_path / "baseline.joblib"
    joblib.dump({"pipeline": res.pipeline, "metrics": res.metrics.__dict__, "class_names": CLASS_NAMES}, bpath)
    monkeypatch.setattr(main, "BASELINE_PATH", bpath)
    monkeypatch.setattr(deep_model, "MODEL_PATH", FIXTURE)
    with TestClient(main.app) as c:
        yield c, data


@pytest.fixture
def empty_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "BASELINE_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(deep_model, "MODEL_PATH", tmp_path / "none.onnx")
    with TestClient(main.app) as c:
        yield c


def test_landing_page_renders(loaded_client):
    c, _ = loaded_client
    r = c.get("/")
    assert r.status_code == 200
    assert "Surface Defect Inspector" in r.text


def test_classes(loaded_client):
    c, _ = loaded_client
    body = c.get("/classes").json()
    assert body["raw"] == CLASS_NAMES
    assert len(body["classes"]) == len(CLASS_NAMES)


def test_health_and_predict_when_loaded(loaded_client):
    c, data = loaded_client
    h = c.get("/health").json()
    assert h["baseline_loaded"] is True
    assert h["deep_loaded"] is True

    img = next((data / "pitted_surface").glob("*.png")).read_bytes()
    r = c.post("/predict", files={"file": ("t.png", img, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["baseline"]["label"] in CLASS_NAMES
    assert body["deep"]["label"] in CLASS_NAMES
    assert abs(sum(body["baseline"]["probabilities"].values()) - 1.0) < 1e-6


def test_baseline_metrics_endpoint(loaded_client):
    c, _ = loaded_client
    m = c.get("/baseline/metrics").json()
    assert m["class_order"] == CLASS_NAMES
    assert 0.0 <= m["accuracy"] <= 1.0


def test_graceful_degradation_when_no_artifacts(empty_client):
    h = empty_client.get("/health").json()
    assert h["baseline_loaded"] is False
    assert h["deep_loaded"] is False
    assert h["baseline_status"] is not None
    # /predict short-circuits to 503 when the baseline isn't loaded.
    r = empty_client.post("/predict", files={"file": ("t.png", b"x", "image/png")})
    assert r.status_code == 503
