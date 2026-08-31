"""Tests for the platform layer: tracing.

The property that matters here is that telemetry can never break the service.
The default path -- no exporter configured -- has to stay a genuine no-op,
because that is the path CI, local development and the Render deployment all
actually run on. Instrumentation that can take down the thing it observes is
worse than no instrumentation.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fresh_telemetry(monkeypatch):
    """A reimported telemetry module with its module-level state reset.

    setup_telemetry() is deliberately idempotent via a module-level flag, so a
    test that wants to exercise configuration must start from a clean module.
    """
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("OTEL_CONSOLE_EXPORT", raising=False)
    from app import telemetry
    return importlib.reload(telemetry)


def test_telemetry_is_noop_when_unconfigured(fresh_telemetry):
    assert fresh_telemetry.setup_telemetry() is False
    assert fresh_telemetry.is_configured() is False

    with fresh_telemetry.span("test.span", attr=1) as current:
        assert current is None
        # Attaching attributes to a no-op span must not raise.
        fresh_telemetry.set_attributes(current, another=2)


def test_telemetry_survives_a_broken_exporter(monkeypatch):
    """A malformed connection string must not take the service down."""
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "this-is-not-valid")
    monkeypatch.delenv("OTEL_CONSOLE_EXPORT", raising=False)
    from app import telemetry
    mod = importlib.reload(telemetry)

    # Either it fails and degrades to a no-op, or it accepts the string and
    # fails later on export. Both are acceptable; raising here is not.
    assert mod.setup_telemetry() in (True, False)
    with mod.span("test.span"):
        pass

    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    importlib.reload(telemetry)


def test_service_starts_and_serves_with_telemetry_off():
    """The instrumented code path must behave identically when tracing is off."""
    from app.main import app
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["baseline_loaded"] is True


def test_predict_is_unchanged_by_instrumentation():
    """Spans wrap the model calls; they must not alter the response shape."""
    import glob

    from app.main import app
    sample = sorted(glob.glob("data/sample/*/*.png"))
    assert sample, "no sample images committed"

    with TestClient(app) as client:
        with open(sample[0], "rb") as handle:
            response = client.post("/predict", files={"file": handle})

    assert response.status_code == 200
    body = response.json()
    assert "baseline" in body
    assert "label" in body["baseline"]
    # "deep" is present either as a prediction or as an explicit null with a
    # status, depending on whether the ONNX model is available.
    assert "deep" in body
