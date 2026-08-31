"""Surface Defect Inspector -- a real classical baseline (LBP texture features +
SVM) shown side by side with a fine-tuned EfficientNet, on the NEU steel-surface
defect benchmark. The deep model has to beat the classical one on measured
numbers, not by assumption.

Both models are loaded from small artifacts at startup (the baseline is a joblib
SVM; the deep model is an ONNX file served by onnxruntime), and the service
degrades gracefully -- reporting exactly which model is missing -- rather than
crashing, when an artifact hasn't been produced yet.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import baseline, telemetry
from app.deep_model import DeepDefectClassifier
from app.labels import CLASS_NAMES, DISPLAY_NAMES

STATIC_DIR = Path(__file__).resolve().parent / "static"
BASELINE_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_lbp_svm.joblib"

_baseline: baseline.BaselineResult | None = None
_baseline_error: str | None = None
_deep: DeepDefectClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models and configure tracing before serving traffic.

    Replaces the deprecated @app.on_event("startup") hook.
    """
    global _baseline, _baseline_error, _deep
    try:
        _baseline = baseline.load_baseline(BASELINE_PATH)
    except Exception as exc:  # noqa: BLE001 -- degrade gracefully if the artifact is absent
        _baseline = None
        _baseline_error = str(exc)
    _deep = DeepDefectClassifier()
    telemetry.setup_telemetry(app)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Surface Defect Inspector",
    description=(
        "Classical LBP+SVM texture baseline vs. a fine-tuned EfficientNet on the NEU "
        "steel-surface defect benchmark (6 defect types) -- measured, not assumed."
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _read_upload_to_gray(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode the uploaded file as an image.")
    return img


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> str:
    # Explicit UTF-8: index.html has real non-ASCII glyphs, and Path.read_text()
    # without an encoding falls back to the OS default (cp1252 on Windows),
    # which silently mangles them -- a bug this portfolio hit for real once.
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "baseline_loaded": _baseline is not None,
        "baseline_status": None if _baseline is not None else _baseline_error,
        "deep_loaded": bool(_deep and _deep.loaded),
        "deep_status": None if (_deep and _deep.loaded) else (_deep.load_error if _deep else "not initialized"),
    }


@app.get("/classes")
def classes() -> dict:
    return {"classes": [DISPLAY_NAMES[c] for c in CLASS_NAMES], "raw": CLASS_NAMES}


@app.get("/baseline/metrics")
def baseline_metrics() -> dict:
    if _baseline is None:
        raise HTTPException(status_code=503, detail=_baseline_error or "Baseline not loaded")
    m = _baseline.metrics
    return {
        "accuracy": m.accuracy,
        "macro_f1": m.macro_f1,
        "per_class_f1": m.per_class_f1,
        "confusion_matrix": m.confusion_matrix,
        "class_order": CLASS_NAMES,
        "n_train": m.n_train,
        "n_test": m.n_test,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _baseline is None:
        raise HTTPException(status_code=503, detail=_baseline_error or "Baseline not loaded")
    gray = _read_upload_to_gray(await file.read())

    # The two models are traced separately so their inference cost and their
    # live agreement rate are both measurable. See app/telemetry.py.
    with telemetry.span("inspect.baseline") as baseline_span:
        baseline_result = _baseline.predict(gray)
        telemetry.set_attributes(
            baseline_span,
            **{
                "inspect.label": baseline_result.get("label"),
                "inspect.confidence": float(baseline_result.get("confidence", 0.0)),
            },
        )

    result: dict = {"baseline": baseline_result}

    if _deep and _deep.loaded:
        with telemetry.span("inspect.deep") as deep_span:
            pred = _deep.predict(gray)
            telemetry.set_attributes(
                deep_span,
                **{
                    "inspect.label": pred.label,
                    "inspect.confidence": float(pred.confidence),
                    # Disagreement between a 97.1% model and a 99.6% model is
                    # the cheapest available signal for images worth a human
                    # look -- and for drift, if the rate starts climbing.
                    "inspect.agrees_with_baseline":
                        pred.label == baseline_result.get("label"),
                },
            )
        result["deep"] = {
            "label": pred.label,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
        }
    else:
        result["deep"] = None
        result["deep_status"] = _deep.load_error if _deep else "Deep model not initialized"

    return result
