"""The classical baseline every deep model has to actually beat before
"we fine-tuned a CNN" earns its complexity: Local Binary Pattern texture
features + a support-vector machine.

LBP is not a strawman here -- it's the canonical, decades-old method for exactly
this kind of task (classifying surface *texture*), so it's a genuinely fair
yardstick, the same role as the TF-IDF baseline in news-topic-classifier or the
greedy heuristic in bedding-franchise-erp. Surface defects (cracks, pits,
streaks, scratches) are texture patterns, and LBP histograms are rotation- and
illumination-robust texture descriptors, so the baseline is expected to do
respectably -- which is what makes "does the CNN actually beat it, and by how
much?" a real question rather than a foregone conclusion.

Feature vector: uniform-LBP histograms at two radii (P=8/R=1 and P=24/R=3),
concatenated -> 10 + 26 = 36 dims, then standardized and fed to an RBF SVM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.labels import BASELINE_IMAGE_SIZE, CLASS_NAMES

# Two (radius, n_points) settings -> multi-scale texture description.
_LBP_SETTINGS = [(1, 8), (3, 24)]


def lbp_features(gray: np.ndarray) -> np.ndarray:
    """Concatenated, L1-normalized uniform-LBP histograms for one grayscale image."""
    if gray.shape != (BASELINE_IMAGE_SIZE, BASELINE_IMAGE_SIZE):
        gray = cv2.resize(gray, (BASELINE_IMAGE_SIZE, BASELINE_IMAGE_SIZE))
    feats = []
    for radius, n_points in _LBP_SETTINGS:
        codes = local_binary_pattern(gray, n_points, radius, method="uniform")
        n_bins = n_points + 2  # uniform LBP -> P+2 distinct codes
        hist, _ = np.histogram(codes.ravel(), bins=n_bins, range=(0, n_bins), density=False)
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-7
        feats.append(hist)
    return np.concatenate(feats)


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Load every image under data_dir/<class_name>/*, returning (features, labels, paths).

    Labels are integer class indices matching CLASS_NAMES order.
    """
    X, y, paths = [], [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.bmp")) + sorted(cls_dir.glob("*.jpg")):
            X.append(lbp_features(_read_gray(p)))
            y.append(idx)
            paths.append(p)
    if not X:
        raise FileNotFoundError(
            f"No images found under {data_dir} (expected {data_dir}/<class_name>/*.png). "
            "Run scripts/fetch_data.py for the real NEU data, or "
            "scripts/make_synthetic_sample.py for a synthetic stand-in."
        )
    return np.asarray(X), np.asarray(y), paths


@dataclass
class BaselineMetrics:
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    n_train: int
    n_test: int


@dataclass
class BaselineResult:
    pipeline: Pipeline = field(repr=False)
    metrics: BaselineMetrics

    def predict(self, gray: np.ndarray) -> dict:
        feats = lbp_features(gray).reshape(1, -1)
        proba = self.pipeline.predict_proba(feats)[0]
        order = np.argsort(-proba)
        return {
            "label": CLASS_NAMES[int(order[0])],
            "confidence": float(proba[order[0]]),
            "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
        }


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)),
        ]
    )


def load_baseline(path: Path) -> BaselineResult:
    """Load a baseline saved by scripts/train_baseline.py back into a BaselineResult.

    Returns None-free: raises if the file is missing so the caller can degrade
    gracefully with a clear message (see app/main.py startup).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No baseline model at {path}. Run scripts/train_baseline.py "
            "(on real NEU data via scripts/fetch_data.py, or on the synthetic stand-in)."
        )
    blob = joblib.load(path)
    return BaselineResult(pipeline=blob["pipeline"], metrics=BaselineMetrics(**blob["metrics"]))


def train_and_evaluate(data_dir: Path, test_size: float = 0.25, seed: int = 42) -> BaselineResult:
    X, y, _ = load_dataset(data_dir)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    pipe = _build_pipeline()
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    class_order = list(range(len(CLASS_NAMES)))
    per_class = f1_score(y_te, pred, labels=class_order, average=None, zero_division=0)
    metrics = BaselineMetrics(
        accuracy=float(accuracy_score(y_te, pred)),
        macro_f1=float(f1_score(y_te, pred, average="macro", zero_division=0)),
        per_class_f1={CLASS_NAMES[c]: float(f) for c, f in zip(class_order, per_class)},
        confusion_matrix=confusion_matrix(y_te, pred, labels=class_order).tolist(),
        n_train=len(y_tr),
        n_test=len(y_te),
    )
    return BaselineResult(pipeline=pipe, metrics=metrics)
