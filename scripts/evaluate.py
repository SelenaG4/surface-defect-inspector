"""Independently reproduce the baseline-vs-EfficientNet comparison on the same
held-out split and log it to MLflow -- the "measured, not assumed" step this
portfolio applies to every model (see swiss-claims-assistant,
bedding-franchise-erp, news-topic-classifier).

It does NOT trust the numbers the Colab notebook printed. It re-splits the real
NEU data with a fixed seed, trains the classical baseline on the train portion,
runs the fine-tuned EfficientNet (ONNX) over the *same* test portion, and reports
both side by side -- so the comparison is genuinely apples-to-apples on one split.

Requires:
  - the real NEU data (scripts/fetch_data.py), laid out as <data-dir>/<class>/*
  - models/model.onnx (from the Colab notebook)
Degrades with a clear message (and logs nothing) if the ONNX model is absent, so
a checkout without it doesn't leave a misleading MLflow run behind.

    python scripts/evaluate.py --data-dir data/NEU-CLS
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import baseline  # noqa: E402
from app.deep_model import DeepDefectClassifier  # noqa: E402
from app.labels import CLASS_NAMES  # noqa: E402

DOCS = ROOT / "docs"
SEED = 42


def _metrics(y_true, y_pred) -> dict:
    order = list(range(len(CLASS_NAMES)))
    per = f1_score(y_true, y_pred, labels=order, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {CLASS_NAMES[c]: float(f) for c, f in zip(order, per)},
        "cm": confusion_matrix(y_true, y_pred, labels=order),
    }


def _plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    DOCS.mkdir(exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_comparison(bm, dm, path):
    x = range(len(CLASS_NAMES)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar([i - w/2 for i in x], [bm["per_class_f1"][c] for c in CLASS_NAMES], w, label="Baseline (LBP + SVM)")
    b2 = ax.bar([i + w/2 for i in x], [dm["per_class_f1"][c] for c in CLASS_NAMES], w, label="Fine-tuned EfficientNet")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.002, f"{r.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    # Both models score high on NEU, so the axis starts at 0.85 to make the gap
    # legible; exact values are labelled on every bar so nothing is hidden.
    ax.set_ylabel("F1 score"); ax.set_ylim(0.85, 1.02)
    ax.set_title("Per-class F1: baseline vs. fine-tuned CNN (NEU surface-defect test set)")
    ax.legend(loc="lower right"); fig.tight_layout()
    DOCS.mkdir(exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    deep = DeepDefectClassifier()
    if not deep.loaded:
        print(f"Deep model not loaded -- {deep.load_error}")
        print("Nothing to compare or log. Run the notebook, copy model.onnx into models/, then retry. "
              "(No MLflow run started.)")
        return

    X, y, paths = baseline.load_dataset(Path(args.data_dir))
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)

    print("Training classical baseline on the train split...")
    pipe = baseline._build_pipeline()
    pipe.fit(X[tr], y[tr])
    bm = _metrics(y[te], pipe.predict(X[te]))

    print("Running the fine-tuned EfficientNet over the same test split...")
    deep_pred = [CLASS_NAMES.index(deep.predict(cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)).label) for i in te]
    dm = _metrics(y[te], deep_pred)

    d_acc = dm["accuracy"] - bm["accuracy"]; d_f1 = dm["macro_f1"] - bm["macro_f1"]
    print("=" * 60)
    print(f"{'Metric':<15}{'Baseline':>12}{'EfficientNet':>15}{'Delta':>12}")
    print("=" * 60)
    print(f"{'Accuracy':<15}{bm['accuracy']:>12.4f}{dm['accuracy']:>15.4f}{d_acc:>+12.4f}")
    print(f"{'Macro F1':<15}{bm['macro_f1']:>12.4f}{dm['macro_f1']:>15.4f}{d_f1:>+12.4f}")
    for c in CLASS_NAMES:
        print(f"  {c:16} baseline={bm['per_class_f1'][c]:.4f}  cnn={dm['per_class_f1'][c]:.4f}")
    print("=" * 60)

    _plot_confusion(bm["cm"], "Baseline (LBP+SVM) confusion", DOCS / "cm_baseline.png")
    _plot_confusion(dm["cm"], "EfficientNet confusion", DOCS / "cm_efficientnet.png")
    _plot_comparison(bm, dm, DOCS / "baseline_vs_cnn.png")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("neu_surface_defect")
    with mlflow.start_run(run_name="baseline_vs_efficientnet"):
        mlflow.log_params({"baseline": "LBP(2-scale)+SVM(rbf)", "deep": "EfficientNet (fine-tuned, ONNX)",
                           "n_test": len(te), "seed": SEED})
        mlflow.log_metrics({
            "baseline_accuracy": bm["accuracy"], "baseline_macro_f1": bm["macro_f1"],
            "cnn_accuracy": dm["accuracy"], "cnn_macro_f1": dm["macro_f1"],
            "delta_accuracy": d_acc, "delta_macro_f1": d_f1,
            **{f"baseline_f1_{c}": v for c, v in bm["per_class_f1"].items()},
            **{f"cnn_f1_{c}": v for c, v in dm["per_class_f1"].items()},
        })
        for p in ["baseline_vs_cnn.png", "cm_baseline.png", "cm_efficientnet.png"]:
            mlflow.log_artifact(str(DOCS / p))
    print("Logged to MLflow (sqlite:///mlflow.db, experiment 'neu_surface_defect').")
    print("Charts saved under docs/.")


if __name__ == "__main__":
    main()
