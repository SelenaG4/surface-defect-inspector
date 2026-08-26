"""Train the classical LBP+SVM baseline and save it as a small joblib artifact.

Run this on the real NEU data (after scripts/fetch_data.py) to produce the
baseline that ships in the app:

    python scripts/train_baseline.py --data-dir data/NEU-CLS

Or on the synthetic stand-in, just to exercise the pipeline:

    python scripts/make_synthetic_sample.py --out data/synthetic --per-class 60
    python scripts/train_baseline.py --data-dir data/synthetic

The saved artifact (models/baseline_lbp_svm.joblib) is tiny -- an SVM over
36-dim LBP histograms -- so it is committed to the repo and loaded at startup,
rather than retrained on every boot (unlike a fast TF-IDF baseline, extracting
LBP over a few thousand images takes long enough that you don't want it in the
request path).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import baseline  # noqa: E402
from app.labels import CLASS_NAMES  # noqa: E402

DEFAULT_OUT = ROOT / "models" / "baseline_lbp_svm.joblib"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with <class_name>/*.png subfolders")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    result = baseline.train_and_evaluate(Path(args.data_dir))
    m = result.metrics

    print(f"Trained LBP+SVM baseline on {args.data_dir}")
    print(f"  train / test images: {m.n_train} / {m.n_test}")
    print(f"  accuracy : {m.accuracy:.4f}")
    print(f"  macro F1 : {m.macro_f1:.4f}")
    for c in CLASS_NAMES:
        print(f"    {c:18} F1={m.per_class_f1[c]:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": result.pipeline,
            "metrics": m.__dict__,
            "class_names": CLASS_NAMES,
        },
        out,
    )
    print(f"Saved baseline -> {out}")


if __name__ == "__main__":
    main()
