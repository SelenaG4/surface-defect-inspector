"""Fetch the real NEU (Northeastern University) surface-defect dataset.

The build sandbox this project was created in cannot reach the dataset host
(same network constraint that pushed the CNN fine-tuning into Colab), so this
script is meant to be run on your own machine or in Colab, where it works.

The NEU-CLS classification set is 1,800 grayscale images (200x200), 300 per
class, across the six defect types in app/labels.py. The most reliable source
is Kaggle.

    # one-time: put your Kaggle API token at ~/.kaggle/kaggle.json, then:
    pip install kagglehub
    python scripts/fetch_data.py --out data/NEU-CLS

The layout this project expects afterwards is data/<class_name>/*.png (or the
raw NEU .bmp files with Cr_/In_/Pa_/PS_/RS_/Sc_ prefixes -- app/baseline.py's
loader handles the class subfolder form; the notebook handles the prefix form).
If you already have the data, just point train_baseline.py / evaluate.py at it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/NEU-CLS")
    args = ap.parse_args()

    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub not installed. Run: pip install kagglehub  (then re-run).")

    print("Downloading NEU surface-defect database from Kaggle...")
    src = Path(kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database"))
    print("Downloaded to:", src)
    print(
        "\nThe download's exact folder layout varies by mirror. Inspect it and, if needed, "
        "arrange images as data/<class_name>/*.png before running scripts/train_baseline.py.\n"
        f"(Raw files are also usable directly by pointing tools at {src}.)"
    )
    out = Path(args.out)
    if not out.exists():
        print(f"Tip: copy or symlink the class folders into {out} for the default paths to work.")


if __name__ == "__main__":
    main()
