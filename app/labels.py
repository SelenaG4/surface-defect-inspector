"""Single source of truth for the six NEU surface-defect classes.

The classical baseline (`app/baseline.py`), the deep model wrapper
(`app/deep_model.py`), the API (`app/main.py`), the training notebook, and the
evaluation script all need to agree on the class order -- the ONNX model's
output logits are ordered, so if any two components disagreed on which index
means "scratches" the predictions would silently be wrong. Import CLASS_NAMES
from here everywhere instead of re-hardcoding the list.

Order matches the alphabetical folder order of the NEU-CLS dataset, which is
also the order the training notebook feeds to the model -- keep them in sync.
"""
from __future__ import annotations

# The six defect types in the NEU (Northeastern University) surface-defect
# benchmark of hot-rolled steel strip, in the canonical alphabetical order.
CLASS_NAMES: list[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Human-readable labels for the UI (the raw folder names are a bit terse).
DISPLAY_NAMES: dict[str, str] = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted surface",
    "rolled-in_scale": "Rolled-in scale",
    "scratches": "Scratches",
}

# NEU images are 200x200 grayscale; the deep model is trained at 224x224 RGB
# (ImageNet-pretrained EfficientNet input). Both sizes live here so the
# preprocessing in baseline/deep/eval never drifts apart.
BASELINE_IMAGE_SIZE = 200
DEEP_IMAGE_SIZE = 224
