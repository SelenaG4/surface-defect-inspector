"""Wraps the fine-tuned EfficientNet defect classifier, served through ONNX
Runtime -- and reports honestly when the model file isn't there yet, rather
than crashing the service.

Why ONNX instead of loading a PyTorch checkpoint: the fine-tuning happens in a
separate Colab notebook (this sandbox has no GPU and can't reach the model hub),
and the resulting EfficientNet, exported to ONNX, is ~20 MB and runs on the
lightweight `onnxruntime` package -- no torch, no torchvision. That is the whole
reason this project's deep model can run *live* on a free-tier host, unlike the
255 MB PyTorch DistilBERT in the news-topic-classifier project, which could only
run locally. Same graceful-degradation contract as that project's
transformer_classifier: if the weights aren't present, `loaded` is False and the
service still serves the baseline.

Preprocessing must match the notebook exactly: grayscale -> 3-channel, resized
to 224x224, scaled to [0,1], then ImageNet mean/std normalization (EfficientNet
was pretrained on ImageNet). If these drift apart, predictions silently degrade.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.labels import CLASS_NAMES, DEEP_IMAGE_SIZE

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.onnx"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class DeepPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def preprocess(image: np.ndarray) -> np.ndarray:
    """Image (grayscale HxW or BGR HxWx3) -> normalized (1,3,224,224) float32 tensor."""
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (DEEP_IMAGE_SIZE, DEEP_IMAGE_SIZE)).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
    return chw[np.newaxis, :, :, :].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class DeepDefectClassifier:
    def __init__(self, model_path: Path | None = None):
        # Resolve at call time (not as a default-arg binding) so tests and the
        # app can point this at a different file by patching MODEL_PATH.
        self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
        self.loaded = False
        self.load_error: str | None = None
        self._session = None
        self._input_name: str | None = None
        self._try_load()

    def _try_load(self) -> None:
        if not self.model_path.exists():
            self.load_error = (
                f"No ONNX model found at {self.model_path}. "
                "Run notebooks/finetune_efficientnet_neu.ipynb and copy model.onnx here."
            )
            return
        try:
            import onnxruntime as ort  # imported lazily so the app boots even if ORT is absent

            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            n_out = self._session.get_outputs()[0].shape[-1]
            if isinstance(n_out, int) and n_out != len(CLASS_NAMES):
                self.load_error = (
                    f"Model at {self.model_path} outputs {n_out} classes but this app expects "
                    f"{len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)}). Retrain/export to match."
                )
                self._session = None
                return
            self.loaded = True
        except Exception as exc:  # noqa: BLE001 -- any load failure should degrade, not crash
            self.load_error = f"Found {self.model_path} but failed to load it: {exc}"

    def predict(self, image: np.ndarray) -> DeepPrediction:
        if not self.loaded:
            raise RuntimeError(self.load_error or "Deep model not loaded")
        tensor = preprocess(image)
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        probs = _softmax(np.asarray(logits, dtype=np.float64))
        best = int(np.argmax(probs))
        return DeepPrediction(
            label=CLASS_NAMES[best],
            confidence=float(probs[best]),
            probabilities={CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        )
