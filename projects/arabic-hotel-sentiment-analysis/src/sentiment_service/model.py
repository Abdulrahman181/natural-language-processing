from pathlib import Path
from typing import Any
import joblib
import numpy as np

from .config import settings

LABELS = ["negative", "neutral", "positive"]

class SentimentModel:
    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path or settings.model_path)
        self.pipeline: Any | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

    def predict_one(self, text: str) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded")
        label = str(self.pipeline.predict([text])[0])
        probabilities = self._probabilities(text, label)
        return {
            "sentiment": label,
            "confidence": max(probabilities.values()),
            "probabilities": probabilities,
            "model_version": settings.model_version,
        }

    def _probabilities(self, text: str, label: str) -> dict[str, float]:
        # LinearSVC has no predict_proba; calibrated softmax over decision scores is a
        # transparent confidence proxy, not a probabilistic guarantee.
        if hasattr(self.pipeline, "decision_function"):
            scores = np.asarray(self.pipeline.decision_function([text]))[0]
            scores = scores - scores.max()
            exp = np.exp(scores)
            probs = exp / exp.sum()
            classes = list(getattr(self.pipeline, "classes_", LABELS))
            return {str(k): float(v) for k, v in zip(classes, probs)}
        return {x: (1.0 if x == label else 0.0) for x in LABELS}

model = SentimentModel()
