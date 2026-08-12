"""Legacy SentenceTransformer + Logistic Regression classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LegacyMLResult:
    category: str
    confidence: float
    explanation: str
    available: bool = True


class LegacySTLRClassifier:
    """Wraps the original all-MiniLM-L6-v2 + LogisticRegression pipeline."""

    def __init__(self) -> None:
        self._embedding_model = None
        self._clf = None
        self._loaded = False
        self._load_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._loaded

    def _ensure_loaded(self) -> None:
        if self._loaded or self._load_error is not None:
            return
        settings = get_settings()
        model_path = settings.resolve(settings.legacy_st_lr_model)
        if not model_path.exists():
            self._load_error = f"Legacy model not found: {model_path}"
            logger.warning(self._load_error)
            return
        try:
            import joblib
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._clf = joblib.load(model_path)
            self._loaded = True
            logger.info("Loaded legacy ST+LR model from %s", model_path)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.exception("Failed to load legacy ST+LR model: %s", exc)

    def classify(self, log_message: str, threshold: Optional[float] = None) -> LegacyMLResult:
        settings = get_settings()
        threshold = settings.legacy_st_lr_threshold if threshold is None else threshold
        self._ensure_loaded()
        if not self._loaded:
            return LegacyMLResult(
                category="Unclassified",
                confidence=0.0,
                explanation=f"Legacy ST+LR unavailable: {self._load_error}",
                available=False,
            )

        embeddings = self._embedding_model.encode([log_message])
        probabilities = self._clf.predict_proba(embeddings)[0]
        confidence = float(max(probabilities))
        predicted = self._clf.predict(embeddings)[0]
        if confidence < threshold:
            return LegacyMLResult(
                category="Unclassified",
                confidence=confidence,
                explanation=(
                    f"Legacy ST+LR confidence {confidence:.3f} below threshold {threshold}."
                ),
            )
        return LegacyMLResult(
            category=str(predicted),
            confidence=confidence,
            explanation=(
                f"Predicted by SentenceTransformer embeddings + Logistic Regression "
                f"(confidence={confidence:.3f})."
            ),
        )


def classify_with_bert(log_message: str) -> str:
    """Backward-compatible alias matching original processor_bert API name."""
    return LegacySTLRClassifier().classify(log_message).category
