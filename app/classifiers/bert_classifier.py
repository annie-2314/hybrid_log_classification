"""Fine-tuned Transformer classifier (DistilBERT/BERT) inference."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _import_hf_auto_classes() -> Tuple[Any, Any]:
    """Import Auto classes in a Transformers v4/v5 compatible way.

    Transformers 5 uses lazy exports; a top-level import can fail during
    concurrent first-load (e.g. uvicorn --reload + /health). Explicit
    submodule imports are more reliable.
    """
    try:
        from transformers.models.auto.modeling_auto import (
            AutoModelForSequenceClassification,
        )
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        return AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        return AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class BertResult:
    category: str
    confidence: float
    explanation: str
    probabilities: Dict[str, float]
    available: bool = True


class BertClassifier:
    """Loads a Hugging Face sequence-classification checkpoint for inference."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.model_dir = Path(model_dir) if model_dir else settings.resolve(settings.bert_model_dir)
        self.max_length = settings.bert_max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._id2label: Dict[int, str] = {}
        self._loaded = False
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._loaded

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            self._load_error = f"Fine-tuned BERT checkpoint not found in {self.model_dir}"
            logger.warning(self._load_error)
            return

        with self._lock:
            if self._loaded:
                return
            try:
                AutoModelForSequenceClassification, AutoTokenizer = _import_hf_auto_classes()
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self._model.to(self.device)
                self._model.eval()

                label_map_path = self.model_dir / "label_map.json"
                if label_map_path.exists():
                    with label_map_path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    self._id2label = {int(k): v for k, v in payload.get("id2label", {}).items()}
                else:
                    self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}

                self._loaded = True
                self._load_error = None
                logger.info("Loaded fine-tuned BERT from %s on %s", self.model_dir, self.device)
            except Exception as exc:  # noqa: BLE001
                # Do not freeze on transient import races; allow the next call to retry.
                self._load_error = str(exc)
                logger.exception("Failed to load BERT classifier: %s", exc)

    def classify(self, log_message: str) -> BertResult:
        self._ensure_loaded()
        if not self._loaded:
            return BertResult(
                category="Unclassified",
                confidence=0.0,
                explanation=f"BERT unavailable: {self._load_error}",
                probabilities={},
                available=False,
            )

        encoded = self._tokenizer(
            log_message,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probs = F.softmax(logits, dim=-1)

        confidence, pred_id = torch.max(probs, dim=-1)
        category = self._id2label.get(int(pred_id.item()), str(int(pred_id.item())))
        prob_map = {
            self._id2label.get(i, str(i)): float(probs[i].item())
            for i in range(probs.shape[0])
        }
        conf = float(confidence.item())
        return BertResult(
            category=category,
            confidence=conf,
            explanation=(
                f"Fine-tuned Transformer predicted '{category}' "
                f"with softmax confidence {conf:.3f}."
            ),
            probabilities=prob_map,
        )

    def predict_batch(self, messages: List[str]) -> List[BertResult]:
        return [self.classify(msg) for msg in messages]
