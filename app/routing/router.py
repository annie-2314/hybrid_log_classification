"""Intelligent hybrid routing layer."""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from app.classifiers.bert_classifier import BertClassifier
from app.classifiers.legacy_st_lr import LegacySTLRClassifier
from app.classifiers.llm_classifier import LLMClassifier
from app.classifiers.regex_classifier import RegexClassifier
from app.core.config import get_settings
from app.core.logging import get_logger
from app.monitoring.metrics import get_metrics_collector, log_classification_event
from app.preprocessing import normalize_log_message
from app.schemas import ClassificationResult
from app.services.severity import map_severity

logger = get_logger(__name__)


class HybridRouter:
    """Route logs through Regex → Fine-tuned BERT → LLM fallback.

    Flow:
    1. Regex for known fixed patterns
    2. Fine-tuned BERT with configurable confidence threshold
    3. Optional legacy ST+LR if BERT unavailable
    4. LLM for low-confidence / complex cases
    """

    def __init__(self) -> None:
        self.regex = RegexClassifier()
        self.bert = BertClassifier()
        self.legacy = LegacySTLRClassifier()
        self.llm = LLMClassifier()
        self.metrics = get_metrics_collector()

    def classify(
        self,
        log_message: str,
        source: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ClassificationResult:
        settings = get_settings()
        request_id = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        routing_path: List[str] = []
        text = normalize_log_message(log_message)
        llm_invoked = False
        llm_latency = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        estimated_cost = 0.0
        error: Optional[str] = None

        try:
            if not text:
                raise ValueError("log_message is empty after preprocessing")

            # Optional source hint from original project: LegacyCRM often needs LLM
            prefer_llm = bool(source and source.strip().lower() == "legacycrm")

            # 1) Regex
            routing_path.append("regex")
            regex_result = self.regex.classify(text)
            if regex_result.category and not prefer_llm:
                method = "regex"
                category = regex_result.category
                confidence = regex_result.confidence
                explanation = regex_result.explanation
                severity = map_severity(category)
            else:
                # 2) Fine-tuned BERT
                method = None
                category = "Unclassified"
                confidence = 0.0
                explanation = ""
                severity = "UNKNOWN"

                if self.bert.is_available and not prefer_llm:
                    routing_path.append("fine_tuned_bert")
                    bert_result = self.bert.classify(text)
                    if (
                        bert_result.available
                        and bert_result.confidence >= settings.bert_confidence_threshold
                    ):
                        method = "fine_tuned_bert"
                        category = bert_result.category
                        confidence = bert_result.confidence
                        explanation = bert_result.explanation
                        severity = map_severity(category)
                    else:
                        routing_path.append("bert_low_confidence")
                        # keep bert suggestion for explanation context
                        if bert_result.available:
                            explanation = (
                                f"BERT confidence {bert_result.confidence:.3f} "
                                f"< threshold {settings.bert_confidence_threshold}; "
                                "falling back to LLM."
                            )
                elif settings.enable_legacy_st_lr and self.legacy.is_available and not prefer_llm:
                    routing_path.append("legacy_st_lr")
                    legacy_result = self.legacy.classify(text)
                    if (
                        legacy_result.available
                        and legacy_result.category != "Unclassified"
                        and legacy_result.confidence >= settings.legacy_st_lr_threshold
                    ):
                        method = "legacy_st_lr"
                        category = legacy_result.category
                        confidence = legacy_result.confidence
                        explanation = legacy_result.explanation
                        severity = map_severity(category)

                # 3) LLM fallback
                if method is None:
                    routing_path.append("llm_fallback")
                    llm_result = self.llm.classify(text)
                    llm_invoked = True
                    llm_latency = llm_result.latency_ms
                    prompt_tokens = llm_result.prompt_tokens
                    completion_tokens = llm_result.completion_tokens
                    estimated_cost = llm_result.estimated_cost_usd
                    method = "llm_fallback"
                    category = llm_result.category
                    confidence = llm_result.confidence
                    explanation = llm_result.explanation
                    severity = llm_result.severity
                    if not llm_result.success:
                        error = llm_result.error

            latency_ms = (time.perf_counter() - started) * 1000.0
            result = ClassificationResult(
                category=category,
                severity=severity,
                confidence=float(confidence),
                classification_method=method,
                explanation=explanation,
                request_id=request_id,
                latency_ms=round(latency_ms, 3),
                llm_invoked=llm_invoked,
                routing_path=routing_path,
            )
            self.metrics.record(
                method=method,
                latency_ms=latency_ms,
                llm_invoked=llm_invoked,
                llm_latency_ms=llm_latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost,
                error=bool(error),
            )
            log_classification_event(
                request_id=request_id,
                method=method,
                category=category,
                confidence=float(confidence),
                latency_ms=latency_ms,
                llm_fallback=llm_invoked,
                error=error,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("Classification failed for request_id=%s", request_id)
            self.metrics.record(
                method="error",
                latency_ms=latency_ms,
                error=True,
            )
            log_classification_event(
                request_id=request_id,
                method="error",
                category="Unclassified",
                confidence=0.0,
                latency_ms=latency_ms,
                llm_fallback=False,
                error=str(exc),
            )
            return ClassificationResult(
                category="Unclassified",
                severity="UNKNOWN",
                confidence=0.0,
                classification_method="error",
                explanation=f"Classification error: {exc}",
                request_id=request_id,
                latency_ms=round(latency_ms, 3),
                llm_invoked=False,
                routing_path=routing_path + ["error"],
            )


_router: Optional[HybridRouter] = None


def get_router() -> HybridRouter:
    global _router
    if _router is None:
        _router = HybridRouter()
    return _router
