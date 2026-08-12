"""In-memory + persisted runtime metrics for cost/latency observability."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RuntimeMetrics:
    total_requests: int = 0
    regex_requests: int = 0
    bert_requests: int = 0
    legacy_st_lr_requests: int = 0
    llm_fallback_requests: int = 0
    llm_invocation_count: int = 0
    total_latency_ms: float = 0.0
    total_llm_latency_ms: float = 0.0
    estimated_llm_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    errors: int = 0

    def to_public_dict(self) -> Dict[str, Any]:
        avg_latency = (
            self.total_latency_ms / self.total_requests if self.total_requests else 0.0
        )
        fallback_pct = (
            (self.llm_fallback_requests / self.total_requests) * 100.0
            if self.total_requests
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "regex_requests": self.regex_requests,
            "bert_requests": self.bert_requests,
            "legacy_st_lr_requests": self.legacy_st_lr_requests,
            "llm_fallback_requests": self.llm_fallback_requests,
            "llm_invocation_count": self.llm_invocation_count,
            "average_latency_ms": round(avg_latency, 3),
            "llm_fallback_percentage": round(fallback_pct, 3),
            "estimated_llm_cost_usd": round(self.estimated_llm_cost_usd, 6),
            "total_llm_latency_ms": round(self.total_llm_latency_ms, 3),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


class MetricsCollector:
    """Thread-safe metrics store with optional JSON persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = RuntimeMetrics()
        self._load()

    def _store_path(self) -> Path:
        settings = get_settings()
        path = settings.resolve(settings.metrics_store)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self) -> None:
        path = self._store_path()
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._metrics = RuntimeMetrics(**{
                k: data.get(k, getattr(RuntimeMetrics(), k))
                for k in asdict(RuntimeMetrics()).keys()
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load metrics store: %s", exc)

    def _persist(self) -> None:
        path = self._store_path()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self._metrics), handle, indent=2)

    def record(
        self,
        *,
        method: str,
        latency_ms: float,
        llm_invoked: bool = False,
        llm_latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        error: bool = False,
    ) -> None:
        with self._lock:
            m = self._metrics
            m.total_requests += 1
            m.total_latency_ms += latency_ms
            if error:
                m.errors += 1
            if method == "regex":
                m.regex_requests += 1
            elif method == "fine_tuned_bert":
                m.bert_requests += 1
            elif method == "legacy_st_lr":
                m.legacy_st_lr_requests += 1
            elif method in {"llm", "llm_fallback"}:
                m.llm_fallback_requests += 1

            if llm_invoked:
                m.llm_invocation_count += 1
                m.total_llm_latency_ms += llm_latency_ms
                m.total_prompt_tokens += prompt_tokens
                m.total_completion_tokens += completion_tokens
                m.estimated_llm_cost_usd += estimated_cost_usd
            self._persist()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._metrics.to_public_dict()

    def reset(self) -> None:
        with self._lock:
            self._metrics = RuntimeMetrics()
            self._persist()


_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def log_classification_event(
    *,
    request_id: str,
    method: str,
    category: str,
    confidence: float,
    latency_ms: float,
    llm_fallback: bool,
    error: Optional[str] = None,
) -> None:
    """Emit a structured classification log line."""
    payload = {
        "event": "classification",
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classification_method": method,
        "predicted_category": category,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 3),
        "llm_fallback": llm_fallback,
        "error": error,
    }
    logger.info("classification_event %s", json.dumps(payload))
