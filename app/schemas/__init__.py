"""Pydantic request/response schemas for IntelliLog AI."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single log classification request."""

    log_message: str = Field(..., min_length=1, description="Raw log line to classify")
    source: Optional[str] = Field(
        default=None,
        description="Optional log source system (e.g. LegacyCRM, ModernHR)",
    )

    @field_validator("log_message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("log_message must not be empty")
        return cleaned


class ClassificationResult(BaseModel):
    """Structured classification output."""

    category: str
    severity: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    classification_method: str
    explanation: str
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None
    llm_invoked: bool = False
    routing_path: Optional[List[str]] = None


class BatchPredictItem(BaseModel):
    log_message: str
    source: Optional[str] = None


class BatchPredictRequest(BaseModel):
    logs: List[BatchPredictItem] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    results: List[ClassificationResult]
    total: int
    avg_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    bert_loaded: bool
    legacy_st_lr_loaded: bool
    llm_configured: bool


class MetricsResponse(BaseModel):
    total_requests: int
    regex_requests: int
    bert_requests: int
    legacy_st_lr_requests: int
    llm_fallback_requests: int
    llm_invocation_count: int
    average_latency_ms: float
    llm_fallback_percentage: float
    estimated_llm_cost_usd: float
    total_llm_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int


class LLMStructuredOutput(BaseModel):
    """Validated LLM JSON payload."""

    category: str
    severity: str
    reason: str

    @field_validator("category", "severity", "reason")
    @classmethod
    def non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty")
        return cleaned
