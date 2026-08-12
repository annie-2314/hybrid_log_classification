"""Production-oriented LLM fallback classifier (Groq)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas import LLMStructuredOutput

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "llm_classify.txt"
_JSON_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class LLMResult:
    category: str
    severity: str
    confidence: float
    explanation: str
    success: bool
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMClassifier:
    """Groq-backed LLM classifier with retries, timeout, and JSON validation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_retries = settings.llm_max_retries
        self.timeout_seconds = settings.llm_timeout_seconds
        self.api_key = settings.groq_api_key
        self.input_cost = settings.llm_input_cost_per_1m
        self.output_cost = settings.llm_output_cost_per_1m
        self._client = None
        self._prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text(encoding="utf-8")
        return (
            "Classify the log. Return JSON with category, severity, reason.\n"
            "Log message:\n{{LOG_MESSAGE}}"
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)
        return self._client

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1_000_000.0) * self.input_cost + (
            completion_tokens / 1_000_000.0
        ) * self.output_cost

    def _parse_response(self, content: str) -> LLMStructuredOutput:
        match = _JSON_RE.search(content)
        if not match:
            raise ValueError("LLM response did not contain JSON")
        payload = json.loads(match.group(0))
        return LLMStructuredOutput.model_validate(payload)

    def classify(self, log_message: str) -> LLMResult:
        if not log_message or not str(log_message).strip():
            return LLMResult(
                category="Unclassified",
                severity="UNKNOWN",
                confidence=0.0,
                explanation="Empty log message",
                success=False,
                error="empty_input",
            )

        if not self.is_configured:
            return LLMResult(
                category="Unclassified",
                severity="UNKNOWN",
                confidence=0.0,
                explanation="LLM not configured (missing GROQ_API_KEY)",
                success=False,
                error="missing_api_key",
            )

        prompt = self._prompt_template.replace("{{LOG_MESSAGE}}", log_message.strip())
        last_error: Optional[str] = None
        started = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                client = self._get_client()
                completion = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or ""
                parsed = self._parse_response(content)
                usage = getattr(completion, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                latency_ms = (time.perf_counter() - started) * 1000.0
                return LLMResult(
                    category=parsed.category,
                    severity=parsed.severity.upper(),
                    confidence=0.75,
                    explanation=parsed.reason,
                    success=True,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=self._estimate_cost(prompt_tokens, completion_tokens),
                    raw_response=content,
                    metadata={"attempt": attempt, "model": self.model},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning("LLM attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                time.sleep(min(0.5 * attempt, 2.0))

        latency_ms = (time.perf_counter() - started) * 1000.0
        return LLMResult(
            category="Unclassified",
            severity="UNKNOWN",
            confidence=0.0,
            explanation=f"LLM classification failed after retries: {last_error}",
            success=False,
            latency_ms=latency_ms,
            error=last_error,
        )


# Backward-compatible thin wrapper matching original API.
def classify_with_llm(log_msg: str) -> str:
    return LLMClassifier().classify(log_msg).category
