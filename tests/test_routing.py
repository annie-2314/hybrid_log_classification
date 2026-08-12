"""Tests for confidence-based routing logic."""

from app.classifiers.bert_classifier import BertResult
from app.classifiers.llm_classifier import LLMResult
from app.classifiers.regex_classifier import RegexMatchResult
from app.routing.router import HybridRouter


class _FakeRegex:
    def __init__(self, category=None):
        self.category = category

    def classify(self, _text):
        if self.category:
            return RegexMatchResult(self.category, 1.0, "matched")
        return RegexMatchResult(None, 0.0, "no match")


class _FakeBert:
    def __init__(self, available=True, category="Error", confidence=0.95):
        self._available = available
        self.category = category
        self.confidence = confidence

    @property
    def is_available(self):
        return self._available

    def classify(self, _text):
        return BertResult(
            category=self.category,
            confidence=self.confidence,
            explanation="fake bert",
            probabilities={self.category: self.confidence},
            available=self._available,
        )


class _FakeLLM:
    is_configured = True

    def classify(self, _text):
        return LLMResult(
            category="Workflow Error",
            severity="HIGH",
            confidence=0.75,
            explanation="fake llm",
            success=True,
            latency_ms=12.0,
            prompt_tokens=10,
            completion_tokens=5,
            estimated_cost_usd=0.0001,
        )


def test_router_uses_regex_first(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    router = HybridRouter()
    router.regex = _FakeRegex("User Action")
    router.bert = _FakeBert()
    router.llm = _FakeLLM()
    result = router.classify("User User1 logged in.")
    assert result.classification_method == "regex"
    assert result.category == "User Action"
    assert result.llm_invoked is False


def test_router_accepts_high_bert_confidence(monkeypatch):
    monkeypatch.setenv("BERT_CONFIDENCE_THRESHOLD", "0.90")
    # Clear cached settings
    from app.core.config import get_settings

    get_settings.cache_clear()
    router = HybridRouter()
    router.regex = _FakeRegex(None)
    router.bert = _FakeBert(confidence=0.94, category="Security Alert")
    router.llm = _FakeLLM()
    result = router.classify("Unauthorized access detected")
    assert result.classification_method == "fine_tuned_bert"
    assert result.category == "Security Alert"
    assert result.llm_invoked is False
    get_settings.cache_clear()


def test_router_falls_back_to_llm_on_low_confidence(monkeypatch):
    monkeypatch.setenv("BERT_CONFIDENCE_THRESHOLD", "0.90")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from app.core.config import get_settings

    get_settings.cache_clear()
    router = HybridRouter()
    router.regex = _FakeRegex(None)
    router.bert = _FakeBert(confidence=0.40, category="Error")
    router.llm = _FakeLLM()
    result = router.classify("Ambiguous strange log")
    assert result.classification_method == "llm_fallback"
    assert result.category == "Workflow Error"
    assert result.llm_invoked is True
    get_settings.cache_clear()
