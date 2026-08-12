"""LLM classifier validation and error-path tests (no live API calls)."""

from app.classifiers.llm_classifier import LLMClassifier
from app.schemas import LLMStructuredOutput


def test_llm_structured_output_validation():
    obj = LLMStructuredOutput(
        category="Workflow Error",
        severity="HIGH",
        reason="Ticket escalation failed",
    )
    assert obj.category == "Workflow Error"


def test_llm_rejects_empty_fields():
    try:
        LLMStructuredOutput(category=" ", severity="HIGH", reason="x")
        assert False, "expected validation error"
    except Exception:
        assert True


def test_llm_missing_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    from app.core.config import get_settings

    get_settings.cache_clear()
    clf = LLMClassifier()
    result = clf.classify("Some log")
    assert result.success is False
    assert result.error == "missing_api_key"
    get_settings.cache_clear()


def test_llm_empty_input(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    from app.core.config import get_settings

    get_settings.cache_clear()
    clf = LLMClassifier()
    result = clf.classify("   ")
    assert result.success is False
    assert result.error == "empty_input"
    get_settings.cache_clear()
