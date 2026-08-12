"""BERT classifier load/inference guard tests."""

from pathlib import Path

from app.classifiers.bert_classifier import BertClassifier


def test_bert_unavailable_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("BERT_MODEL_DIR", str(tmp_path / "missing"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    clf = BertClassifier(model_dir=tmp_path / "missing")
    result = clf.classify("test log")
    assert result.available is False
    assert result.category == "Unclassified"
    get_settings.cache_clear()


def test_bert_loads_if_checkpoint_exists():
    """If a trained checkpoint exists locally, smoke-test inference."""
    clf = BertClassifier()
    if not clf.is_available:
        # Not a failure — training may not have been run yet.
        return
    result = clf.classify("Unauthorized access to data was attempted")
    assert result.available is True
    assert result.confidence >= 0.0
    assert result.category
