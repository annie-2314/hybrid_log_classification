"""Dataset pipeline tests."""

from app.preprocessing import normalize_log_message
from app.preprocessing.dataset import analyze_class_distribution, load_raw_dataset


def test_normalize_collapses_whitespace():
    assert normalize_log_message("  hello\t\nworld  ") == "hello world"


def test_load_and_analyze_dataset():
    df = load_raw_dataset()
    assert len(df) > 0
    analysis = analyze_class_distribution(df)
    assert analysis["num_classes"] >= 2
    assert "counts" in analysis
