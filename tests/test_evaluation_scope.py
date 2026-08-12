"""Tests for fair/scoped evaluation helpers."""

import pandas as pd

from training.evaluate import _subset_regex_scope, eval_regex


def test_regex_scoped_accuracy_is_high_on_regex_logs():
    df = pd.read_csv("training/data/splits/test.csv")
    result = eval_regex(df)
    assert result["eval_scope"] == "regex_applicable_subset"
    assert result["n_samples"] > 0
    # Regex is designed for these rows and should be near-perfect.
    assert result["accuracy"] >= 0.95
    # Full-set reference remains available and is lower.
    ref = result["full_test_set_reference"]
    assert ref["accuracy"] < result["accuracy"]


def test_regex_subset_helper():
    df = pd.read_csv("training/data/splits/test.csv")
    scoped = _subset_regex_scope(df)
    assert len(scoped) > 0
    assert set(scoped["complexity"].str.lower().unique()) == {"regex"}
