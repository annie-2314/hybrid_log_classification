"""Backward-compatible entrypoints preserving original module names."""

from app.classifiers.regex_classifier import classify_with_regex

__all__ = ["classify_with_regex"]
