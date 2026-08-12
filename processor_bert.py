"""Backward-compatible wrapper for the legacy ST+LR classifier."""

from app.classifiers.legacy_st_lr import classify_with_bert

__all__ = ["classify_with_bert"]
