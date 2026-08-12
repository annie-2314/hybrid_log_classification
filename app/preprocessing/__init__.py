"""Text preprocessing utilities for log messages."""

from __future__ import annotations

import re
from typing import Optional


_WHITESPACE_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_log_message(text: Optional[str]) -> str:
    """Normalize a raw log string for classification.

    Steps:
    - coerce to string
    - strip control characters
    - collapse whitespace
    - trim
    """
    if text is None:
        return ""
    value = str(text)
    value = _CTRL_RE.sub(" ", value)
    value = value.replace("\u00a0", " ")
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def preprocess_for_model(text: str, lowercase: bool = False) -> str:
    """Apply model-facing preprocessing on top of normalization."""
    value = normalize_log_message(text)
    if lowercase:
        value = value.lower()
    return value
