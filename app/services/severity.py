"""Severity mapping helpers."""

from __future__ import annotations

from app.core.config import get_settings


def map_severity(category: str) -> str:
    """Map a predicted category to a severity level via config."""
    settings = get_settings()
    return settings.severity_map.get(category, settings.severity_map.get("Unclassified", "UNKNOWN"))
