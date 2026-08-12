"""Regex classifier for predictable log patterns (preserved & improved)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple


@dataclass
class RegexMatchResult:
    category: Optional[str]
    confidence: float
    explanation: str
    matched_pattern: Optional[str] = None


class RegexClassifier:
    """Rule-based classifier for fixed/predictable log patterns."""

    def __init__(self, patterns: Optional[Dict[str, str]] = None) -> None:
        # Preserve original patterns from processor_regex.py and add a few safe variants.
        self._patterns: List[Tuple[Pattern[str], str, str]] = []
        raw = patterns or {
            r"User User\d+ logged (in|out).": "User Action",
            r"Backup (started|ended) at .*": "System Notification",
            r"Backup completed successfully.": "System Notification",
            r"System updated to version .*": "System Notification",
            r"File .* uploaded successfully by user .*": "System Notification",
            r"Disk cleanup completed successfully.": "System Notification",
            r"System reboot initiated by user .*": "System Notification",
            r"Account with ID .* created by .*": "User Action",
        }
        for pattern, label in raw.items():
            compiled = re.compile(pattern)
            explanation = f"Matched regex pattern for '{label}'."
            self._patterns.append((compiled, label, explanation))

    def classify(self, log_message: str) -> RegexMatchResult:
        for compiled, label, explanation in self._patterns:
            if compiled.search(log_message):
                return RegexMatchResult(
                    category=label,
                    confidence=1.0,
                    explanation=explanation,
                    matched_pattern=compiled.pattern,
                )
        return RegexMatchResult(
            category=None,
            confidence=0.0,
            explanation="No regex pattern matched.",
        )


# Backward-compatible function used by legacy scripts/tests.
_default_regex = RegexClassifier()


def classify_with_regex(log_message: str) -> Optional[str]:
    result = _default_regex.classify(log_message)
    return result.category
