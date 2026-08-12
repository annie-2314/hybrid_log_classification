"""Tests for regex classification."""

from app.classifiers.regex_classifier import RegexClassifier, classify_with_regex


def test_regex_user_action():
    clf = RegexClassifier()
    result = clf.classify("User User123 logged in.")
    assert result.category == "User Action"
    assert result.confidence == 1.0


def test_regex_system_notification():
    assert classify_with_regex("Backup completed successfully.") == "System Notification"


def test_regex_no_match():
    clf = RegexClassifier()
    result = clf.classify("Something completely unexpected happened.")
    assert result.category is None
    assert result.confidence == 0.0
