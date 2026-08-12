"""Pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.routing.router import HybridRouter
from app.classifiers.regex_classifier import RegexClassifier


@pytest.fixture()
def client(monkeypatch):
    """API client with LLM forced off for deterministic unit tests."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def regex_classifier():
    return RegexClassifier()


@pytest.fixture()
def router(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    # Fresh router instance
    return HybridRouter()
