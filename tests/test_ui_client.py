"""Tests for the Streamlit API client."""

from unittest.mock import MagicMock, patch

import pytest

from ui.api_client import IntelliLogAPIError, IntelliLogClient


def test_predict_posts_json():
    client = IntelliLogClient(base_url="http://example.test")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "category": "System Notification",
        "severity": "LOW",
        "confidence": 1.0,
        "classification_method": "regex",
        "explanation": "matched",
        "llm_invoked": False,
    }

    with patch("ui.api_client.httpx.Client") as mock_cls:
        mock_cls.return_value.__enter__.return_value.request.return_value = mock_response
        result = client.predict("Backup completed successfully.")

    assert result["category"] == "System Notification"
    call_kwargs = mock_cls.return_value.__enter__.return_value.request.call_args
    assert call_kwargs.args[0] == "POST"
    assert call_kwargs.args[1].endswith("/predict")


def test_health_connection_error():
    import httpx

    client = IntelliLogClient(base_url="http://127.0.0.1:9")
    with patch("ui.api_client.httpx.Client") as mock_cls:
        mock_cls.return_value.__enter__.return_value.request.side_effect = httpx.ConnectError("fail")
        with pytest.raises(IntelliLogAPIError):
            client.health()


def test_api_error_status():
    client = IntelliLogClient(base_url="http://example.test")
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "bad"
    mock_response.json.return_value = {"detail": "File must be a CSV."}
    with patch("ui.api_client.httpx.Client") as mock_cls:
        mock_cls.return_value.__enter__.return_value.request.return_value = mock_response
        with pytest.raises(IntelliLogAPIError, match="400"):
            client.metrics()
