"""HTTP client for the IntelliLog FastAPI backend."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class IntelliLogAPIError(RuntimeError):
    """Raised when the backend API call fails."""


class IntelliLogClient:
    """Thin wrapper around IntelliLog REST endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout_seconds: float = 60.0) -> None:
        self.base_url = (base_url or os.getenv("INTELLILOG_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            raise IntelliLogAPIError(
                f"Cannot reach IntelliLog API at {self.base_url}. Start the backend first."
            ) from exc

        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = str(payload.get("detail", payload))
            except ValueError:
                pass
            raise IntelliLogAPIError(f"API {response.status_code}: {detail}")
        return response

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health").json()

    def metrics(self) -> Dict[str, Any]:
        return self._request("GET", "/metrics").json()

    def predict(self, log_message: str, source: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"log_message": log_message}
        if source:
            body["source"] = source
        return self._request("POST", "/predict", json=body).json()

    def batch_predict(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", "/batch-predict", json={"logs": logs}).json()
