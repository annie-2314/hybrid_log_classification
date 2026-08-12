"""API validation and health/metrics tests."""

import io


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "bert_loaded" in body


def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "total_requests" in body
    assert "llm_fallback_percentage" in body


def test_predict_validation_empty(client):
    response = client.post("/predict", json={"log_message": "   "})
    assert response.status_code == 422


def test_predict_ok(client, monkeypatch):
    # Ensure deterministic path via regex
    response = client.post(
        "/predict",
        json={"log_message": "Backup completed successfully."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "System Notification"
    assert body["classification_method"] == "regex"
    assert "severity" in body
    assert "confidence" in body
    assert "explanation" in body


def test_batch_predict(client):
    response = client.post(
        "/batch-predict",
        json={
            "logs": [
                {"log_message": "Backup completed successfully."},
                {"log_message": "User User42 logged out."},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["results"]) == 2


def test_batch_csv(client):
    csv_content = "source,log_message\nModernHR,Backup completed successfully.\n"
    response = client.post(
        "/batch-predict/csv",
        files={"file": ("logs.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 200
    assert "predicted_category" in response.text


def test_batch_csv_invalid(client):
    response = client.post(
        "/batch-predict/csv",
        files={"file": ("logs.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
