import json
import os
import pytest
from app import app as flask_app  # imports the Flask app instance


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("status") == "ok"


def test_predict_spam(client):
    payload = {"text": "Congratulations! You have won free entry to a prize!"}
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert "prediction" in data
    assert "label" in data
    # label should be 0 or 1
    assert data["label"] in (0, 1)
    # confidence present (might be None if model doesn't support predict_proba)
    assert "confidence" in data
