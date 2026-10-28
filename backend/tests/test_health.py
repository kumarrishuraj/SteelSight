import pytest
from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "SteelSight API"

def test_model_status():
    response = client.get("/api/model-status")
    assert response.status_code == 200
    data = response.json()
    assert data["loaded"] is True
    assert data["input_shape"] == [256, 1600]
    assert len(data["classes"]) == 4
    assert data["epoch"] == 18
