import io
import pytest
from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_empty_file_upload():
    response = client.post(
        "/api/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_invalid_extension():
    response = client.post(
        "/api/predict",
        files={"file": ("malicious.exe", b"test content", "application/octet-stream")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()

def test_invalid_threshold_out_of_bounds():
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", b"fake", "image/jpeg")},
        data={"threshold": "1.5"}
    )
    assert response.status_code == 422
    assert "threshold" in response.json()["detail"].lower()

def test_corrupt_image():
    response = client.post(
        "/api/predict",
        files={"file": ("broken.jpg", b"not-an-image-data-string", "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code in [400, 500]
