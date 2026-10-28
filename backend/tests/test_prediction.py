import io
from pathlib import Path
import pytest
from PIL import Image
from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_prediction_with_real_image():
    # Pick a real image from dataset/train_images or generate a test image
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset" / "train_images"
    images = list(dataset_dir.glob("*.jpg")) if dataset_dir.exists() else []
    
    if images:
        with open(images[0], "rb") as f:
            file_bytes = f.read()
        filename = images[0].name
    else:
        # Create synthetic 256x1600 test image
        img = Image.new("RGB", (1600, 256), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        file_bytes = buf.getvalue()
        filename = "synthetic_test.jpg"

    response = client.post(
        "/api/predict",
        files={"file": (filename, file_bytes, "image/jpeg")},
        data={"threshold": "0.5"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == filename
    assert "original_image" in data
    assert "overlay_image" in data
    assert "heatmap_image" in data
    assert len(data["masks"]) == 4
    assert len(data["classes"]) == 4
    assert "any_defect_detected" in data
    assert data["threshold"] == 0.5
