import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pth"
MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_THRESHOLD = 0.5
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95

INPUT_HEIGHT = 256
INPUT_WIDTH = 1600
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    "Defect Class 1",
    "Defect Class 2",
    "Defect Class 3",
    "Defect Class 4"
]

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
).split(",")
