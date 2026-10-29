import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
def _get_config_value(key: str, default: str = "") -> str:
    """Safely retrieves configuration from Streamlit Secrets or Environment Variables."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass
    val = os.getenv(key)
    if val is not None and val.strip():
        return val.strip()
    return default

DEFAULT_MODEL_PATH = BASE_DIR / "model" / "model.pth"
MODEL_PATH = Path(_get_config_value("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

DEFAULT_MODEL_URL = "https://github.com/kumarrishuraj/SteelSight/releases/download/v1.0.0/model.pth"
MODEL_URL = _get_config_value("MODEL_URL", DEFAULT_MODEL_URL)

MODEL_AUTH_TOKEN = _get_config_value("MODEL_AUTH_TOKEN", "")

DEFAULT_MODEL_SHA256 = "076f04466dd37ae0b6e9a9e4bd61810921a9f92996b04ef5c966d827b9eaefe8"
EXPECTED_MODEL_SHA256 = _get_config_value("EXPECTED_MODEL_SHA256", DEFAULT_MODEL_SHA256).lower()

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

# CORS allowed origins (configurable via ALLOWED_ORIGINS env)
raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
)
ALLOWED_ORIGINS = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]
