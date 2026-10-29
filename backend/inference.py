import sys
import io
import os
import base64
import hashlib
import urllib.request
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import torch
import cv2
import numpy as np
from PIL import Image

from backend import config

# Ensure root workspace is in sys.path so model imports work
if str(config.BASE_DIR) not in sys.path:
    sys.path.insert(0, str(config.BASE_DIR))

from model.model import Unet

def verify_file_sha256(filepath: Path, expected_hash: str) -> bool:
    """Computes SHA-256 of a local file in chunks and compares with expected_hash."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    computed = sha256.hexdigest().lower()
    return computed == expected_hash.lower()

def ensure_model_checkpoint() -> Path:
    """
    Ensures model checkpoint exists at config.MODEL_PATH.
    If absent and config.MODEL_URL is set, downloads it securely (with optional auth header)
    and validates its SHA-256 checksum.
    """
    target_path = config.MODEL_PATH
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path

    if not config.MODEL_URL:
        raise FileNotFoundError(
            f"Model checkpoint not found at {target_path} and MODEL_URL is not configured. "
            "Please provide a local checkpoint or set the MODEL_URL environment variable."
        )

    print(f"Model checkpoint not found locally. Initiating download from MODEL_URL...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download_path = target_path.with_suffix(".tmp")

    req = urllib.request.Request(config.MODEL_URL)
    req.add_header("User-Agent", "SteelSight-ModelLoader/1.0")
    if config.MODEL_AUTH_TOKEN:
        # Supports Bearer token (Hugging Face / private storage)
        req.add_header("Authorization", f"Bearer {config.MODEL_AUTH_TOKEN}")

    try:
        with urllib.request.urlopen(req, timeout=180) as response, open(temp_download_path, "wb") as out_file:
            block_size = 1024 * 1024
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)

        # Validate checksum if configured
        if config.EXPECTED_MODEL_SHA256:
            print("Verifying downloaded model SHA-256 checksum...")
            if not verify_file_sha256(temp_download_path, config.EXPECTED_MODEL_SHA256):
                temp_download_path.unlink(missing_ok=True)
                raise ValueError("Downloaded checkpoint SHA-256 does not match expected checksum. File may be corrupted.")
            print("Checksum verified successfully.")

        temp_download_path.replace(target_path)
        print(f"Model successfully downloaded and saved to {target_path}")
        return target_path
    except Exception as e:
        if temp_download_path.exists():
            temp_download_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed downloading model checkpoint: {str(e)}") from e

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.epoch: Optional[int] = None
        self.best_loss: Optional[float] = None
        self.is_loaded: bool = False

    def load(self):
        if self.is_loaded:
            return

        checkpoint_path = ensure_model_checkpoint()

        # Instantiate architecture
        net = Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=4,
            activation=None
        )

        # Load weights
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False
        )
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        net.to(self.device)
        net.eval()

        self.model = net
        self.epoch = checkpoint.get("epoch")
        self.best_loss = checkpoint.get("best_loss")
        self.is_loaded = True

model_manager = ModelManager()

def np_to_base64_png(image_np: np.ndarray) -> str:
    """Converts RGB/RGBA/Grayscale uint8 numpy image to a base64 data URI."""
    if len(image_np.shape) == 2:
        img = Image.fromarray(image_np, mode="L")
    elif image_np.shape[2] == 3:
        img = Image.fromarray(image_np, mode="RGB")
    elif image_np.shape[2] == 4:
        img = Image.fromarray(image_np, mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape {image_np.shape}")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

# Distinct high-contrast colors for defect channels (RGB)
CLASS_COLORS = [
    (255, 65, 84),    # Class 1: Coral Red
    (69, 212, 154),   # Class 2: Mint Green
    (66, 184, 255),   # Class 3: Sky Blue
    (240, 180, 92)    # Class 4: Amber Yellow
]

def run_inference(image_bytes: bytes, filename: str, threshold: float = 0.5) -> Dict[str, Any]:
    if not model_manager.is_loaded:
        model_manager.load()

    # 1. Decode original image
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not decode image: {str(e)}")

    orig_w, orig_h = pil_img.size
    orig_np = np.array(pil_img)

    # 2. Resize to exact training resolution: (1600, 256) (Width x Height)
    resized_rgb = cv2.resize(orig_np, (config.INPUT_WIDTH, config.INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # 3. Normalize as per original notebook training pipeline
    scaled = resized_rgb.astype(np.float32) / 255.0
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(config.IMAGENET_STD, dtype=np.float32)
    norm_img = (scaled - mean) / std

    # 4. CHW & Batch Tensor
    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).to(model_manager.device)

    # 5. Forward pass
    with torch.no_grad():
        logits = model_manager.model(tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy() # Shape: (4, 256, 1600)

    # 6. Apply threshold to get binary masks
    binary_masks = (probabilities > threshold).astype(np.uint8) # (4, 256, 1600)

    # 7. Compute per-class stats
    class_stats = []
    any_detected = False
    masks_b64 = []

    for i in range(4):
        mask_ch = binary_masks[i]
        pix_count = int(np.sum(mask_ch))
        detected = pix_count > 0
        if detected:
            any_detected = True
        area_pct = float(pix_count / (config.INPUT_HEIGHT * config.INPUT_WIDTH) * 100.0)

        class_stats.append({
            "class_id": i + 1,
            "name": config.CLASS_NAMES[i],
            "detected": detected,
            "pixels": pix_count,
            "area_percent": round(area_pct, 4)
        })

        # Mask visual: 0 -> black, 255 -> white
        mask_visual = (mask_ch * 255).astype(np.uint8)
        masks_b64.append(np_to_base64_png(mask_visual))

    # 8. Create combined overlay on the resized RGB image
    overlay_rgb = resized_rgb.copy()
    for i in range(4):
        mask_ch = binary_masks[i]
        if np.sum(mask_ch) == 0:
            continue
        color = CLASS_COLORS[i]
        colored_layer = np.zeros_like(resized_rgb)
        colored_layer[mask_ch == 1] = color
        # Blend 45% alpha
        alpha = 0.45
        mask_bool = mask_ch == 1
        overlay_rgb[mask_bool] = (
            (1 - alpha) * overlay_rgb[mask_bool] + alpha * colored_layer[mask_bool]
        ).astype(np.uint8)

    # 9. Create Heatmap (Max over channels -> Inferno colormap)
    max_prob = np.max(probabilities, axis=0) # (256, 1600) float 0..1
    heatmap_gray = (np.clip(max_prob, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return {
        "filename": filename,
        "original_width": orig_w,
        "original_height": orig_h,
        "device": str(model_manager.device),
        "threshold": threshold,
        "original_image": np_to_base64_png(resized_rgb),
        "overlay_image": np_to_base64_png(overlay_rgb),
        "heatmap_image": np_to_base64_png(heatmap_rgb),
        "masks": masks_b64,
        "classes": class_stats,
        "any_defect_detected": any_detected
    }
