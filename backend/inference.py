import sys
import io
import base64
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
import cv2
import numpy as np
from PIL import Image

from backend import config

# Ensure root workspace is in sys.path so model imports work
if str(config.BASE_DIR) not in sys.path:
    sys.path.insert(0, str(config.BASE_DIR))

from model.model import Unet

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

        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {config.MODEL_PATH}")

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
            str(config.MODEL_PATH),
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
