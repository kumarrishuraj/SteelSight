from typing import List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "SteelSight API"
    version: str = "1.0.0"

class ModelStatusResponse(BaseModel):
    loaded: bool
    device: str
    epoch: Optional[int] = None
    best_loss: Optional[float] = None
    input_shape: List[int]
    classes: List[str]

class ClassDetectionStat(BaseModel):
    class_id: int
    name: str
    detected: bool
    pixels: int
    area_percent: float

class PredictionResponse(BaseModel):
    filename: str
    original_width: int
    original_height: int
    device: str
    threshold: float
    original_image: str  # Data URI base64
    overlay_image: str   # Data URI base64
    heatmap_image: str   # Data URI base64
    masks: List[str]     # 4 Data URI base64 masks
    classes: List[ClassDetectionStat]
    any_defect_detected: bool

class ErrorResponse(BaseModel):
    detail: str
