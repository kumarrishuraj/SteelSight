import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config
from backend.schemas import (
    HealthResponse,
    ModelStatusResponse,
    PredictionResponse,
    ErrorResponse
)
from backend.inference import model_manager, run_inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model once
    try:
        model_manager.load()
        print(f"Model successfully loaded on {model_manager.device}")
    except Exception as e:
        print(f"Warning: Model could not be loaded at startup: {e}")
    yield

# Auto-load on import if not loaded to support test clients and worker initialization
try:
    if not model_manager.is_loaded:
        model_manager.load()
except Exception as e:
    print(f"Initial model load deferred: {e}")

app = FastAPI(
    title="SteelSight API",
    description="AI-Powered Industrial Steel Surface Defect Segmentation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits local development ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check endpoint"
)
async def health_check():
    return HealthResponse()

@app.get(
    "/api/model-status",
    response_model=ModelStatusResponse,
    summary="Model status and metadata"
)
async def model_status():
    return ModelStatusResponse(
        loaded=model_manager.is_loaded,
        device=str(model_manager.device),
        epoch=model_manager.epoch,
        best_loss=model_manager.best_loss,
        input_shape=[config.INPUT_HEIGHT, config.INPUT_WIDTH],
        classes=config.CLASS_NAMES
    )

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Execute steel defect segmentation inference"
)
async def predict_endpoint(
    file: UploadFile = File(...),
    threshold: float = Form(default=config.DEFAULT_THRESHOLD)
):
    # 1. Validate threshold
    if not (config.MIN_THRESHOLD <= threshold <= config.MAX_THRESHOLD):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Threshold must be between {config.MIN_THRESHOLD} and {config.MAX_THRESHOLD}."
        )

    # 2. Validate filename and extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(list(config.ALLOWED_EXTENSIONS))}"
        )

    # 3. Read content and validate size
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(content) > config.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed upload size of {config.MAX_UPLOAD_SIZE // (1024*1024)} MB."
        )

    # 4. Run inference
    try:
        result = run_inference(
            image_bytes=content,
            filename=file.filename,
            threshold=threshold
        )
        return PredictionResponse(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        # Don't leak stack trace to client
        print(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed while processing the image. Please verify image format and retry."
        )
