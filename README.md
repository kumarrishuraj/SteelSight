# SteelSight

**AI-powered industrial steel surface inspection and defect segmentation.**

SteelSight delivers real-time pixel-level defect segmentation on continuous rolled steel strip photographs, using a custom ResNet-18 U-Net deep convolutional neural network pre-trained on the Severstal steel defect dataset.

---

## Interactive Streamlit Web Application - https://steelsight-euz7hu4qudbz6ydybfkxmf.streamlit.app/

Run the standalone Streamlit application locally:

```powershell
cd E:\steelsight
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

### Streamlit Community Cloud Deployment
- **Repository:** `https://github.com/kumarrishuraj/SteelSight`
- **Branch:** `main`
- **Main file path:** `streamlit_app.py`
- **App Secrets (Optional for external weights):**
  - `MODEL_URL`: Direct HTTPS link to `model.pth`
  - `EXPECTED_MODEL_SHA256`: `076f04466dd37ae0b6e9a9e4bd61810921a9f92996b04ef5c966d827b9eaefe8`


---

## System Overview

- **Frontend**: React 18 + Vite, Lucide React icons, responsive industrial dark interface.
- **Backend**: FastAPI with async multi-part payload validation and in-memory base64 response pipeline.
- **Deep Learning**: PyTorch 2.13.0, torchvision, custom ResNet-18 U-Net architecture (`epoch: 18`, `best_loss: 0.01176`).
- **Computer Vision**: OpenCV, Pillow, NumPy.

---

## Quickstart Instructions

### 1. Launch the Backend API

Open PowerShell, activate the virtual environment, and launch the Uvicorn server:

```powershell
cd E:\steelsight
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/api/health`
- Model Status: `http://127.0.0.1:8000/api/model-status`

### 2. Launch the Frontend Application

Open a second PowerShell terminal:

```powershell
cd E:\steelsight\frontend
npm run dev
```

- Web Application URL: `http://127.0.0.1:5173`

---

## Running the Automated Test Suite

Run backend test cases (verifying endpoints, validation rules, and live model prediction):

```powershell
cd E:\steelsight
.\.venv\Scripts\python.exe -m pytest backend/tests -v
```

Test production frontend compilation:

```powershell
cd E:\steelsight\frontend
npm run build
```

---

## API Documentation

### `POST /api/predict`
Accepts a multipart file upload (`.jpg`, `.png`, `.webp`, `.bmp` up to 15MB) and an optional detection threshold (`0.05` to `0.95`, default `0.50`).

Returns:
- `original_image`: Base64 PNG of standardized 256×1600 input strip.
- `overlay_image`: Multi-channel color-blended defect visualization.
- `heatmap_image`: Combined confidence gradient map.
- `masks`: 4 discrete binary mask images (Defect Classes 1–4).
- `classes`: Array detailing per-class detected status, pixel count, and percentage surface area.
- `any_defect_detected`: Boolean flag indicating presence of any defects.
