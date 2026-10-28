# SteelSight Automated Verification & Test Report

**Execution Environment:** Windows | Python 3.13.14 (.venv) | Node v25.6.0 | PyTorch 2.13.0+cpu

---

## 1. Backend Pytest Test Execution
All 7 unit and integration tests executed against the live model checkpoint (`E:\steelsight\model\model.pth`).

| Test Suite | Test Identifier | Result | Description |
|---|---|---|---|
| `test_health.py` | `test_health_check` | **PASSED** | Verifies `GET /api/health` returns status 200 and valid JSON |
| `test_health.py` | `test_model_status` | **PASSED** | Verifies `GET /api/model-status` reports loaded status, epoch 18, and 4 classes |
| `test_prediction.py` | `test_prediction_with_real_image` | **PASSED** | Executes real forward pass on Severstal test sample image, asserting valid 4-channel segmentation, heatmap, overlay, and pixel statistics |
| `test_validation.py` | `test_empty_file_upload` | **PASSED** | Asserts HTTP 400 when empty 0-byte payload is transmitted |
| `test_validation.py` | `test_invalid_extension` | **PASSED** | Asserts HTTP 400 when non-image binary (`.exe`) is uploaded |
| `test_validation.py` | `test_invalid_threshold_out_of_bounds` | **PASSED** | Asserts HTTP 422 when threshold > 0.95 or < 0.05 |
| `test_validation.py` | `test_corrupt_image` | **PASSED** | Asserts HTTP 400 rejection for corrupted image byte sequences |

**Summary:** 7 Passed, 0 Failed, 0 Skipped, 0 Blocked.

---

## 2. Real Ground Truth Validation (Dice & IoU Evaluation)
Using Severstal ground-truth Run-Length Encoded (RLE) annotations decoded directly from `dataset/train.csv` (1-based Fortran column-major order, shape `256 × 1600`) and the model's stratified validation split (`random_state=69`), Dice and IoU metrics were calculated across all 4 defect channels with sensitivity threshold = 0.50:

- **Validation Samples Evaluated**: 60 stratified validation images containing real defects
- **Class 1 (Defect Class 1)**: Dice = **0.9159** | IoU = **0.8952**
- **Class 2 (Defect Class 2)**: Dice = **0.9000** | IoU = **0.9000**
- **Class 3 (Defect Class 3)**: Dice = **0.7315** | IoU = **0.6286**
- **Class 4 (Defect Class 4)**: Dice = **0.9789** | IoU = **0.9682**
- **Overall Mean Dice**: **0.8816**
- **Overall Mean IoU**: **0.8480**

---

## 3. End-to-End Browser Automation Test (Playwright)
A full browser lifecycle test was conducted (`frontend/e2e_test.js`) connecting the React client with the active Uvicorn/FastAPI backend:

1. **Homepage Navigation**: Loaded `http://127.0.0.1:5173/` and verified hero heading: *"See defects before they become problems."* — **PASSED**
2. **Navigation to Inspect**: Clicked CTA and routed to `/inspect` — **PASSED**
3. **File Selection**: Attached real Severstal dataset image (`0002cc93b.jpg`, 256×1600 px) — **PASSED**
4. **Image Preview & Metadata**: Real-time rendering of dimension badge (`1600 × 256 px`) and threshold adjustment to 0.45 — **PASSED**
5. **Real Model Inference**: Triggered `Run Defect Inspection` calling `POST /api/predict`, executing PyTorch CPU forward pass, and routing to `/results` — **PASSED**
6. **Multi-View Inspection Tabs**: Verified active rendering and layer switching between Color Overlay, Confidence Heatmap, and discrete Class 1–4 binary masks — **PASSED**
7. **Artifact Export**: Verified PNG download trigger (`0002cc93b.jpg_overlay.png`) — **PASSED**
8. **Contact Form Validation**: Verified required field and email format validation on `/contact` — **PASSED**
9. **Responsive Layouts Tested**:
   - Desktop (1440 × 900) — **PASSED**
   - Laptop (1280 × 800) — **PASSED**
   - Tablet (768 × 1024) — **PASSED**
   - Mobile (390 × 844) — **PASSED**
10. **Browser Console Errors**: **0 errors captured**

---

## 4. Frontend Production Build Verification
- Engine: Vite 5.4.21
- Modules transformed: 1,980
- Result: **SUCCESSFUL PRODUCTION BUNDLE** (Built in 1.49s without errors)
  - `dist/index.html` (0.95 kB)
  - `dist/assets/index-CucWdTmR.css` (2.91 kB)
  - `dist/assets/index-obAz4Mrx.js` (213.15 kB)
