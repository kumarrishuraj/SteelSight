# SteelSight Architecture & Technical Reference

## 1. System Topology
SteelSight is composed of two decoupled services:
- **Backend Service**: An asynchronous FastAPI application orchestrating image validation, PyTorch GPU/CPU inference, and multi-channel artifact encoding.
- **Frontend Service**: A modern single-page React 18 application built with Vite and Lucide React, adhering to an industrial dark design paradigm.

```
+-----------------------------------------------------------+
|                   SteelSight React UI                     |
|  (Home | Inspect Console | Multi-View Results | About)    |
+-----------------------------------------------------------+
                             |
                   HTTP POST /api/predict
                             v
+-----------------------------------------------------------+
|                   FastAPI Backend                         |
|  - Payload & Dimension Validation (< 15MB, decodable)     |
|  - Preprocessing (256x1600 ResNet18 Normalization)        |
+-----------------------------------------------------------+
                             |
                     Forward Pass (eval)
                             v
+-----------------------------------------------------------+
|           PyTorch U-Net (ResNet-18 Encoder)               |
|  - 4 Independent Sigmoid Defect Output Channels           |
|  - Threshold Filtering (Default: 0.50)                    |
|  - Max Probability Projection (Inferno Colormap)          |
+-----------------------------------------------------------+
```

## 2. Preprocessing & Normalization Protocol
To guarantee strict mathematical compatibility with the pre-trained weights (`epoch: 18`, `best_loss: 0.01176`):
1. **Color Alignment**: BGR to RGB conversion.
2. **Resizing**: Scaled to `1600 px (W) × 256 px (H)` using bilinear interpolation.
3. **Floating Point Scaling**: Scaled to `[0.0, 1.0]` as `float32`.
4. **Standardization**:
   - `Mean`: `(0.485, 0.456, 0.406)`
   - `Std`: `(0.229, 0.224, 0.225)`
5. **Tensor Formatting**: Transposed to shape `(1, 3, 256, 1600)`.

## 3. Post-Processing & Mask Generation
- Logits produced by the final 1×1 convolution are activated with `torch.sigmoid(logits)`.
- Binary thresholding partitions active defect regions.
- Colormap overlays blend defect channels at 45% alpha over original raw strip textures:
  - Class 1: Coral Red (`#ff4154`)
  - Class 2: Mint Green (`#45d49a`)
  - Class 3: Sky Blue (`#42b8ff`)
  - Class 4: Amber Yellow (`#f0b45c`)
- A maximum probability projection across all 4 channels produces an informative heatmap via OpenCV `COLORMAP_INFERNO`.
