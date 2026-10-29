"""
SteelSight — Industrial Steel Surface Defect Segmentation
Streamlit Community Cloud Application
Powered by PyTorch ResNet-18 U-Net Deep Convolutional Neural Network
"""

import os
import sys
import io
import base64
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Bridge Streamlit Cloud Secrets to Environment Variables if configured
try:
    if hasattr(st, "secrets"):
        for secret_key in ["MODEL_URL", "MODEL_AUTH_TOKEN", "EXPECTED_MODEL_SHA256"]:
            if secret_key in st.secrets and not os.getenv(secret_key):
                os.environ[secret_key] = str(st.secrets[secret_key])
except Exception:
    # No secrets file locally; environment variables or local paths will be used
    pass

from backend import config
from backend.inference import model_manager, run_inference, CLASS_COLORS

# Streamlit Page Configuration
st.set_page_config(
    page_title="SteelSight | AI Steel Defect Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }
    .badge-bar {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .spec-badge {
        background-color: #f0f4f8;
        border: 1px solid #d0dbe5;
        color: #2c3e50;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #fafbfc;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing SteelSight ResNet-18 U-Net Engine...")
def load_cached_model():
    """
    Loads and caches the model weights into memory.
    Executes once per Streamlit session/process.
    """
    if not model_manager.is_loaded:
        model_manager.load()
    return {
        "device": str(model_manager.device),
        "epoch": model_manager.epoch,
        "best_loss": model_manager.best_loss,
        "is_loaded": model_manager.is_loaded
    }

def b64_to_image(b64_str: str) -> Image.Image:
    """Converts a base64 data URI to a PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def main():
    # Header Section
    st.markdown('<div class="main-title">🔬 SteelSight AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real-Time Pixel-Level Defect Segmentation for Continuous Rolled Steel Manufacturing</div>',
        unsafe_allow_html=True
    )

    # Load Model Checkpoint with Cached Resource
    try:
        model_info = load_cached_model()
    except Exception as e:
        st.error(
            f"**Model Initialization Failed:** {str(e)}\n\n"
            "If running on Streamlit Cloud, ensure `model/model.pth` is available or configure `MODEL_URL` in Streamlit Secrets."
        )
        st.stop()

    # Model Spec Badges
    st.markdown(f"""
    <div class="badge-bar">
        <span class="spec-badge">Architecture: ResNet-18 U-Net</span>
        <span class="spec-badge">Checkpoint Epoch: {model_info.get('epoch', 18)}</span>
        <span class="spec-badge">Val Loss: {model_info.get('best_loss', 0.01176):.5f}</span>
        <span class="spec-badge">Compute: {model_info.get('device', 'cpu').upper()}</span>
        <span class="spec-badge">Input: 1600 × 256 px</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Inspection Settings")

        threshold = st.slider(
            "Detection Sensitivity Threshold",
            min_value=float(config.MIN_THRESHOLD),
            max_value=float(config.MAX_THRESHOLD),
            value=float(config.DEFAULT_THRESHOLD),
            step=0.05,
            help="Defect segmentation cutoff probability. Lower values increase sensitivity; higher values reduce false positives."
        )

        st.markdown("---")
        st.subheader("Defect Class Legend")
        class_colors_hex = ["#FF4154", "#45D49A", "#42B8FF", "#F0B45C"]
        for i in range(4):
            st.markdown(
                f'<div style="display:flex;align-items:center;margin-bottom:6px;">'
                f'<span style="background-color:{class_colors_hex[i]};width:16px;height:16px;display:inline-block;border-radius:3px;margin-right:8px;"></span>'
                f'<b>Class {i+1}:</b>&nbsp;{config.CLASS_NAMES[i]}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Technical Overview")
        st.markdown(
            "- **Encoder:** Pre-trained ResNet-18 feature extractor\n"
            "- **Decoder:** 5-stage skip-connected convolution\n"
            "- **Output:** 4 binary defect masks (Sigmoid activation)\n"
            "- **Resolution:** Standardized $1600 \\times 256$ strip format\n"
        )

    # Image Input Area
    st.subheader("1. Upload Steel Surface Image")

    # Check for local sample images
    sample_images = []
    dataset_dir = BASE_DIR / "dataset" / "train_images"
    if dataset_dir.exists():
        sample_images = list(dataset_dir.glob("*.jpg"))[:5]

    selected_sample = None
    if sample_images:
        sample_options = ["(Upload my own image)"] + [f.name for f in sample_images]
        chosen = st.selectbox("Or choose a sample production image from dataset:", sample_options)
        if chosen != "(Upload my own image)":
            selected_sample = dataset_dir / chosen

    uploaded_file = st.file_uploader(
        "Upload continuous strip surface image (.jpg, .jpeg, .png, .bmp, .webp)",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    image_bytes = None
    image_name = "sample.jpg"

    if uploaded_file is not None:
        if uploaded_file.size > config.MAX_UPLOAD_SIZE:
            st.error(f"File size exceeds maximum allowed limit ({config.MAX_UPLOAD_SIZE / (1024*1024):.0f} MB).")
            st.stop()
        image_bytes = uploaded_file.getvalue()
        image_name = uploaded_file.name
    elif selected_sample is not None and selected_sample.exists():
        with open(selected_sample, "rb") as f:
            image_bytes = f.read()
        image_name = selected_sample.name

    if image_bytes is None:
        st.info("👆 Please upload a steel surface image or select a sample image above to begin inspection.")
        st.stop()

    # Run Inference
    st.markdown("---")
    st.subheader("2. Surface Inspection Results")

    with st.spinner("Analyzing steel surface for defects..."):
        try:
            results = run_inference(image_bytes=image_bytes, filename=image_name, threshold=threshold)
        except Exception as e:
            st.error(f"Inference pipeline error: {str(e)}")
            st.stop()

    # Defect Status Banner
    any_defect = results.get("any_defect_detected", False)
    classes_info = results.get("classes", [])
    detected_classes = [c["name"] for c in classes_info if c["detected"]]

    if any_defect:
        st.error(f"⚠️ **Defects Detected:** {', '.join(detected_classes)} (Threshold: {threshold:.2f})")
    else:
        st.success(f"✅ **Clean Steel Strip:** No defects detected across all 4 classes at threshold {threshold:.2f}.")

    # Key Metrics Display
    total_pixels = config.INPUT_HEIGHT * config.INPUT_WIDTH
    total_defect_pixels = sum(c["pixels"] for c in classes_info)
    total_defect_pct = round((total_defect_pixels / total_pixels) * 100.0, 3)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Detection Status", "DEFECT" if any_defect else "CLEAN")
    with col_m2:
        st.metric("Total Defect Area", f"{total_defect_pct:.3f} %")
    with col_m3:
        st.metric("Classes Detected", f"{len(detected_classes)} / 4")
    with col_m4:
        st.metric("Inference Device", results.get("device", "cpu").upper())

    # Visualizations Tabs
    st.markdown("### Visual Inspection Outputs")
    tab_overlay, tab_heatmap, tab_masks, tab_original = st.tabs([
        "🎨 Defect Color Overlay",
        "🔥 Confidence Heatmap",
        "🔲 Discrete Channel Masks",
        "📷 Standardized Input Strip"
    ])

    with tab_overlay:
        st.image(
            b64_to_image(results["overlay_image"]),
            caption="Multi-Channel Defect Overlay (45% Blended Alpha on 256×1600 Continuous Strip)",
            use_container_width=True
        )

    with tab_heatmap:
        st.image(
            b64_to_image(results["heatmap_image"]),
            caption="Max-Probability Confidence Heatmap (Inferno Colormap)",
            use_container_width=True
        )

    with tab_masks:
        st.caption("Discrete binary segmentation masks for each defect class (White = Defect, Black = Background):")
        mask_cols = st.columns(4)
        for i in range(4):
            with mask_cols[i]:
                c_data = classes_info[i]
                mask_img = b64_to_image(results["masks"][i])
                st.image(
                    mask_img,
                    caption=f"{c_data['name']}\n({c_data['pixels']:,} px | {c_data['area_percent']:.3f}%)",
                    use_container_width=True
                )

    with tab_original:
        st.image(
            b64_to_image(results["original_image"]),
            caption=f"Standardized Input Strip ({results['original_width']}×{results['original_height']} px resized to {config.INPUT_WIDTH}×{config.INPUT_HEIGHT} px)",
            use_container_width=True
        )

    # Detailed Quantitative Summary Table
    st.markdown("### Quantitative Defect Statistics")
    df_display = pd.DataFrame([
        {
            "Class ID": c["class_id"],
            "Defect Class": c["name"],
            "Detected": "⚠️ Yes" if c["detected"] else "No",
            "Defect Pixels": f"{c['pixels']:,} px",
            "Surface Area %": f"{c['area_percent']:.4f} %"
        }
        for c in classes_info
    ])
    st.dataframe(df_display, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
