"""
SteelSight — Industrial Steel Surface Defect Segmentation
Streamlit Community Cloud Application · Premium UI
Powered by PyTorch ResNet-18 U-Net Deep Convolutional Neural Network
"""

import os
import sys
import io
import base64
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

# ─── Project root on sys.path ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ─── Bridge Streamlit Secrets → env vars BEFORE importing backend ────────────
try:
    if hasattr(st, "secrets"):
        for _k in ("MODEL_URL", "MODEL_AUTH_TOKEN", "EXPECTED_MODEL_SHA256", "MODEL_PATH"):
            if _k in st.secrets and str(st.secrets[_k]).strip():
                os.environ[_k] = str(st.secrets[_k]).strip()
except Exception:
    pass

from backend import config
from backend.inference import model_manager, run_inference  # noqa: E402

# ─── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="SteelSight | Industrial Defect Segmentation",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Session state defaults ──────────────────────────────────────────────────
st.session_state.setdefault("page", "home")
st.session_state.setdefault("result", None)
st.session_state.setdefault("uploaded_bytes", None)
st.session_state.setdefault("uploaded_name", None)
st.session_state.setdefault("threshold", 0.5)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset & tokens ── */
:root {
  --bg:            #080B10;
  --surface:       #0D1117;
  --surface-el:    #171E29;
  --border:        #1C2A3A;
  --text:          #E8EDF2;
  --text-muted:    #5A7190;
  --accent:        #42B8FF;
  --accent-glow:   rgba(66,184,255,0.18);
  --success:       #45D49A;
  --warning:       #F0B45C;
  --danger:        #FF4154;
  --mono:          'JetBrains Mono', 'Fira Code', monospace;
  --sans:          'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* App background */
.stApp {
  background-color: var(--bg) !important;
  background-image:
    linear-gradient(rgba(28,42,58,0.35) 1px, transparent 1px),
    linear-gradient(90deg, rgba(28,42,58,0.35) 1px, transparent 1px);
  background-size: 32px 32px;
  font-family: var(--sans);
  color: var(--text);
}

/* Override Streamlit block container */
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* Generic elements */
h1, h2, h3, h4, h5 { color: var(--text); font-family: var(--sans); }

/* ── Navbar ── */
.ss-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2.5rem;
  height: 64px;
  background: rgba(13,17,23,0.92);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(12px);
}
.ss-logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 800;
  font-size: 1.15rem;
  color: var(--text);
  letter-spacing: -0.02em;
}
.ss-logo span.eye { color: var(--accent); font-size: 1.2rem; }
.ss-ai-badge {
  font-family: var(--mono);
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
  background: rgba(66,184,255,0.10);
  border: 1px solid rgba(66,184,255,0.30);
  border-radius: 4px;
  padding: 2px 7px;
}
.ss-navlinks { display: flex; align-items: center; gap: 0.25rem; }
.ss-navlink {
  background: none;
  border: none;
  color: var(--text-muted);
  font-family: var(--sans);
  font-size: 0.9rem;
  font-weight: 500;
  padding: 0.4rem 0.85rem;
  border-radius: 6px;
  cursor: pointer;
  transition: color 0.15s, background 0.15s;
}
.ss-navlink:hover { color: var(--text); background: var(--surface-el); }
.ss-navlink.active { color: var(--accent); }
.ss-launch-btn {
  background: var(--accent);
  color: #04070A;
  border: none;
  font-family: var(--sans);
  font-size: 0.88rem;
  font-weight: 700;
  padding: 0.5rem 1.25rem;
  border-radius: 6px;
  cursor: pointer;
  transition: opacity 0.15s, box-shadow 0.15s;
  box-shadow: 0 0 16px rgba(66,184,255,0.30);
}
.ss-launch-btn:hover { opacity: 0.88; box-shadow: 0 0 24px rgba(66,184,255,0.50); }

/* ── Badge ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-family: var(--mono);
  font-size: 0.70rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 9px;
  border-radius: 4px;
  border: 1px solid;
}
.badge-accent  { color: var(--accent);  border-color: rgba(66,184,255,0.35);  background: rgba(66,184,255,0.08); }
.badge-success { color: var(--success); border-color: rgba(69,212,154,0.35);  background: rgba(69,212,154,0.08); }
.badge-danger  { color: var(--danger);  border-color: rgba(255,65,84,0.35);   background: rgba(255,65,84,0.08);  }
.badge-warn    { color: var(--warning); border-color: rgba(240,180,92,0.35);  background: rgba(240,180,92,0.08); }
.badge-neutral { color: var(--text-muted); border-color: var(--border); background: var(--surface-el); }

/* ── Card ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.5rem;
}
.card-el {
  background: var(--surface-el);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
}

/* ── Buttons ── */
.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--accent);
  color: #04070A;
  border: none;
  font-family: var(--sans);
  font-size: 1rem;
  font-weight: 700;
  padding: 0.75rem 1.75rem;
  border-radius: 7px;
  cursor: pointer;
  transition: opacity 0.15s, box-shadow 0.15s;
  box-shadow: 0 0 18px rgba(66,184,255,0.28);
  text-decoration: none;
}
.btn-primary:hover { opacity: 0.88; box-shadow: 0 0 28px rgba(66,184,255,0.45); }
.btn-primary:disabled { opacity: 0.40; cursor: not-allowed; box-shadow: none; }
.btn-secondary {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: transparent;
  color: var(--text);
  border: 1px solid var(--border);
  font-family: var(--sans);
  font-size: 0.9rem;
  font-weight: 600;
  padding: 0.6rem 1.25rem;
  border-radius: 7px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}
.btn-secondary:hover { background: var(--surface-el); border-color: var(--accent); }

/* ── Section layout helpers ── */
.container { max-width: 1200px; margin: 0 auto; padding: 0 2rem; }
.section    { padding: 4rem 0; }
.section-alt {
  padding: 4rem 0;
  background: var(--surface);
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}

/* ── Hero ── */
.hero-badge { margin-bottom: 1.25rem; }
.hero-heading {
  font-size: clamp(2rem, 4.5vw, 3.4rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.15;
  margin-bottom: 1.2rem;
}
.hero-heading .accent {
  color: var(--accent);
  text-shadow: 0 0 28px rgba(66,184,255,0.40);
}
.hero-desc {
  font-size: 1.1rem;
  color: var(--text-muted);
  line-height: 1.65;
  margin-bottom: 2.25rem;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
}

/* ── Feature card icon box ── */
.icon-box {
  width: 44px;
  height: 44px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.25rem;
  font-size: 1.3rem;
}

/* ── Pipeline step circle ── */
.step-circle {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: var(--surface-el);
  border: 2px solid var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 1rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0 auto 1.25rem;
}

/* ── Upload dropzone ── */
.dropzone {
  background: var(--surface);
  border: 2px dashed var(--border);
  border-radius: 8px;
  padding: 2.5rem 1.5rem;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  margin-bottom: 1.25rem;
}
.dropzone:hover { border-color: var(--accent); background: rgba(66,184,255,0.04); }

/* ── Tab bar ── */
.tab-bar {
  display: flex;
  gap: 0.5rem;
  overflow-x: auto;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
}
.tab-btn {
  padding: 0.42rem 0.9rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-muted);
  font-size: 0.83rem;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s;
  font-family: var(--sans);
}
.tab-btn.active {
  border-color: var(--accent);
  background: var(--surface-el);
  color: var(--accent);
}

/* ── Class cards ── */
.class-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem;
}

/* ── Spec grid item ── */
.spec-item {
  background: var(--surface-el);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
}
.spec-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-family: var(--mono); letter-spacing: 0.06em; }
.spec-value { font-weight: 700; font-size: 1rem; margin-top: 0.25rem; }

/* ── CTA banner ── */
.cta-banner {
  background: linear-gradient(135deg, var(--surface-el) 0%, var(--surface) 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 3rem 2rem;
  text-align: center;
}

/* ── Disclaimer box ── */
.disclaimer {
  background: rgba(23,30,41,0.6);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  font-size: 0.85rem;
  color: var(--text-muted);
}

/* ── Footer ── */
.ss-footer {
  border-top: 1px solid var(--border);
  background: var(--surface);
  padding: 3rem 0 2rem;
  margin-top: auto;
}

/* ── Streamlit widget overrides ── */
/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 8px !important;
  padding: 1.5rem !important;
}
[data-testid="stFileUploader"] label { color: var(--text) !important; }
[data-testid="stFileUploader"] small { color: var(--text-muted) !important; }

/* Slider */
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }
div[data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }
div[data-baseweb="slider"] div[data-testid="stSliderThumbValue"] { color: var(--accent) !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  color: var(--text-muted) !important;
  font-family: var(--sans) !important;
  font-size: 0.85rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* Buttons */
.stButton > button {
  background: var(--accent) !important;
  color: #04070A !important;
  border: none !important;
  font-weight: 700 !important;
  border-radius: 7px !important;
  box-shadow: 0 0 18px rgba(66,184,255,0.25) !important;
  transition: opacity 0.15s !important;
  font-family: var(--sans) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:disabled { opacity: 0.40 !important; cursor: not-allowed !important; box-shadow: none !important; }

/* Secondary buttons via class */
.secondary-btn > button {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}
.secondary-btn > button:hover {
  background: var(--surface-el) !important;
  border-color: var(--accent) !important;
  color: var(--text) !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
  font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: var(--surface-el) !important;
  border-color: var(--accent) !important;
}

/* Images */
[data-testid="stImage"] img { border-radius: 6px; }

/* Error/success messages */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* Metric */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--mono) !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* Horizontal rule */
hr { border-color: var(--border) !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  overflow: hidden;
}

/* Text / markdown elements */
p { color: var(--text); }
.stMarkdown p { color: var(--text); }

/* Mono labels */
.mono { font-family: var(--mono); }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def b64_to_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def b64_to_bytes(b64_str: str) -> bytes:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)


def nav_to(page: str):
    st.session_state["page"] = page
    st.rerun()


# ─── Model loader (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing SteelSight ResNet-18 U-Net engine…")
def load_model():
    if not model_manager.is_loaded:
        model_manager.load()
    return {
        "device": str(model_manager.device),
        "epoch": model_manager.epoch,
        "best_loss": model_manager.best_loss,
        "is_loaded": model_manager.is_loaded,
    }


# ─── Navbar ──────────────────────────────────────────────────────────────────

def render_navbar():
    cur = st.session_state["page"]
    active = lambda p: "active" if cur == p else ""  # noqa: E731

    nav_cols = st.columns([3, 5, 2])

    with nav_cols[0]:
        st.markdown(
            '<div class="ss-logo"><span class="eye">👁</span>SteelSight'
            '&nbsp;<span class="ss-ai-badge">AI</span></div>',
            unsafe_allow_html=True,
        )

    with nav_cols[1]:
        nc = st.columns(4)
        pages = [("home", "Home"), ("inspection", "Inspection"), ("about", "About"), ("contact", "Contact")]
        for col, (page_id, label) in zip(nc, pages):
            with col:
                cls = "secondary-btn" if cur != page_id else ""
                st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                    nav_to(page_id)
                st.markdown("</div>", unsafe_allow_html=True)

    with nav_cols[2]:
        if st.button("🚀 Launch System", key="nav_launch", use_container_width=True):
            nav_to("inspection")

    st.markdown('<hr style="margin:0;border-color:var(--border);">', unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────

def render_footer():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:var(--border);">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div style="font-weight:800;font-size:1.05rem;margin-bottom:0.5rem;">👁 SteelSight</div>'
            '<div style="font-size:0.88rem;color:var(--text-muted);max-width:280px;">'
            "High-throughput deep learning segmentation for industrial steel manufacturing and automated quality inspection."
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div style="font-weight:700;font-size:0.9rem;margin-bottom:0.75rem;">System Capabilities</div>'
            '<ul style="list-style:none;padding:0;margin:0;color:var(--text-muted);font-size:0.85rem;display:flex;flex-direction:column;gap:0.4rem;">'
            "<li>4-Class Defect Segmentation</li>"
            "<li>Multi-Channel Confidence Heatmaps</li>"
            "<li>Pixel-level Area Quantification</li>"
            "<li>Real-time PyTorch CPU/CUDA Inference</li>"
            "</ul>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div style="font-weight:700;font-size:0.9rem;margin-bottom:0.75rem;">Architecture Stack</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;">'
            '<span class="badge badge-accent">Streamlit</span>'
            '<span class="badge badge-accent">PyTorch 2.x</span>'
            '<span class="badge badge-accent">ResNet-18 U-Net</span>'
            '<span class="badge badge-accent">FastAPI</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="margin-top:1.5rem;border-top:1px solid var(--border);padding-top:1rem;'
        'display:flex;justify-content:space-between;font-size:0.82rem;color:var(--text-muted);">'
        "<span>© 2025 SteelSight System. Designed for precision metallurgical quality assurance.</span>"
        '<span style="font-family:var(--mono);">⚡ Industrial CV Inference Node</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# ─── PAGE: HOME ──────────────────────────────────────────────────────────────

def page_home():
    # Hero
    st.markdown(
        """
<div style="border-bottom:1px solid var(--border);padding:5rem 2rem 4rem;text-align:center;">
  <div class="hero-badge">
    <span class="badge badge-accent">✦ Production Deep Learning Pipeline</span>
  </div>
  <div class="hero-heading">
    See defects before they become <span class="accent">problems.</span>
  </div>
  <div class="hero-desc">
    High-resolution pixel-level semantic segmentation for hot and cold rolled sheet steel.
    Powered by a custom ResNet-18 U-Net encoder-decoder trained for multi-class metallurgical defect isolation.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Hero CTA buttons
    hcol1, hcol2, hcol3 = st.columns([2, 1, 1])
    with hcol2:
        if st.button("▶  Start Inspection", key="hero_inspect", use_container_width=True):
            nav_to("inspection")
    with hcol3:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("Explore the System", key="hero_about", use_container_width=True):
            nav_to("about")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Capabilities section
    st.markdown(
        '<div style="text-align:center;margin-bottom:2rem;">'
        '<span class="badge badge-accent">Core Capabilities</span><br><br>'
        '<span style="font-size:1.9rem;font-weight:700;">Engineered for Industrial Inspection</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    cap_cols = st.columns(3)
    capabilities = [
        {
            "icon": "⊞",
            "icon_color": "var(--accent)",
            "icon_bg": "rgba(66,184,255,0.10)",
            "icon_border": "rgba(66,184,255,0.25)",
            "title": "4-Channel Defect Decomposition",
            "desc": (
                "Deconstructs surface anomalies into four distinct defect categories simultaneously "
                "via sigmoid-activated prediction heads."
            ),
        },
        {
            "icon": "◎",
            "icon_color": "var(--success)",
            "icon_bg": "rgba(69,212,154,0.10)",
            "icon_border": "rgba(69,212,154,0.25)",
            "title": "Fine Spatial Precision",
            "desc": (
                "Processes extreme aspect-ratio continuous steel strip images (1600×256 pixels) "
                "without loss of longitudinal edge fidelity."
            ),
        },
        {
            "icon": "〰",
            "icon_color": "var(--warning)",
            "icon_bg": "rgba(240,180,92,0.10)",
            "icon_border": "rgba(240,180,92,0.25)",
            "title": "Confidence Heatmaps",
            "desc": (
                "Generates perceptual density gradient maps revealing subtle micro-fissures and "
                "marginal surface irregularities before total failure."
            ),
        },
    ]
    for col, cap in zip(cap_cols, capabilities):
        with col:
            st.markdown(
                f"""
<div class="card">
  <div class="icon-box" style="background:{cap['icon_bg']};border:1px solid {cap['icon_border']};color:{cap['icon_color']};">
    <span style="font-size:1.3rem;">{cap['icon']}</span>
  </div>
  <div style="font-size:1.1rem;font-weight:700;margin-bottom:0.6rem;">{cap['title']}</div>
  <div style="color:var(--text-muted);font-size:0.9rem;line-height:1.6;">{cap['desc']}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline section
    st.markdown(
        '<div style="background:var(--surface);border-top:1px solid var(--border);border-bottom:1px solid var(--border);'
        'padding:3.5rem 2rem;margin:0 -2rem;">'
        '<div style="text-align:center;margin-bottom:2.5rem;">'
        '<span class="badge badge-accent">Inspection Pipeline</span><br><br>'
        '<span style="font-size:1.9rem;font-weight:700;">How SteelSight Operates</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    pipe_cols = st.columns(3)
    steps = [
        ("01", "Image Ingestion", "Upload high-definition surface captures directly from shop floor cameras or line scanners."),
        ("02", "U-Net Forward Pass", "ResNet-18 features are extracted across 5 stages and reconstructed via skip connections."),
        ("03", "Quantitative Analysis", "Instant breakdown of affected pixel area, defect overlays, and channel-by-channel mask exports."),
    ]
    for col, (num, title, desc) in zip(pipe_cols, steps):
        with col:
            st.markdown(
                f"""
<div style="text-align:center;padding:0 1rem;">
  <div class="step-circle">{num}</div>
  <div style="font-size:1.05rem;font-weight:700;margin-bottom:0.5rem;">{title}</div>
  <div style="color:var(--text-muted);font-size:0.88rem;line-height:1.6;">{desc}</div>
</div>
""",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # CTA banner
    st.markdown(
        '<div class="cta-banner">'
        '<div style="font-size:2rem;font-weight:700;margin-bottom:1rem;">Initiate Steel Surface Inspection</div>'
        '<div style="color:var(--text-muted);max-width:580px;margin:0 auto 2rem;font-size:1rem;line-height:1.6;">'
        "Test your industrial production captures against the loaded model checkpoint to receive instant segmentation analytics."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    cta1, cta2, cta3 = st.columns([2, 2, 2])
    with cta2:
        if st.button("▶  Open Inspection Console →", key="cta_btn", use_container_width=True):
            nav_to("inspection")

    st.markdown("<br>", unsafe_allow_html=True)


# ─── PAGE: INSPECTION ────────────────────────────────────────────────────────

def page_inspection(model_info: dict):
    st.markdown("<div style='padding:2.5rem 2rem 0;'>", unsafe_allow_html=True)

    # Header badges
    device_label = model_info.get("device", "cpu").upper()
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">'
        f'<span class="badge badge-accent">Inspection Console</span>'
        f'<span class="badge badge-success">Model Ready ({device_label})</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:2rem;font-weight:800;margin-bottom:0.4rem;">Steel Surface Inspection</div>'
        '<div style="color:var(--text-muted);font-size:0.98rem;margin-bottom:2rem;">'
        "Upload high-resolution continuous strip images to perform real-time pixel-level segmentation across 4 defect channels."
        "</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        # File uploader styled as dropzone
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
            'text-transform:uppercase;color:var(--text-muted);margin-bottom:0.5rem;">Surface Image Upload</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Drag and drop surface image here — JPG, PNG, WEBP, BMP up to 15 MB",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="inspect_uploader",
            label_visibility="visible",
        )

        if uploaded is not None:
            if uploaded.size > config.MAX_UPLOAD_SIZE:
                st.error(
                    f"File exceeds {config.MAX_UPLOAD_SIZE // (1024*1024)} MB limit. Please select a smaller image."
                )
            else:
                st.session_state["uploaded_bytes"] = uploaded.getvalue()
                st.session_state["uploaded_name"] = uploaded.name

                # Preview card
                img = Image.open(io.BytesIO(st.session_state["uploaded_bytes"]))
                w, h = img.size
                size_kb = uploaded.size // 1024
                st.markdown(
                    f'<div class="card" style="margin-top:1rem;">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">'
                    f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                    f'<span style="color:var(--accent);">🖼</span>'
                    f'<span style="font-weight:600;font-size:0.92rem;">{uploaded.name}</span>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<div style="display:flex;gap:2rem;font-size:0.80rem;color:var(--text-muted);'
                    f'font-family:var(--mono);margin-top:0.5rem;">'
                    f'<span>Dimensions: <span style="color:var(--text);">{w} × {h} px</span></span>'
                    f'<span>File Size: <span style="color:var(--text);">{size_kb} KB</span></span>'
                    f'<span>Model Target: <span style="color:var(--accent);">1600 × 256 px</span></span>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    with right_col:
        # Threshold card
        current_thresh = st.session_state["threshold"]
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">'
            f'<div style="font-weight:600;font-size:0.95rem;">⚙ Detection Threshold</div>'
            f'<span class="badge badge-accent" style="font-family:var(--mono);">{current_thresh:.2f}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        new_thresh = st.slider(
            "Detection Threshold",
            min_value=0.05,
            max_value=0.95,
            value=current_thresh,
            step=0.05,
            key="thresh_slider",
            label_visibility="collapsed",
        )
        st.session_state["threshold"] = new_thresh
        st.markdown(
            '<div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--text-muted);margin-top:0.25rem;">'
            "<span>High Sensitivity (0.05)</span><span>Recommended (0.50)</span><span>High Specificity (0.95)</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # Class legend
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card">'
            '<div style="font-weight:600;font-size:0.92rem;margin-bottom:0.9rem;">Defect Class Legend</div>',
            unsafe_allow_html=True,
        )
        class_colors = ["#FF4154", "#45D49A", "#42B8FF", "#F0B45C"]
        for i, (color, name) in enumerate(zip(class_colors, config.CLASS_NAMES)):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;font-size:0.88rem;">'
                f'<span style="width:14px;height:14px;border-radius:3px;background:{color};flex-shrink:0;display:inline-block;"></span>'
                f'<span style="color:var(--text-muted);">Class {i+1}:</span>'
                f'<span style="font-weight:600;">{name}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Run button — full width below columns
    st.markdown("<br>", unsafe_allow_html=True)
    has_image = st.session_state.get("uploaded_bytes") is not None

    run_clicked = st.button(
        "▶  Run Defect Inspection" if has_image else "Upload an Image to Begin",
        key="run_inspect_btn",
        disabled=not has_image,
        use_container_width=True,
    )

    if run_clicked and has_image:
        with st.spinner("Executing PyTorch segmentation — ResNet-18 U-Net forward pass…"):
            try:
                result = run_inference(
                    image_bytes=st.session_state["uploaded_bytes"],
                    filename=st.session_state["uploaded_name"],
                    threshold=st.session_state["threshold"],
                )
                st.session_state["result"] = result
                nav_to("results")
            except Exception as exc:
                st.error(f"Inference pipeline error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


# ─── PAGE: RESULTS ────────────────────────────────────────────────────────────

def page_results():
    result = st.session_state.get("result")

    st.markdown("<div style='padding:2rem 2rem 0;'>", unsafe_allow_html=True)

    if result is None:
        st.markdown(
            '<div style="max-width:500px;margin:4rem auto 0;">'
            '<div class="card" style="text-align:center;padding:3rem 2rem;">'
            '<div style="font-size:2.5rem;margin-bottom:1rem;color:var(--warning);">⚠</div>'
            '<div style="font-size:1.35rem;font-weight:700;margin-bottom:0.75rem;">No Active Inspection Data</div>'
            '<div style="color:var(--text-muted);font-size:0.9rem;line-height:1.6;margin-bottom:1.75rem;">'
            "No prior segmentation inference session was found. Upload a steel surface image in the inspection console to view results."
            "</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("▶  Initiate New Inspection", key="results_go_inspect"):
            nav_to("inspection")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Top bar ──────────────────────────────────────────────────────────────
    top_left, top_right = st.columns([1, 3])
    with top_left:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("← New Inspection", key="results_back"):
            nav_to("inspection")
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        defect = result.get("any_defect_detected", False)
        dev = result.get("device", "cpu").upper()
        thr = result.get("threshold", 0.5)
        defect_badge = (
            '<span class="badge badge-danger">⚠ Defects Detected</span>'
            if defect
            else '<span class="badge badge-success">✓ No Anomalies Above Threshold</span>'
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;">'
            f'<span class="badge badge-accent">⚡ {dev}</span>'
            f'<span class="badge badge-neutral">Threshold: {thr:.2f}</span>'
            f"{defect_badge}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visualizer card with tabs ─────────────────────────────────────────────
    TAB_LABELS = [
        "Color Overlay",
        "Original Strip",
        "Confidence Heatmap",
        "Defect Class 1",
        "Defect Class 2",
        "Defect Class 3",
        "Defect Class 4",
    ]
    tabs = st.tabs(TAB_LABELS)

    tab_images = [
        result.get("overlay_image"),
        result.get("original_image"),
        result.get("heatmap_image"),
        result["masks"][0] if result.get("masks") else None,
        result["masks"][1] if result.get("masks") else None,
        result["masks"][2] if result.get("masks") else None,
        result["masks"][3] if result.get("masks") else None,
    ]
    tab_captions = [
        "Multi-Channel Defect Overlay (45% blended alpha on 256×1600 strip)",
        f"Standardized Input Strip ({result.get('original_width','?')}×{result.get('original_height','?')} px → 1600×256 px)",
        "Max-Probability Confidence Heatmap (Inferno colormap)",
        "Class 1 Binary Mask — Crazing",
        "Class 2 Binary Mask — Inclusion",
        "Class 3 Binary Mask — Patches",
        "Class 4 Binary Mask — Pitted Surface",
    ]
    tab_filenames = [
        "overlay", "original", "heatmap", "class1", "class2", "class3", "class4",
    ]

    for tab, b64_img, caption, fname in zip(tabs, tab_images, tab_captions, tab_filenames):
        with tab:
            if b64_img:
                img_obj = b64_to_image(b64_img)
                st.image(img_obj, caption=caption, use_container_width=True)
                # Source info + download
                dl_col1, dl_col2 = st.columns([3, 1])
                with dl_col1:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:var(--text-muted);font-family:var(--mono);">'
                        f'Source: <span style="color:var(--text);">{result.get("filename","image")}</span>'
                        f' ({result.get("original_width","?")} × {result.get("original_height","?")} px)'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with dl_col2:
                    st.download_button(
                        "⬇ Download",
                        data=b64_to_bytes(b64_img),
                        file_name=f'steelsight_{result.get("filename","image")}_{fname}.png',
                        mime="image/png",
                        key=f"dl_{fname}",
                        use_container_width=True,
                    )
            else:
                st.warning("Image data unavailable.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Class metric cards ────────────────────────────────────────────────────
    classes_info = result.get("classes", [])
    class_colors_hex = ["#FF4154", "#45D49A", "#42B8FF", "#F0B45C"]

    if classes_info:
        st.markdown(
            '<div style="font-size:1.2rem;font-weight:700;margin-bottom:1rem;">Quantitative Defect Analysis</div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(4)
        for col, cls_data, color in zip(metric_cols, classes_info, class_colors_hex):
            with col:
                detected = cls_data.get("detected", False)
                badge = (
                    f'<span class="badge badge-danger">Present</span>'
                    if detected
                    else '<span class="badge badge-success">Not Detected</span>'
                )
                st.markdown(
                    f'<div class="class-card" style="border-left:3px solid {color};">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">'
                    f'<span style="font-weight:700;color:{color};font-size:0.95rem;">{cls_data.get("name","Class")}</span>'
                    f"{badge}"
                    f"</div>"
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.85rem;">'
                    f'<div><div class="spec-label">Defect Pixels</div>'
                    f'<div style="font-weight:700;font-family:var(--mono);font-size:0.95rem;">{cls_data.get("pixels",0):,} px</div></div>'
                    f'<div><div class="spec-label">Surface Area %</div>'
                    f'<div style="font-weight:700;font-family:var(--mono);font-size:0.95rem;">{cls_data.get("area_percent",0.0):.3f}%</div></div>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        "<strong>Responsible Use Note:</strong> Predictions are generated using a pre-trained deep convolutional neural network "
        "for automated quality screening assistance. Critical metallurgical certifications must follow standard "
        "non-destructive testing (NDT) verification protocols."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ─── PAGE: ABOUT ─────────────────────────────────────────────────────────────

def page_about():
    st.markdown("<div style='max-width:900px;margin:0 auto;padding:3.5rem 2rem 5rem;'>", unsafe_allow_html=True)

    st.markdown(
        '<span class="badge badge-accent">System Architecture</span><br><br>'
        '<div style="font-size:2.2rem;font-weight:800;margin-bottom:0.75rem;">About SteelSight</div>'
        '<div style="color:var(--text-muted);font-size:1rem;line-height:1.6;margin-bottom:2.5rem;">'
        "SteelSight is an industrial-grade computer vision system engineered to localize, decompose, and quantify "
        "surface defects on flat steel strips during manufacturing operations."
        "</div>",
        unsafe_allow_html=True,
    )

    # Architecture specs card
    st.markdown(
        '<div class="card" style="margin-bottom:2rem;">'
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.25rem;">'
        '<span style="color:var(--accent);font-size:1.2rem;">⚙</span>'
        '<span style="font-size:1.2rem;font-weight:700;">Deep Learning Architecture Specifications</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    spec_cols = st.columns(4)
    specs = [
        ("Network Backbone", "ResNet-18 Encoder"),
        ("Decoder Topology", "U-Net (256, 128, 64, 32, 16)"),
        ("Input Resolution", "256 × 1600 px (RGB)"),
        ("Channel Activation", "Independent Sigmoids (4 classes)"),
    ]
    for col, (label, val) in zip(spec_cols, specs):
        with col:
            st.markdown(
                f'<div class="spec-item"><div class="spec-label">{label}</div>'
                f'<div class="spec-value">{val}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div style="color:var(--text-muted);font-size:0.9rem;line-height:1.6;margin-top:1.25rem;">'
        "The model utilizes five stages of residual convolutions with lateral skip connections that route high-frequency boundary "
        "information directly from encoder feature hierarchies to expanding decoder blocks. This preserves microscopic defect "
        "contours even across continuous strip rolling lines."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Dataset & training card
    st.markdown(
        '<div class="card" style="margin-bottom:2rem;">'
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.25rem;">'
        '<span style="color:var(--success);font-size:1.2rem;">⊞</span>'
        '<span style="font-size:1.2rem;font-weight:700;">Dataset &amp; Training Baseline</span>'
        "</div>"
        '<div style="color:var(--text-muted);font-size:0.9rem;line-height:1.6;margin-bottom:1rem;">'
        "Model weights were trained on the Severstal Steel Defect Detection benchmark corpus containing thousands of annotated "
        "256×1600 surface photographs. Optimization utilized Binary Cross Entropy with Logits combined with Dice coefficient penalization."
        "</div>"
        '<div style="display:flex;flex-wrap:wrap;gap:0.5rem;">'
        '<span class="badge badge-accent">Epoch: 18</span>'
        '<span class="badge badge-accent">Best Validation Loss: 0.01176</span>'
        '<span class="badge badge-accent">Independent Multi-Label Formulation</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Operational constraints card
    st.markdown(
        '<div class="card" style="border-left:3px solid var(--warning);">'
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem;">'
        '<span style="color:var(--warning);font-size:1.2rem;">⚠</span>'
        '<span style="font-size:1.2rem;font-weight:700;">Operational Constraints &amp; Disclaimers</span>'
        "</div>"
        '<ul style="color:var(--text-muted);font-size:0.9rem;padding-left:1.25rem;line-height:1.7;margin:0;">'
        "<li>Inference performance depends on consistent lighting and surface cleanliness comparable to typical rolling mill inspection hoods.</li>"
        "<li>Extreme camera blur, uncalibrated scaling, or non-steel surfaces can produce spurious activations.</li>"
        "<li>SteelSight provides automated detection assistance and does not supersede ISO 9001 metallurgical certification standards without human oversight.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ─── PAGE: CONTACT ───────────────────────────────────────────────────────────

def page_contact():
    st.markdown("<div style='max-width:680px;margin:0 auto;padding:3.5rem 2rem 5rem;'>", unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;margin-bottom:2rem;">'
        '<span class="badge badge-accent">Direct Inquiries</span><br><br>'
        '<div style="font-size:2rem;font-weight:800;margin-bottom:0.5rem;">Connect with Engineering</div>'
        '<div style="color:var(--text-muted);font-size:0.95rem;">'
        "Submit questions regarding deep learning deployment, camera calibration, or custom mill integration."
        "</div></div>",
        unsafe_allow_html=True,
    )

    submitted = st.session_state.get("contact_submitted", False)

    if submitted:
        name = st.session_state.get("contact_name", "")
        st.markdown(
            f'<div class="card" style="padding:2.5rem;text-align:center;">'
            f'<div style="width:56px;height:56px;border-radius:50%;background:rgba(69,212,154,0.10);'
            f'border:1px solid var(--success);display:flex;align-items:center;justify-content:center;'
            f'margin:0 auto 1.25rem;font-size:1.6rem;">✓</div>'
            f'<div style="font-size:1.35rem;font-weight:700;margin-bottom:0.75rem;">Inquiry Registered</div>'
            f'<div style="color:var(--text-muted);font-size:0.9rem;line-height:1.6;margin-bottom:1.5rem;">'
            f"Thank you, <strong>{name}</strong>. Your feedback has been verified in the front-end simulation interface. "
            f"(Note: Email delivery is not configured on this standalone instance)."
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Submit Another Inquiry", key="contact_reset"):
            for k in ("contact_submitted", "contact_name"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        with st.form("contact_form"):
            st.markdown('<div class="card" style="padding:2rem;">', unsafe_allow_html=True)
            name_input = st.text_input("Full Name *", placeholder="e.g. Elena Rostova")
            email_input = st.text_input("Corporate Email *", placeholder="e.g. e.rostova@steelforge.com")
            subject_input = st.text_input("Subject Line *", placeholder="e.g. Model inference on 1.2mm gauge strip")
            message_input = st.text_area(
                "Inquiry / Message *",
                placeholder="Specify rolling conditions, camera hardware, and deployment objectives…",
                height=130,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            submitted_btn = st.form_submit_button(
                "📨 Submit Technical Inquiry", use_container_width=True
            )

        if submitted_btn:
            errors = []
            if not name_input.strip():
                errors.append("Full name is required.")
            if not email_input.strip():
                errors.append("Email address is required.")
            elif "@" not in email_input or "." not in email_input.split("@")[-1]:
                errors.append("Please provide a valid email address.")
            if not subject_input.strip():
                errors.append("Subject line is required.")
            if not message_input.strip():
                errors.append("Inquiry message cannot be empty.")
            elif len(message_input.strip()) < 15:
                errors.append("Message must contain at least 15 characters.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.session_state["contact_submitted"] = True
                st.session_state["contact_name"] = name_input.strip()
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load model (cached, runs once)
    try:
        model_info = load_model()
    except Exception as exc:
        st.error(
            f"**Model Initialization Failed:** {exc}\n\n"
            "If running on Streamlit Cloud, ensure `MODEL_URL` is configured in Streamlit Secrets and "
            "points to the public `model.pth` release asset."
        )
        st.stop()

    # Navbar
    render_navbar()

    # Route to page
    page = st.session_state["page"]
    if page == "home":
        page_home()
    elif page == "inspection":
        page_inspection(model_info)
    elif page == "results":
        page_results()
    elif page == "about":
        page_about()
    elif page == "contact":
        page_contact()
    else:
        page_home()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
