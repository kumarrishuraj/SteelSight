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

import streamlit as st
from PIL import Image

# ── Project root on sys.path ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Bridge Streamlit Secrets → env vars BEFORE importing backend ─────────────
try:
    if hasattr(st, "secrets"):
        for _k in ("MODEL_URL", "MODEL_AUTH_TOKEN", "EXPECTED_MODEL_SHA256", "MODEL_PATH"):
            if _k in st.secrets and str(st.secrets[_k]).strip():
                os.environ[_k] = str(st.secrets[_k]).strip()
except Exception:
    pass

from backend import config
from backend.inference import model_manager, run_inference  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SteelSight | Surface Defect Inspection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ─────────────────────────────────────────────────────────────
st.session_state.setdefault("page", "home")
st.session_state.setdefault("result", None)
st.session_state.setdefault("uploaded_bytes", None)
st.session_state.setdefault("uploaded_name", None)
st.session_state.setdefault("threshold", 0.5)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Barlow:wght@400;500;600;700;800;900&family=Barlow+Condensed:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ══════════════════════════════════════════
   DESIGN TOKENS — Forge / Heavy-Industrial
   ══════════════════════════════════════════ */
:root {
  /* Surfaces */
  --bg:          #111110;
  --surface:     #1A1917;
  --surface-2:   #242320;
  --surface-3:   #2E2C29;
  --border:      #3A3733;
  --border-soft: #2A2825;

  /* Typography */
  --text:        #F0EDE8;
  --text-2:      #B8B3AC;
  --text-3:      #787066;
  --text-inv:    #111110;

  /* Accent — molten amber / forge orange */
  --amber:       #E8972A;
  --amber-dim:   #C47E1F;
  --amber-bright:#F5B042;
  --amber-bg:    rgba(232, 151, 42, 0.08);
  --amber-border:rgba(232, 151, 42, 0.25);

  /* Status */
  --ok:          #5A9E6F;
  --ok-bg:       rgba(90, 158, 111, 0.10);
  --ok-border:   rgba(90, 158, 111, 0.30);
  --warn:        #C4762A;
  --warn-bg:     rgba(196, 118, 42, 0.10);
  --warn-border: rgba(196, 118, 42, 0.30);
  --err:         #B84040;
  --err-bg:      rgba(184, 64, 64, 0.10);
  --err-border:  rgba(184, 64, 64, 0.30);

  /* Class colours (kept from model — these are fixed) */
  --c1: #C94040;
  --c2: #3D8C60;
  --c3: #C4762A;
  --c4: #7A5C8A;

  /* Typography */
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'Barlow', sans-serif;
  --cond: 'Barlow Condensed', sans-serif;
}

/* ── Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]       { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stDecoration"]    { display: none !important; }
.stApp > header                 { display: none !important; }

/* ── App background ── */
.stApp {
  background: var(--bg) !important;
  font-family: var(--sans);
  color: var(--text);
}
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Typography base ── */
h1, h2, h3, h4, h5 {
  font-family: var(--cond);
  color: var(--text);
  font-weight: 700;
  letter-spacing: 0.01em;
}
p { color: var(--text-2); line-height: 1.65; }

/* ══════════════ NAV ══════════════ */
.ss-nav {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 2rem;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 999;
}
.ss-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.ss-brand-mark {
  width: 32px;
  height: 32px;
  background: var(--amber);
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  color: var(--text-inv);
  font-weight: 900;
  font-family: var(--cond);
  letter-spacing: -0.05em;
  flex-shrink: 0;
}
.ss-brand-name {
  font-family: var(--cond);
  font-size: 1.3rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text);
}
.ss-brand-name span {
  color: var(--amber);
}
.ss-status-pill {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 2px;
  border: 1px solid var(--ok-border);
  color: var(--ok);
  background: var(--ok-bg);
}

/* ══════════════ PANELS / CARDS ══════════════ */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.5rem;
}
.panel-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border-soft);
}
.panel-title {
  font-family: var(--cond);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-3);
}
.panel-num {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--amber);
  border: 1px solid var(--amber-border);
  background: var(--amber-bg);
  padding: 1px 6px;
  border-radius: 2px;
}

/* ══════════════ STAT BLOCKS ══════════════ */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}
.stat-cell {
  background: var(--surface-2);
  padding: 1rem 1.25rem;
}
.stat-label {
  font-family: var(--mono);
  font-size: 0.63rem;
  font-weight: 500;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: 0.4rem;
}
.stat-value {
  font-family: var(--cond);
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--text);
  line-height: 1;
}
.stat-unit {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--text-3);
  margin-top: 0.2rem;
}

/* ══════════════ TAGS / BADGES ══════════════ */
.tag {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-family: var(--mono);
  font-size: 0.63rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 2px;
  border: 1px solid;
}
.tag-amber  { color: var(--amber-bright); border-color: var(--amber-border); background: var(--amber-bg); }
.tag-ok     { color: var(--ok);           border-color: var(--ok-border);    background: var(--ok-bg);    }
.tag-err    { color: var(--err);          border-color: var(--err-border);   background: var(--err-bg);   }
.tag-warn   { color: var(--warn);         border-color: var(--warn-border);  background: var(--warn-bg);  }
.tag-muted  { color: var(--text-3);       border-color: var(--border);       background: var(--surface-2);}

/* ══════════════ BUTTONS ══════════════ */
/* Primary */
.stButton > button {
  background: var(--amber) !important;
  color: var(--text-inv) !important;
  border: none !important;
  border-radius: 4px !important;
  font-family: var(--cond) !important;
  font-size: 0.95rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 0.65rem 1.5rem !important;
  transition: background 0.15s !important;
  box-shadow: none !important;
}
.stButton > button:hover  { background: var(--amber-bright) !important; }
.stButton > button:active { background: var(--amber-dim) !important; }
.stButton > button:disabled {
  background: var(--surface-3) !important;
  color: var(--text-3) !important;
  cursor: not-allowed !important;
}

/* Secondary via wrapper */
.btn-ghost > button {
  background: transparent !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border) !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  font-size: 0.82rem !important;
}
.btn-ghost > button:hover {
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border-color: var(--amber) !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
  background: var(--surface-3) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  box-shadow: none !important;
  padding: 0.4rem 1rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: var(--surface-2) !important;
  border-color: var(--amber) !important;
  color: var(--text) !important;
}

/* ══════════════ SLIDER ══════════════ */
[data-testid="stSlider"] > div > div > div > div {
  background: var(--amber) !important;
}
[data-testid="stSlider"] > div > div > div {
  background: var(--surface-3) !important;
}

/* ══════════════ FILE UPLOADER ══════════════ */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--border) !important;
  border-radius: 4px !important;
  background: var(--surface-2) !important;
  padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface-2) !important;
  border: none !important;
  padding: 2rem 1.5rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
  color: var(--text-3) !important;
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
}
[data-testid="stFileUploader"] button {
  background: var(--surface-3) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  box-shadow: none !important;
}

/* ══════════════ TABS ══════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  color: var(--text-3) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.65rem 1rem !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--amber) !important;
  border-bottom-color: var(--amber) !important;
  background: transparent !important;
}

/* ══════════════ IMAGE ══════════════ */
[data-testid="stImage"] img {
  border-radius: 3px;
  display: block;
}

/* ══════════════ ALERTS ══════════════ */
[data-testid="stAlert"] {
  border-radius: 4px !important;
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
}

/* ══════════════ FORM ══════════════ */
[data-testid="stForm"] {
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
}
input[type="text"], input[type="email"], textarea {
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  font-family: var(--mono) !important;
  font-size: 0.85rem !important;
}
input:focus, textarea:focus {
  border-color: var(--amber) !important;
  outline: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
}

/* ══════════════ MISC ══════════════ */
hr { border-color: var(--border) !important; margin: 0 !important; }
.stMarkdown p { color: var(--text-2); }
.stMarkdown li { color: var(--text-2); }

/* Ruler / divider */
.h-rule {
  height: 1px;
  background: var(--border);
  margin: 1.5rem 0;
}

/* Section wrapper */
.section-wrap {
  padding: 3rem 2.5rem;
}
.section-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--amber);
  margin-bottom: 0.5rem;
}
.section-title {
  font-family: var(--cond);
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: 0.01em;
  color: var(--text);
  line-height: 1.1;
  text-transform: uppercase;
}
.section-sub {
  font-family: var(--sans);
  font-size: 1rem;
  color: var(--text-3);
  line-height: 1.65;
  margin-top: 0.75rem;
  max-width: 640px;
}

/* Class strip indicator */
.class-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.65rem 0;
  border-bottom: 1px solid var(--border-soft);
}
.class-swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}
.class-name {
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--text-2);
  font-weight: 500;
  flex: 1;
}
.class-metric {
  font-family: var(--mono);
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text);
}

/* ══════ HERO NUMBER ══════ */
.hero-num {
  font-family: var(--cond);
  font-size: clamp(5rem, 12vw, 10rem);
  font-weight: 900;
  color: var(--surface-3);
  line-height: 0.9;
  position: absolute;
  right: 2rem;
  top: 50%;
  transform: translateY(-50%);
  user-select: none;
  pointer-events: none;
  letter-spacing: -0.04em;
}

/* Inspection result bar */
.result-bar {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1.25rem;
  border-radius: 4px;
  margin-bottom: 1.5rem;
  border-left: 3px solid;
}
.result-bar.defect {
  background: var(--err-bg);
  border-color: var(--err);
}
.result-bar.clean {
  background: var(--ok-bg);
  border-color: var(--ok);
}
.result-bar-icon { font-size: 1.1rem; }
.result-bar-text {
  font-family: var(--cond);
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.result-bar.defect .result-bar-text { color: var(--err); }
.result-bar.clean  .result-bar-text { color: var(--ok);  }

/* Footer */
.ss-footer {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 2rem 2.5rem;
  margin-top: 4rem;
}
.footer-brand {
  font-family: var(--cond);
  font-size: 1rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-2);
  margin-bottom: 0.4rem;
}
.footer-copy {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--text-3);
  letter-spacing: 0.06em;
}

/* Mono data display */
.data-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border-soft);
}
.data-key {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--text-3);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.data-val {
  font-family: var(--mono);
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text);
}

/* Checkerboard placeholder bg for image viewer */
.img-canvas {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}

/* Class card big */
.cls-card {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.1rem;
  position: relative;
  overflow: hidden;
}
.cls-card-accent {
  position: absolute;
  top: 0; left: 0;
  width: 3px;
  height: 100%;
}
.cls-card-name {
  font-family: var(--cond);
  font-size: 1rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text);
  margin-bottom: 0.6rem;
}
.cls-stat-row {
  display: flex;
  gap: 1.5rem;
  margin-top: 0.5rem;
}
.cls-stat { }
.cls-stat-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.cls-stat-num {
  font-family: var(--mono);
  font-size: 1rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1.1;
  margin-top: 0.1rem;
}

/* Contact */
.contact-field-label {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.10em;
  margin-bottom: 0.3rem;
}

/* Page content area */
.page-body {
  max-width: 1240px;
  margin: 0 auto;
  padding: 2.5rem 2rem 4rem;
}

/* Notice/disclaimer */
.notice {
  background: var(--surface-2);
  border-left: 3px solid var(--border);
  border-radius: 0 3px 3px 0;
  padding: 0.85rem 1.1rem;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--text-3);
  line-height: 1.6;
}
.notice strong { color: var(--text-2); }

/* Spec table */
.spec-table { width: 100%; border-collapse: collapse; }
.spec-table tr { border-bottom: 1px solid var(--border-soft); }
.spec-table td {
  padding: 0.65rem 0.75rem;
  font-family: var(--mono);
  font-size: 0.78rem;
  vertical-align: top;
}
.spec-table td:first-child { color: var(--text-3); width: 40%; text-transform: uppercase; letter-spacing: 0.06em; }
.spec-table td:last-child  { color: var(--text); font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

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


@st.cache_resource(show_spinner="Loading inspection model…")
def load_model():
    if not model_manager.is_loaded:
        model_manager.load()
    return {
        "device":    str(model_manager.device),
        "epoch":     model_manager.epoch,
        "best_loss": model_manager.best_loss,
        "is_loaded": model_manager.is_loaded,
    }


# ═══════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════

def render_navbar():
    cur = st.session_state["page"]

    st.markdown(
        '<div class="ss-nav">'
        '<div class="ss-brand">'
        '  <div class="ss-brand-mark">SS</div>'
        '  <div class="ss-brand-name">Steel<span>Sight</span></div>'
        '  <span class="ss-status-pill">● ONLINE</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    n1, n2, n3, n4, _, n5 = st.columns([1, 1, 1, 1, 3, 1.4])
    with n1:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("Home", key="nav_home", use_container_width=True):
            nav_to("home")
        st.markdown('</div>', unsafe_allow_html=True)
    with n2:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("Inspect", key="nav_inspect", use_container_width=True):
            nav_to("inspection")
        st.markdown('</div>', unsafe_allow_html=True)
    with n3:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("About", key="nav_about", use_container_width=True):
            nav_to("about")
        st.markdown('</div>', unsafe_allow_html=True)
    with n4:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("Contact", key="nav_contact", use_container_width=True):
            nav_to("contact")
        st.markdown('</div>', unsafe_allow_html=True)
    with n5:
        if st.button("▶  Open Inspection Console", key="nav_launch", use_container_width=True):
            nav_to("inspection")

    st.markdown('<hr>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

def render_footer():
    st.markdown(
        '<div class="ss-footer">'
        '<div class="footer-brand">SteelSight &mdash; Surface Defect Inspection System</div>'
        '<div class="footer-copy">'
        '© 2025 SteelSight &nbsp;·&nbsp; ResNet-18 U-Net &nbsp;·&nbsp; Severstal Corpus &nbsp;·&nbsp;'
        ' Epoch 18 &nbsp;·&nbsp; Best Loss 0.01176'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAGE — HOME
# ═══════════════════════════════════════════════════════════════

def page_home():
    # ── Hero ──────────────────────────────────────────────────
    st.markdown(
        '<div style="background:var(--surface);border-bottom:1px solid var(--border);'
        'padding:4rem 2.5rem 3.5rem;position:relative;overflow:hidden;">'
        '<div style="max-width:720px;">'
        '<div class="section-label">Industrial Inspection System · Rev 1.0</div>'
        '<div class="section-title">Steel Surface<br>Defect Detection</div>'
        '<div class="section-sub">'
        'High-throughput pixel-level segmentation for continuous rolled sheet steel. '
        'Decomposes surface anomalies into four independently classified defect channels '
        'with quantitative coverage metrics.'
        '</div>'
        '</div>'
        '<div class="hero-num">01</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    h1, h2, _, h3 = st.columns([1.2, 1.2, 4, 1])
    with h1:
        if st.button("▶  Start Inspection", key="hero_start", use_container_width=True):
            nav_to("inspection")
    with h2:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("System Specs", key="hero_specs", use_container_width=True):
            nav_to("about")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Key numbers ───────────────────────────────────────────
    st.markdown(
        '<div class="stat-grid">'

        '<div class="stat-cell">'
        '<div class="stat-label">Defect Classes</div>'
        '<div class="stat-value">4</div>'
        '<div class="stat-unit">Independent channels</div>'
        '</div>'

        '<div class="stat-cell">'
        '<div class="stat-label">Input Resolution</div>'
        '<div class="stat-value">1600<span style="font-size:1rem;color:var(--text-3)">×256</span></div>'
        '<div class="stat-unit">Pixels (RGB)</div>'
        '</div>'

        '<div class="stat-cell">'
        '<div class="stat-label">Encoder Backbone</div>'
        '<div class="stat-value" style="font-size:1.1rem;">ResNet-18</div>'
        '<div class="stat-unit">ImageNet pre-trained</div>'
        '</div>'

        '<div class="stat-cell">'
        '<div class="stat-label">Training Epochs</div>'
        '<div class="stat-value">18</div>'
        '<div class="stat-unit">Best loss: 0.01176</div>'
        '</div>'

        '<div class="stat-cell">'
        '<div class="stat-label">Max Upload</div>'
        '<div class="stat-value">15</div>'
        '<div class="stat-unit">MB per image</div>'
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<br><br>', unsafe_allow_html=True)

    # ── Three capability columns ───────────────────────────────
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.65rem;letter-spacing:0.14em;'
        'text-transform:uppercase;color:var(--amber);margin-bottom:1.25rem;">'
        '// System Capabilities'
        '</div>',
        unsafe_allow_html=True,
    )

    cap1, cap2, cap3 = st.columns(3, gap="small")

    caps = [
        ("01", "Multi-channel Decomposition",
         "Segments four co-occurring defect classes in a single forward pass using independent "
         "sigmoid-activated output heads. No class competition."),
        ("02", "Spatial Confidence Maps",
         "Projects maximum per-pixel probability across all channels using an inferno colormap, "
         "surfacing stress concentrations below the binary threshold."),
        ("03", "Quantitative Coverage",
         "Reports defective pixel counts and percentage surface area per class, enabling "
         "quantitative pass/fail decisions against production tolerances."),
    ]

    for col, (num, title, desc) in zip([cap1, cap2, cap3], caps):
        with col:
            st.markdown(
                f'<div class="panel" style="height:100%;">'
                f'<div style="font-family:var(--mono);font-size:0.65rem;color:var(--amber);'
                f'letter-spacing:0.12em;margin-bottom:0.75rem;">{num}</div>'
                f'<div style="font-family:var(--cond);font-size:1.05rem;font-weight:800;'
                f'text-transform:uppercase;letter-spacing:0.03em;color:var(--text);'
                f'margin-bottom:0.6rem;">{title}</div>'
                f'<div style="font-size:0.9rem;color:var(--text-3);line-height:1.65;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<br><br>', unsafe_allow_html=True)

    # ── Pipeline steps ─────────────────────────────────────────
    st.markdown(
        '<div style="background:var(--surface);border-top:1px solid var(--border);'
        'border-bottom:1px solid var(--border);padding:2.5rem 2rem;margin:0 -2rem;">'

        '<div style="font-family:var(--mono);font-size:0.65rem;letter-spacing:0.14em;'
        'text-transform:uppercase;color:var(--amber);margin-bottom:1.25rem;">// Inspection Pipeline</div>'

        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;'
        'background:var(--border);border:1px solid var(--border);border-radius:4px;overflow:hidden;">'

        '<div style="background:var(--surface-2);padding:1.25rem;">'
        '<div style="font-family:var(--mono);font-size:0.6rem;color:var(--amber);letter-spacing:0.1em;margin-bottom:0.5rem;">STEP 01</div>'
        '<div style="font-family:var(--cond);font-size:0.95rem;font-weight:800;text-transform:uppercase;'
        'color:var(--text);margin-bottom:0.4rem;">Image Ingestion</div>'
        '<div style="font-size:0.85rem;color:var(--text-3);line-height:1.6;">'
        'Upload a surface capture from a line scanner, industrial camera, or inspection station.</div>'
        '</div>'

        '<div style="background:var(--surface-2);padding:1.25rem;">'
        '<div style="font-family:var(--mono);font-size:0.6rem;color:var(--amber);letter-spacing:0.1em;margin-bottom:0.5rem;">STEP 02</div>'
        '<div style="font-family:var(--cond);font-size:0.95rem;font-weight:800;text-transform:uppercase;'
        'color:var(--text);margin-bottom:0.4rem;">Neural Segmentation</div>'
        '<div style="font-size:0.85rem;color:var(--text-3);line-height:1.6;">'
        'ResNet-18 encoder extracts features; U-Net decoder reconstructs defect masks per class.</div>'
        '</div>'

        '<div style="background:var(--surface-2);padding:1.25rem;">'
        '<div style="font-family:var(--mono);font-size:0.6rem;color:var(--amber);letter-spacing:0.1em;margin-bottom:0.5rem;">STEP 03</div>'
        '<div style="font-family:var(--cond);font-size:0.95rem;font-weight:800;text-transform:uppercase;'
        'color:var(--text);margin-bottom:0.4rem;">Report & Export</div>'
        '<div style="font-size:0.85rem;color:var(--text-3);line-height:1.6;">'
        'Download visualizations, review per-class pixel counts, and assess coverage percentage.</div>'
        '</div>'

        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<br><br>', unsafe_allow_html=True)

    # ── CTA ────────────────────────────────────────────────────
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div class="panel" style="text-align:center;padding:2.5rem 2rem;">'
            '<div style="font-family:var(--mono);font-size:0.65rem;color:var(--amber);'
            'letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.75rem;">'
            '// Ready to inspect'
            '</div>'
            '<div style="font-family:var(--cond);font-size:1.6rem;font-weight:800;'
            'text-transform:uppercase;color:var(--text);margin-bottom:0.5rem;">'
            'Begin Surface Analysis'
            '</div>'
            '<div style="font-size:0.9rem;color:var(--text-3);margin-bottom:1.5rem;">'
            'Upload an image and receive pixel-level defect classification within seconds.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("▶  Open Inspection Console", key="cta_inspect", use_container_width=True):
            nav_to("inspection")


# ═══════════════════════════════════════════════════════════════
# PAGE — INSPECTION
# ═══════════════════════════════════════════════════════════════

def page_inspection(model_info: dict):
    st.markdown('<div class="page-body">', unsafe_allow_html=True)

    # Header row
    dev = model_info.get("device", "cpu").upper()
    ep  = model_info.get("epoch", 18)
    bl  = model_info.get("best_loss", 0.01176)

    st.markdown(
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        f'margin-bottom:2rem;gap:1rem;flex-wrap:wrap;">'
        f'<div>'
        f'<div class="section-label">Inspection Console</div>'
        f'<div class="section-title">Surface Analysis</div>'
        f'</div>'
        f'<div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;padding-top:0.5rem;">'
        f'<span class="tag tag-ok">● Model Loaded</span>'
        f'<span class="tag tag-muted">{dev}</span>'
        f'<span class="tag tag-muted">Epoch {ep}</span>'
        f'<span class="tag tag-muted">Loss {bl:.5f}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([3, 2], gap="large")

    # ── Left: upload ──────────────────────────────────────────
    with left_col:
        st.markdown(
            '<div class="panel-header">'
            '<span class="panel-num">01</span>'
            '<span class="panel-title">Image Upload</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Drop a surface image here — JPG · PNG · WEBP · BMP  ·  Max 15 MB",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="inspect_uploader",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            if uploaded.size > config.MAX_UPLOAD_SIZE:
                st.error(f"File exceeds 15 MB limit ({uploaded.size // (1024*1024):.1f} MB uploaded).")
            else:
                st.session_state["uploaded_bytes"] = uploaded.getvalue()
                st.session_state["uploaded_name"]  = uploaded.name

                img = Image.open(io.BytesIO(st.session_state["uploaded_bytes"]))
                w, h = img.size
                size_kb = uploaded.size // 1024

                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="panel-header">'
                    '<span class="panel-num">02</span>'
                    '<span class="panel-title">Image Preview</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="img-canvas">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div style="display:flex;gap:2rem;padding:0.6rem 0;'
                    f'border-top:1px solid var(--border-soft);margin-top:0.5rem;">'
                    f'<div><div class="data-key">Filename</div>'
                    f'<div class="data-val">{uploaded.name}</div></div>'
                    f'<div><div class="data-key">Dimensions</div>'
                    f'<div class="data-val">{w} × {h} px</div></div>'
                    f'<div><div class="data-key">File size</div>'
                    f'<div class="data-val">{size_kb} KB</div></div>'
                    f'<div><div class="data-key">Model input</div>'
                    f'<div class="data-val" style="color:var(--amber);">1600 × 256 px</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Right: controls ───────────────────────────────────────
    with right_col:
        # Threshold
        st.markdown(
            '<div class="panel">'
            '<div class="panel-header">'
            '<span class="panel-num">03</span>'
            '<span class="panel-title">Detection Threshold</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        cur_thresh = st.session_state["threshold"]
        new_thresh = st.slider(
            "threshold",
            min_value=0.05,
            max_value=0.95,
            value=cur_thresh,
            step=0.05,
            key="thresh_slider",
            label_visibility="collapsed",
        )
        st.session_state["threshold"] = new_thresh

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:var(--mono);font-size:0.65rem;color:var(--text-3);margin-top:0.4rem;">'
            f'<span>High Recall (0.05)</span>'
            f'<span style="color:var(--amber);font-weight:600;">{new_thresh:.2f}</span>'
            f'<span>High Precision (0.95)</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<br>', unsafe_allow_html=True)

        # Class legend
        st.markdown(
            '<div class="panel">'
            '<div class="panel-header">'
            '<span class="panel-num">04</span>'
            '<span class="panel-title">Defect Class Map</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        class_colors = ["#C94040", "#3D8C60", "#C4762A", "#7A5C8A"]
        class_labels = ["Crazing", "Inclusion", "Patches", "Pitted Surface"]
        for i, (color, label) in enumerate(zip(class_colors, class_labels)):
            is_last = i == len(class_colors) - 1
            border_style = "" if is_last else "border-bottom:1px solid var(--border-soft);"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.75rem;'
                f'padding:0.6rem 0;{border_style}">'
                f'<div style="width:10px;height:10px;border-radius:2px;background:{color};flex-shrink:0;"></div>'
                f'<div style="font-family:var(--mono);font-size:0.72rem;color:var(--text-3);'
                f'flex:1;text-transform:uppercase;letter-spacing:0.06em;">Class {i+1}</div>'
                f'<div style="font-family:var(--sans);font-size:0.85rem;font-weight:600;color:var(--text-2);">'
                f'{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    has_image = st.session_state.get("uploaded_bytes") is not None

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    btn_label = "▶  Run Defect Segmentation" if has_image else "Upload an image above to begin"
    run_clicked = st.button(
        btn_label,
        key="run_btn",
        disabled=not has_image,
        use_container_width=True,
    )

    if run_clicked and has_image:
        with st.spinner("Running ResNet-18 U-Net forward pass…"):
            try:
                result = run_inference(
                    image_bytes=st.session_state["uploaded_bytes"],
                    filename=st.session_state["uploaded_name"],
                    threshold=st.session_state["threshold"],
                )
                st.session_state["result"] = result
                nav_to("results")
            except Exception as exc:
                st.error(f"Inference error: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE — RESULTS
# ═══════════════════════════════════════════════════════════════

def page_results():
    result = st.session_state.get("result")

    st.markdown('<div class="page-body">', unsafe_allow_html=True)

    if result is None:
        st.markdown(
            '<div style="max-width:480px;margin:4rem auto 0;">'
            '<div class="panel" style="text-align:center;padding:2.5rem 2rem;">'
            '<div style="font-size:2rem;margin-bottom:0.75rem;">⚠</div>'
            '<div style="font-family:var(--cond);font-size:1.2rem;font-weight:800;'
            'text-transform:uppercase;color:var(--text);margin-bottom:0.6rem;">No Inspection Data</div>'
            '<div style="font-size:0.88rem;color:var(--text-3);line-height:1.6;margin-bottom:1.5rem;">'
            'No completed inspection was found. Submit an image in the inspection console.'
            '</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("← Go to Inspection", key="results_fallback"):
            nav_to("inspection")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Top status bar ────────────────────────────────────────
    defect = result.get("any_defect_detected", False)
    dev    = result.get("device", "cpu").upper()
    thr    = result.get("threshold", 0.5)
    fname  = result.get("filename", "image")
    ow     = result.get("original_width", "?")
    oh     = result.get("original_height", "?")

    top_l, top_r = st.columns([1, 3])
    with top_l:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("← New Inspection", key="results_back"):
            nav_to("inspection")
        st.markdown('</div>', unsafe_allow_html=True)

    with top_r:
        status_tag = (
            '<span class="tag tag-err">⚠ Defects Detected</span>'
            if defect else
            '<span class="tag tag-ok">✓ No Defects Above Threshold</span>'
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">'
            f'{status_tag}'
            f'<span class="tag tag-muted">{dev}</span>'
            f'<span class="tag tag-muted">Threshold {thr:.2f}</span>'
            f'<span class="tag tag-muted">{fname}</span>'
            f'<span class="tag tag-muted">{ow}×{oh} px source</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Result banner ─────────────────────────────────────────
    bar_cls  = "defect" if defect else "clean"
    bar_icon = "⚠" if defect else "✓"
    bar_text = "SURFACE DEFECTS DETECTED" if defect else "SURFACE WITHIN THRESHOLD — NO DEFECTS DETECTED"
    st.markdown(
        f'<div class="result-bar {bar_cls}">'
        f'<span class="result-bar-icon">{bar_icon}</span>'
        f'<span class="result-bar-text">{bar_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Visualizer ────────────────────────────────────────────
    st.markdown(
        '<div class="panel-header">'
        '<span class="panel-num">01</span>'
        '<span class="panel-title">Visual Analysis</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    TAB_NAMES  = ["COLOR OVERLAY", "ORIGINAL STRIP", "CONF. HEATMAP",
                  "CLASS 1", "CLASS 2", "CLASS 3", "CLASS 4"]
    b64_images = [
        result.get("overlay_image"),
        result.get("original_image"),
        result.get("heatmap_image"),
        result["masks"][0] if result.get("masks") else None,
        result["masks"][1] if result.get("masks") else None,
        result["masks"][2] if result.get("masks") else None,
        result["masks"][3] if result.get("masks") else None,
    ]
    dl_names = ["overlay", "original", "heatmap", "class1", "class2", "class3", "class4"]
    captions = [
        "Multi-channel color overlay — 45% alpha blend over raw strip texture",
        f"Standardized strip input ({ow}×{oh} px → 1600×256 px)",
        "Max-probability confidence map — Inferno colormap",
        "Class 1: Crazing — Binary segmentation mask",
        "Class 2: Inclusion — Binary segmentation mask",
        "Class 3: Patches — Binary segmentation mask",
        "Class 4: Pitted Surface — Binary segmentation mask",
    ]

    tabs = st.tabs(TAB_NAMES)
    for tab, b64, caption, dl_name in zip(tabs, b64_images, captions, dl_names):
        with tab:
            if b64:
                st.markdown('<div class="img-canvas">', unsafe_allow_html=True)
                st.image(b64_to_image(b64), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                dl_col, cap_col = st.columns([1, 4])
                with dl_col:
                    st.download_button(
                        "⬇ Export PNG",
                        data=b64_to_bytes(b64),
                        file_name=f"steelsight_{fname}_{dl_name}.png",
                        mime="image/png",
                        key=f"dl_{dl_name}",
                        use_container_width=True,
                    )
                with cap_col:
                    st.markdown(
                        f'<div style="font-family:var(--mono);font-size:0.68rem;'
                        f'color:var(--text-3);padding-top:0.6rem;">{caption}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Image data unavailable.")

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Class breakdown ───────────────────────────────────────
    st.markdown(
        '<div class="panel-header" style="margin-top:0.5rem;">'
        '<span class="panel-num">02</span>'
        '<span class="panel-title">Per-Class Defect Metrics</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    classes_info  = result.get("classes", [])
    cls_colors    = ["#C94040", "#3D8C60", "#C4762A", "#7A5C8A"]
    cls_labels    = ["Crazing", "Inclusion", "Patches", "Pitted Surface"]

    if classes_info:
        c_cols = st.columns(4)
        for col, cls_data, color, label in zip(c_cols, classes_info, cls_colors, cls_labels):
            detected = cls_data.get("detected", False)
            px       = cls_data.get("pixels", 0)
            area     = cls_data.get("area_percent", 0.0)
            with col:
                badge = (
                    f'<span class="tag tag-err">PRESENT</span>'
                    if detected else
                    '<span class="tag tag-muted">CLEAR</span>'
                )
                st.markdown(
                    f'<div class="cls-card">'
                    f'<div class="cls-card-accent" style="background:{color};"></div>'
                    f'<div class="cls-card-name" style="color:{color};">{label}</div>'
                    f'{badge}'
                    f'<div class="cls-stat-row">'
                    f'  <div class="cls-stat">'
                    f'    <div class="cls-stat-label">Defect px</div>'
                    f'    <div class="cls-stat-num">{px:,}</div>'
                    f'  </div>'
                    f'  <div class="cls-stat">'
                    f'    <div class="cls-stat-label">Area</div>'
                    f'    <div class="cls-stat-num">{area:.3f}%</div>'
                    f'  </div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown(
        '<div class="notice">'
        '<strong>Responsible Use —</strong> '
        'These predictions are generated by a convolutional neural network trained on the Severstal benchmark corpus. '
        'Results are intended as automated screening assistance and must be validated against certified '
        'Non-Destructive Testing (NDT) protocols before use in critical quality-control decisions.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE — ABOUT
# ═══════════════════════════════════════════════════════════════

def page_about():
    st.markdown('<div class="page-body" style="max-width:960px;">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">System Documentation</div>'
        '<div class="section-title">Architecture &amp; Specifications</div>'
        '<div class="section-sub">'
        'SteelSight localizes and quantifies surface anomalies on flat-rolled steel strip '
        'using a fully convolutional encoder-decoder network.'
        '</div>'
        '<br>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        # Architecture table
        st.markdown(
            '<div class="panel">'
            '<div class="panel-header">'
            '<span class="panel-num">01</span>'
            '<span class="panel-title">Model Specifications</span>'
            '</div>'
            '<table class="spec-table">'
            '<tr><td>Encoder backbone</td><td>ResNet-18 (ImageNet)</td></tr>'
            '<tr><td>Decoder topology</td><td>U-Net — 5 stages</td></tr>'
            '<tr><td>Decoder channels</td><td>256 · 128 · 64 · 32 · 16</td></tr>'
            '<tr><td>Input resolution</td><td>1600 × 256 px (RGB)</td></tr>'
            '<tr><td>Output heads</td><td>4 × independent Sigmoid</td></tr>'
            '<tr><td>Loss function</td><td>BCEWithLogitsLoss</td></tr>'
            '<tr><td>Optimizer</td><td>Adam · lr = 5e-4</td></tr>'
            '<tr><td>LR scheduler</td><td>ReduceLROnPlateau · patience 3</td></tr>'
            '<tr><td>Training corpus</td><td>Severstal Steel Defect Dataset</td></tr>'
            '<tr><td>Best epoch</td><td>18 / 20</td></tr>'
            '<tr><td>Best validation loss</td><td>0.01176</td></tr>'
            '<tr><td>Checkpoint SHA-256</td><td style="word-break:break-all;font-size:0.68rem;">076f044…efe8</td></tr>'
            '</table>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        # Preprocessing
        st.markdown(
            '<div class="panel" style="margin-bottom:1rem;">'
            '<div class="panel-header">'
            '<span class="panel-num">02</span>'
            '<span class="panel-title">Preprocessing Protocol</span>'
            '</div>'
            '<table class="spec-table">'
            '<tr><td>Color order</td><td>BGR → RGB conversion</td></tr>'
            '<tr><td>Resize</td><td>Bilinear → 1600 × 256 px</td></tr>'
            '<tr><td>Scaling</td><td>uint8 → float32 / 255.0</td></tr>'
            '<tr><td>Mean (R, G, B)</td><td>0.485 · 0.456 · 0.406</td></tr>'
            '<tr><td>Std  (R, G, B)</td><td>0.229 · 0.224 · 0.225</td></tr>'
            '<tr><td>Tensor shape</td><td>(1, 3, 256, 1600)</td></tr>'
            '</table>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Class colours
        st.markdown(
            '<div class="panel">'
            '<div class="panel-header">'
            '<span class="panel-num">03</span>'
            '<span class="panel-title">Output Class Mapping</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for i, (color, label) in enumerate([
            ("#C94040", "Crazing"),
            ("#3D8C60", "Inclusion"),
            ("#C4762A", "Patches"),
            ("#7A5C8A", "Pitted Surface"),
        ]):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.55rem 0;'
                f'border-bottom:1px solid var(--border-soft);">'
                f'<div style="width:10px;height:10px;border-radius:2px;'
                f'background:{color};flex-shrink:0;"></div>'
                f'<div style="font-family:var(--mono);font-size:0.68rem;'
                f'color:var(--text-3);width:52px;letter-spacing:0.08em;">CLASS {i+1}</div>'
                f'<div style="font-size:0.88rem;font-weight:600;color:var(--text-2);">{label}</div>'
                f'<div style="margin-left:auto;font-family:var(--mono);font-size:0.68rem;'
                f'color:{color};">{color.upper()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Operational constraints
    st.markdown(
        '<div class="panel" style="border-left:3px solid var(--warn);">'
        '<div class="panel-header">'
        '<span class="panel-num">04</span>'
        '<span class="panel-title">Operational Constraints</span>'
        '</div>'
        '<ul style="color:var(--text-3);font-size:0.88rem;padding-left:1.25rem;'
        'line-height:1.75;margin:0;">'
        '<li>Performance assumes lighting conditions consistent with rolling-mill inspection hoods. '
        'Non-uniform illumination or lens flare may produce spurious activations.</li>'
        '<li>The model is validated on 1600 × 256 px flat-rolled steel captures. '
        'Non-ferrous or textured alloy surfaces fall outside the training distribution.</li>'
        '<li>Output predictions are automated screening assistance. '
        'Critical quality-control sign-off must follow certified NDT verification protocols (ISO 9001).</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE — CONTACT
# ═══════════════════════════════════════════════════════════════

def page_contact():
    st.markdown('<div class="page-body" style="max-width:680px;">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Direct Inquiries</div>'
        '<div class="section-title">Engineering Contact</div>'
        '<div class="section-sub" style="margin-bottom:2rem;">'
        'Submit questions about deployment, camera calibration, model integration, '
        'or custom mill configuration.'
        '</div>',
        unsafe_allow_html=True,
    )

    submitted = st.session_state.get("contact_submitted", False)

    if submitted:
        name = st.session_state.get("contact_name", "")
        st.markdown(
            f'<div class="panel" style="text-align:center;padding:2.5rem 2rem;">'
            f'<div style="font-family:var(--mono);font-size:0.65rem;color:var(--ok);'
            f'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.75rem;">// Inquiry Registered</div>'
            f'<div style="font-family:var(--cond);font-size:1.3rem;font-weight:800;'
            f'text-transform:uppercase;color:var(--text);margin-bottom:0.6rem;">'
            f'Message Received, {name}'
            f'</div>'
            f'<div style="font-size:0.88rem;color:var(--text-3);line-height:1.6;margin-bottom:1.5rem;">'
            f'Your inquiry has been logged. Note: email delivery is not configured '
            f'on this standalone instance.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("Submit Another", key="contact_reset"):
            st.session_state.pop("contact_submitted", None)
            st.session_state.pop("contact_name", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.form("contact_form"):
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            name_in    = st.text_input("Full Name *",        placeholder="e.g. Elena Rostova",                     key="cf_name")
            email_in   = st.text_input("Corporate Email *",  placeholder="e.g. e.rostova@steelforge.com",           key="cf_email")
            subject_in = st.text_input("Subject *",          placeholder="e.g. Inference on 1.2 mm gauge strip",    key="cf_subject")
            msg_in     = st.text_area ("Inquiry / Message *",
                                       placeholder="Specify rolling conditions, camera hardware, and deployment objectives…",
                                       height=120, key="cf_msg")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            sub_btn = st.form_submit_button("Submit Technical Inquiry", use_container_width=True)

        if sub_btn:
            errs = []
            if not name_in.strip():    errs.append("Full name is required.")
            if not email_in.strip():   errs.append("Email is required.")
            elif "@" not in email_in:  errs.append("Please enter a valid email address.")
            if not subject_in.strip(): errs.append("Subject is required.")
            if len(msg_in.strip()) < 15: errs.append("Message must be at least 15 characters.")
            if errs:
                for e in errs: st.error(e)
            else:
                st.session_state["contact_submitted"] = True
                st.session_state["contact_name"]      = name_in.strip()
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    try:
        model_info = load_model()
    except Exception as exc:
        st.error(
            f"**Model initialization failed:** {exc}\n\n"
            "On Streamlit Cloud, set `MODEL_URL` and `EXPECTED_MODEL_SHA256` in Secrets."
        )
        st.stop()

    render_navbar()

    page = st.session_state["page"]
    if   page == "home":       page_home()
    elif page == "inspection":  page_inspection(model_info)
    elif page == "results":     page_results()
    elif page == "about":       page_about()
    elif page == "contact":     page_contact()
    else:                       page_home()

    render_footer()


if __name__ == "__main__":
    main()
