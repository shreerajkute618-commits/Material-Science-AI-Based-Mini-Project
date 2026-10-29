"""
AI-Based Identification and Sorting of Recyclable Materials
Streamlit Web Application for Materials Science and Engineering Laboratory.
Upgraded with ultra-modern custom sidebar navigation cards, pulsating status lamps,
and glassmorphic engineering styling.
"""

import os
import sys
import json
import io
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

import config
import predict
import torch

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Recyclable Material Identifier & Sorter",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Engineering & Glassmorphism Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }

    /* =========================================================
       SIDEBAR & MODERN NAVIGATION STYLING
       ========================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 50%, #E2E8F0 100%) !important;
        border-right: 1px solid #CBD5E1 !important;
        padding-top: 10px;
    }

    /* Sidebar Brand Header Box */
    .sidebar-brand-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 18px 16px;
        margin-bottom: 20px;
        color: #FFFFFF;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .sidebar-brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin-top: 8px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    .sidebar-brand-sub {
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
        margin-right: 5px;
    }
    @keyframes pulse {
        to {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
    }

    /* Nav Menu Category Header */
    .nav-section-label {
        font-size: 0.72rem;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-left: 6px;
        margin-bottom: 8px;
    }

    /* HIDE DEFAULT STREAMLIT RADIO CIRCLE BUTTONS */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 6px;
        padding: 0;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* TRANSFORM RADIO LABELS INTO MODERN FLOATING NAV PILLS */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 11px 15px !important;
        margin-bottom: 5px !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        margin: 0 !important;
        transition: color 0.2s ease !important;
    }

    /* NAV HOVER INTERACTION */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #F0FDF4 !important;
        border-color: #86EFAC !important;
        transform: translateX(4px) !important;
        box-shadow: 0 6px 14px rgba(16, 185, 129, 0.12) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
        color: #047857 !important;
    }

    /* ACTIVE NAV ITEM (GLOWING EMERALD PILL) */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, #0F766E 0%, #047857 100%) !important;
        border-color: #059669 !important;
        box-shadow: 0 8px 18px -3px rgba(15, 118, 110, 0.45) !important;
        transform: translateX(5px) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Sidebar Status Footer Card */
    .sidebar-footer-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px;
        margin-top: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.78rem;
        margin-bottom: 6px;
    }
    .status-label {
        color: #64748B;
        font-weight: 600;
    }
    .status-val {
        color: #0F172A;
        font-weight: 700;
    }

    /* =========================================================
       HERO & DASHBOARD STYLING
       ========================================================= */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        box-shadow: 0 16px 36px -10px rgba(15, 23, 42, 0.35);
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-tag {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 60%, #6EE7B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        color: #CBD5E1;
        font-size: 1.02rem;
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 0px;
    }

    /* Stat Cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .stat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .stat-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-icon {
        font-size: 1.4rem;
        padding: 8px;
        border-radius: 10px;
        background: #F1F5F9;
    }
    .stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1;
        margin-bottom: 6px;
    }
    .stat-sub {
        font-size: 0.78rem;
        color: #059669;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Material Showcase Cards */
    .material-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        transition: all 0.25s ease-in-out;
        height: 100%;
    }
    .material-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.09);
    }
    .material-icon-box {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        margin: 0 auto 14px auto;
    }
    .material-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .material-sci {
        font-size: 0.76rem;
        color: #64748B;
        margin-bottom: 12px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1.3;
    }
    .material-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .material-chute {
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569;
        border-top: 1px dashed #E2E8F0;
        padding-top: 10px;
    }

    /* Result Panel Styling */
    .result-card-main {
        background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%);
        border: 2px solid #3B82F6;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.12);
        margin-bottom: 20px;
    }
    .result-badge-top {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 10px;
    }
    .result-class-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .result-confidence-text {
        font-size: 1.05rem;
        color: #059669;
        font-weight: 700;
    }

    /* Uncertainty Warning Box */
    .uncertainty-box {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 10px;
        color: #92400E;
        margin: 16px 0;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Industrial Sorting Chutes */
    .sorting-chute {
        border-radius: 14px;
        padding: 16px 12px;
        text-align: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
        height: 100%;
    }
    .sorting-chute-active {
        border: 3px solid #10B981 !important;
        background: linear-gradient(180deg, #ECFDF5 0%, #FFFFFF 100%) !important;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.25) !important;
        transform: scale(1.04);
    }
    .conveyor-belt {
        background: repeating-linear-gradient(
            45deg,
            #1E293B,
            #1E293B 15px,
            #334155 15px,
            #334155 30px
        );
        height: 24px;
        border-radius: 12px;
        margin: 20px 0;
        border: 2px solid #0F172A;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }

    /* Scientific Cards */
    .sci-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }
    .sci-card-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 10px;
        margin-bottom: 14px;
    }

    /* Section Subheaders */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 24px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = predict.load_prediction_history()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def render_sidebar():
    """Renders the ultra-modern styled sidebar with custom brand card and interactive nav pills."""
    # Top Brand Header Card
    st.sidebar.markdown("""
    <div class='sidebar-brand-card'>
        <div style='font-size: 2.2rem;'>♻️</div>
        <div class='sidebar-brand-title'>
            RECYCLE·AI PRO
        </div>
        <div class='sidebar-brand-sub'>
            <span class='pulse-dot'></span>SYSTEM ONLINE • MSE LAB
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<div class='nav-section-label'>NAVIGATION MODULES</div>", unsafe_allow_html=True)

    pages = [
        "🏠 Dashboard",
        "🔍 Material Identification",
        "⚙️ Sorting Simulation",
        "📊 Model Performance",
        "📜 Prediction History",
        "🔬 Materials Science Guide",
        "ℹ️ Viva & Demo Prep"
    ]

    # Render Styled Radio (CSS turns this into modern cards with hover/glow)
    page = st.sidebar.radio("Navigation Menu", pages, label_visibility="collapsed")

    # Bottom System Telemetry Card
    model_exists = config.MODEL_PATH.exists()
    st.sidebar.markdown(f"""
    <div class='sidebar-footer-card'>
        <div class='status-row'>
            <span class='status-label'>AI Engine</span>
            <span class='status-val'>MobileNetV2</span>
        </div>
        <div class='status-row'>
            <span class='status-label'>Accuracy</span>
            <span class='status-val' style='color: #059669;'>75.78% (Test)</span>
        </div>
        <div class='status-row'>
            <span class='status-label'>Device</span>
            <span class='status-val'>{'GPU Accelerated' if torch.cuda.is_available() else 'CPU Engine'}</span>
        </div>
        <div class='status-row' style='margin-bottom: 0;'>
            <span class='status-label'>Status</span>
            <span class='status-val' style='color: #059669;'>{'Ready' if model_exists else 'Untrained'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return page


def page_dashboard():
    """Ultra-modern Overview Dashboard."""
    # Hero Banner Section
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-tag'>✨ Materials Science & Engineering Laboratory</div>
        <div class='hero-title'>AI-Based Identification & Sorting of Recyclable Materials</div>
        <p class='hero-desc'>
            An intelligent computer vision framework powered by MobileNetV2 deep transfer learning. 
            Automates the classification of post-consumer waste, links visual signatures to thermodynamic & microstructural properties, 
            and demonstrates pneumatic/magnetic sorting line diversion for the circular economy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Model Accuracy</span>
                <span class='stat-icon'>🎯</span>
            </div>
            <div>
                <div class='stat-value'>75.78%</div>
                <div class='stat-sub'>✓ Measured on 384 Unseen Test Images</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Neural Backbone</span>
                <span class='stat-icon'>🧠</span>
            </div>
            <div>
                <div class='stat-value'>MobileNetV2</div>
                <div class='stat-sub'>✓ Inverted Residuals • ~3.4M Params</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Material Streams</span>
                <span class='stat-icon'>♻️</span>
            </div>
            <div>
                <div class='stat-value'>6 Classes</div>
                <div class='stat-sub'>✓ Polymers, Metals, Glass, Cellulose</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Sorting Twin</span>
                <span class='stat-icon'>🏭</span>
            </div>
            <div>
                <div class='stat-value'>6 Chutes</div>
                <div class='stat-sub'>✓ Pneumatic, Eddy & Optical Diverters</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🗂️ Supported Circular Material Streams</div>", unsafe_allow_html=True)

    # 6 Material Stream Cards with Color Palettes
    cols = st.columns(6)
    card_styles = {
        "Plastic": {
            "icon": "🧴", "bg": "#FEF3C7", "border": "#F59E0B", "color": "#B45309",
            "desc": "Synthetic Polymer Chains (PET/HDPE)", "badge": "4-6 Cycles", "badge_bg": "#FDE68A", "badge_col": "#92400E",
            "chute": "Chute 1 • Optical NIR & Sink-Float"
        },
        "Metal": {
            "icon": "🥫", "bg": "#F1F5F9", "border": "#94A3B8", "color": "#475569",
            "desc": "Crystalline Metallic Lattice (Al/Steel)", "badge": "Infinitely Recyclable", "badge_bg": "#E2E8F0", "badge_col": "#334155",
            "chute": "Chute 2 • Eddy Current & Magnet"
        },
        "Paper": {
            "icon": "📄", "bg": "#EFF6FF", "border": "#60A5FA", "color": "#1D4ED8",
            "desc": "Hydrophilic Cellulose Microfibrils", "badge": "5-7 Cycles", "badge_bg": "#DBEAFE", "badge_col": "#1E40AF",
            "chute": "Chute 3 • Pneumatic Baling Line"
        },
        "Cardboard": {
            "icon": "📦", "bg": "#FFFBEB", "border": "#D97706", "color": "#92400E",
            "desc": "Corrugated Fluted Composite Board", "badge": "High Recovery", "badge_bg": "#FEF3C7", "badge_col": "#78350F",
            "chute": "Chute 4 • Heavy Shredder"
        },
        "Glass": {
            "icon": "🍾", "bg": "#ECFDF5", "border": "#34D399", "color": "#047857",
            "desc": "Amorphous Silicate Network (SiO2)", "badge": "100% Infinitely", "badge_bg": "#D1FAE5", "badge_col": "#065F46",
            "chute": "Chute 5 • Cullet Crusher"
        },
        "Trash": {
            "icon": "🗑️", "bg": "#F8FAFC", "border": "#64748B", "color": "#334155",
            "desc": "Multi-material Composites & Refuse", "badge": "Refuse Fuel (RDF)", "badge_bg": "#E2E8F0", "badge_col": "#1E293B",
            "chute": "Chute 6 • Energy Recovery / Landfill"
        }
    }

    for idx, (cat_key, col) in enumerate(zip(config.CLASS_NAMES, cols)):
        info = card_styles[cat_key]
        disp_name = config.DISPLAY_NAMES[cat_key]
        with col:
            st.markdown(f"""
            <div class='material-card'>
                <div class='material-icon-box' style='background: {info["bg"]}; color: {info["color"]};'>
                    {info["icon"]}
                </div>
                <div class='material-name'>{disp_name}</div>
                <div class='material-sci'>{info["desc"]}</div>
                <div>
                    <span class='material-badge' style='background: {info["badge_bg"]}; color: {info["badge_col"]};'>
                        {info["badge"]}
                    </span>
                </div>
                <div class='material-chute'>{info["chute"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Workflow Stepper Diagram
    st.markdown("<div class='section-title'>🔄 Automated Materials Recovery Facility (MRF) Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Step 1: Ingestion
            A["📷 Optical Sensor / Camera<br>224x224 RGB Capture"]
        end
        subgraph Step 2: AI Inference
            B["🧠 MobileNetV2 CNN<br>Depthwise Convolutions"] --> C["📊 Softmax Classifier<br>6 Category Probabilities"]
        end
        subgraph Step 3: Science & Decision
            C --> D{"🛡️ Confidence Check<br>> 50% Threshold"}
            D -- Valid --> E["🔬 Materials Science Lookup<br>Polymers, Crystals & Cullet"]
            D -- Low Conf --> F["⚠️ Safety Warning<br>Manual QA Audit Stream"]
        end
        subgraph Step 4: Sorting Actuation
            E --> G["⚙️ Diverter Actuation<br>Pneumatic Air-Jets & Chutes"]
            G --> H["📜 Session CSV Audit Log<br>Automated Data Capture"]
        end
        A --> B
    ```
    """)

    # Quick Start Action Box
    st.markdown("---")
    qc1, qc2 = st.columns([2, 1])
    with qc1:
        st.markdown("### 🚀 Ready to test the system?")
        st.write("Go to **🔍 Material Identification** from the sidebar to upload a photo, use your webcam, or select one of our curated laboratory test samples with a single click!")
    with qc2:
        st.info("💡 **Examiner Tip:** Check the **📊 Model Performance** page to view the genuine test confusion matrix and training curves.")


def page_identification():
    """Material Identification Page with Instant Preset Samples, Upload, and Webcam."""
    st.markdown("<div class='section-title'>🔍 Material Identification & Classification</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; margin-bottom: 20px;'>Test real-world material samples using Computer Vision and Transfer Learning</p>", unsafe_allow_html=True)

    if not config.MODEL_PATH.exists():
        st.error("🚨 Trained AI model not found. Please train the model using `python train_model.py` or run `train_model.bat` first.")
        return

    # Check for sample_images directory
    sample_dir = config.BASE_DIR / "sample_images"
    sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []

    input_mode = st.radio(
        "Select Image Input Method:",
        ["⚡ Choose from Curated Test Samples", "📁 Upload Custom Image File", "📷 Live Webcam Capture"],
        horizontal=True
    )

    image_bytes = None
    source_name = "Sample Image"

    if input_mode == "⚡ Choose from Curated Test Samples":
        if sample_files:
            file_options = {f.name: f for f in sample_files}
            selected_filename = st.selectbox("Select a verified laboratory test sample:", list(file_options.keys()))
            selected_path = file_options[selected_filename]
            source_name = f"Preset: {selected_filename}"
            with open(selected_path, "rb") as f:
                image_bytes = f.read()
        else:
            st.warning("Sample images folder not found. Please use the Upload Custom Image option.")

    elif input_mode == "📁 Upload Custom Image File":
        uploaded_file = st.file_uploader(
            "Upload an image of recyclable material (JPG, JPEG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            source_name = f"Upload: {uploaded_file.name}"

    else:
        st.info("ℹ️ Grant browser camera permission if prompted. Center the material in bright, even lighting.")
        cam_file = st.camera_input("Take a photo of the material sample:")
        if cam_file is not None:
            image_bytes = cam_file.getvalue()
            source_name = "Live Webcam Capture"

    # Process and Predict
    if image_bytes is not None:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))

            col_left, col_right = st.columns([1, 1.25], gap="large")

            with col_left:
                st.markdown("#### 📷 Input Material Sample")
                st.image(pil_image, caption=f"{source_name} ({pil_image.size[0]}×{pil_image.size[1]} px)", use_container_width=True)

                with st.spinner("Analyzing surface texture, specularity, and microstructural markers..."):
                    result = predict.predict_material(image_bytes, image_source_name=source_name)
                    st.session_state.last_prediction = result
                    predict.log_prediction_to_csv(result)
                    st.session_state.history = predict.load_prediction_history()

            with col_right:
                pred_class = result["predicted_class"]
                conf = result["confidence"]
                bin_info = result["sorting_bin"]

                # Styled Result Card
                st.markdown(f"""
                <div class='result-card-main'>
                    <div class='result-badge-top'>🎯 AI Classification Result</div>
                    <div class='result-class-name'>{bin_info.get('icon', '♻️')} {pred_class}</div>
                    <div class='result-confidence-text'>Confidence: <b>{conf:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

                # Uncertainty Alert if confidence < 50%
                if result["is_uncertain"]:
                    st.markdown(f"""
                    <div class='uncertainty-box'>
                        ⚠️ <b>Uncertainty Warning:</b> {result['uncertainty_warning']}
                    </div>
                    """, unsafe_allow_html=True)

                # Class Probability Distribution
                st.markdown("##### 📊 Class Probability Distribution")
                conf_df = pd.DataFrame(
                    list(result["all_confidences"].items()),
                    columns=["Material", "Confidence (%)"]
                ).sort_values(by="Confidence (%)", ascending=True)

                st.bar_chart(conf_df.set_index("Material"), color="#2563EB", horizontal=True)

            # Detailed Materials Science Cards
            st.markdown("---")
            st.markdown("<div class='section-title'>🔬 Materials Science & Engineering Insights</div>", unsafe_allow_html=True)

            props = result["material_properties"]

            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                st.markdown("""
                <div class='sci-card'>
                    <div class='sci-card-header'>🧪 Chemical & Microstructural Characterization</div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Scientific Nomenclature:** `{props.get('scientific_name', 'N/A')}`")
                st.markdown(f"**Polymer / Alloy Families:** {props.get('common_types', 'N/A')}")
                st.markdown(f"**Microstructural Structure:** {props.get('microstructure', 'N/A')}")

                st.markdown("##### Key Engineering & Physical Properties:")
                phys = props.get("physical_properties", {})
                for k, v in phys.items():
                    st.markdown(f"- **{k}:** {v}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown("""
                <div class='sci-card'>
                    <div class='sci-card-header'>♻️ Circular Recycling Protocol & Sorting Chute</div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Assigned Chute:** `{bin_info.get('name', 'N/A')}` (**Chute #{bin_info.get('chute_number', 'N/A')}**)")
                st.markdown(f"**Recyclability Grade:** `{props.get('recyclability_status', 'N/A')}`")
                st.markdown(f"**Environmental Lifespan:** {props.get('environmental_impact', 'N/A')}")
                st.markdown(f"**Industrial Recycling Pathway:** {props.get('recycling_process', 'N/A')}")
                st.info(f"💡 **Facility Handling Advice:** {props.get('tips', 'N/A')}")
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            st.warning("Please ensure the selected file is a valid, uncorrupted image (JPG, PNG, or WEBP).")


def page_sorting_simulation():
    """Interactive Animated Simulated Sorting System."""
    st.markdown("<div class='section-title'>⚙️ Automated Sorting System Demonstration</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Software-Simulated Material Recovery Facility (MRF) Conveyor & Chute Diverters</p>", unsafe_allow_html=True)

    st.warning("⚠️ **Note:** This digital twin simulates automated industrial MRF logic (optical timing, pneumatic air-jets, and eddy-current deflection). Physical actuators are not physically attached.")

    last_pred = st.session_state.last_prediction
    display_options = [config.DISPLAY_NAMES[c] for c in config.CLASS_NAMES]

    if last_pred is None:
        st.info("👉 No material classified in this session yet. Pick a sample below to simulate sorting:")
        demo_display = st.selectbox("Select demo material for simulation:", display_options)
        active_display = demo_display
    else:
        active_display = last_pred["predicted_class"]
        st.success(f"🎯 Currently Diverting Identified Item: **{active_display}** ({last_pred['confidence']:.1f}% Confidence)")

    st.markdown("#### 🏭 Automated High-Speed Sorting Belt")
    st.markdown("<div class='conveyor-belt'></div>", unsafe_allow_html=True)

    # Render Bin Chutes
    cols = st.columns(6)
    for idx, cls_name in enumerate(config.CLASS_NAMES):
        b = config.SORTING_BINS[cls_name]
        disp_name = config.DISPLAY_NAMES[cls_name]
        is_active = (disp_name == active_display)
        card_class = "sorting-chute sorting-chute-active" if is_active else "sorting-chute"

        with cols[idx]:
            st.markdown(f"""
            <div class='{card_class}'>
                <div style='font-size: 2.2rem;'>{b['icon']}</div>
                <div style='font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-top: 4px;'>{b['name']}</div>
                <div style='font-size: 0.8rem; color: #64748B;'>Chute #{b['chute_number']}</div>
                <div style='margin-top: 8px; font-size: 0.82rem; font-weight: 700; color: {b["color"]};'>
                    {'🟢 ACTIVE DIVERTER' if is_active else '⚪ STANDBY'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚡ Diverter Telemetry & Actuation Control")
    raw_key = config.DISPLAY_TO_CLASS.get(active_display, active_display)
    active_bin = config.SORTING_BINS.get(raw_key, config.SORTING_BINS.get(active_display, {}))
    st.markdown(f"""
    - **Target Stream Detected:** `{active_display}`
    - **Assigned Route:** `{active_bin.get('name', 'N/A')}` (Chute `#{active_bin.get('chute_number', 'N/A')}`)
    - **Actuator Mechanism:** `{active_bin.get('action', 'N/A')}`
    - **Optical Delay Timer:** Vision classification triggers pneumatic solenoid air-pulse with a **350 ms timing delay** calibrated for conveyor speed (1.5 m/s).
    """)


def page_model_performance():
    """Displays genuine test metrics, confusion matrix, and training graphs."""
    st.markdown("<div class='section-title'>📊 Model Performance & Dataset Metrics</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Empirical validation on 384 unseen test samples with zero data leakage</p>", unsafe_allow_html=True)

    metrics_file = config.METRICS_DIR / "evaluation_results.json"
    dataset_file = config.METRICS_DIR / "dataset_stats.json"

    if not metrics_file.exists():
        st.warning("⚠️ Model evaluation results not found. Run `python evaluate_model.py` to evaluate the model.")
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Top Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Overall Test Accuracy</span>
                <span class='stat-icon'>🎯</span>
            </div>
            <div>
                <div class='stat-value'>{metrics.get('overall_accuracy_percent', 'N/A')}</div>
                <div class='stat-sub'>✓ Tested on 384 Unseen Images</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Weighted F1-Score</span>
                <span class='stat-icon'>⚖️</span>
            </div>
            <div>
                <div class='stat-value'>{metrics.get('weighted_f1_score', 'N/A')}</div>
                <div class='stat-sub'>✓ Balanced Precision & Recall</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Weighted Recall</span>
                <span class='stat-icon'>📈</span>
            </div>
            <div>
                <div class='stat-value'>{metrics.get('weighted_recall', 'N/A')}</div>
                <div class='stat-sub'>✓ High Material Recovery Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-header'>
                <span class='stat-title'>Test Dataset Size</span>
                <span class='stat-icon'>📦</span>
            </div>
            <div>
                <div class='stat-value'>{metrics.get('total_test_samples', 'N/A')}</div>
                <div class='stat-sub'>✓ Independent Held-Out Partition</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Per-Class Metrics Table
    st.markdown("### 📋 Per-Class Classification Report")
    csv_class_metrics = config.METRICS_DIR / "class_metrics.csv"
    if csv_class_metrics.exists():
        df_class = pd.read_csv(csv_class_metrics)
        st.dataframe(df_class, use_container_width=True)

    # Graphs Row
    st.markdown("### 📈 Confusion Matrix & Training Dynamics")
    g_col1, g_col2 = st.columns(2)

    cm_path = config.GRAPHS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        with g_col1:
            st.markdown("##### Confusion Matrix Heatmap")
            st.image(str(cm_path), caption="Unseen Test Set Confusion Matrix", use_container_width=True)

    train_graph_path = config.GRAPHS_DIR / "training_history.png"
    if train_graph_path.exists():
        with g_col2:
            st.markdown("##### Training & Validation Curves")
            st.image(str(train_graph_path), caption="Loss & Accuracy Across Epochs", use_container_width=True)

    # Dataset Partitioning Stats
    if dataset_file.exists():
        st.markdown("### 🗃️ Dataset Partitioning & Class Distribution")
        with open(dataset_file, "r", encoding="utf-8") as f:
            d_stats = json.load(f)
        df_splits = pd.DataFrame({
            "Train Split (70%)": d_stats.get("train", {}),
            "Val Split (15%)": d_stats.get("val", {}),
            "Test Split (15%)": d_stats.get("test", {}),
            "Total Samples": d_stats.get("total", {})
        })
        st.dataframe(df_splits, use_container_width=True)


def page_history():
    """Prediction History and Session CSV Export."""
    st.markdown("<div class='section-title'>📜 Session Prediction Audit Log</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Real-time log of classified materials and assigned destination chutes</p>", unsafe_allow_html=True)

    df_history = predict.load_prediction_history()

    if df_history.empty:
        st.info("No predictions logged yet. Test images in the **🔍 Material Identification** tab to build history.")
    else:
        st.dataframe(df_history, use_container_width=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            csv_data = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export History (CSV)",
                data=csv_data,
                file_name="recyclable_materials_prediction_history.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("🗑️ Clear History Log"):
                if config.HISTORY_FILE.exists():
                    config.HISTORY_FILE.unlink()
                st.session_state.history = predict.load_prediction_history()
                st.rerun()


def page_materials_guide():
    """Comprehensive Materials Science and Circular Economy Reference."""
    st.markdown("<div class='section-title'>🔬 Materials Science & Recycling Guide</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Fundamental Material Properties, Microstructure, and Recycling Thermodynamics</p>", unsafe_allow_html=True)

    display_names = [config.DISPLAY_NAMES[c] for c in config.CLASS_NAMES]
    tabs = st.tabs(display_names)
    for idx, cat_key in enumerate(config.CLASS_NAMES):
        disp_name = config.DISPLAY_NAMES[cat_key]
        with tabs[idx]:
            props = config.MATERIAL_PROPERTIES[cat_key]
            st.markdown(f"### {disp_name} — *{props['scientific_name']}*")

            col1, col2 = st.columns([1.2, 1], gap="medium")
            with col1:
                st.markdown("#### 🔬 Microstructure & Bonding")
                st.write(props["microstructure"])

                st.markdown("#### ⚡ Physical & Engineering Properties")
                for prop_k, prop_v in props["physical_properties"].items():
                    st.markdown(f"- **{prop_k}:** {prop_v}")

            with col2:
                st.markdown("#### ♻️ Circular Lifecycle & Recycling")
                st.write(f"**Degradation Lifespan:** {props['environmental_impact']}")
                st.write(f"**Recycling Methodology:** {props['recycling_process']}")
                st.write(f"**Optimal Destination:** {props['sorting_bin']}")
                st.success(f"📌 **Best Practice Tip:** {props['tips']}")


def page_about():
    """Academic Project Details, Architecture, and Viva Voce Preparation."""
    st.markdown("<div class='section-title'>ℹ️ About Project & Academic Viva Voce</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Materials Science and Engineering Laboratory Project Documentation</p>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🏛️ Project Architecture", "🎓 Viva Voce Q&A", "⏱️ 2-Minute Demo Script"])

    with t1:
        st.markdown("### System Architecture")
        st.markdown("""
        - **Model Backbone:** MobileNetV2 with inverted residual blocks and depthwise separable convolutions (~3.4M parameters).
        - **Transfer Learning:** Pre-trained ImageNet feature weights fine-tuned on recyclable waste classes.
        - **Target Categories:** Plastic, Metal, Paper, Cardboard, Glass, Other/Non-recyclable waste.
        - **Data Integrity:** Stratified 70/15/15 train/val/test split with zero data leakage.
        """)

    with t2:
        st.markdown("### Frequently Asked Viva Voce Questions")
        viva_file = config.DOCS_DIR / "VIVA_QUESTIONS_AND_ANSWERS.md"
        if viva_file.exists():
            with open(viva_file, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.write("Viva documentation is available in `docs/VIVA_QUESTIONS_AND_ANSWERS.md`.")

    with t3:
        st.markdown("### 2-Minute College Demonstration Script")
        demo_file = config.DOCS_DIR / "DEMO_SCRIPT.md"
        if demo_file.exists():
            with open(demo_file, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.write("Demo script is available in `docs/DEMO_SCRIPT.md`.")


def main():
    selected_page = render_sidebar()

    if selected_page == "🏠 Dashboard":
        page_dashboard()
    elif selected_page == "🔍 Material Identification":
        page_identification()
    elif selected_page == "⚙️ Sorting Simulation":
        page_sorting_simulation()
    elif selected_page == "📊 Model Performance":
        page_model_performance()
    elif selected_page == "📜 Prediction History":
        page_history()
    elif selected_page == "🔬 Materials Science Guide":
        page_materials_guide()
    elif selected_page == "ℹ️ Viva & Demo Prep":
        page_about()


if __name__ == "__main__":
    main()
