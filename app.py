
# ===================================================================================================================
# app.py
# Main entry point for PoliPulseCA web app. Handles hardware status check and
# links user to desired analysis mode (General Trend Analysis or Group Comparison).
# Dependencies: streamlit, torch, mode_general.py, mode_group_comparison.py
# Author: P. Jost
# Date: May 2025
# License: MIT License (code), CC BY 4.0 (training data/tables/docs) https://creativecommons.org/licenses/by/4.0/
#          Transformer models: Apache 2.0; data: US public domain/open
# ===================================================================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PoliPulseCA - MAIN APP: MODE SELECTION AND LAUNCHER
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import torch

st.set_page_config(page_title="PoliPulse CA", layout="wide")
st.image("images/polipulseca_main_log500.png")

# --- GPU Status ---
gpu_available = torch.cuda.is_available()
if gpu_available:
    try:
        device_name = torch.cuda.get_device_name(0)
    except:
        device_name = "Unknown CUDA Device"
    gpu_status = f"🟢 GPU Detected: {device_name}"
else:
    gpu_status = "🔴 GPU Not Available"
st.markdown(f"**🖥️ Hardware Status:** {gpu_status}")

# --- Mode Selection ---
mode = st.selectbox(
    "Choose Trend Analysis Mode:",
    (
        "📈 General Trend Analysis",
        "📊 Compare Groups Analysis",
        "🏋️‍♂️ Fine-tune Transformer",
    )
)

if mode == "📈 General Trend Analysis":
    import mode_general as app
    app.run()

elif mode == "📊 Compare Groups Analysis":
    import mode_group_comparison as app
    app.run()
        
elif mode == "🏋️‍♂️ Fine-tune Transformer":
    import mode_train_transformer as app
    app.run()