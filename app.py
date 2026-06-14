import streamlit as st
import pandas as pd
import os
from predict import SIAInferenceEngine

st.set_page_config(page_title="SIA Dashboard", layout="wide")
st.title("🛡️ Support Integrity Auditor (SIA) UI")

@st.cache_resource
def load_engine():
    return SIAInferenceEngine() if os.path.exists("./sia_deberta_lora") else None

engine = load_engine()

col1, col2 = st.columns([1, 1])
with col1:
    st.header("Audit Request Input")
    t_id = st.text_input("Ticket ID", "TC-101")
    subj = st.text_input("Subject", "Database Outage")
    desc = st.text_area("Description", "Production database is crashing completely under heavy load.")
    prio = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])
    chan = st.selectbox("Channel", ["Email", "Chat", "Web-Portal"])
    rest = st.slider("Resolution Time (Hours)", 0.0, 100.0, 50.0)
    
    if st.button("Run Audit") and engine:
        row = {'Ticket ID': t_id, 'Ticket Subject': subj, 'Ticket Description': desc, 'Ticket Priority': prio, 'Ticket Channel': chan, 'Resolution Time': rest}
        _, dossier = engine.predict_and_document(row)
        with col2:
            st.header("Generated Evidence Dossier")
            st.json(dossier)
    elif engine is None:
        st.warning("Please run train_pipeline.py first to create and save the model weights directory locally.")
