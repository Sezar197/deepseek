# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load models and scaler
@st.cache_resource
def load_models():
    rf = joblib.load('rf_model.pkl')
    xgb = joblib.load('xgb_model.pkl')
    ann = joblib.load('ann_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf, xgb, ann, scaler

rf_model, xgb_model, ann_model, scaler = load_models()

st.set_page_config(page_title="Beam Deflection Predictor", layout="centered")
st.title("📐 Timoshenko Beam Deflection Predictor")
st.markdown("Enter the beam properties and load to predict deflection using **Random Forest**, **XGBoost**, and **Neural Network** models.")

# Input fields
col1, col2 = st.columns(2)
with col1:
    youngs = st.number_input("Young's Modulus (MPa)", value=210000.0, step=1000.0)
    width = st.number_input("Width (mm)", value=150.0, step=10.0)
    load = st.number_input("Load (N)", value=-1000.0, step=100.0)
with col2:
    poisson = st.number_input("Poisson's Ratio", value=0.3, step=0.01, format="%.3f")
    depth = st.number_input("Depth (mm)", value=150.0, step=10.0)

if st.button("Predict Deflection"):
    # Prepare input array
    input_data = np.array([[youngs, poisson, width, depth, load]])
    input_scaled = scaler.transform(input_data)

    # Predictions
    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]
    ann_pred = ann_model.predict(input_scaled)[0]

    st.subheader("📊 Predicted Deflection")
    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest", f"{rf_pred:.6f}")
    col2.metric("XGBoost", f"{xgb_pred:.6f}")
    col3.metric("Neural Network", f"{ann_pred:.6f}")

    st.caption("All values are in mm (absolute deflection).")