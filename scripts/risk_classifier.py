# scripts/risk_classifier.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_artifacts():
    # Compute project root (parent of scripts/)
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Build absolute paths into models/
    model_path  = os.path.join(BASE, "models", "risk_classification_rfc.pkl")
    scaler_path = os.path.join(BASE, "models", "scaler_rc.pkl")
    # Debug: verify paths
    print("Loading RFC from:", model_path)
    print("Loading scaler from:", scaler_path)
    # Load
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def run():
    st.header("📈 Risk Classification (Low / Medium / High)")

    # load model & scaler
    model, scaler = load_artifacts()

    # ——— Inputs ———
    customer_age     = st.number_input("Customer Age", 18, 100, 30)
    annual_income    = st.number_input("Annual Income", 0.0, 1e7, 500000.0)
    property_age     = st.number_input("Property Age", 0, 100, 5)
    claim_history    = st.number_input("Claim History (count)", 0, 50, 1)
    premium_amount   = st.number_input("Premium Amount", 0.0, 1e6, 10000.0)
    claim_amount     = st.number_input("Claim Amount", 0.0, 1e7, 5000.0)
    fraudulent_claim = st.selectbox("Fraudulent Claim", ["No","Yes"])
    gender           = st.selectbox("Gender", ["Male","Female","Other"])
    policy_type      = st.selectbox("Policy Type", ["Health","Auto","Life","Property"])
    claim_to_income  = st.number_input("Claim to Income Ratio", 0.0, 10.0, 0.05, step=0.01)
    age_risk_factor  = st.number_input("Age Risk Factor", 0.0, 5.0, 1.0, step=0.1)

    # ——— Encode & one-hot ———
    risk_map    = {"Low":0, "Medium":1, "High":2}
    fraud_map   = {"No":0, "Yes":1}
    genders     = ["Male","Female"]
    policies    = ["Health","Auto","Life","Property"]

    fc = fraud_map[fraudulent_claim]
    gender_ohe = [1 if gender==g else 0 for g in genders]
    policy_ohe = [1 if policy_type==p else 0 for p in policies]

    # ——— Assemble in exact order ———
    X = np.array([[
        customer_age,
        annual_income,
        property_age,
        claim_history,
        premium_amount,
        claim_amount,
        fc
    ] + gender_ohe + policy_ohe + [
        claim_to_income,
        age_risk_factor
    ]], dtype=float)

    # ——— Predict & display ———
    if st.button("Predict Risk Category"):
        Xs       = scaler.transform(X)
        pred_num = model.predict(Xs)[0]
        proba    = model.predict_proba(Xs)[0]
        inv_map  = {0:"Low",1:"Medium",2:"High"}
        pred_lbl = inv_map[pred_num]

        st.markdown(f"### 🔮 Predicted Risk Category: **{pred_lbl}**")
        st.markdown("#### Class Probabilities:")
        st.dataframe(pd.DataFrame([proba], columns=[inv_map[i] for i in range(len(proba))]))
