# scripts/claim_predictor.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_artifacts():
    # Compute project root (parent of scripts/)
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Pickles in models/
    model_path  = os.path.join(BASE, "models", "claim_prediction_rfr.pkl")
    scaler_path = os.path.join(BASE, "models", "scaler_cp.pkl")
    # Load artifacts
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def run():
    st.header("💰 Claim Amount Prediction")

    # Load model & scaler
    model, scaler = load_artifacts()

    # ——— Inputs ———
    age               = st.number_input("Age", 18, 100, 30)
    annual_income     = st.number_input("Annual Income", 0.0, 1e7, 500000.0, step=1000.0)
    property_age      = st.number_input("Property Age", 0, 100, 5)
    claim_history     = st.number_input("Claim History (count)", 0, 50, 1)
    risk_score        = st.selectbox("Risk Score", ["Low","Medium","High"])
    premium_amount    = st.number_input("Premium Amount", 0.0, 1e6, 10000.0, step=100.0)
    fraudulent        = st.selectbox("Fraudulent Claim", ["No","Yes"])
    
    # Newly added features
    claim_to_income   = st.number_input("Claim to Income Ratio", 0.0, 10.0, 0.05, step=0.01)
    age_risk_factor   = st.number_input("Age Risk Factor", 0.0, 5.0, 1.0, step=0.1)
    
    gender            = st.selectbox("Gender", ["Male","Female"])
    policy_type       = st.selectbox("Policy Type", ["Auto","Health","Life","Property"])

    # ——— Encode ———
    risk_map   = {"Low":0, "Medium":1, "High":2}
    fraud_map  = {"No":0, "Yes":1}

    rs = risk_map[risk_score]
    fc = fraud_map[fraudulent]

    # one-hot for gender: [Female, Male]
    gender_ohe = [1 if gender=="Female" else 0,
                  1 if gender=="Male"   else 0]

    # one-hot for policy: [Auto, Health, Life, Property]
    policies   = ["Auto","Health","Life","Property"]
    policy_ohe = [1 if policy_type==p else 0 for p in policies]

    # ——— Assemble exactly 15 features ———
    X = np.array([[
        age,
        annual_income,
        property_age,
        claim_history,
        rs,
        premium_amount,
        fc,
        claim_to_income,
        age_risk_factor
    ] + gender_ohe + policy_ohe], dtype=float)

    # ——— Predict & display ———
    if st.button("Predict Claim Amount"):
        Xs   = scaler.transform(X)
        pred = model.predict(Xs)[0]
        st.success(f"🔮 Predicted Claim Amount: ₹{pred:,.2f}")

        # Optionally show the feature values
        df = pd.DataFrame({
            "Feature": [
                "age","annual_income","property_age","claim_history","risk_score",
                "premium_amount","fraudulent_claim","claim_to_income","age_risk_factor",
                "gender_Female","gender_Male",
                "policy_Auto","policy_Health","policy_Life","policy_Property"
            ],
            "Value": X.flatten().tolist()
        })
        st.table(df)
