# scripts/fraud_detection_manual_v2.py

import os
import numpy as np
import joblib
import streamlit as st

@st.cache_resource
def load_artifacts():
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(BASE, "models", "fraud_detection_if.pkl")
    scaler_path = os.path.join(BASE, "models", "scaler_fd.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_fraud(features, model, scaler):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prediction_label = "Fraud" if prediction == -1 else "Normal"
    return prediction_label

def run():
    st.header("🚨 Real-Time Fraud Detection (Improved Version)")
    st.subheader("Enter Claim Details Below")

    model, scaler = load_artifacts()

    # ------- User Input Form --------
    claim_amount = st.number_input("Claim Amount (₹)", min_value=0.0, step=1000.0, value=5000.0)

    suspicious_flag = st.selectbox("Suspicious Flags Present?", ["No", "Yes"])
    suspicious_flag_num = 1 if suspicious_flag == "Yes" else 0

    claim_type = st.selectbox("Claim Type", ["Medical", "Vehicle", "Home Damage"])

    # ------- Feature Engineering --------
    # High_Claim calculation (compared to median later, but for demo we simulate)
    # You can define a threshold manually if you want
    # For now assume: > 50000 is high
    high_claim = 1 if claim_amount > 50000 else 0

    medical_claim = 1 if claim_type == "Medical" else 0
    vehicle_claim = 1 if claim_type == "Vehicle" else 0
    home_damage_claim = 1 if claim_type == "Home Damage" else 0

    # Arrange features in same order as training
    # Claim_Amount, High_Claim, Medical_Claim, Vehicle_Claim, Home_Damage_Claim, Suspicious_Flags
    X = np.array([[claim_amount, high_claim, medical_claim, vehicle_claim, home_damage_claim, suspicious_flag_num]])

    if st.button("Predict Fraud"):
        result = predict_fraud(X, model, scaler)

        if result == "Fraud":
            st.error(f"🚨 Prediction: {result}")
        else:
            st.success(f"✅ Prediction: {result}")

if __name__ == "__main__":
    run()
