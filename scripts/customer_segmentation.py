# scripts/customer_segmentation.py

import os
import joblib
import numpy as np
import streamlit as st

@st.cache_resource
def load_artifacts():
    BASE        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scaler_path = os.path.join(BASE, "models", "scaler_cs.pkl")
    kmeans_path = os.path.join(BASE, "models", "customer_segmentation_kmeans.pkl")
    scaler      = joblib.load(scaler_path)
    kmeans      = joblib.load(kmeans_path)
    return scaler, kmeans

def run():
    st.header("👥 Customer Segmentation")

    scaler, kmeans = load_artifacts()

    # ——— Inputs ———
    age                 = st.number_input("Age", 18, 100, 35)
    gender              = st.selectbox("Gender", ["Male","Female"])
    location            = st.selectbox("Location", ["Urban","Suburban","Rural"])
    income_level        = st.selectbox("Income Level", ["Low","Medium","High"])
    active_policies     = st.number_input("Number of Active Policies", 0, 10, 2)
    total_premium_paid  = st.number_input("Total Premium Paid", 0.0, 1e7, 20000.0, step=100.0)
    claim_frequency     = st.number_input("Claim Frequency", 0, 50, 1)
    policy_upgrades     = st.number_input("Policy Upgrades", 0, 10, 0)
    occupation          = st.selectbox("Occupation", ["Salaried","Self-Employed","Retired","Student","Unemployed"])
    coverage_amount     = st.number_input("Coverage Amount", 0.0, 1e7, 100000.0, step=1000.0)
    policy_type         = st.selectbox("Policy Type", ["Auto","Health","Life","Property"])

    # ——— Label-encode every categorical exactly as used in training ———
    gender_map  = {"Male":0, "Female":1}
    loc_map     = {"Urban":0, "Suburban":1, "Rural":2}
    inc_map     = {"Low":0, "Medium":1, "High":2}
    occ_map     = {"Salaried":0, "Self-Employed":1, "Retired":2, "Student":3, "Unemployed":4}
    policy_map  = {"Auto":0, "Health":1, "Life":2, "Property":3}

    gender_num       = gender_map[gender]
    loc_num          = loc_map[location]
    inc_num          = inc_map[income_level]
    occ_num          = occ_map[occupation]
    policy_type_num  = policy_map[policy_type]

    # ——— Build feature vector in exact order ———
    # [age, gender_num, loc_num, inc_num, active_policies,
    #  total_premium_paid, claim_frequency, policy_upgrades,
    #  occ_num, coverage_amount, policy_type_num]
    X = np.array([[
        age,
        gender_num,
        loc_num,
        inc_num,
        active_policies,
        total_premium_paid,
        claim_frequency,
        policy_upgrades,
        occ_num,
        coverage_amount,
        policy_type_num
    ]], dtype=float)

    if st.button("Assign Segment"):
        Xs      = scaler.transform(X)
        cluster = kmeans.predict(Xs)[0]

        cluster_labels = {
            0: "Young Professionals",
            1: "High-Value Clients",
            2: "Frequent Claimers",
            3: "Passive Buyers",
            4: "Engaged Mid-Income"
        }
        segment = cluster_labels.get(cluster, f"Cluster {cluster}")

        st.markdown(f"### 🏷️ Segment: **{segment}** (Cluster #{cluster})")

        # show distance to each center
        dists = kmeans.transform(Xs)[0]
        st.table({
            "Segment": list(cluster_labels.values()),
            "Distance": np.round(dists, 2)
        })
