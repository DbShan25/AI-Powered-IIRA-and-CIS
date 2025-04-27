# deployment/streamlit_main.py

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ✅ FIRST Streamlit command
st.set_page_config(page_title="AI Insurance Assistant", layout="wide")

# ─── 1) Ensure project root (parent of deployment/) is on Python path ─────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── 2) Try normal import; if that fails, fallback to loading by file ───────────
imported_normally = True
try:
    from scripts import (
        risk_classifier,
        claim_predictor,
        customer_segmentation,
        fraud_detector,
        multilingual_translator,
        sentiment_predictor,
        summarizer,
        insurance_chatbot
    )
except ImportError:
    imported_normally = False
    import importlib.machinery, importlib.util

    def load_script(name, fname):
        path = os.path.join(ROOT, "scripts", fname)
        loader = importlib.machinery.SourceFileLoader(name, path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod

    risk_classifier        = load_script("risk_classifier",       "risk_classifier.py")
    claim_predictor        = load_script("claim_predictor",       "claim_predictor.py")
    customer_segmentation  = load_script("customer_segmentation", "customer_segmentation.py")
    fraud_detector         = load_script("fraud_detector",        "fraud_detector.py")
    multilingual_translator = load_script("multilingual_translator", "multilingual_translator.py")
    sentiment_predictor    = load_script("sentiment_predictor",   "sentiment_predictor.py")
    summarizer             = load_script("summarizer",            "summarizer.py")
    insurance_chatbot      = load_script("insurance_chatbot",     "insurance_chatbot.py")

# ─── 3) Build Streamlit UI ──────────────────────────────────────────────────────

menu = [
    "🏠 Overview",
    "📈 Risk Classification",
    "💰 Claim Prediction",
    "👥 Customer Segmentation",
    "🕵️ Fraud Detection",
    "🌍 Multilingual Translator",
    "💬 Sentiment Analysis",
    "📄 Policy Summarizer",
    "🤖 Insurance Chatbot"
]
choice = st.sidebar.selectbox("🔎 Choose Module", menu)

if choice == "🏠 Overview":
    st.title("AI-Powered Insurance Risk Assessment & Customer Insights")
    st.markdown("Welcome👋")
    st.markdown("Use the sidebar to navigate between modules.")

    # ----------- Load Main Dataset -----------
    data_path_FD = r"C:\Users\Hxtreme\Jupyter_Notebook_Learning\Final_Project\Dataset\Insurance_FD.csv"  # Your cleaned insurance dataset
    FD_df = pd.read_csv(data_path_FD)

    # Load Reviews Dataset
    review_data_path = r"C:\Users\Hxtreme\Jupyter_Notebook_Learning\Final_Project\Dataset\Insurance_CFS.csv"
    review_df = pd.read_csv(review_data_path)
    # See first few rows to confirm

    # Load Customer Segmentation dataset
    data_path_CS = r"C:\Users\Hxtreme\Jupyter_Notebook_Learning\Final_Project\Dataset\Insurance_CS.csv"
    CS_df = pd.read_csv(data_path_CS)

    # Load Customer Segmentation dataset
    data_path_RC = r"C:\Users\Hxtreme\Jupyter_Notebook_Learning\Final_Project\Dataset\Insurance_RC_CP.csv"
    RC_df = pd.read_csv(data_path_RC)

    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Policyholders", f"{FD_df['Policyholder_ID'].nunique()}")
    with col2:
        st.metric("Total Claims", f"{FD_df.shape[0]}")
    with col3:
        st.metric("Total Claim Amount (₹)", f"{int(FD_df['Claim_Amount'].sum()):,}")
    with col4:
        fraud_percent = (FD_df['Fraud_Label'].sum() / FD_df.shape[0]) * 100
        st.metric("Fraud Cases (%)", f"{fraud_percent:.2f}%")


    st.divider()

    # ----------- Some Quick Visualizations -----------
    # —————————— 2 Columns Layout ———————————

    left_col, right_col = st.columns(2)
    
    # —————————— Row 1 ———————————
    with left_col:
        st.subheader("💬 Positive Reviews")
        if "Sentiment_Label" in review_df.columns and "Review_Text" in review_df.columns:
            text_positive = " ".join(review_df[review_df["Sentiment_Label"] == "Positive"]["Review_Text"].dropna())
            if text_positive.strip():
                wc = WordCloud(width=400, height=200, background_color="white", colormap="Greens").generate(text_positive)
                fig, ax = plt.subplots(figsize=(4,2))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
    
    with right_col:
        st.subheader("📊 Risk Levels by Policy Type")
        if "Policy_Type" in RC_df.columns and "Risk_Score" in RC_df.columns:
            risk_policy = RC_df.groupby(["Policy_Type", "Risk_Score"]).size().unstack(fill_value=0)
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            risk_policy.plot(kind="bar", stacked=True, ax=ax1)
            ax1.set_ylabel("Number of Policies")
            ax1.set_title("Risk Distribution")
            st.pyplot(fig1)
    
    # —————————— Gap ———————————
    st.markdown("<br>", unsafe_allow_html=True)
    
    # —————————— Row 2 ———————————
    left_col2, right_col2 = st.columns(2)
    
    with left_col2:
        st.subheader("💬 Negative Reviews")
        text_negative = " ".join(review_df[review_df["Sentiment_Label"] == "Negative"]["Review_Text"].dropna())
        if text_negative.strip():
            wc = WordCloud(width=400, height=200, background_color="white", colormap="Reds").generate(text_negative)
            fig, ax = plt.subplots(figsize=(4,2))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    
    with right_col2:
        st.subheader("🌎 Policy Type by Location")
        if "Policy Type" in CS_df.columns and "Location" in CS_df.columns:
            policy_location = CS_df.groupby(["Location", "Policy Type"]).size().unstack(fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            policy_location.plot(kind="bar", stacked=True, ax=ax2)
            ax2.set_ylabel("Number of Policies")
            ax2.set_title("Policy vs Location")
            st.pyplot(fig2)
    
    # —————————— Gap ———————————
    st.markdown("<br>", unsafe_allow_html=True)
    
    # —————————— Row 3 ———————————
    left_col3, right_col3 = st.columns(2)
    
    with left_col3:
        st.subheader("💬 Neutral Reviews")
        text_neutral = " ".join(review_df[review_df["Sentiment_Label"] == "Neutral"]["Review_Text"].dropna())
        if text_neutral.strip():
            wc = WordCloud(width=400, height=200, background_color="white", colormap="Greys").generate(text_neutral)
            fig, ax = plt.subplots(figsize=(4,2))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    
    with right_col3:
        st.subheader("👩‍💼 Policy Type by Occupation")
        if "Policy Type" in CS_df.columns and "Occupation" in CS_df.columns:
            policy_occupation = CS_df.groupby(["Occupation", "Policy Type"]).size().unstack(fill_value=0)
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            policy_occupation.plot(kind="bar", stacked=True, ax=ax3)
            ax3.set_ylabel("Number of Policies")
            ax3.set_title("Policy vs Occupation")
            st.pyplot(fig3)



elif choice == "📈 Risk Classification":
    risk_classifier.run()

elif choice == "💰 Claim Prediction":
    claim_predictor.run()

elif choice == "👥 Customer Segmentation":
    customer_segmentation.run()

elif choice == "🕵️ Fraud Detection":
    fraud_detector.run()

elif choice == "🌍 Multilingual Translator":
    multilingual_translator.run()

elif choice == "💬 Sentiment Analysis":
    sentiment_predictor.run()

elif choice == "📄 Policy Summarizer":
    summarizer.run()

elif choice == "🤖 Insurance Chatbot":
    insurance_chatbot.run()

# ─── 4) Footer info if fallback was used ───────────────────────────────────────
if not imported_normally:
    st.info("⚙️ Modules loaded via dynamic fallback loader (importlib).")


#streamlit run C:\Users\Hxtreme\Jupyter_Notebook_Learning\Final_Project\Deployment\streamlit_main.py
