# scripts/sentiment_analyzer_textinput.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@st.cache_resource
def load_sentiment_model():
    model_path = r"C:\Users\Hxtreme\Jupyter_Notebook_Learning\Final_Project\models\saved_sentiment_cardiff"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def map_label(label):
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    label_index = int(label.split("_")[-1])
    return label_map[label_index]

def run():
    st.header("💬 Live Customer Feedback Sentiment Analyzer")
    st.subheader("Enter feedback text below:")

    sentiment_pipeline = load_sentiment_model()

    user_input = st.text_area("✍️ Type your feedback here", height=200)

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some feedback text to analyze.")
        else:
            result = sentiment_pipeline(user_input)[0]
            sentiment = map_label(result['label'])
            confidence = result['score']

            st.success(f"✅ Predicted Sentiment: **{sentiment.upper()}** (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    run()
