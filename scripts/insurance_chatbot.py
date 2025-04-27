# scripts/insurance_chatbot.py

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

@st.cache_resource
def load_insurance_bot():
    model_path = r"C:\Users\Hxtreme\Jupyter_Notebook_Learning\Final_Project\models\flan_t5_insurance"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return generator

def generate_insurance_response(generator, query):
    prompt = f"Answer the following INSURANCE-RELATED question accurately and politely:\n\nQuestion: {query}\n\nAnswer:"
    result = generator(prompt, max_length=100, do_sample=False)
    return result[0]['generated_text']


def run():
    st.title("🤖 AI Insurance Chatbot")
    st.subheader("Ask your insurance-related questions!")

    generator = load_insurance_bot()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("💬 You:", "")

    if st.button("Ask"):
        if user_input.strip() == "":
            st.warning("⚠️ Please type a question.")
        else:
            with st.spinner("Thinking... 🤔"):
                bot_response = generate_insurance_response(generator, user_input)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", bot_response))

    # Display conversation history
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"🧑‍💼 **You:** {message}")
        else:
            st.markdown(f"🤖 **Bot:** {message}")

if __name__ == "__main__":
    run()
