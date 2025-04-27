# scripts/document_translator_app.py

import os
import pdfplumber
import docx
import time
import streamlit as st
from deep_translator import GoogleTranslator

# --------- Extract Text Functions ---------
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())

# --------- Safe Translation Function ---------
def translate_line_safe(line, translator, retries=3, delay=1):
    for attempt in range(retries):
        try:
            return translator.translate(line)
        except Exception as e:
            st.warning(f"⚠️ Retry {attempt+1} for line '{line[:30]}...' ➜ Error: {e}")
            time.sleep(delay)
    return line  # fallback

# --------- Main Translation Pipeline ---------
def translate_text(full_text, src_lang, dest_lang):
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
    translator = GoogleTranslator(source=src_lang, target=dest_lang, timeout=5)
    translated_lines = [translate_line_safe(line, translator) for line in lines]
    return "\n\n".join(translated_lines)

# --------- Streamlit App ---------
def run():
    st.title("🌎 Document Translator App")
    st.subheader("Upload a PDF or DOCX and Translate to Any Language!")

    uploaded_file = st.file_uploader("📂 Upload your PDF or DOCX file", type=["pdf", "docx"])
    src_lang = st.selectbox("Source Language", ["en", "hi", "ta", "fr", "de", "es", "zh"])
    dest_lang = st.selectbox("Target Language", ["hi", "ta", "fr", "de", "es", "zh", "en"])

    if uploaded_file and st.button("Translate"):
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Save file temporarily
        temp_input_path = f"temp_upload{ext}"
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text
        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(temp_input_path)
        elif ext == ".docx":
            extracted_text = extract_text_from_docx(temp_input_path)
        else:
            st.error("❌ Unsupported file type!")
            return

        if not extracted_text:
            st.warning("⚠️ No text found in uploaded file.")
            return

        st.info("🔄 Translating... Please wait.")

        translated_text = translate_text(extracted_text, src_lang, dest_lang)

        st.success("✅ Translation Completed!")
        st.download_button(
            label="📥 Download Translated Text",
            data=translated_text,
            file_name="translated_output.txt",
            mime="text/plain",
        )

        # Clean up
        os.remove(temp_input_path)

if __name__ == "__main__":
    run()
