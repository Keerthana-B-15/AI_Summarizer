import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_summarizer():
    """Initialize the summarization pipeline."""
    logging.info("Initializing the summarization model...")
    return pipeline("summarization", model="facebook/bart-large-xsum")

# Function to truncate document to fit model's max sequence length
def truncate_document(document, max_length=1024):
    tokens = document.split()
    truncated_tokens = tokens[:max_length]
    return ' '.join(truncated_tokens)

# Summarize document with error handling
def summarize_document(summarizer, document):
    try:
        truncated_document = truncate_document(document, max_length=1024)
        with st.spinner("Generating summary, please wait..."):
            summary = summarizer(
                truncated_document,
                max_length=50,
                min_length=25,
                do_sample=False,
                truncation=True
            )
        return summary[0]['summary_text']
    except Exception as e:
        logging.error(f"Error summarizing document: {e}")
        return "Unable to generate summary"

# Streamlit Web App
st.set_page_config(page_title="AI Document Summarizer", page_icon="🔖", layout="wide")

# Sidebar Instructions
st.sidebar.header("How to Use")
st.sidebar.write("1. Paste a document or upload a text file.\n" +
                 "2. Click the **Summarize** button to generate a summary.\n" +
                 "3. View both the original and summarized text below.")


# App Title and Header
st.title("🔖 AI Document Summarizer")
st.markdown("### Quickly summarize long documents into concise text!")

# Initialize summarizer
summarizer = initialize_summarizer()

# Input Section
st.header("Input Document")
st.markdown("**Choose one of the following options to input your document:**")

# Document Input
document_input = st.text_area("Paste your document below:", placeholder="Paste your document here...")

# File Uploader
uploaded_file = st.file_uploader("Or upload a text file:", type=["txt"], help="Only .txt files are supported.")
if uploaded_file:
    try:
        document_input = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Summarize Button
st.divider()
st.header("Generate Summary")
if st.button("🔄 Summarize"):
    if document_input:
        st.subheader("Original Text")
        st.text_area("Original Document:", value=document_input, height=300, disabled=True)

        st.subheader("Summarized Text")
        summary = summarize_document(summarizer, document_input)
        st.text_area("Summary:", value=summary, height=200, disabled=True)
    else:
        st.error("Please input or upload a document.")

# Footer
st.divider()
st.markdown("---")
st.markdown(
    "<small>Powered by [Hugging Face Transformers](https://huggingface.co/transformers) and [Streamlit](https://streamlit.io)</small>",
    unsafe_allow_html=True
)