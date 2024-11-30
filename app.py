import streamlit as st
from transformers import pipeline
from datasets import load_dataset
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import logging
import os
from collections import Counter
import networkx as nx
from datetime import datetime

# Ensure required NLTK resources are downloaded
nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load abstractive summarization model
def initialize_abstractive_summarizer():
    logging.info("Initializing the abstractive summarization model...")
    return pipeline("summarization", model="facebook/bart-large-xsum")

# Perform extractive summarization
def perform_extractive_summarization(document, num_sentences=3):
    parser = PlaintextParser.from_string(document, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join(str(sentence) for sentence in summary)

# Truncate document for abstractive summarization
def truncate_document(document, max_length=1024):
    tokens = document.split()
    truncated_tokens = tokens[:max_length]
    return ' '.join(truncated_tokens)

# Summarize document using abstractive summarization
def perform_abstractive_summarization(summarizer, document):
    try:
        truncated_document = truncate_document(document)
        summary = summarizer(
            truncated_document,
            max_length=50,
            min_length=25,
            do_sample=False,
            truncation=True
        )
        return summary[0]['summary_text']
    except Exception as e:
        logging.error(f"Error in abstractive summarization: {e}")
        return "Unable to generate summary"

# Generate word cloud
def generate_word_cloud(document):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(document)
    return wordcloud

# Generate bar chart for word frequencies
def generate_bar_chart(text):
    words = text.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(10)
    return common_words

# Generate network graph for text
def generate_network_graph(text):
    words = text.split()
    G = nx.Graph()
    for i in range(len(words) - 1):
        G.add_edge(words[i], words[i + 1])
    return G

# Generate timeline from text (dummy implementation for demo purposes)
def generate_timeline(text):
    dates = []
    for word in text.split():
        try:
            dates.append(datetime.strptime(word, "%Y"))
        except ValueError:
            pass
    return dates

# Save results to CSV
def save_to_csv(file_path, original_text, abstractive_summary, extractive_summary):
    data = {
        "Original Text": [original_text],
        "Abstractive Summary": [abstractive_summary],
        "Extractive Summary": [extractive_summary]
    }
    df = pd.DataFrame(data)

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

# Streamlit Web App
st.set_page_config(page_title="AI Document Summarizer", page_icon="📜", layout="wide")

# Sidebar Instructions
st.sidebar.header("How to Use")
st.sidebar.write("1. Paste a document or upload a text file.\n" +
                 "2. Click the **Summarize** button to generate summaries and visualizations.\n" +
                 "3. View the original text, summaries, and visualizations below.\n" +
                 "4. Summaries will be saved to a CSV file for future reference.")

# App Title
st.title("📜 AI Document Summarizer")
st.markdown("### Generate abstractive and extractive summaries for your documents, plus visualizations!")

# Initialize summarizers
abstractive_summarizer = initialize_abstractive_summarizer()

# Input Section
st.header("Input Document")
document_input = st.text_area("Paste your document below:", placeholder="Paste your document here...")

# File Uploader
uploaded_file = st.file_uploader("Or upload a text file:", type=["txt"])
if uploaded_file:
    try:
        document_input = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Summarize Button
csv_file_path = "summaries.csv"
st.header("Generate Summaries and Visualizations")
if st.button("🔄 Summarize"):
    if document_input:
        st.subheader("Original Text")
        st.text_area("Original Document:", value=document_input, height=300, disabled=True)

        # Abstractive Summary
        st.subheader("Abstractive Summary")
        abstractive_summary = perform_abstractive_summarization(abstractive_summarizer, document_input)
        st.text_area("Abstractive Summary:", value=abstractive_summary, height=200, disabled=True)

        # Extractive Summary
        st.subheader("Extractive Summary")
        extractive_summary = perform_extractive_summarization(document_input)
        st.text_area("Extractive Summary:", value=extractive_summary, height=200, disabled=True)

        # Word Cloud
        st.subheader("Word Cloud")
        wordcloud = generate_word_cloud(extractive_summary)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Bar Chart
        st.subheader("Bar Chart of Word Frequencies")
        common_words = generate_bar_chart(extractive_summary)
        words, frequencies = zip(*common_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, frequencies, color='skyblue')
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title("Top 10 Word Frequencies in Extractive Summary")
        st.pyplot(plt)

        # Network Graph
        st.subheader("Network Graph of Words")
        G = generate_network_graph(extractive_summary)
        plt.figure(figsize=(10, 7))
        nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold")
        st.pyplot(plt)

        # Timeline
        st.subheader("Timeline Visualization")
        dates = generate_timeline(extractive_summary)
        if dates:
            st.line_chart([date.year for date in dates])
        else:
            st.write("No temporal data found in the summary.")

        # Save results to CSV
        save_to_csv(csv_file_path, document_input, abstractive_summary, extractive_summary)
        st.success(f"Summaries saved to {csv_file_path}!")
    else:
        st.error("Please input or upload a document.")

# Footer
st.markdown("---")
st.markdown(
    "<small>Powered by [Hugging Face Transformers](https://huggingface.co/transformers), [Sumy](https://github.com/miso-belica/sumy), [WordCloud](https://github.com/amueller/word_cloud), and [Streamlit](https://streamlit.io)</small>",
    unsafe_allow_html=True
)
