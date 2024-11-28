# AI Document Summarizer

## Overview
The AI Document Summarizer is a user-friendly web application that simplifies lengthy documents into concise and meaningful summaries. Built with the powerful Hugging Face Transformers library and Streamlit, it leverages state-of-the-art AI models to deliver high-quality summarizations in seconds.


## Features
- **Flexible Input Options**:
  - Paste text directly or upload `.txt` files.
- **Interactive Summarization**:
  - Generates summaries with a click of a button.
  - Includes a copy feature for both original and summarized texts.
- **Advanced AI Model**:
  - Powered by Facebook BART ("facebook/bart-large-xsum") for abstractive summarization.
- **Real-time Feedback**:
  - Displays progress while generating summaries.
- **Error Handling**:
  - Provides feedback if the input cannot be processed.


## Installation

### Setting Up Virtual Environment

1. Create a virtual environment:
   ```bash
   python -m venv env
   ```

2. Activate the virtual environment:
   - **On Windows**:
     ```bash
     .\env\Scripts\activate
     ```
   - **On macOS/Linux**:
     ```bash
     source env/bin/activate
     ```

1. Clone the repository:
   ```bash
   git clone https://github.com/Keerthana-B-15/AI_Summarizer.git
   ```
2. Navigate to the project directory:
   ```bash
   cd AI_Summarizer
   ```
3. Make sure your virtual environment is activated, then Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Open the app in your web browser (URL will be displayed in the terminal).

3. Input a document via text area or upload a `.txt` file.

4. Click the **Summarize** button to generate a summary.

5. Use the copy icons to copy the original or summarized text.


## How It Works

- The application uses the Facebook BART model ("facebook/bart-large-xsum") for generating abstractive summaries.
- It processes documents up to 1024 tokens, truncating longer texts to fit the model's input size.
- Summaries are constrained to a length of 25 to 50 words.


## Dependencies

- **Python 3.8+**
- **Streamlit**: For building the interactive web application.
- **Transformers**: For leveraging the BART model.
- **Datasets**: For dataset management.

Install dependencies using the following command:
```bash
pip install -r requirements.txt
```


## Contributing

We welcome contributions to improve the AI Document Summarizer! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.


## License
This project is licensed under the [MIT License](LICENSE).


## Acknowledgments

- **Hugging Face**: For providing cutting-edge NLP models.
- **Streamlit**: For the seamless creation of data apps.


## Contact

For questions or feedback, please contact:
- **Email**: [keerthanab610@gmail.com]
- **GitHub**: [[your-github-profile](https://github.com/Keerthana-B-15/)]
- **LinkedIn**: [[your-linkedin-profile](https://www.linkedin.com/in/keerthana-b-904b79256/)]


**Try the AI Document Summarizer and save time summarizing lengthy texts!**

