from flask import Flask, render_template, request, jsonify, send_from_directory
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline
import os

app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the summarization model
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (better than PyPDF2)."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        text = " ".join(text.split())  # Remove extra spaces & newlines
    except Exception as e:
        print("Error extracting text:", e)
        return None

    return text if text else None

def summarize_text(text, max_length=150, min_length=50):
    """Summarizes long text by splitting into smaller chunks if needed."""
    if len(text) > 1000:  # If text is too long, split into chunks
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1000:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk)

        summarized_chunks = [summarizer("summarize: " + chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] for chunk in chunks]
        return " ".join(summarized_chunks)
    else:
        return summarizer("summarize: " + text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return jsonify({'error': 'Could not extract text'}), 400

    # Summarize the extracted text
    summary = summarize_text(text)
    
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)

