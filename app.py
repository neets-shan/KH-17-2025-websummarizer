from flask import Flask, render_template, request, jsonify, send_file
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline
import os
from gtts import gTTS  # Import gTTS for text-to-speech

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load the summarization model
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
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
    return summarizer("summarize: " + text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

def generate_tts(text, filename="output.mp3"):
    """Converts text to speech and saves it as an MP3 file."""
    tts = gTTS(text=text, lang="en")
    file_path = os.path.join(AUDIO_FOLDER, filename)
    tts.save(file_path)
    return file_path

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

    # Generate speech file
    audio_path = generate_tts(summary)

    return jsonify({'summary': summary, 'audio_url': f'/audio/{os.path.basename(audio_path)}'})

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serves the generated audio file."""
    return send_file(os.path.join(AUDIO_FOLDER, filename), as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
