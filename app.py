from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline

app = Flask(__name__)

# Load the summarization model


summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")




import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:  # Only add text if extraction is successful
                text += extracted + "\n"
    
    print("Extracted Text:\n", text[:500])  # Print first 500 characters for debugging
    return text if text else "Text extraction failed. Try another PDF."

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

    pdf_path = f"uploads/{pdf_file.filename}"
    pdf_file.save(pdf_path)

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return jsonify({'error': 'Could not extract text'}), 400

    # Summarize (limit input length)
    summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)
    
    return jsonify({'summary': summary[0]['summary_text']})

if __name__ == '__main__':
    app.run(debug=True)
