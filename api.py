from flask import Flask, request, send_file
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import zipfile
import tempfile
import os

app = Flask(__name__)


genai.configure(api_key="AIzaSyCwhuP5ymrMdJCaOMlJDE_30RIyidfMQ2M")
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Load SentenceTransformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_answers(data):
    words = data['words']
    labels = data['labels']
    answers = []
    current_label = []
    current_answer = []
    in_label = False
    in_answer = False

    for word, label in zip(words, labels):
        if label == 'B-ANSNUM':
            if in_answer and current_answer:
                answer_text = ' '.join(current_answer)
                answers.append({'label': ' '.join(current_label), 'answer': answer_text})
                current_answer = []
            current_label = [word]
            in_label = True
            in_answer = False
        elif label == 'I-ANSNUM':
            if in_label:
                current_label.append(word)
        elif label == 'B-ANSWER':
            if in_label:
                in_label = False
                in_answer = True
                current_answer = [word]
        elif label == 'I-ANSWER':
            if in_answer:
                current_answer.append(word)
        elif label in ['O', 'SECTION', 'B-SUBPOINT']:
            if in_answer and current_answer:
                answer_text = ' '.join(current_answer)
                answers.append({'label': ' '.join(current_label), 'answer': answer_text})
                current_answer = []
                in_answer = False
            current_label = []
            in_label = False

    if in_answer and current_answer:
        answer_text = ' '.join(current_answer)
        answers.append({'label': ' '.join(current_label), 'answer': answer_text})

    return answers

def get_keywords(text):
    prompt = f"Extract key terms or phrases from the following text and list them separated by commas: {text}"
    response = gemini_model.generate_content(prompt)
    return response.text.split(', ')

def get_summary(text):
    prompt = f"Summarize the following text in one or two sentences: {text}"
    response = gemini_model.generate_content(prompt)
    return response.text

@app.route('/process', methods=['POST'])
def process_file():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    if not file.filename.endswith('.json'):
        return 'File must be a JSON file', 400

    # Use a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = os.path.join(temp_dir, 'input.json')
        file.save(json_path)

        # Load and validate JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'words' not in data or 'labels' not in data:
            return 'JSON must contain "words" and "labels" fields', 400

        # Process the data
        answers = extract_answers(data)
        answer_texts = [answer['answer'] for answer in answers]
        embeddings = st_model.encode(answer_texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Generate metadata
        metadata = []
        for answer in answers:
            label = answer['label']
            text = answer['answer']
            keywords = get_keywords(text)
            summary = get_summary(text)
            metadata.append({
                'label': label,
                'answer': text,
                'keywords': keywords,
                'summary': summary
            })

        # Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss_path = os.path.join(temp_dir, 'embeddings.faiss')
        faiss.write_index(index, faiss_path)

        # Save metadata to JSON
        metadata_path = os.path.join(temp_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create zip file with outputs
        zip_path = os.path.join(temp_dir, 'output.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(faiss_path, 'embeddings.faiss')
            zipf.write(metadata_path, 'metadata.json')

        # Return the zip file
        return send_file(zip_path, as_attachment=True, download_name='output.zip')

if __name__ == '__main__':
    app.run(debug=True)
