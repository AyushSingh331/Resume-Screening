import os
import re
import pickle
import string
import io

from flask import Flask, request, render_template, jsonify

import nltk
import pdfplumber
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

app = Flask(__name__)

model = pickle.load(open('knn.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

def resumeKeywords(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    cleanText = cleanText.strip()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cleanText)

    stop_words = set(stopwords.words('english'))
    words_new = [word.lower() for word in tokens if word.lower() not in stop_words]

    wn = WordNetLemmatizer()
    lemm_text = [wn.lemmatize(word) for word in words_new]

    return ' '.join(lemm_text)

def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return " ".join([page.extract_text() or "" for page in pdf.pages])
    elif filename.endswith('.docx'):
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode('utf-8', errors='ignore')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']

        raw_text = extract_text(file)

        processed_text = resumeKeywords(raw_text)

        vectorized_text = tfidf.transform([processed_text])

        prediction_id = model.predict(vectorized_text)
        category_name = encoder.inverse_transform(prediction_id)[0]
        
        return jsonify({'category': category_name})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
