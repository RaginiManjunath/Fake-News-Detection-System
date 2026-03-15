from flask import Flask, render_template, request
import pickle
import os
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (runs once)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "Models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ProcessedData", "tfidf_vectorizer.pkl")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open(MODEL_PATH, "rb"))
tfidf = pickle.load(open(VECTORIZER_PATH, "rb"))

# ---------------- TEXT CLEANING ----------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']

    # Limit text length (important for LIAR dataset)
    news_text = ". ".join(news_text.split(".")[:3])

    cleaned_text = clean_text(news_text)
    vector = tfidf.transform([cleaned_text])

    probabilities = model.predict_proba(vector)[0]
    confidence = np.max(probabilities)
    prediction = model.predict(vector)[0]

    if prediction == 1:
        category = "REAL NEWS"
        emoji = "✅"
    else:
        category = "FAKE NEWS"
        emoji = "❌"

    percentage = int(confidence * 100)

    return render_template(
        "index.html",
        category=category,
        percentage=percentage,
        emoji=emoji,
        text=news_text
    )

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
