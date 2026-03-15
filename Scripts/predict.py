import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download nltk resources if not present
nltk.download('stopwords')
nltk.download('wordnet')

# ---------- Load Model & Vectorizer ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "Models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_DIR, "ProcessedData", "tfidf_vectorizer.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
tfidf = pickle.load(open(VECTORIZER_PATH, "rb"))

# ---------- Text Cleaning ----------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------- Prediction Function ----------
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence_scores = model.predict_proba(vector)[0]
    confidence = confidence_scores[prediction]
    
    class_labels = ['FAKE NEWS', 'BARELY TRUE', 'HALF TRUE', 'MOSTLY TRUE', 'TRUE']
    return class_labels[prediction], confidence

# ---------- Test ----------
if __name__ == "__main__":
    sample_text = input("Enter news text: ")
    result, confidence = predict_news(sample_text)
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence*100:.2f}%")
