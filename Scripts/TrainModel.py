import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = r"C:\Users\ragin\OneDrive\Pictures\Desktop\FakeNewsDetectionSystem"
DATA_DIR = BASE_DIR + r"\processedData"

# -------- LOAD DATA --------
X_train = pickle.load(open(DATA_DIR + r"\X_train.pkl", "rb"))
y_train = pickle.load(open(DATA_DIR + r"\y_train.pkl", "rb"))

X_valid = pickle.load(open(DATA_DIR + r"\X_valid.pkl", "rb"))
y_valid = pickle.load(open(DATA_DIR + r"\y_valid.pkl", "rb"))

# Load tfidf vectorizer
tfidf = pickle.load(open(DATA_DIR + r"\tfidf_vectorizer.pkl", "rb"))

# -------- CHECK CLASS DISTRIBUTION --------
unique_classes, counts = np.unique(y_train, return_counts=True)
print("Training label distribution:")
class_names = ['Fake News', 'True News']
for c, n in zip(unique_classes, counts):
    print(f"Class {c} ({class_names[c]}): {n} samples")

# -------- SAFETY CHECK --------
if len(unique_classes) < 2:
    print("\n❌ ERROR: Only one class present in training data.")
    print("Your preprocessing label mapping made all labels same.")
    print("No model can be trained on one class.")
    exit()

# -------- TRAIN MODEL --------
model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(X_train, y_train)

# -------- VALIDATE --------
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)

print("\n" + "="*50)
print("Classification Report:")
print(classification_report(y_valid, y_pred, target_names=class_names))
print("="*50)

# -------- SAVE MODEL --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Create Models folder if it doesn't exist
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# Also save the vectorizer in Models folder
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
pickle.dump(tfidf, open(VECTORIZER_PATH, "wb"))

# Save model inside Models folder
MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_model.pkl")
pickle.dump(model, open(MODEL_PATH, "wb"))

print("\n✅ Model trained successfully!")
print("Validation Accuracy:", acc)
print("Model saved at:", MODEL_PATH)

