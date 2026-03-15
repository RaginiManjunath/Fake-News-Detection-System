import pandas as pd
import re
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- PATHS --------------------
TRAIN_PATH = r"C:\Users\ragin\OneDrive\Pictures\Desktop\FakeNewsDetectionSystem\Datasets\train.tsv"
VALID_PATH = r"C:\Users\ragin\OneDrive\Pictures\Desktop\FakeNewsDetectionSystem\Datasets\valid.tsv"
TEST_PATH  = r"C:\Users\ragin\OneDrive\Pictures\Desktop\FakeNewsDetectionSystem\Datasets\test.tsv"

# -------------------- LOAD DATA --------------------
def load_data(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df = df[[1,2]]  # Column 2 = label, Column 3 = text
    df.columns = ['label', 'text']
    return df

train_df = load_data(TRAIN_PATH)
valid_df = load_data(VALID_PATH)
test_df  = load_data(TEST_PATH)

# -------------------- LABEL CONVERSION --------------------
def label_to_multiclass(label):
    label = str(label).strip().lower()
    # Binary classification: only 'false' is Fake News, everything else is True News
    if label == 'false':
        return 0  # Fake News
    else:  # barely-true, half-true, mostly-true, true
        return 1  # True News

train_df['label'] = train_df['label'].apply(label_to_multiclass)
valid_df['label'] = valid_df['label'].apply(label_to_multiclass)
test_df['label']  = test_df['label'].apply(label_to_multiclass)

# -------------------- TEXT CLEANING --------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):   # Handle missing values
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

train_df['clean_text'] = train_df['text'].apply(clean_text)
valid_df['clean_text'] = valid_df['text'].apply(clean_text)
test_df['clean_text']  = test_df['text'].apply(clean_text)

# -------------------- TF-IDF VECTORIZE --------------------
tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(train_df['clean_text'])
X_valid = tfidf.transform(valid_df['clean_text'])
X_test  = tfidf.transform(test_df['clean_text'])

y_train = train_df['label']
y_valid = valid_df['label']
y_test  = test_df['label']


print("\nLabel distribution in training data:")
print(train_df['label'].value_counts())

# -------------------- SAVE PROCESSED DATA --------------------

# Get absolute path of current script folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up → project root folder
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Create ProcessedData folder inside project
PROCESSED_DIR = os.path.join(PROJECT_DIR, "ProcessedData")

# Create folder safely
if not os.path.exists(PROCESSED_DIR):
    os.mkdir(PROCESSED_DIR)

# Save files
pickle.dump(X_train, open(os.path.join(PROCESSED_DIR, "X_train.pkl"), "wb"))
pickle.dump(X_valid, open(os.path.join(PROCESSED_DIR, "X_valid.pkl"), "wb"))
pickle.dump(X_test,  open(os.path.join(PROCESSED_DIR, "X_test.pkl"), "wb"))

pickle.dump(y_train, open(os.path.join(PROCESSED_DIR, "y_train.pkl"), "wb"))
pickle.dump(y_valid, open(os.path.join(PROCESSED_DIR, "y_valid.pkl"), "wb"))
pickle.dump(y_test,  open(os.path.join(PROCESSED_DIR, "y_test.pkl"), "wb"))

pickle.dump(tfidf, open(os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"), "wb"))

print("✅ Preprocessing completed successfully! ProcessedData folder created.")

