# Fake News Detection System

A machine learning project that detects fake news articles using text classification.

## Project Overview

This system uses **Logistic Regression** to classify news articles as fake or real based on processed text features (TF-IDF vectors). The project pipeline includes data preprocessing, model training, and a Flask web interface for predictions.

## Project Structure

```
FakeNewsDetectionSystem/
├── app/                    # Flask web application
│   ├── app.py             # Main Flask app
│   ├── static/
│   │   └── style.css      # Website styling
│   └── templates/
│       └── index.html     # Web interface
├── Datasets/              # Raw datasets (TSV format)
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
├── Scripts/               # Python scripts
│   ├── preprocess.py      # Data cleaning and TF-IDF vectorization
│   ├── TrainModel.py      # Model training and validation
│   └── predict.py         # Prediction functionality
├── Models/                # Trained model storage
├── ProcessedData/         # Vectorized dataset pickles
└── README.md             # This file
```

## Code Explanation: TrainModel.py

This script trains a Logistic Regression classifier on preprocessed data.

### Key Steps:

1. **Load Processed Data** - Reads pickled TF-IDF vectors from `ProcessedData/` folder
   - `X_train.pkl`, `y_train.pkl` - Training features and labels
   - `X_valid.pkl`, `y_valid.pkl` - Validation features and labels

2. **Check Class Distribution** - Prints the number of fake vs. real news samples in training data

3. **Safety Validation** - Ensures both classes exist (prevents training on single class)

4. **Train Model** - Fits Logistic Regression with 1000 iterations
   ```python
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   ```

5. **Evaluate Performance** - Tests on validation set and calculates accuracy

6. **Save Model** - Stores trained model as `fake_news_model.pkl` in the `Models/` folder

## How to Use

### 1. Preprocess Data
```bash
python Scripts/preprocess.py
```
Cleans raw data and converts text to TF-IDF vectors.

### 2. Train Model
```bash
python Scripts/TrainModel.py
```
Trains the classifier and saves it to `Models/`.

### 3. Run Web App
```bash
python app/app.py
```
Launches the Flask web interface for making predictions.

## Technologies Used

- **scikit-learn** - Machine learning library (LogisticRegression, TF-IDF)
- **NumPy** - Numerical computing
- **Flask** - Web framework
- **Pickle** - Model serialization

## Model Details

- **Algorithm**: Logistic Regression
- **Feature Type**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classes**: 2 (Fake = 0, Real = 1)
- **Validation Metric**: Accuracy Score
