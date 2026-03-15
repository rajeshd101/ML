import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load sentiment dataset from sklearn
from sklearn.datasets import load_files
import os

# Create synthetic sentiment data
texts = [
    "This movie is absolutely amazing and brilliant", "Best film I have ever seen",
    "Incredible performance by the actors", "I loved every second of it",
    "Fantastic story and great cinematography", "This is a masterpiece",
    "Wonderful experience, highly recommended", "Outstanding movie overall",

    "This movie is terrible and boring", "Worst film I have ever seen",
    "Awful performance by the actors", "I hated every second of it",
    "Terrible story and poor cinematography", "This is a disaster",
    "Dreadful experience, not recommended", "Disappointing movie overall",
]
labels = [1] * 8 + [0] * 8

for _ in range(1250):
    texts.extend([
        "Amazing film, great acting and plot",
        "Loved it, would watch again",
        "Terrible movie, waste of time",
        "Awful script and bad directing"
    ])
    labels.extend([1, 1, 0, 0])

df = pd.DataFrame({'text': texts, 'label': labels})

X = df['text'].values
y = df['label'].values

# For speed, use a subset (first 10000 rows)
X = X[:10000]
y = y[:10000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

joblib.dump(pipeline, 'sentiment_model.pkl')
print("Saved sentiment_model.pkl")