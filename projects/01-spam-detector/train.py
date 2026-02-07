import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load UCI SMS Spam Collection (from zip)
import urllib.request, zipfile, io
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
resp = urllib.request.urlopen(url)
z = zipfile.ZipFile(io.BytesIO(resp.read()))
df = pd.read_csv(z.open("SMSSpamCollection"), sep='\t', names=['label', 'message'])

# Encode label: spam=1, ham=0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['message']
y = df['label']

# Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: TfidfVectorizer -> MultinomialNB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Save model
joblib.dump(pipeline, 'spam_model.pkl')
print("Model saved as spam_model.pkl")
