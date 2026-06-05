# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Generate synthetic fake news dataset
print("Generating synthetic fake news dataset...")
np.random.seed(42)
real_texts = [
    "Breaking News: Scientists discover new species", "Market closes with gains today",
    "Government announces new policy", "Medical breakthrough in treatment",
    "World leaders meet for summit", "University research shows progress",
    "Economic report shows growth", "Environmental protection law passed"
]
fake_texts = [
    "Amazing trick doctors hate", "Cure found doctors won't tell",
    "Shocking conspiracy revealed", "Secret discovered by scientists",
    "Incredible discovery hidden from public", "Truth about government lies",
    "Unbelievable incident caught on camera", "Experts shocked by finding"
]
texts = real_texts * 15 + fake_texts * 15
labels = ['REAL'] * (len(real_texts) * 15) + ['FAKE'] * (len(fake_texts) * 15)
df = pd.DataFrame({'text': texts, 'label': labels})
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Encode labels: REAL=1, FAKE=0
df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})
print(f"Label distribution:\n{df['label'].value_counts()}")

X = df['text'].fillna('')
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorizer + Classifier Pipeline
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = PassiveAggressiveClassifier()
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== Results ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])}")

# Save model and vectorizer together
model_data = {
    'classifier': clf,
    'vectorizer': vectorizer
}
joblib.dump(model_data, 'fake_news_model.pkl')
print("\n✅ Model saved as 'fake_news_model.pkl'")