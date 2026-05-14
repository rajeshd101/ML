"""
Digit Recognizer - Training Script
Uses MNIST digits from sklearn
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load digits dataset
digits = load_digits()
n_samples = len(digits.images)
X = digits.images.reshape(n_samples, -1)
y = digits.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {digits.target_names}")
print(f"Samples per class: {np.bincount(y)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Random Forest ---
print("\n--- Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, digits=3))

# --- Logistic Regression ---
print("\n--- Logistic Regression Classifier ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, digits=3))

# Save both models
joblib.dump(rf_model, 'digit_rf.pkl')
print(f"\nRandom Forest model saved: digit_rf.pkl (accuracy: {rf_accuracy:.4f})")

joblib.dump(lr_model, 'digit_lr.pkl')
print(f"Logistic Regression model saved: digit_lr.pkl (accuracy: {lr_accuracy:.4f})")

print("\nTraining complete!")