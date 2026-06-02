import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib

# Load Credit Card Fraud dataset (first 20000 rows)
print("Loading Credit Card Fraud dataset...")
url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
df = pd.read_csv(url, nrows=20000)
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} out of {len(df)} ({df['Class'].mean()*100:.3f}%)")

# Features: V1-V28, Amount (skip Time)
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
X = df[feature_cols].values
y = df['Class'].values

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline: StandardScaler -> RandomForestClassifier with class_weight='balanced'
print("\nTraining Random Forest with class_weight='balanced'...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Print metrics for fraud class (Class=1)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

print("\n--- Fraud Class Metrics ---")
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save model
joblib.dump(pipeline, 'fraud_model.pkl')
print("\nModel saved: fraud_model.pkl")
print("Training complete!")