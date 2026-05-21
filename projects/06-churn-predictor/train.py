"""
Customer Churn Predictor - Training Script
Telco Customer Churn dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json

# Generate synthetic Telco Customer Churn dataset
print("Generating synthetic customer churn dataset...")
np.random.seed(42)
n_samples = 5000
df = pd.DataFrame({
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    'TotalCharges': np.random.uniform(0, 10000, n_samples),
    'MonthlyCharges': np.random.uniform(20, 150, n_samples),
    'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.26, 0.74])
})
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Handle TotalCharges - replace empty strings with NaN, drop NaNs, convert to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
print(f"Shape after cleaning: {df.shape}")

# Binary encode simple binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    if col == 'gender':
        df[col] = df[col].map({'Male': 1, 'Female': 0})
    else:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# Multi-category columns to one-hot encode
multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']

df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=False)
print(f"Shape after one-hot encoding: {df.shape}")

# Separate features and target
y = df['Churn'].values
X = df.drop('Churn', axis=1).values
feature_names = list(df.drop('Churn', axis=1).columns)

print(f"Number of features: {len(feature_names)}")
print(f"Class distribution: {np.bincount(y)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
print("\n--- Random Forest Classifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay (0)', 'Churn (1)'], digits=3))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(cm)

# Save model
joblib.dump(model, 'churn_model.pkl')
print(f"\nModel saved: churn_model.pkl")

# Save feature names
with open('churn_features.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"Feature names saved: churn_features.json ({len(feature_names)} features)")

print("\nTraining complete!")