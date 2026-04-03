"""
House Price Predictor - Training Script
Uses California Housing dataset from sklearn
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import json

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target: MedHouseVal")
print(f"Sample size: {len(X)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel: RandomForestRegressor(n_estimators=100)")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: ${rmse:.2f} (in units of $100,000)")
print(f"RMSE (actual): ${rmse * 100000:.2f}")

# Save model
joblib.dump(model, 'housing_model.pkl')
print(f"\nModel saved: housing_model.pkl")

# Save feature names
feature_names = list(X.columns)
with open('features.json', 'w') as f:
    json.dump(feature_names, f)
print(f"Features saved: features.json")
print("Training complete!")