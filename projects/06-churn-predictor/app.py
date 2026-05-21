"""
Customer Churn Predictor - FastAPI Web Application
"""

import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Customer Churn Predictor")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and features
model = joblib.load("churn_model.pkl")
with open("churn_features.json", "r") as f:
    feature_names = json.load(f)

# Define feature metadata for the form
# Categorical features that need dropdown/select
categorical_fields = {
    "gender": {
        "type": "select",
        "options": {"Male": 1, "Female": 0},
        "label": "Gender",
        "default": "Male"
    },
    "Partner": {
        "type": "select",
        "options": {"Yes": 1, "No": 0},
        "label": "Has Partner",
        "default": "No"
    },
    "Dependents": {
        "type": "select",
        "options": {"Yes": 1, "No": 0},
        "label": "Has Dependents",
        "default": "No"
    },
    "PhoneService": {
        "type": "select",
        "options": {"Yes": 1, "No": 0},
        "label": "Phone Service",
        "default": "Yes"
    },
    "MultipleLines": {
        "type": "onehot",
        "prefix": "MultipleLines",
        "categories": ["No", "No phone service", "Yes"],
        "label": "Multiple Lines",
        "default": "No"
    },
    "InternetService": {
        "type": "onehot",
        "prefix": "InternetService",
        "categories": ["DSL", "Fiber optic", "No"],
        "label": "Internet Service",
        "default": "DSL"
    },
    "OnlineSecurity": {
        "type": "onehot",
        "prefix": "OnlineSecurity",
        "categories": ["No", "Yes", "No internet service"],
        "label": "Online Security",
        "default": "No"
    },
    "OnlineBackup": {
        "type": "onehot",
        "prefix": "OnlineBackup",
        "categories": ["No", "Yes", "No internet service"],
        "label": "Online Backup",
        "default": "No"
    },
    "DeviceProtection": {
        "type": "onehot",
        "prefix": "DeviceProtection",
        "categories": ["No", "Yes", "No internet service"],
        "label": "Device Protection",
        "default": "No"
    },
    "TechSupport": {
        "type": "onehot",
        "prefix": "TechSupport",
        "categories": ["No", "Yes", "No internet service"],
        "label": "Tech Support",
        "default": "No"
    },
    "StreamingTV": {
        "type": "onehot",
        "prefix": "StreamingTV",
        "categories": ["No", "Yes", "No internet service"],
        "label": "Streaming TV",
        "default": "No"
    },
    "StreamingMovies": {
        "type": "onehot",
        "prefix": "StreamingMovies",
        "categories": ["No", "Yes", "No internet service"],
        "label": "Streaming Movies",
        "default": "No"
    },
    "Contract": {
        "type": "onehot",
        "prefix": "Contract",
        "categories": ["Month-to-month", "One year", "Two year"],
        "label": "Contract",
        "default": "Month-to-month"
    },
    "PaperlessBilling": {
        "type": "select",
        "options": {"Yes": 1, "No": 0},
        "label": "Paperless Billing",
        "default": "Yes"
    },
    "PaymentMethod": {
        "type": "onehot",
        "prefix": "PaymentMethod",
        "categories": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
        "label": "Payment Method",
        "default": "Electronic check"
    }
}

numeric_fields = {
    "tenure": {"label": "Tenure (months)", "default": 1, "min": 0, "max": 100},
    "MonthlyCharges": {"label": "Monthly Charges ($)", "default": 50.0, "min": 0, "max": 200},
    "TotalCharges": {"label": "Total Charges ($)", "default": 100.0, "min": 0, "max": 10000}
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "categorical_fields": categorical_fields,
        "numeric_fields": numeric_fields
    })

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    
    # Build feature vector
    input_values = {}
    
    # Handle simple binary selects
    for name, meta in categorical_fields.items():
        if meta["type"] == "select":
            input_values[name] = int(meta["options"].get(data.get(name, meta["default"]), 0))
    
    # Handle numeric fields
    for name, meta in numeric_fields.items():
        val = float(data.get(name, meta["default"]))
        input_values[name] = val
    
    # Handle one-hot encoded fields
    for name, meta in categorical_fields.items():
        if meta["type"] == "onehot":
            selected = data.get(name, meta["default"])
            for cat in meta["categories"]:
                col_name = f"{meta['prefix']}_{cat}"
                input_values[col_name] = 1 if selected == cat else 0
    
    # Build vector in correct feature order
    X = np.array([input_values.get(f, 0) for f in feature_names]).reshape(1, -1)
    
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])  # Probability of churn
    
    if pred == 1:
        result_text = "⚠️ Likely to Churn"
        probability = proba
    else:
        result_text = "✅ Likely to Stay"
        probability = 1 - proba
    
    return JSONResponse({
        "prediction": pred,
        "churn_probability": proba,
        "result_text": result_text,
        "probability": probability,
        "probability_pct": f"{probability * 100:.1f}%"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8006)