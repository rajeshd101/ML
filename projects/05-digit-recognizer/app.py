"""
Digit Recognizer - FastAPI Web Application
"""

import json
import numpy as np
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Digit Recognizer")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load both models
rf_model = joblib.load("digit_rf.pkl")
lr_model = joblib.load("digit_lr.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-draw")
async def predict_draw(request: Request):
    data = await request.json()
    
    # Get pixel values - 64 values expected
    pixels = data.get("pixels", [])
    if not pixels or len(pixels) != 64:
        return JSONResponse({
            "error": "Expected 64 pixel values (0-16)"
        }, status_code=400)
    
    # Convert to numpy array and reshape
    X = np.array(pixels, dtype=np.float64).reshape(1, -1)
    
    # Scale pixels: dataset is 0-16, ensure we handle it properly
    X = X / 16.0  # Normalize to 0-1 range
    
    # RF prediction
    rf_pred = int(rf_model.predict(X)[0])
    rf_probs = rf_model.predict_proba(X)[0]
    rf_confidence = float(rf_probs[rf_pred])
    
    # LR prediction
    lr_pred = int(lr_model.predict(X)[0])
    lr_probs = lr_model.predict_proba(X)[0]
    lr_confidence = float(lr_probs[lr_pred])
    
    # Get top 3 for each
    rf_top3_indices = np.argsort(rf_probs)[-3:][::-1]
    rf_top3 = [{"digit": int(i), "confidence": float(rf_probs[i])} for i in rf_top3_indices]
    
    lr_top3_indices = np.argsort(lr_probs)[-3:][::-1]
    lr_top3 = [{"digit": int(i), "confidence": float(lr_probs[i])} for i in lr_top3_indices]
    
    return JSONResponse({
        "random_forest": {
            "prediction": rf_pred,
            "confidence": rf_confidence,
            "top3": rf_top3
        },
        "logistic_regression": {
            "prediction": lr_pred,
            "confidence": lr_confidence,
            "top3": lr_top3
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)