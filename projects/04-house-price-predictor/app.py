"""
House Price Predictor - FastAPI Web Application
"""

import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="House Price Predictor")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and features
model = joblib.load("housing_model.pkl")
with open("features.json", "r") as f:
    feature_names = json.load(f)

FEATURE_DISPLAY_NAMES = {
    "MedInc": "Median Income ($10k)",
    "HouseAge": "House Age (years)",
    "AveRooms": "Avg Rooms per Household",
    "AveBedrms": "Avg Bedrooms per Household",
    "Population": "Block Population",
    "AveOccup": "Avg Occupants per Household",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}

FEATURE_DEFAULTS = {
    "MedInc": 3.87,
    "HouseAge": 28.6,
    "AveRooms": 5.43,
    "AveBedrms": 1.10,
    "Population": 1425,
    "AveOccup": 3.07,
    "Latitude": 35.63,
    "Longitude": -119.57
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": feature_names,
        "display_names": FEATURE_DISPLAY_NAMES,
        "defaults": FEATURE_DEFAULTS
    })

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    input_data = []
    for feat in feature_names:
        val = float(data.get(feat, 0))
        input_data.append(val)
    
    X = np.array(input_data).reshape(1, -1)
    pred = model.predict(X)[0]
    price = pred * 100000  # Convert from $100k units to dollars
    
    return JSONResponse({
        "prediction": float(pred),
        "price": float(price),
        "price_formatted": f"${price:,.2f}"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)