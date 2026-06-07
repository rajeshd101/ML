# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

app = FastAPI(title="Airline Passenger Forecasting Service")
templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')

# Load the real-world trained model
try:
    model = joblib.load("arima_model.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

@app.get('/', response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get("/predict")
def predict(months: int = 6):
    """
    Predict future monthly international airline passenger counts.
    'months' is the number of future months to forecast.
    """
    if months < 1 or months > 24:
        raise HTTPException(status_code=400, detail="Please request between 1 and 24 months.")
    
    try:
        # Generate out-of-sample monthly forecasts
        forecast_results = model.forecast(steps=months)
        
        # Structure payload mapping future Month Start strings to predicted numbers
        predictions = {
            date.strftime("%Y-%m-%d"): int(round(float(val)))
            for date, val in zip(forecast_results.index, forecast_results)
        }
        
        return {
            "status": "success",
            "target_variable": "total_international_passengers_thousands",
            "forecast_months": months,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=True)

