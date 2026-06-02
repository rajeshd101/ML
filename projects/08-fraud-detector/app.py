from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import numpy as np

app = FastAPI(title="Credit Card Fraud Detector")
templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')

# Load model
model = None
try:
    model = joblib.load('fraud_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run 'python train.py' first to train the model.")

@app.get('/', response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/predict')
async def predict(request: Request):
    data = await request.json()
    amount = float(data.get('amount', 0))
    
    if model is None:
        return JSONResponse({'error': 'Model not loaded. Run train.py first.'}, status_code=500)
    
    features = np.zeros((1, 29))
    features[0, -1] = amount  # Amount is the last feature
    
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])
    
    return {
        'prediction': 'FRAUD' if pred == 1 else 'NORMAL',
        'probability': round(prob * 100, 2),
        'is_fraud': pred == 1
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8008)