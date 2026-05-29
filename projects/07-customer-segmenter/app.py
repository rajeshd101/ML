from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import numpy as np
import os

app = FastAPI(title="Customer Segmenter")
templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')

# Load models
model = None
scaler = None
try:
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run 'python train.py' first to train the model.")

SEGMENT_DESCRIPTIONS = {
    0: {'name': 'Budget Shoppers', 'emoji': '💰', 'desc': 'Low income, low spending — value-conscious customers who hunt for deals.'},
    1: {'name': 'High Spenders', 'emoji': '💎', 'desc': 'High income, high spending — premium customers who love to shop.'},
    2: {'name': 'Average Joes', 'emoji': '🙂', 'desc': 'Medium income, medium spending — the everyday balanced shoppers.'},
    3: {'name': 'Careful Planners', 'emoji': '📋', 'desc': 'High income, low spending — savers who plan their purchases carefully.'},
    4: {'name': 'Young Explorers', 'emoji': '🌟', 'desc': 'Low income, high spending — young adventurous shoppers living in the moment.'}
}

@app.get('/', response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/segment')
def segment(age: int = Form(...), annual_income: int = Form(...), spending_score: int = Form(...)):
    if model is None or scaler is None:
        return JSONResponse({'error': 'Model not loaded. Run train.py first.'}, status_code=500)
    
    features = np.array([[age, annual_income, spending_score]])
    features_scaled = scaler.transform(features)
    cluster = int(model.predict(features_scaled)[0])
    segment_info = SEGMENT_DESCRIPTIONS.get(cluster, {'name': 'Unknown', 'emoji': '❓', 'desc': 'No description'})
    
    return {
        'cluster': cluster,
        'name': segment_info['name'],
        'emoji': segment_info['emoji'],
        'description': segment_info['desc']
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8007)