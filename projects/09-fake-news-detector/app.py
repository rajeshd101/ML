# app.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI(title="Fake News Detector")
templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')

# Load model
try:
    model_data = joblib.load('fake_news_model.pkl')
    classifier = model_data['classifier']
    vectorizer = model_data['vectorizer']
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Run 'python train.py' first to train the model.")
    classifier = None
    vectorizer = None

@app.get('/', response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/predict')
async def predict(request: Request):
    data = await request.json()
    news_text = data.get('news_text', '')
    
    if classifier is None or vectorizer is None:
        return JSONResponse({'error': 'Model not loaded. Run train.py first.'}, status_code=500)
    
    # Vectorize input
    text_vec = vectorizer.transform([news_text])
    
    # Predict
    prediction = int(classifier.predict(text_vec)[0])
    prob = classifier.decision_function(text_vec)[0]
    # Convert decision function to probability-like confidence (0-1)
    confidence = 1 / (1 + np.exp(-abs(prob)))
    
    label = "REAL" if prediction == 1 else "FAKE"
    
    return JSONResponse({
        'prediction': label,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8009)