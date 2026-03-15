from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import numpy as np

app = FastAPI(title='Sentiment Analyzer')
templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')

model = joblib.load('sentiment_model.pkl')


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict')
async def predict(text: str = Form(...)):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = round(float(max(proba)) * 100, 2)
    label = 'Positive' if pred == 1 else 'Negative'
    return JSONResponse({
        'prediction': label,
        'confidence': confidence
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8003)