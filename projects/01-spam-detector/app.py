from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import numpy as np

app = FastAPI(title="Spam SMS Detector")

# Mount static files
app.mount('/static', StaticFiles(directory='static'), name='static')

# Templates
templates = Jinja2Templates(directory='templates')

# Load model
model = joblib.load('spam_model.pkl')


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict')
async def predict(message: str = Form(...)):
    pred = model.predict([message])[0]
    proba = model.predict_proba([message])[0]

    if pred == 1:
        result = 'spam'
        confidence = round(proba[1] * 100, 2)
    else:
        result = 'ham'
        confidence = round(proba[0] * 100, 2)

    return JSONResponse({
        'prediction': result,
        'confidence': confidence
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001)