from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import json
import numpy as np

app = FastAPI(title='Movie Recommender')
templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')

# Load model and titles
item_sim = joblib.load('movie_sim.pkl')
with open('movie_titles.json', 'r') as f:
    movie_titles = json.load(f)

# Build sorted list of (id, title)
movie_list = sorted([(int(k), v) for k, v in movie_titles.items()], key=lambda x: x[1])


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'movies': movie_list
    })


@app.post('/recommend')
async def recommend(movie_id: int = Form(...)):
    if movie_id not in item_sim.index:
        return JSONResponse({'error': 'Movie not found'})

    sim_scores = item_sim[movie_id].sort_values(ascending=False)
    # Exclude the movie itself, take top 5
    top = sim_scores.iloc[1:6]
    recommendations = []
    for mid, score in top.items():
        title = movie_titles.get(int(mid), f"Movie {mid}")
        recommendations.append({
            'id': int(mid),
            'title': title,
            'similarity': round(float(score), 4)
        })
    return JSONResponse({'recommendations': recommendations})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8002)