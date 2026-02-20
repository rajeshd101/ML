import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Download MovieLens 100k dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
zip_path = "ml-100k.zip"

if not os.path.exists(zip_path):
    print("Downloading MovieLens 100k dataset...")
    urllib.request.urlretrieve(url, zip_path)

# Extract zip
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall()

# Read ratings data
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Read movie titles
items = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                     names=['item_id', 'title', 'release_date', 'video_release_date',
                            'imdb_url'] + [f'genre_{i}' for i in range(19)])

# Keep only item_id and title
items = items[['item_id', 'title']]

# Build user-item matrix
user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Compute cosine similarity between items (transpose matrix to get item-item)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity,
                                   index=user_item_matrix.columns,
                                   columns=user_item_matrix.columns)

# Create movie titles mapping
movie_titles = items.set_index('item_id')['title'].to_dict()

# Save
joblib.dump(item_similarity_df, 'movie_sim.pkl')
with open('movie_titles.json', 'w') as f:
    json.dump(movie_titles, f)

print(f"Saved movie_sim.pkl (shape: {item_similarity_df.shape})")
print(f"Saved movie_titles.json ({len(movie_titles)} movies)")
print("Training complete!")