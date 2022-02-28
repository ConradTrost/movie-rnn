import pandas as pd
import joblib
import numpy as np
from tensorflow import keras

model = keras.models.load_model('../_models/movie')
history = joblib.load('../_models/history.json')

df = pd.read_csv('../_clean-data/tmdbRatings.csv')

# Run this for 2293 (MallRats)

user_ids = df['userId'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df['tmdbId'].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df['user'] = df['userId'].map(user2user_encoded)
df['movie'] = df['tmdbId'].map(movie2movie_encoded)
df['rating'] = df['rating'].values.astype(np.float32)
df = df.sample(frac=1, random_state=42)

movie_df = pd.read_csv('../_clean-data/tmdbMovies.csv')

# Let us get a user and see the top recommendations.
# user_id = df.userId.sample(1).iloc[0]
user_id = 263
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[
    ~movie_df['tmdbId'].isin(movies_watched_by_user.tmdbId.values)
]['tmdbId']
movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
)
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
)
ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for user: {}'.format(user_id))
print('====' * 9)
print('Movies with high ratings from user')
print('----' * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by='rating', ascending=False)
    .head(5)
    .tmdbId.values
)
movie_df_rows = movie_df[movie_df['tmdbId'].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ':', row.genres)

print('----' * 8)
print('Top 10 movie recommendations')
print('----' * 8)
recommended_movies = movie_df[movie_df['tmdbId'].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ':', row.genres)
