"""
Clean up data from Movie Lens
"""
import pandas as pd

tmdb_links = pd.read_csv('../_data/links.csv')
tmdb_links.drop(labels=['imdbId'], axis=1, inplace=True)
tmdb_links = tmdb_links[pd.to_numeric(tmdb_links['tmdbId'], errors='coerce').notnull()]
tmdb_links['tmdbId'] = tmdb_links['tmdbId'].astype(int)

ratings = pd.read_csv('../_data/ratings.csv')
ratings.drop(labels='timestamp', axis=1, inplace=True)

df = pd.merge(tmdb_links, ratings, how='inner', on='movieId')
df = df.sort_values('userId', ascending=True).reset_index(drop=True)

df.drop(labels='movieId', axis=1, inplace=True)

df.to_csv('../_clean-data/tmdbRatings.csv', index=False)
