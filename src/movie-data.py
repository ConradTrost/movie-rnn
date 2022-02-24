"""
Clean up data from Movie Lens
"""
import pandas as pd

tmdb_links = pd.read_csv('../_data/links.csv')
tmdb_links.drop(labels=['imdbId'], axis=1, inplace=True)
tmdb_links = tmdb_links[pd.to_numeric(tmdb_links['tmdbId'], errors='coerce').notnull()]
tmdb_links['tmdbId'] = tmdb_links['tmdbId'].astype(int)

movies = pd.read_csv('../_data/movies.csv')
df = pd.merge(tmdb_links, movies, how='inner', on='movieId')

df.drop(labels='movieId', axis=1, inplace=True)

df.to_csv('../_clean-data/tmdbMovies.csv', index=False)
