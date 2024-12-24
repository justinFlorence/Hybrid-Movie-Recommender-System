# src/preprocessing.py

import pandas as pd

def clean_movie_titles(movies_df):
    # Strip whitespace from titles
    movies_df['title'] = movies_df['title'].str.strip()
    return movies_df

def split_genres(movies_df):
    # Split genres into a list
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    return movies_df

def merge_ratings_movies(ratings_df, movies_df):
    # Merge ratings with movie titles and genres
    merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')
    return merged_df

def save_processed_data(df, path='data/processed/ratings_movies_merged.csv'):
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")

def merge_with_imdb(movies_df, imdb_basics_df, imdb_ratings_df):
    """
    Merge MovieLens movies with IMDb data based on title matching.
    Note: This is a simplistic approach and may require more sophisticated matching.
    """
    # Normalize titles for better matching
    movies_df['normalized_title'] = movies_df['title'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    imdb_basics_df['normalized_title'] = imdb_basics_df['primaryTitle'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    # Merge on normalized titles
    merged_df = pd.merge(movies_df, imdb_basics_df, on='normalized_title', how='left', suffixes=('', '_imdb'))

    return merged_df
