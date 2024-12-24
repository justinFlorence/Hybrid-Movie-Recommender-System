# src/scripts/cf_prepare_data.py

import pandas as pd
from scipy import sparse
import joblib
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def load_ratings(path='/share/blondin/jrfloren/movie-recommender/data/raw/ml-32m/ratings.csv'):
    print(f"Loading ratings data from {path}...")
    dtype = {
        'userId': 'int32',
        'movieId': 'int32',
        'rating': 'float32',
        'timestamp': 'int64'
    }
    ratings = pd.read_csv(path, dtype=dtype)
    print(f"Ratings data loaded: {ratings.shape[0]} entries.")
    return ratings

def load_movies(path='/share/blondin/jrfloren/movie-recommender/data/raw/ml-32m/movies.csv'):
    print(f"Loading movies data from {path}...")
    movies = pd.read_csv(path)
    print(f"Movies data loaded: {movies.shape[0]} entries.")
    return movies

def merge_data(ratings, movies):
    print("Merging ratings with movie titles...")
    merged = pd.merge(ratings, movies, on='movieId', how='left')
    missing_titles = merged['title'].isnull().sum()
    if missing_titles > 0:
        print(f"Warning: {missing_titles} ratings have missing movie titles.")
    else:
        print("All ratings have corresponding movie titles.")
    return merged

def encode_ids(merged):
    print("Encoding user and movie IDs...")
    # Encode user IDs
    user_ids = merged['userId'].unique()
    user_id_mapping = {id: index for index, id in enumerate(user_ids)}
    merged['user_index'] = merged['userId'].map(user_id_mapping)
    
    # Encode movie IDs
    movie_ids = merged['movieId'].unique()
    movie_id_mapping = {id: index for index, id in enumerate(movie_ids)}
    merged['movie_index'] = merged['movieId'].map(movie_id_mapping)
    
    print(f"Number of unique users: {len(user_id_mapping)}")
    print(f"Number of unique movies: {len(movie_id_mapping)}")
    
    return merged, user_id_mapping, movie_id_mapping

def create_interaction_matrix(merged):
    print("Creating interaction matrix...")
    # Assuming implicit interactions: ratings > 0
    interactions = merged[merged['rating'] > 0]
    
    # Creating a sparse matrix
    interaction_matrix = sparse.coo_matrix(
        (interactions['rating'].astype(np.float32),
         (interactions['user_index'], interactions['movie_index'])),
        shape=(merged['user_index'].nunique(), merged['movie_index'].nunique())
    )
    
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    return interaction_matrix

def create_item_features(movies, movie_id_mapping):
    print("Creating item features based on genres...")
    # Extract genres and split into lists
    movies['genres'] = movies['genres'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))
    
    # Initialize MultiLabelBinarizer to one-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres'])
    genre_feature_names = mlb.classes_
    
    print(f"Number of genre features: {len(genre_feature_names)}")
    
    # Create a DataFrame for genres
    genre_df = pd.DataFrame(genre_features, columns=genre_feature_names)
    genre_df['movieId'] = movies['movieId']
    
    # Map movieId to movie_index
    genre_df['movie_index'] = genre_df['movieId'].map(movie_id_mapping)
    
    # Handle missing movie_index
    missing_indices = genre_df['movie_index'].isnull().sum()
    if missing_indices > 0:
        print(f"Warning: {missing_indices} movies have no corresponding movie_index.")
        genre_df = genre_df.dropna(subset=['movie_index'])
    
    # Drop movieId as it's no longer needed
    genre_df = genre_df.drop(columns=['movieId'])
    
    # Convert movie_index to integer
    genre_df['movie_index'] = genre_df['movie_index'].astype(int)
    
    # Create a sparse matrix for item features
    item_features_matrix = sparse.coo_matrix(
        (genre_df[genre_feature_names].values.flatten(),
         (np.repeat(genre_df['movie_index'].values, len(genre_feature_names)),
          np.tile(np.arange(len(genre_feature_names)), len(genre_df)))),
        shape=(len(movie_id_mapping), len(genre_feature_names))
    )
    
    print(f"Item features matrix shape: {item_features_matrix.shape}")
    return item_features_matrix, genre_feature_names

def save_processed_data(merged, user_id_mapping, movie_id_mapping, interaction_matrix, item_features_matrix, genre_feature_names):
    os.makedirs('/share/blondin/jrfloren/movie-recommender/data/processed/', exist_ok=True)
    
    print("Saving processed data...")
    merged.to_csv('/share/blondin/jrfloren/movie-recommender/data/processed/ratings_movies_merged.csv', index=False)
    joblib.dump(user_id_mapping, '/share/blondin/jrfloren/movie-recommender/data/processed/user_id_mapping.joblib')
    joblib.dump(movie_id_mapping, '/share/blondin/jrfloren/movie-recommender/data/processed/movie_id_mapping.joblib')
    sparse.save_npz('/share/blondin/jrfloren/movie-recommender/data/processed/interaction_matrix.npz', interaction_matrix)
    sparse.save_npz('/share/blondin/jrfloren/movie-recommender/data/processed/item_features.npz', item_features_matrix)
    
    # Save genre feature names
    with open('/share/blondin/jrfloren/movie-recommender/data/processed/genre_feature_names.txt', 'w') as f:
        for genre in genre_feature_names:
            f.write(f"{genre}\n")
    
    print("Processed data saved successfully.")

def main():
    ratings = load_ratings()
    movies = load_movies()
    merged = merge_data(ratings, movies)
    merged, user_id_mapping, movie_id_mapping = encode_ids(merged)
    interaction_matrix = create_interaction_matrix(merged)
    item_features_matrix, genre_feature_names = create_item_features(movies, movie_id_mapping)
    save_processed_data(merged, user_id_mapping, movie_id_mapping, interaction_matrix, item_features_matrix, genre_feature_names)

if __name__ == "__main__":
    main()
