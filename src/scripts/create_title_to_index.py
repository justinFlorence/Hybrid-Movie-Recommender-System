# src/scripts/create_title_to_index.py

import pandas as pd
import joblib
import logging
import sys
import os
import yaml

def setup_logging(log_path='create_title_to_index.log'):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def load_config(config_path='/share/blondin/jrfloren/movie-recommender/config.yml'):
    logging.info(f"Loading configuration from {config_path}...")
    if not os.path.exists(config_path):
        logging.error(f"Configuration file does not exist at {config_path}.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    return config

def load_data(ratings_movies_path, movie_mapping_path):
    logging.info(f"Loading ratings_movies_imdb_merged.csv from {ratings_movies_path}...")
    try:
        ratings_movies = pd.read_csv(ratings_movies_path)
        logging.info(f"Loaded {len(ratings_movies)} records from ratings_movies_imdb_merged.csv.")
    except Exception as e:
        logging.error(f"Error loading ratings_movies_imdb_merged.csv: {e}")
        sys.exit(1)
    
    logging.info(f"Loading movie_mapping.csv from {movie_mapping_path}...")
    try:
        movie_mapping = pd.read_csv(movie_mapping_path)
        logging.info(f"Loaded {len(movie_mapping)} records from movie_mapping.csv.")
    except Exception as e:
        logging.error(f"Error loading movie_mapping.csv: {e}")
        sys.exit(1)
    
    return ratings_movies, movie_mapping

def create_title_to_index(ratings_movies, movie_mapping):
    logging.info("Filtering ratings_movies to include only unique movies (titleType == 'movie')...")
    # Filter for entries where titleType is 'movie'
    if 'titleType' in ratings_movies.columns:
        movies = ratings_movies[ratings_movies['titleType'] == 'movie'].copy()
    else:
        logging.warning("'titleType' column not found. Proceeding without filtering.")
        movies = ratings_movies.copy()
    
    logging.info(f"Number of unique movie entries after filtering: {len(movies)}")
    
    # Remove duplicates based on 'movieId' and 'primaryTitle'
    movies = movies.drop_duplicates(subset=['movieId', 'primaryTitle'])
    logging.info(f"Number of movie entries after removing duplicates: {len(movies)}")
    
    # Merge with movie_mapping to get movie_index
    logging.info("Merging with movie_mapping to get movie_index...")
    movies = movies.merge(movie_mapping, on='movieId', how='left')
    
    # Check for any movies without a movie_index
    missing_indices = movies[movies['movie_index'].isnull()]
    if not missing_indices.empty:
        logging.warning(f"{len(missing_indices)} movies do not have a corresponding movie_index. These will be excluded.")
        movies = movies.dropna(subset=['movie_index'])
    
    # Create a mapping from primaryTitle (lowercased) to movie_index
    logging.info("Creating mapping from primaryTitle to movie_index...")
    title_to_index = pd.Series(movies.movie_index.values, index=movies.primaryTitle.str.lower()).to_dict()
    
    logging.info(f"Total unique movie titles mapped: {len(title_to_index)}")
    
    return title_to_index

def save_mapping(title_to_index, path='/share/blondin/jrfloren/movie-recommender/data/processed/title_to_index.pkl'):
    logging.info(f"Saving title_to_index mapping to {path}...")
    try:
        joblib.dump(title_to_index, path)
        logging.info("title_to_index mapping saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save title_to_index mapping: {e}")
        sys.exit(1)

def main():
    setup_logging()
    logging.info("Starting Title to Index Mapping Creation Script.")
    
    # Load configuration
    config = load_config()
    
    # Load data
    ratings_movies_path = config['imdb_datasets']['ratings_movies_imdb_merged']
    movie_mapping_path = config['collaborative_filtering']['movie_mapping']
    ratings_movies, movie_mapping = load_data(ratings_movies_path, movie_mapping_path)
    
    # Create title to index mapping
    title_to_index = create_title_to_index(ratings_movies, movie_mapping)
    
    # Save the mapping
    save_mapping(title_to_index, path=config['web_app']['title_to_index_path'])
    
    logging.info("Title to Index Mapping Creation Script Completed Successfully.")

if __name__ == "__main__":
    main()
