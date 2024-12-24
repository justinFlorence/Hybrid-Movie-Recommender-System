# src/scripts/create_id_mappings.py

import joblib
import pandas as pd
from scipy import sparse
import logging
import sys
import os
import yaml

def setup_logging(log_path='create_id_mappings.log'):
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

def load_config(config_path='config.yml'):
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

def load_movie_data(movies_path):
    logging.info(f"Loading movie data from {movies_path}...")
    try:
        movies = pd.read_csv(movies_path)
        logging.info(f"Loaded {len(movies)} movies.")
    except Exception as e:
        logging.error(f"Failed to load movie data: {e}")
        sys.exit(1)
    return movies

def create_mappings(interaction_matrix_path, movies):
    logging.info("Creating user and movie ID mappings...")
    # Load interaction matrix to determine number of users and movies
    try:
        interaction_matrix = sparse.load_npz(interaction_matrix_path)
        num_users, num_movies = interaction_matrix.shape
        logging.info(f"Interaction matrix has {num_users} users and {num_movies} movies.")
    except Exception as e:
        logging.error(f"Failed to load interaction matrix: {e}")
        sys.exit(1)
    
    # Create user ID mapping (assuming user IDs are 0-indexed)
    user_id_mapping = {f"user_{i}": i for i in range(num_users)}
    
    # Create movie ID mapping based on the order in the interaction matrix
    movie_id_mapping = {row['tconst']: idx for idx, row in movies.iterrows()}
    
    # Save mappings
    logging.info("Saving user ID mapping...")
    joblib.dump(user_id_mapping, 'data/processed/user_id_mapping.joblib')
    logging.info("Saving movie ID mapping...")
    joblib.dump(movie_id_mapping, 'data/processed/movie_id_mapping.joblib')
    logging.info("User and Movie ID mappings created successfully.")

def main():
    setup_logging()
    logging.info("Starting ID Mappings Creation Script.")
    
    # Load configuration
    config = load_config()
    
    # Load movie data
    movies_path = config['imdb_datasets']['movies_metadata_processed']  # Adjust path if necessary
    movies = load_movie_data(movies_path)
    
    # Create mappings
    interaction_matrix_path = config['collaborative_filtering']['interaction_matrix_path']
    create_mappings(interaction_matrix_path, movies)
    
    logging.info("ID Mappings Creation Script Completed Successfully.")

if __name__ == "__main__":
    main()
