# src/scripts/content_based_filtering.py

import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import logging
import sys
import os

def setup_logging(log_path='content_based_filtering.log'):
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

def load_movie_metadata(metadata_path):
    logging.info(f"Loading movie metadata from {metadata_path}...")
    if not os.path.exists(metadata_path):
        logging.error(f"Movie metadata file does not exist at {metadata_path}.")
        sys.exit(1)
    try:
        movies_df = pd.read_csv(metadata_path)
        logging.info(f"Loaded {len(movies_df)} movies.")
    except Exception as e:
        logging.error(f"Failed to load movie metadata: {e}")
        sys.exit(1)
    return movies_df

def preprocess_features(movies_df):
    logging.info("Preprocessing movie features...")
    # Combine relevant features into a single string
    movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['overview'] + ' ' + movies_df['keywords']
    logging.info("Combined features created.")
    return movies_df

def vectorize_features(movies_df, n_features=1000):
    logging.info("Vectorizing movie features using TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=n_features)
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, tfidf

def compute_similarity(tfidf_matrix):
    logging.info("Computing cosine similarity matrix...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    logging.info("Cosine similarity matrix computed.")
    return cosine_sim

def save_content_based_model(movies_df, cosine_sim, tfidf, model_path='content_based_model.joblib'):
    logging.info(f"Saving content-based model to {model_path}...")
    try:
        joblib.dump({
            'movies_df': movies_df,
            'cosine_sim': cosine_sim,
            'tfidf': tfidf
        }, model_path)
        logging.info("Content-based model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save content-based model: {e}")
        sys.exit(1)

def main():
    setup_logging()
    logging.info("Starting content-based filtering script.")
    
    # Load configuration
    config = load_config()
    
    # Extract paths from config
    cf_config = config.get('collaborative_filtering', {})
    metadata_path = cf_config.get('metadata_path', 
        '/share/blondin/jrfloren/movie-recommender/data/raw/movies_metadata.csv')
    content_model_path = cf_config.get('content_model_path', 
        '/share/blondin/jrfloren/movie-recommender/models/content_based_model.joblib')
    
    # Load movie metadata
    movies_df = load_movie_metadata(metadata_path)
    
    # Preprocess features
    movies_df = preprocess_features(movies_df)
    
    # Vectorize features
    tfidf_matrix, tfidf = vectorize_features(movies_df)
    
    # Compute similarity
    cosine_sim = compute_similarity(tfidf_matrix)
    
    # Save the content-based model
    save_content_based_model(movies_df, cosine_sim, tfidf, content_model_path)
    
    logging.info("Content-based filtering script completed successfully.")

if __name__ == "__main__":
    main()
