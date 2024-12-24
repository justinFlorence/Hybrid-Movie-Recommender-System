# src/scripts/cf_train_model.py

import yaml
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
import joblib
from scipy import sparse
import logging
import sys

def setup_logging(log_path='/share/blondin/jrfloren/movie-recommender/data/processed/cf_train_model.log'):
    """
    Sets up logging to both a file and the console.
    """
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def load_config(config_path='/share/blondin/jrfloren/movie-recommender/config.yml'):
    """
    Loads the configuration from a YAML file.
    """
    logging.info(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    return config

def load_interaction_matrix(path='/share/blondin/jrfloren/movie-recommender/data/processed/interaction_matrix.npz'):
    """
    Loads the interaction matrix from a .npz file.
    """
    logging.info(f"Loading interaction matrix from {path}...")
    try:
        interaction_matrix = sparse.load_npz(path)
        logging.info(f"Interaction matrix loaded with shape {interaction_matrix.shape}.")
    except Exception as e:
        logging.error(f"Failed to load interaction matrix: {e}")
        sys.exit(1)
    return interaction_matrix

def load_item_features(path='/share/blondin/jrfloren/movie-recommender/data/processed/item_features_reduced.npz'):
    """
    Loads the reduced item features from a .npz file.
    Ensures that the loaded matrix is sparse.
    """
    logging.info(f"Loading reduced item features from {path}...")
    try:
        item_features = sparse.load_npz(path)
        logging.info(f"Reduced item features loaded with shape {item_features.shape}.")
        logging.info(f"Reduced item features non-zero entries (nnz): {item_features.nnz}")
        sparsity = item_features.nnz / (item_features.shape[0] * item_features.shape[1])
        logging.info(f"Reduced item features sparsity: {sparsity:.6f}")
        
        if not sparse.isspmatrix(item_features):
            logging.error("Reduced item features are not in a sparse matrix format.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load reduced item features: {e}")
        sys.exit(1)
    return item_features

def load_mappings(users_path='/share/blondin/jrfloren/movie-recommender/data/processed/user_id_mapping.joblib', 
                movies_path='/share/blondin/jrfloren/movie-recommender/data/processed/movie_id_mapping.joblib'):
    """
    Loads user and movie ID mappings from joblib files.
    """
    logging.info("Loading user and movie ID mappings...")
    try:
        user_id_mapping = joblib.load(users_path)
        movie_id_mapping = joblib.load(movies_path)
        logging.info(f"Loaded {len(user_id_mapping)} users and {len(movie_id_mapping)} movies.")
        
        if len(user_id_mapping) == 0 or len(movie_id_mapping) == 0:
            logging.error("User or Movie ID mapping is empty.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load mappings: {e}")
        sys.exit(1)
    return user_id_mapping, movie_id_mapping

def check_data_consistency(interaction_matrix, user_id_mapping, movie_id_mapping):
    """
    Ensures that the interaction matrix dimensions match the user and movie mappings.
    """
    num_users, num_items = interaction_matrix.shape
    logging.info(f"Interaction matrix has {num_users} users and {num_items} items.")
    
    if num_users != len(user_id_mapping):
        logging.error(f"Number of users in interaction matrix ({num_users}) does not match user ID mapping ({len(user_id_mapping)}).")
        sys.exit(1)
    if num_items != len(movie_id_mapping):
        logging.error(f"Number of items in interaction matrix ({num_items}) does not match movie ID mapping ({len(movie_id_mapping)}).")
        sys.exit(1)
    logging.info("Interaction matrix dimensions match the ID mappings.")

def train_model(config, interaction_matrix, item_features):
    """
    Initializes and trains the LightFM model using the provided configuration and data.
    """
    logging.info("Initializing LightFM model...")
    try:
        model = LightFM(
            no_components=config['no_components'],
            learning_rate=config['learning_rate'],
            loss=config['loss']
        )
        logging.info("LightFM model initialized successfully.")
    except TypeError as e:
        logging.error(f"Failed to initialize LightFM model: {e}")
        sys.exit(1)
    
    logging.info("Starting model training...")
    try:
        model.fit(
            interaction_matrix,
            item_features=item_features,
            epochs=config['epochs'],
            num_threads=config['num_threads'],
            verbose=True
        )
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        sys.exit(1)
    
    return model

def evaluate_model(model, interaction_matrix, item_features):
    """
    Evaluates the trained model using Precision@5 and AUC metrics.
    """
    logging.info("Evaluating the model...")
    try:
        precision = precision_at_k(model, interaction_matrix, item_features=item_features, k=5).mean()
        auc = auc_score(model, interaction_matrix, item_features=item_features).mean()
        logging.info(f"Precision@5: {precision:.4f}")
        logging.info(f"AUC Score: {auc:.4f}")
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        sys.exit(1)
    return precision, auc

def save_model(model, path='/share/blondin/jrfloren/movie-recommender/data/processed/lightfm_model.joblib'):
    """
    Saves the trained LightFM model to a file.
    """
    logging.info(f"Saving the model to {path}...")
    try:
        joblib.dump(model, path)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")
        sys.exit(1)

def save_evaluation_metrics(precision, auc, path='/share/blondin/jrfloren/movie-recommender/data/processed/cf_evaluation_metrics.txt'):
    """
    Saves the evaluation metrics to a text file.
    """
    logging.info(f"Saving evaluation metrics to {path}...")
    try:
        with open(path, 'w') as f:
            f.write(f"Precision@5: {precision:.4f}\n")
            f.write(f"AUC Score: {auc:.4f}\n")
        logging.info("Evaluation metrics saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save evaluation metrics: {e}")
        sys.exit(1)

def main():
    setup_logging()
    logging.info("Starting the LightFM training script.")
    
    # Load configuration
    config = load_config()
    
    # Load interaction matrix and reduced item features
    interaction_matrix = load_interaction_matrix()
    item_features = load_item_features()
    
    # Verify sparsity and format of item features
    if not sparse.isspmatrix(item_features):
        logging.error("Item features matrix is not in sparse format.")
        sys.exit(1)
    if item_features.nnz == 0:
        logging.error("Item features matrix is empty.")
        sys.exit(1)
    logging.info(f"Item features sparsity: {item_features.nnz / (item_features.shape[0] * item_features.shape[1]):.6f}")
    
    # Load mappings
    user_id_mapping, movie_id_mapping = load_mappings()
    
    # Check data consistency
    check_data_consistency(interaction_matrix, user_id_mapping, movie_id_mapping)
    
    # Train the model
    model = train_model(config['collaborative_filtering'], interaction_matrix, item_features)
    
    # Evaluate the model
    precision, auc = evaluate_model(model, interaction_matrix, item_features)
    
    # Save the model and metrics
    save_model(model)
    save_evaluation_metrics(precision, auc)
    
    logging.info("LightFM training script completed successfully.")

if __name__ == "__main__":
    main()
