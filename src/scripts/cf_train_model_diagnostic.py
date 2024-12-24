# cf_train_model_diagnostic.py

import yaml
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
import joblib
from scipy import sparse
import logging
import sys
import os

def setup_logging(log_path='cf_train_model_diagnostic.log'):
    """
    Sets up logging to both a file and the console.
    """
    logging.basicConfig(
        filename=log_path,
        filemode='w',  # Overwrite the log file each run
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

def validate_config(config):
    """
    Validates the presence and correctness of required configuration parameters.
    """
    logging.info("Validating configuration parameters...")
    required_params = ['no_components', 'learning_rate', 'loss', 'epochs', 'num_threads']
    cf_config = config.get('collaborative_filtering', {})
    missing_params = [param for param in required_params if param not in cf_config]
    if missing_params:
        logging.error(f"Missing configuration parameters: {missing_params}")
        sys.exit(1)
    
    # Validate parameter types and values
    if not isinstance(cf_config['no_components'], int) or cf_config['no_components'] <= 0:
        logging.error("Parameter 'no_components' must be a positive integer.")
        sys.exit(1)
    if not isinstance(cf_config['learning_rate'], float) or cf_config['learning_rate'] <= 0:
        logging.error("Parameter 'learning_rate' must be a positive float.")
        sys.exit(1)
    if cf_config['loss'] not in ['warp', 'warp-kos', 'bpr', 'logistic']:
        logging.error("Parameter 'loss' must be one of ['warp', 'warp-kos', 'bpr', 'logistic'].")
        sys.exit(1)
    if not isinstance(cf_config['epochs'], int) or cf_config['epochs'] <= 0:
        logging.error("Parameter 'epochs' must be a positive integer.")
        sys.exit(1)
    if not isinstance(cf_config['num_threads'], int) or cf_config['num_threads'] <= 0:
        logging.error("Parameter 'num_threads' must be a positive integer.")
        sys.exit(1)
    
    logging.info("Configuration parameters are valid.")

def load_sparse_matrix(path, matrix_name="Matrix"):
    """
    Loads a sparse matrix from a .npz file and logs its properties.
    """
    logging.info(f"Loading {matrix_name} from {path}...")
    if not os.path.exists(path):
        logging.error(f"{matrix_name} file does not exist at {path}.")
        sys.exit(1)
    try:
        matrix = sparse.load_npz(path)
        logging.info(f"{matrix_name} loaded successfully.")
        logging.info(f"{matrix_name} shape: {matrix.shape}")
        logging.info(f"{matrix_name} non-zero entries (nnz): {matrix.nnz}")
        sparsity = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        logging.info(f"{matrix_name} sparsity: {sparsity:.6f}")
        if matrix.nnz == 0:
            logging.error(f"{matrix_name} is empty.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load {matrix_name}: {e}")
        sys.exit(1)
    return matrix

def load_mappings(users_path='/share/blondin/jrfloren/movie-recommender/data/processed/user_id_mapping.joblib', 
                 movies_path='/share/blondin/jrfloren/movie-recommender/data/processed/movie_id_mapping.joblib'):
    """
    Loads user and movie ID mappings and validates them.
    """
    logging.info("Loading user and movie ID mappings...")
    if not os.path.exists(users_path):
        logging.error(f"User ID mapping file does not exist at {users_path}.")
        sys.exit(1)
    if not os.path.exists(movies_path):
        logging.error(f"Movie ID mapping file does not exist at {movies_path}.")
        sys.exit(1)
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
    Checks if the interaction matrix dimensions match the mappings.
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

def run_toy_model():
    """
    Trains a LightFM model on a small toy dataset to ensure the training process works.
    """
    logging.info("Running toy model training to verify LightFM functionality...")
    try:
        from scipy.sparse import csr_matrix
        import numpy as np

        # Create a small toy interaction matrix
        toy_interactions = csr_matrix(np.array([
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0]
        ]))
        # Create toy item features
        toy_item_features = csr_matrix(np.eye(3))

        toy_model = LightFM(no_components=2, learning_rate=0.05, loss='warp')
        toy_model.fit(toy_interactions, item_features=toy_item_features, epochs=10, num_threads=2, verbose=True)

        # Evaluate toy model
        precision = precision_at_k(toy_model, toy_interactions, item_features=toy_item_features, k=2).mean()
        auc = auc_score(toy_model, toy_interactions, item_features=toy_item_features).mean()
        logging.info(f"Toy Model Precision@2: {precision:.4f}")
        logging.info(f"Toy Model AUC Score: {auc:.4f}")
    except Exception as e:
        logging.error(f"Toy model training failed: {e}")
        sys.exit(1)
    logging.info("Toy model training succeeded.")

def train_actual_model(config, interaction_matrix, item_features):
    """
    Attempts to train the LightFM model on the actual data with enhanced logging.
    """
    logging.info("Initializing LightFM model with actual configuration...")
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
    
    logging.info("Starting model training on actual data...")
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
    Evaluates the trained model and logs the metrics.
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

def main():
    setup_logging()
    logging.info("Starting the LightFM diagnostic script.")
    
    # Load and validate configuration
    config = load_config()
    validate_config(config)
    cf_config = config['collaborative_filtering']
    logging.info(f"Configuration Parameters: {cf_config}")
    
    # Load interaction matrix and item features
    interaction_matrix = load_sparse_matrix(
        path='/share/blondin/jrfloren/movie-recommender/data/processed/interaction_matrix.npz',
        matrix_name="Interaction Matrix"
    )
    item_features = load_sparse_matrix(
        path='/share/blondin/jrfloren/movie-recommender/data/processed/item_features.npz',
        matrix_name="Item Features"
    )
    
    # Load and validate mappings
    user_id_mapping, movie_id_mapping = load_mappings()
    
    # Check data consistency
    check_data_consistency(interaction_matrix, user_id_mapping, movie_id_mapping)
    
    # Run a toy model to ensure LightFM works
    run_toy_model()
    
    # Attempt to train the actual model with enhanced logging
    model = train_actual_model(cf_config, interaction_matrix, item_features)
    
    # Evaluate the model
    precision, auc = evaluate_model(model, interaction_matrix, item_features)
    
    logging.info("LightFM diagnostic script completed successfully.")
    logging.info(f"Final Model Precision@5: {precision:.4f}")
    logging.info(f"Final Model AUC Score: {auc:.4f}")

if __name__ == "__main__":
    main()
