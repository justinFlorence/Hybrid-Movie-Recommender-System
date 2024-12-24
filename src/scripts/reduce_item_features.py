# reduce_item_features.py

import yaml
from sklearn.decomposition import PCA
from scipy import sparse
import joblib
import numpy as np
import logging
import sys
import os

def setup_logging(log_path='reduce_item_features.log'):
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

def load_item_features(path):
    """
    Loads the item features from a .npz file.
    """
    logging.info(f"Loading item features from {path}...")
    if not os.path.exists(path):
        logging.error(f"Item features file does not exist at {path}.")
        sys.exit(1)
    try:
        item_features = sparse.load_npz(path)
        logging.info(f"Item features loaded successfully with shape {item_features.shape}.")
    except Exception as e:
        logging.error(f"Failed to load item features: {e}")
        sys.exit(1)
    return item_features

def apply_pca_with_desired_sparsity(item_features, n_components=100, desired_sparsity=0.15):
    """
    Applies PCA to reduce the dimensionality of the item features and introduces sparsity
    based on the desired sparsity level.
    
    Parameters:
    - item_features: scipy.sparse matrix
    - n_components: int, number of PCA components
    - desired_sparsity: float, desired sparsity level (e.g., 0.15 for 15% non-zeros)
    
    Returns:
    - reduced_item_features_sparse: scipy.sparse matrix
    """
    logging.info(f"Applying PCA to reduce item features to {n_components} components...")
    
    # Convert to dense for PCA
    if sparse.isspmatrix(item_features):
        item_features_dense = item_features.toarray()
    else:
        item_features_dense = item_features
    
    logging.info("Fitting PCA model...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(item_features_dense)
    logging.info("PCA transformation completed.")
    
    # Calculate the threshold to achieve desired sparsity
    # desired_sparsity: proportion of non-zero entries, e.g., 0.15 means 15% non-zeros
    total_elements = reduced_features.size
    desired_nonzeros = int(total_elements * desired_sparsity)
    
    logging.info(f"Applying threshold to achieve desired sparsity of {desired_sparsity*100}% ({desired_nonzeros} non-zero entries)...")
    
    # Flatten the absolute values to find the cutoff
    abs_values = np.abs(reduced_features).flatten()
    # Get the threshold value
    if desired_nonzeros == 0:
        threshold = np.inf
    elif desired_nonzeros >= len(abs_values):
        threshold = 0.0
    else:
        # Partition the array to find the desired threshold
        threshold = np.partition(abs_values, -desired_nonzeros)[-desired_nonzeros]
    
    logging.info(f"Computed threshold: {threshold}")
    
    # Set values below the threshold to zero
    reduced_features[np.abs(reduced_features) < threshold] = 0.0
    logging.info("Thresholding completed.")
    
    # Convert to sparse matrix
    reduced_item_features_sparse = sparse.csr_matrix(reduced_features)
    logging.info(f"Reduced item features shape: {reduced_item_features_sparse.shape}")
    logging.info(f"Reduced item features non-zero entries (nnz): {reduced_item_features_sparse.nnz}")
    sparsity = reduced_item_features_sparse.nnz / (reduced_item_features_sparse.shape[0] * reduced_item_features_sparse.shape[1])
    logging.info(f"Reduced item features sparsity: {sparsity:.6f}")
    
    return reduced_item_features_sparse

def save_reduced_item_features(reduced_item_features, path):
    """
    Saves the reduced item features to a .npz file.
    """
    logging.info(f"Saving reduced item features to {path}...")
    try:
        sparse.save_npz(path, reduced_item_features)
        logging.info("Reduced item features saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save reduced item features: {e}")
        sys.exit(1)

def main():
    setup_logging()
    logging.info("Starting the item features reduction script.")
    
    # Load configuration
    config = load_config()
    
    # Extract item features path and PCA parameters
    cf_config = config.get('collaborative_filtering', {})
    original_item_features_path = cf_config.get('item_features_path', 
        '/share/blondin/jrfloren/movie-recommender/data/processed/item_features.npz')
    reduced_item_features_path = cf_config.get('reduced_item_features_path', 
        '/share/blondin/jrfloren/movie-recommender/data/processed/item_features_reduced.npz')
    n_components = cf_config.get('pca_components', 100)
    desired_sparsity = cf_config.get('pca_desired_sparsity', 0.15)  # Desired sparsity level (30%)
    
    # Load original item features
    item_features = load_item_features(original_item_features_path)
    
    # Apply PCA and introduce desired sparsity
    reduced_item_features = apply_pca_with_desired_sparsity(item_features, n_components=n_components, desired_sparsity=desired_sparsity)
    
    # Save reduced item features
    save_reduced_item_features(reduced_item_features, reduced_item_features_path)
    
    logging.info("Item features reduction script completed successfully.")

if __name__ == "__main__":
    main()
