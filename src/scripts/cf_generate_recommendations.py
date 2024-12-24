# src/scripts/cf_generate_recommendations.py

import pandas as pd
from scipy import sparse
from lightfm import LightFM
import joblib
import numpy as np

def load_model(path='../../data/processed/lightfm_model.joblib'):
    print(f"Loading model from {path}...")
    model = joblib.load(path)
    return model

def load_interaction_matrix(path='../../data/processed/interaction_matrix.npz'):
    print(f"Loading interaction matrix from {path}...")
    interaction_matrix = sparse.load_npz(path)
    return interaction_matrix

def load_mappings(users_path='../../data/processed/user_mapping.csv', movies_path='../../data/processed/movie_mapping.csv'):
    print("Loading user and movie mappings...")
    users = pd.read_csv(users_path)
    movies = pd.read_csv(movies_path)
    return users, movies

def get_movie_title(movie_id, movies_df):
    try:
        return movies_df.loc[movies_df['movieId'] == movie_id, 'title'].values[0]
    except IndexError:
        return "Unknown Movie"

def recommend_movies(model, interaction_matrix, user_id, users_df, movies_df, num_recommendations=10):
    user_indices = users_df.loc[users_df['userId'] == user_id, 'user_index'].values
    if len(user_indices) == 0:
        print(f"User ID {user_id} not found.")
        return []
    user_index = user_indices[0]
    
    # Create a list of user indices repeated for each item
    user_ids = [user_index] * interaction_matrix.shape[1]
    item_ids = np.arange(interaction_matrix.shape[1])
    
    # Predict scores for all items for the given user
    scores = model.predict(user_ids, item_ids).flatten()
    
    # Get indices of movies the user has already interacted with
    user_interactions = interaction_matrix.tocsr()[user_index].indices
    
    # Exclude already interacted movies by setting their scores to -inf
    scores[user_interactions] = -np.inf
    
    # Get top scoring movie indices
    top_indices = np.argpartition(scores, -num_recommendations)[-num_recommendations:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    
    # Map movie indices back to movie IDs
    recommended_movie_ids = movies_df.iloc[top_indices]['movieId'].values
    
    # Get movie titles
    recommendations = [get_movie_title(mid, movies_df) for mid in recommended_movie_ids]
    
    return recommendations

def main():
    model = load_model()
    interaction_matrix = load_interaction_matrix()
    users_df, movies_df = load_mappings()
    
    # Example: Recommend for user ID 1
    user_id = 1  # Change as needed
    print(f"Generating recommendations for User ID {user_id}...")
    recommendations = recommend_movies(model, interaction_matrix, user_id, users_df, movies_df)
    
    if not recommendations:
        print("No recommendations available.")
    else:
        print("Top Recommendations:")
        for idx, movie in enumerate(recommendations, start=1):
            print(f"{idx}. {movie}")
        
        # Save recommendations to file
        recommendations_df = pd.DataFrame({
            'userId': [user_id]*len(recommendations),
            'Recommendation Rank': list(range(1, len(recommendations)+1)),
            'Movie Title': recommendations
        })
        recommendations_df.to_csv(f'../../data/processed/user_{user_id}_recommendations.csv', index=False)
        print(f"Recommendations saved at ../../data/processed/user_{user_id}_recommendations.csv")

if __name__ == "__main__":
    main()
