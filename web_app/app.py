# web_app/app.py

import streamlit as st
import pandas as pd
import joblib
from scipy import sparse
import numpy as np
import yaml
import os
import requests
import datetime

def load_config(config_path='../config.yml'):
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    if not os.path.exists(config_path):
        st.error(f"Configuration file does not exist at {config_path}.")
        st.stop()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@st.cache_resource
def load_models_and_data(config):
    """
    Loads the hybrid LightFM model and necessary data files.

    Args:
        config (dict): Configuration parameters.

    Returns:
        tuple: Loaded model, user and movie mappings, title to index mapping,
               movies dataframe, interaction matrix, and item features.
    """
    # Load LightFM hybrid model
    cf_model_path = config['collaborative_filtering']['model_path']
    if not os.path.exists(cf_model_path):
        st.error(f"Collaborative Filtering model not found at {cf_model_path}.")
        st.stop()
    cf_model = joblib.load(cf_model_path)
    
    # Load user and movie mappings
    user_mapping_path = config['collaborative_filtering']['user_mapping']
    movie_mapping_path = config['collaborative_filtering']['movie_mapping']
    if not os.path.exists(user_mapping_path) or not os.path.exists(movie_mapping_path):
        st.error("User or Movie ID mapping files are missing.")
        st.stop()
    user_mapping = pd.read_csv(user_mapping_path)
    movie_mapping = pd.read_csv(movie_mapping_path)
    
    # Load title to index mapping
    title_to_index_path = config['web_app']['title_to_index_path']
    if not os.path.exists(title_to_index_path):
        st.error(f"Title to Index mapping file not found at {title_to_index_path}.")
        st.stop()
    title_to_index = joblib.load(title_to_index_path)
    
    # Load movies metadata
    movies_metadata_path = config['imdb_datasets']['ratings_movies_imdb_merged']
    if not os.path.exists(movies_metadata_path):
        st.error(f"Movies metadata file not found at {movies_metadata_path}.")
        st.stop()
    movies_df = pd.read_csv(movies_metadata_path)
    
    # Load interaction matrix
    interaction_matrix_path = config['collaborative_filtering']['interaction_matrix_path']
    if not os.path.exists(interaction_matrix_path):
        st.error(f"Interaction matrix not found at {interaction_matrix_path}.")
        st.stop()
    interaction_matrix = sparse.load_npz(interaction_matrix_path)
    
    # Load item features
    item_features_path = config['collaborative_filtering']['item_features_path']
    if not os.path.exists(item_features_path):
        st.error(f"Item features file not found at {item_features_path}.")
        st.stop()
    item_features = sparse.load_npz(item_features_path)
    
    return cf_model, user_mapping, movie_mapping, title_to_index, movies_df, interaction_matrix, item_features

def get_movie_index(title, title_to_index):
    """
    Retrieves the movie index for a given movie title.

    Args:
        title (str): The movie title input by the user.
        title_to_index (dict): Mapping from movie titles to indices.

    Returns:
        int or None: The corresponding movie index or None if not found.
    """
    return title_to_index.get(title.lower(), None)

def generate_recommendations(cf_model, interaction_matrix, item_features, movie_indices, top_n=10):
    """
    Generates movie recommendations using the hybrid LightFM model.

    Args:
        cf_model (LightFM): The trained LightFM model.
        interaction_matrix (sparse matrix): User-item interaction matrix.
        item_features (sparse matrix): Item features matrix.
        movie_indices (list): List of movie indices input by the user.
        top_n (int): Number of recommendations to generate.

    Returns:
        list: List of recommended movie indices.
    """
    user_id = interaction_matrix.shape[0]  # Assign a new user ID

    # Create a temporary interaction vector for the new user
    new_user_interactions = sparse.lil_matrix((1, interaction_matrix.shape[1]))
    new_user_interactions[0, movie_indices] = 1

    # Append the new user to the interaction matrix
    interaction_matrix_extended = sparse.vstack([interaction_matrix, new_user_interactions])

    # Predict scores for all movies for the new user
    scores = cf_model.predict(
        user_ids=np.array([user_id]),
        item_ids=np.arange(interaction_matrix_extended.shape[1]),
        item_features=item_features,
        num_threads=4
    )

    # Exclude already liked movies by setting their scores to -inf
    scores[0, movie_indices] = -np.inf

    # Get top N movie indices
    top_indices = scores.argsort()[0][-top_n:][::-1]

    return top_indices.tolist()

def get_movie_details(movie_indices, movies_df):
    """
    Retrieves movie details for the recommended movie indices.

    Args:
        movie_indices (list): List of recommended movie indices.
        movies_df (DataFrame): DataFrame containing movies metadata.

    Returns:
        DataFrame: DataFrame with details of recommended movies.
    """
    # Ensure movie_indices are within the dataframe
    valid_indices = [idx for idx in movie_indices if idx < len(movies_df)]
    recommended_movies = movies_df.iloc[valid_indices][['primaryTitle', 'genres_imdb', 'startYear']].copy()
    recommended_movies.reset_index(drop=True, inplace=True)
    return recommended_movies

def get_movie_poster(title, tmdb_api_key):
    """
    Fetches the movie poster URL from TMDb API.

    Args:
        title (str): The movie title.
        tmdb_api_key (str): TMDb API key.

    Returns:
        str or None: URL of the movie poster or None if not found.
    """
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': tmdb_api_key,
        'query': title
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w200{poster_path}"
    except Exception as e:
        st.warning(f"Failed to fetch poster for {title}: {e}")
    return None

def log_feedback(movie, feedback, log_path='data/processed/user_feedback.csv'):
    """
    Logs user feedback for a recommended movie.

    Args:
        movie (str): The movie title.
        feedback (str): Feedback type ('like' or 'dislike').
        log_path (str): Path to the feedback log CSV file.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {'movie': movie, 'feedback': feedback, 'timestamp': timestamp}
    feedback_df = pd.DataFrame([log_entry])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.exists(log_path):
        feedback_df.to_csv(log_path, index=False)
    else:
        feedback_df.to_csv(log_path, mode='a', header=False, index=False)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="🎬 Movie Recommender System", layout="wide")
    st.title("🎬 Movie Recommender System")
    st.markdown("""
    ## Get Personalized Movie Recommendations
    Enter **at least 5 movies** you really like, and we'll recommend **10 movies** you'll enjoy.
    """)

    # Load configuration
    config = load_config()

    # Load models and data
    cf_model, user_mapping, movie_mapping, title_to_index, movies_df, interaction_matrix, item_features = load_models_and_data(config)

    # Get TMDb API key if available (for fetching movie posters)
    tmdb_api_key = config.get('tmdb', {}).get('api_key', None)

    # Get list of all movie titles
    if 'titleType' in movies_df.columns:
        movie_titles = movies_df[movies_df['titleType'] == 'movie']['primaryTitle'].tolist()
    else:
        movie_titles = movies_df['primaryTitle'].tolist()

    # User Input with Multiselect (Searchable)
    favorite_movies = st.multiselect(
        "Enter your favorite movies (at least 5):",
        options=movie_titles,
        default=[],
        max_selections=5,
        help="Select at least 5 movies you really like."
    )

    if st.button("Get Recommendations"):
        if not favorite_movies:
            st.error("Please enter at least 5 movie titles.")
        else:
            if len(favorite_movies) < 5:
                st.error("Please enter at least 5 movie titles.")
            else:
                # Map movie titles to indices
                movie_indices = []
                not_found = []
                for title in favorite_movies:
                    idx = get_movie_index(title, title_to_index)
                    if idx is not None:
                        movie_indices.append(int(idx))
                    else:
                        not_found.append(title)

                if not_found:
                    st.warning(f"The following movies were not found and will be ignored: {', '.join(not_found)}")

                if len(movie_indices) < 5:
                    st.error("Please enter at least 5 valid movie titles.")
                else:
                    # Generate recommendations
                    with st.spinner('Generating recommendations...'):
                        top_indices = generate_recommendations(
                            cf_model,
                            interaction_matrix,
                            item_features,
                            movie_indices,
                            top_n=10
                        )

                    # Get movie details
                    recommended_movies = get_movie_details(top_indices, movies_df)

                    # Display recommendations
                    st.success("### Recommended Movies for You:")
                    for idx, row in recommended_movies.iterrows():
                        title = row['primaryTitle']
                        genres = row['genres_imdb'] if pd.notnull(row['genres_imdb']) else 'N/A'
                        release_year = int(row['startYear']) if pd.notnull(row['startYear']) else 'N/A'

                        # Create two columns: one for the poster, one for the details
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            if tmdb_api_key:
                                poster_url = get_movie_poster(title, tmdb_api_key)
                                if poster_url:
                                    st.image(poster_url, width=100)
                                else:
                                    st.write("No Poster Available")
                            else:
                                st.write("Poster Not Enabled")

                        with col2:
                            st.markdown(f"**{idx + 1}. {title}** ({release_year})")
                            st.markdown(f"**Genres:** {genres}")
                            
                            # Optional: Display plot and ratings if available
                            if 'plot' in row:
                                plot = row['plot']
                                st.markdown(f"**Plot:** {plot}")
                            if 'averageRating' in row:
                                rating = row['averageRating']
                                st.markdown(f"**Rating:** {rating}")

                            # Feedback Buttons
                            like, dislike = st.columns(2)
                            with like:
                                if st.button(f"👍 Like", key=f"like_{idx}"):
                                    log_feedback(title, 'like')
                                    st.success(f"You liked **{title}**!")
                            with dislike:
                                if st.button(f"👎 Dislike", key=f"dislike_{idx}"):
                                    log_feedback(title, 'dislike')
                                    st.warning(f"You disliked **{title}**!")

    # Optional: Display all available movie titles for reference
    # st.write("### Available Movies:")
    # st.write(movie_titles)

if __name__ == "__main__":
    main()
