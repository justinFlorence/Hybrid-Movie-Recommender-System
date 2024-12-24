# src/preprocess.py

import sys
import os
from data_loading import load_ratings, load_movies, load_tags, load_links, load_imdb_basics, load_imdb_ratings
from preprocessing import clean_movie_titles, split_genres, merge_ratings_movies, save_processed_data, merge_with_imdb

def main():
    # Load MovieLens data
    print("Loading MovieLens data...")
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()
    links = load_links()
    print("MovieLens data loaded successfully.")

    # Clean movie titles
    print("Cleaning movie titles...")
    movies = clean_movie_titles(movies)

    # Split genres
    print("Splitting genres...")
    movies = split_genres(movies)

    # Merge ratings with movies
    print("Merging ratings with movies...")
    merged = merge_ratings_movies(ratings, movies)

    # Check for missing values
    print("Checking for missing values in merged data...")
    missing_values = merged.isnull().sum()
    print(missing_values)

    # Handle missing values if any (for simplicity, drop rows with missing values)
    if missing_values.any():
        print("Dropping rows with missing values...")
        merged = merged.dropna()
        print("Missing values handled.")

    # Save processed data
    print("Saving processed data...")
    save_processed_data(merged)
    print("Processed data saved successfully.")

    # Optional: Integrate IMDb data
    print("\n=== Optional: Integrating IMDb Data ===")
    try:
        print("Loading IMDb data...")
        imdb_basics = load_imdb_basics()
        imdb_ratings = load_imdb_ratings()
        print("IMDb data loaded successfully.")

        print("Merging MovieLens data with IMDb data...")
        merged_with_imdb = merge_with_imdb(merged, imdb_basics, imdb_ratings)
        print("IMDb integration completed.")

        # Check for missing values after IMDb merge
        print("Checking for missing values after IMDb merge...")
        missing_values_imdb = merged_with_imdb.isnull().sum()
        print(missing_values_imdb)

        # Save the IMDb integrated data
        print("Saving IMDb integrated data...")
        merged_with_imdb.to_csv('data/processed/ratings_movies_imdb_merged.csv', index=False)
        print("IMDb integrated data saved successfully.")

    except Exception as e:
        print(f"An error occurred during IMDb integration: {e}")
        print("Proceeding without IMDb integration.")

if __name__ == "__main__":
    main()
