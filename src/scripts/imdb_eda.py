# src/scripts/imdb_eda.py

import pandas as pd

BASICS_PATH = 'data/raw/imdb/title.basics.tsv.gz'
RATINGS_PATH = 'data/raw/imdb/title.ratings.tsv.gz'

def load_imdb_data():
    print("Loading IMDb basics data...")
    basics = pd.read_csv(BASICS_PATH, sep='\t', na_values='\\N', low_memory=False)
    print("Loading IMDb ratings data...")
    ratings = pd.read_csv(RATINGS_PATH, sep='\t', na_values='\\N', low_memory=False)
    return basics, ratings

def basic_info(basics, ratings):
    print("\n=== IMDb Basics DataFrame ===")
    print(basics.head())
    print("\n=== IMDb Ratings DataFrame ===")
    print(ratings.head())
    print("\n=== DataFrame Shapes ===")
    print(f"Basics: {basics.shape}")
    print(f"Ratings: {ratings.shape}")

def main():
    basics, ratings = load_imdb_data()
    basic_info(basics, ratings)

if __name__ == "__main__":
    main()
