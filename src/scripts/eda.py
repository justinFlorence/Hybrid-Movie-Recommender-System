# src/scripts/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Increase display options for better readability in logs
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Paths to data
RATINGS_PATH = 'data/raw/ml-32m/ratings.csv'
MOVIES_PATH = 'data/raw/ml-32m/movies.csv'
TAGS_PATH = 'data/raw/ml-32m/tags.csv'
LINKS_PATH = 'data/raw/ml-32m/links.csv'

def load_data():
    print("Loading ratings data...")
    ratings = pd.read_csv(RATINGS_PATH)
    print("Loading movies data...")
    movies = pd.read_csv(MOVIES_PATH)
    print("Loading tags data...")
    tags = pd.read_csv(TAGS_PATH)
    print("Loading links data...")
    links = pd.read_csv(LINKS_PATH)
    return ratings, movies, tags, links

def basic_info(ratings, movies, tags, links):
    print("\n=== Ratings DataFrame ===")
    print(ratings.head())
    print("\n=== Movies DataFrame ===")
    print(movies.head())
    print("\n=== Tags DataFrame ===")
    print(tags.head())
    print("\n=== Links DataFrame ===")
    print(links.head())
    print("\n=== DataFrame Shapes ===")
    print(f"Ratings: {ratings.shape}")
    print(f"Movies: {movies.shape}")
    print(f"Tags: {tags.shape}")
    print(f"Links: {links.shape}")

def plot_rating_distribution(ratings):
    plt.figure(figsize=(8,6))
    sns.countplot(x='rating', data=ratings, palette='viridis')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('data/processed/rating_distribution.png')
    plt.close()
    print("Rating distribution plot saved at data/processed/rating_distribution.png")

def plot_top_genres(movies):
    plt.figure(figsize=(12,8))
    genres = movies['genres'].str.split('|', expand=True).stack().value_counts()
    sns.barplot(x=genres.values[:10], y=genres.index[:10], palette='magma')
    plt.title('Top 10 Genres')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.savefig('data/processed/top_genres.png')
    plt.close()
    print("Top genres plot saved at data/processed/top_genres.png")

def main():
    ratings, movies, tags, links = load_data()
    basic_info(ratings, movies, tags, links)
    plot_rating_distribution(ratings)
    plot_top_genres(movies)

if __name__ == "__main__":
    main()
