# src/data_loading.py

import pandas as pd

def load_ratings(path='data/raw/ml-32m/ratings.csv'):
    print(f"Loading ratings data from {path}...")
    ratings = pd.read_csv(path)
    return ratings

def load_movies(path='data/raw/ml-32m/movies.csv'):
    print(f"Loading movies data from {path}...")
    movies = pd.read_csv(path)
    return movies

def load_tags(path='data/raw/ml-32m/tags.csv'):
    print(f"Loading tags data from {path}...")
    tags = pd.read_csv(path)
    return tags

def load_links(path='data/raw/ml-32m/links.csv'):
    print(f"Loading links data from {path}...")
    links = pd.read_csv(path)
    return links

def load_imdb_basics(path='data/raw/imdb/title.basics.tsv.gz'):
    print(f"Loading IMDb basics data from {path}...")
    basics = pd.read_csv(path, sep='\t', na_values='\\N', low_memory=False)
    return basics

def load_imdb_ratings(path='data/raw/imdb/title.ratings.tsv.gz'):
    print(f"Loading IMDb ratings data from {path}...")
    ratings = pd.read_csv(path, sep='\t', na_values='\\N', low_memory=False)
    return ratings
