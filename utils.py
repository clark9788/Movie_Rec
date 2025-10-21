"""
Utility functions for MovieLens recommendation system.
Handles data downloading, loading, and preprocessing.
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path


# Data paths
DATA_DIR = Path("data_cache/data")
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_FILE = DATA_DIR.parent / "ml-latest-small.zip"


def download_movielens_data():
    """
    Download and extract the MovieLens small dataset.
    """
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    if (DATA_DIR / "movies.csv").exists() and (DATA_DIR / "ratings.csv").exists():
        print("MovieLens dataset already downloaded.")
        return
    
    print("Downloading MovieLens small dataset...")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()
    
    # Save zip file
    with open(ZIP_FILE, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        # Extract to parent directory
        zip_ref.extractall(DATA_DIR.parent)
    
    # Move files from ml-latest-small to data directory
    extracted_dir = DATA_DIR.parent / "ml-latest-small"
    for file in extracted_dir.glob("*"):
        file.rename(DATA_DIR / file.name)
    
    # Clean up
    extracted_dir.rmdir()
    ZIP_FILE.unlink()
    
    print(f"Dataset downloaded and extracted to {DATA_DIR}")


def load_movies():
    """
    Load movies data from CSV file.
    
    Returns:
        pd.DataFrame: Movies dataframe with columns [movieId, title, genres]
    """
    movies_file = DATA_DIR / "movies.csv"
    if not movies_file.exists():
        raise FileNotFoundError(f"Movies file not found: {movies_file}")
    
    movies = pd.read_csv(movies_file)
    return movies


def load_ratings():
    """
    Load ratings data from CSV file.
    
    Returns:
        pd.DataFrame: Ratings dataframe with columns [userId, movieId, rating, timestamp]
    """
    ratings_file = DATA_DIR / "ratings.csv"
    if not ratings_file.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
    
    ratings = pd.read_csv(ratings_file)
    return ratings


def get_movie_by_id(movie_id, movies_df):
    """
    Get movie information by movie ID.
    
    Args:
        movie_id (int): Movie ID to lookup
        movies_df (pd.DataFrame): Movies dataframe
    
    Returns:
        pd.Series or None: Movie information if found, None otherwise
    """
    result = movies_df[movies_df['movieId'] == movie_id]
    if len(result) > 0:
        return result.iloc[0]
    return None


def format_genres(genres_str):
    """
    Format genres string for display.
    
    Args:
        genres_str (str): Genres string with pipe separators (e.g., "Action|Adventure|Sci-Fi")
    
    Returns:
        str: Formatted genres string
    """
    if pd.isna(genres_str) or genres_str == "(no genres listed)":
        return "N/A"
    return genres_str.replace("|", ", ")


def create_user_item_matrix(ratings_df):
    """
    Create a user-item matrix from ratings dataframe.
    
    Args:
        ratings_df (pd.DataFrame): Ratings dataframe
    
    Returns:
        tuple: (user_item_matrix, user_mapping, item_mapping)
    """
    # Create mappings
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}
    
    # Create matrix
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    
    user_item_matrix = np.zeros((n_users, n_movies))
    
    for _, row in ratings_df.iterrows():
        user_idx = user_to_idx[row['userId']]
        movie_idx = movie_to_idx[row['movieId']]
        user_item_matrix[user_idx, movie_idx] = row['rating']
    
    return user_item_matrix, user_to_idx, movie_to_idx, idx_to_user, idx_to_movie


def initialize_data():
    """
    Initialize the data by downloading (if needed) and loading the dataset.
    
    Returns:
        tuple: (movies_df, ratings_df)
    """
    download_movielens_data()
    movies_df = load_movies()
    ratings_df = load_ratings()
    
    print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings.")
    print(f"Number of users: {ratings_df['userId'].nunique()}")
    print(f"Rating scale: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
    
    return movies_df, ratings_df


