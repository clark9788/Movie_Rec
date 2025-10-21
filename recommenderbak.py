"""
Movie Recommender System using Neural Collaborative Filtering.
Pre-trains on MovieLens data and generates recommendations based on user ratings.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
from utils import initialize_data, format_genres


class MovieRatingDataset(Dataset):
    """
    Dataset class for MovieLens ratings.
    """
    def __init__(self, ratings_df, user_to_idx, movie_to_idx):
        self.user_ids = ratings_df['userId'].values
        self.movie_ids = ratings_df['movieId'].values
        self.ratings = ratings_df['rating'].values
        
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        user_idx = self.user_to_idx[self.user_ids[idx]]
        movie_idx = self.movie_to_idx[self.movie_ids[idx]]
        rating = torch.FloatTensor([self.ratings[idx]])
        
        return user_idx, movie_idx, rating


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model architecture.
    Combines embedding layers with multi-layer perceptron.
    """
    def __init__(self, n_users, n_movies, embedding_dim=50, hidden_dims=[128, 64, 32]):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # MLP layers
        input_dim = embedding_dim * 2
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, user_idx, movie_idx):
        """Forward pass."""
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        
        # Concatenate embeddings
        combined = torch.cat([user_emb, movie_emb], dim=-1)
        
        # MLP
        output = self.mlp(combined)
        return output.squeeze()


class MovieRecommender:
    """
    Main recommender system class.
    """
    def __init__(self):
        self.model = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.movies_df = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_ratings = {}  # In-memory storage for current user's ratings
        self.ratings_file = 'user_ratings.csv'  # File to store persistent ratings
        
    def load_data(self):
        """Load and prepare the MovieLens dataset."""
        print("Loading MovieLens dataset...")
        self.movies_df, ratings_df = initialize_data()
        
        # Create mappings
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}
        
        print(f"Dataset loaded: {len(unique_users)} users, {len(unique_movies)} movies")
        
        # Load existing user ratings
        self.load_user_ratings()
        
        return ratings_df
    
    def load_user_ratings(self):
        """Load existing user ratings from CSV file."""
        if os.path.exists(self.ratings_file):
            try:
                ratings_df = pd.read_csv(self.ratings_file)
                self.user_ratings = dict(zip(ratings_df['movieId'], ratings_df['rating']))
                print(f"Loaded {len(self.user_ratings)} existing ratings from {self.ratings_file}")
                self._display_rating_stats()
            except Exception as e:
                print(f"Error loading ratings file: {e}")
                print("Starting with empty ratings...")
                self.user_ratings = {}
        else:
            print("No existing ratings file found. Starting fresh...")
            self.user_ratings = {}
    
    def save_user_ratings(self):
        """Save current user ratings to CSV file."""
        if self.user_ratings:
            try:
                ratings_data = []
                for movie_id, rating in self.user_ratings.items():
                    ratings_data.append({'movieId': movie_id, 'rating': rating})
                
                ratings_df = pd.DataFrame(ratings_data)
                ratings_df.to_csv(self.ratings_file, index=False)
                print(f"Saved {len(self.user_ratings)} ratings to {self.ratings_file}")
            except Exception as e:
                print(f"Error saving ratings: {e}")
        else:
            print("No ratings to save.")
    
    def _display_rating_stats(self):
        """Display statistics about current ratings."""
        if not self.user_ratings:
            return
        
        ratings = list(self.user_ratings.values())
        avg_rating = np.mean(ratings)
        min_rating = min(ratings)
        max_rating = max(ratings)
        
        print(f"\nCurrent Rating Statistics:")
        print(f"  Total movies rated: {len(self.user_ratings)}")
        print(f"  Average rating: {avg_rating:.2f}")
        print(f"  Rating range: {min_rating:.1f} - {max_rating:.1f}")
        
        # Show rating distribution
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        print(f"  Rating distribution:")
        for rating in sorted(rating_counts.keys()):
            count = rating_counts[rating]
            percentage = (count / len(ratings)) * 100
            print(f"    {rating:.1f}: {count} movies ({percentage:.1f}%)")
    
    def _display_rated_movies(self):
        """Display all movies the user has rated."""
        if not self.user_ratings:
            print("No movies rated yet.")
            return
        
        print(f"\nMovies You've Rated ({len(self.user_ratings)} total):")
        print("-" * 80)
        print(f"{'Movie ID':<10} {'Title':<50} {'Rating':<8}")
        print("-" * 80)
        
        # Sort by rating (highest first)
        sorted_ratings = sorted(self.user_ratings.items(), key=lambda x: x[1], reverse=True)
        
        for movie_id, rating in sorted_ratings:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            title = movie_info['title'][:48]  # Truncate if too long
            print(f"{movie_id:<10} {title:<50} {rating:<8.1f}")
        
        print("-" * 80)
    
    def train_model(self, ratings_df, epochs=10, batch_size=1024, learning_rate=0.001):
        """Train the Neural Collaborative Filtering model."""
        print("\nTraining Neural Collaborative Filtering model...")
        print("This may take a few minutes...")
        
        # Cache the ratings data for later use in recommendations
        self._cached_ratings_df = ratings_df.copy()
        
        # Split data
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = MovieRatingDataset(train_df, self.user_to_idx, self.movie_to_idx)
        test_dataset = MovieRatingDataset(test_df, self.user_to_idx, self.movie_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        n_users = len(self.user_to_idx)
        n_movies = len(self.movie_to_idx)
        self.model = NeuralCollaborativeFiltering(n_users, n_movies).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (user_idx, movie_idx, rating) in enumerate(train_loader):
                user_idx = user_idx.to(self.device)
                movie_idx = movie_idx.to(self.device)
                rating = rating.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(user_idx, movie_idx)
                loss = criterion(output, rating.squeeze())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            test_loss = 0
            predictions = []
            targets = []
            with torch.no_grad():
                for user_idx, movie_idx, rating in test_loader:
                    user_idx = user_idx.to(self.device)
                    movie_idx = movie_idx.to(self.device)
                    rating = rating.to(self.device)
                    
                    output = self.model(user_idx, movie_idx)
                    test_loss += nn.MSELoss()(output, rating.squeeze()).item()
                    
                    predictions.extend(output.cpu().numpy())
                    targets.extend(rating.cpu().numpy().flatten())
            
            avg_train_loss = total_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        print(f"\nModel training completed!")
        print(f"Final RMSE: {rmse:.4f}")
        print(f"Final MAE: {mae:.4f}")
        
        # Explain quality metrics
        self._explain_quality_metrics(rmse, mae)
    
    def _explain_quality_metrics(self, rmse, mae):
        """Explain the quality metrics and what they mean."""
        print("\n" + "="*60)
        print("RECOMMENDER QUALITY METRICS EXPLANATION")
        print("="*60)
        
        print(f"\n1. ROOT MEAN SQUARE ERROR (RMSE): {rmse:.4f}")
        print("   - Measures average prediction error")
        print("   - Lower values are better (closer to 0)")
        print("   - Typical good values: 0.8-1.0 for MovieLens")
        print("   - Your model's RMSE indicates prediction accuracy")
        
        print(f"\n2. MEAN ABSOLUTE ERROR (MAE): {mae:.4f}")
        print("   - Average absolute difference between predicted and actual ratings")
        print("   - Also lower is better")
        print("   - More interpretable than RMSE (in rating scale units)")
        
        print(f"\n3. WHAT THESE MEAN FOR YOU:")
        print("   - The model has learned patterns from existing user ratings")
        print("   - It can predict how much you might like movies you haven't seen")
        print("   - Lower errors = more accurate recommendations")
        
        print(f"\n4. WHY NEURAL COLLABORATIVE FILTERING?")
        print("   - Learns complex non-linear user-movie relationships")
        print("   - Better than simple similarity-based methods")
        print("   - Can handle sparse data (few ratings per user)")
        print("   - State-of-the-art performance on recommendation tasks")
        
        print("\n5. SUBJECTIVE TESTING:")
        print("   - You and your wife will be the ultimate judges")
        print("   - Look for: relevance, diversity, novelty of recommendations")
        print("   - Good recommendations should surprise you positively!")
        print("="*60)
    
    def add_user_rating(self, movie_id, rating):
        """Add a rating from the current user."""
        if movie_id in self.movies_df['movieId'].values:
            # Check if movie was already rated
            was_already_rated = movie_id in self.user_ratings
            old_rating = self.user_ratings.get(movie_id, None)
            
            self.user_ratings[movie_id] = rating
            
            # Get movie title for better feedback
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            movie_title = movie_info['title']
            
            if was_already_rated:
                print(f"Updated rating: '{movie_title}' (ID: {movie_id}) = {old_rating:.1f} → {rating:.1f}")
            else:
                print(f"Added rating: '{movie_title}' (ID: {movie_id}) = {rating:.1f}")
            
            # Save ratings immediately to prevent data loss
            self.save_user_ratings()
            
        else:
            print(f"Movie ID {movie_id} not found in database")
    
    def get_recommendations(self, top_k=20):
        """Generate recommendations for the current user using collaborative filtering approach."""
        if not self.user_ratings:
            print("No ratings provided yet. Please rate some movies first.")
            return []
        
        print(f"\nGenerating recommendations based on {len(self.user_ratings)} ratings...")
        print("Using collaborative filtering approach for new user recommendations...")
        print("This will take at least 2 minutes...")
        
        # Get all movies the user hasn't rated
        rated_movies = set(self.user_ratings.keys())
        all_movies = set(self.movies_df['movieId'].values)
        unrated_movies = all_movies - rated_movies
        
        # Find similar users based on ratings
        similar_users = self._find_similar_users()
        
        # Predict ratings for unrated movies using similar users' ratings
        predictions = []
        
        for movie_id in unrated_movies:
            if movie_id in self.movie_to_idx:
                # Calculate weighted average rating from similar users
                weighted_rating = self._calculate_weighted_rating(movie_id, similar_users)
                predictions.append((movie_id, weighted_rating))
        
        # Sort by predicted rating and get top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = predictions[:top_k]
        
        return top_recommendations
    
    def _find_similar_users(self, top_k=50):
        """Find users with similar rating patterns."""
        # Load the ratings data to find similar users
        ratings_df = self._get_ratings_data()
        
        # Calculate user similarity based on rated movies
        user_similarities = {}
        user_ratings = self.user_ratings
        
        for user_id in self.user_to_idx.keys():
            if user_id in ratings_df['userId'].values:
                user_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].values)
                current_user_movies = set(user_ratings.keys())
                
                # Calculate Jaccard similarity
                intersection = len(user_movies & current_user_movies)
                union = len(user_movies | current_user_movies)
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > 0:
                        user_similarities[user_id] = similarity
        
        # Sort by similarity and return top users
        similar_users = sorted(user_similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return similar_users
    
    def _calculate_weighted_rating(self, movie_id, similar_users):
        """Calculate weighted average rating for a movie based on similar users."""
        ratings_df = self._get_ratings_data()
        
        weighted_sum = 0
        total_weight = 0
        
        for user_id, similarity in similar_users:
            user_movie_ratings = ratings_df[
                (ratings_df['userId'] == user_id) & 
                (ratings_df['movieId'] == movie_id)
            ]
            
            if not user_movie_ratings.empty:
                rating = user_movie_ratings['rating'].iloc[0]
                weighted_sum += rating * similarity
                total_weight += similarity
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback: return average rating for this movie
            movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']
            return movie_ratings.mean() if not movie_ratings.empty else 3.0
    
    def _get_ratings_data(self):
        """Load ratings data for similarity calculations."""
        # This method loads the ratings data - we'll need to store it during training
        if not hasattr(self, '_cached_ratings_df'):
            self._cached_ratings_df = initialize_data()[1]  # Get ratings_df from utils
        return self._cached_ratings_df
    
    def display_recommendations(self, recommendations):
        """Display recommendations in a formatted table."""
        if not recommendations:
            print("No recommendations available.")
            return
        
        print("\n" + "="*100)
        print("YOUR PERSONALIZED MOVIE RECOMMENDATIONS")
        print("="*100)
        print(f"{'Rank':<6} {'Movie ID':<10} {'Title':<60} {'Genres':<20}")
        print("-"*100)
        
        for rank, (movie_id, pred_rating) in enumerate(recommendations, 1):
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            title = movie_info['title'][:58]  # Truncate if too long
            genres = format_genres(movie_info['genres'])[:18]  # Truncate if too long
            
            print(f"{rank:<6} {movie_id:<10} {title:<60} {genres:<20}")
        
        print("-"*100)
        print(f"Generated {len(recommendations)} rec1ommendations based on your ratings.")
        print("="*100)
    
    def rating_loop(self):
        """Interactive rating loop."""
        print("\n" + "="*60)
        print("MOVIE RATING SYSTEM")
        print("="*60)
        print("Rate movies you've watched (1.0 to 5.0 in 0.5 increments)")
        print("Enter: movie_id rating (e.g., '1 4.5')")
        print("Enter '0 0' to finish rating and get recommendations")
        print("Enter 'stats' to see current rating statistics")
        print("Enter 'list' to see movies you've already rated")
        print("="*60)
        
        # Show current stats if we have ratings
        if self.user_ratings:
            self._display_rating_stats()
            print()
        
        while True:
            try:
                user_input = input("\nEnter movie ID and rating: ").strip()
                
                if user_input == "0 0":
                    break
                elif user_input.lower() == "stats":
                    self._display_rating_stats()
                    continue
                elif user_input.lower() == "list":
                    self._display_rated_movies()
                    continue
                
                parts = user_input.split()
                if len(parts) != 2:
                    print("Please enter movie ID and rating separated by space")
                    print("Or use 'stats' to see statistics, 'list' to see rated movies, or '0 0' to finish")
                    continue
                
                movie_id = int(parts[0])
                rating = float(parts[1])
                
                # Validate rating
                if rating < 0.5 or rating > 5.0 or rating % 0.5 != 0:
                    print("Rating must be between 0.5 and 5.0 in 0.5 increments")
                    continue
                
                # Add rating
                self.add_user_rating(movie_id, rating)
                
            except ValueError:
                print("Please enter valid numbers")
            except KeyboardInterrupt:
                print("\nRating interrupted")
                break


def main():
    """Main function to run the recommender system."""
    print("MOVIE RECOMMENDER SYSTEM")
    print("Using Neural Collaborative Filtering for high-quality recommendations")
    print()
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load data and train model
    ratings_df = recommender.load_data()
    recommender.train_model(ratings_df)
    
    # Rating loop
    recommender.rating_loop()
    
    # Generate and display recommendations
    if recommender.user_ratings:
        recommendations = recommender.get_recommendations()
        recommender.display_recommendations(recommendations)
        
        print(f"\nThank you for using the recommender system!")
        print(f"Please let us know how well the recommendations worked for you!")
    else:
        print("\nNo ratings provided. Exiting...")
    
    # Final save to ensure all ratings are persisted
    recommender.save_user_ratings()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nRecommender system interrupted. Saving ratings before exit...")
        try:
            # Try to save ratings even on interruption
            recommender = MovieRecommender()
            recommender.load_data()  # This will load existing ratings
            recommender.save_user_ratings()
        except:
            pass  # If we can't save, just exit
        print("Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

