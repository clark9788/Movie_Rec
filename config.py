"""
Configuration file for MovieLens Recommender System.
Contains default and optimal hyperparameters.
"""

# Default configuration (baseline)
DEFAULT_CONFIG = {
    'learning_rate': 0.001,
    'patience': 5,
    'batch_size': 1024,
    'epochs': 50,
    'embedding_dim': 50,
    'hidden_dims': [128, 64, 32],
    'dropout_rate': 0.2
}

# This will be updated after hyperparameter testing
# Initially set to default, will be overwritten by update_optimal_config.py
OPTIMAL_CONFIG = DEFAULT_CONFIG.copy()

# For reference: explanation of each parameter
PARAMETER_DESCRIPTIONS = {
    'learning_rate': 'Step size for gradient descent optimization (0.0001-0.01)',
    'patience': 'Number of epochs to wait for improvement before early stopping (3-10)',
    'batch_size': 'Number of samples per training batch (512-2048)',
    'epochs': 'Maximum number of training epochs (20-100)',
    'embedding_dim': 'Dimension of user/movie embeddings (25-100)',
    'hidden_dims': 'List of hidden layer sizes in MLP (e.g., [128,64,32])',
    'dropout_rate': 'Dropout probability for regularization (0.1-0.3)'
}


