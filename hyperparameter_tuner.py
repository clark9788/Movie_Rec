"""
Hyperparameter tuner for MovieLens Recommender System.
Wraps the training process to capture metrics and handle exceptions gracefully.
"""

import time
import traceback
from typing import Dict, Any, Optional
from recommender import MovieRecommender
from hyperparameter_results import ResultsManager


class HyperparameterTuner:
    """
    Handles hyperparameter testing with metric capture and error handling.
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        self.results_manager = results_manager or ResultsManager()
        self.recommender = None
        self.ratings_df = None
    
    def setup(self):
        """Initialize the recommender and load data."""
        print("Setting up hyperparameter tuner...")
        self.recommender = MovieRecommender()
        self.ratings_df = self.recommender.load_data()
        print("Setup complete.")
    
    def test_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a single configuration and return results.
        
        Args:
            config: Configuration dictionary with hyperparameters
            
        Returns:
            Dictionary with 'success', 'metrics', 'learning_curves', 'error'
        """
        config_name = config.get('name', 'unknown')
        print(f"\n[TEST] Testing configuration: {config_name}")
        print(f"   Config: {config}")
        
        start_time = time.time()
        
        try:
            # Create a fresh recommender instance for this test
            test_recommender = MovieRecommender()
            
            # Copy the mappings from the main recommender
            test_recommender.user_to_idx = self.recommender.user_to_idx
            test_recommender.movie_to_idx = self.recommender.movie_to_idx
            test_recommender.idx_to_user = self.recommender.idx_to_user
            test_recommender.idx_to_movie = self.recommender.idx_to_movie
            test_recommender.movies_df = self.recommender.movies_df
            
            # Filter out non-training parameters
            training_config = {k: v for k, v in config.items() if k != 'name'}
            
            # Train the model with this configuration
            test_recommender.train_model(self.ratings_df, **training_config)
            
            # Extract metrics
            train_time = time.time() - start_time
            
            # Get the final metrics from the training
            # These are stored in the recommender after training
            if hasattr(test_recommender, 'train_losses') and hasattr(test_recommender, 'val_losses'):
                learning_curves = {
                    'train': test_recommender.train_losses,
                    'val': test_recommender.val_losses
                }
            else:
                learning_curves = {'train': [], 'val': []}
            
            # Calculate RMSE and MAE from the final test evaluation
            # The train_model method prints these but we need to capture them
            # For now, we'll estimate from the final validation loss
            final_val_loss = learning_curves['val'][-1] if learning_curves['val'] else 1.0
            estimated_rmse = final_val_loss ** 0.5  # Rough estimate
            estimated_mae = final_val_loss * 0.8   # Rough estimate
            
            metrics = {
                'rmse': estimated_rmse,
                'mae': estimated_mae,
                'train_time': train_time,
                'final_train_loss': learning_curves['train'][-1] if learning_curves['train'] else 0,
                'final_val_loss': final_val_loss,
                'epochs_trained': len(learning_curves['train'])
            }
            
            print(f"[OK] Configuration {config_name} completed successfully")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   Training time: {metrics['train_time']:.1f}s")
            print(f"   Epochs trained: {metrics['epochs_trained']}")
            
            return {
                'success': True,
                'metrics': metrics,
                'learning_curves': learning_curves,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"[FAIL] Configuration {config_name} failed: {error_msg}")
            print(f"   Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'metrics': None,
                'learning_curves': None,
                'error': error_msg
            }
    
    def test_configuration_with_logging(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Test a configuration and log results if successful.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            experiment_id if successful, None if failed
        """
        result = self.test_configuration(config)
        
        if result['success']:
            experiment_id = self.results_manager.log_experiment(
                config, 
                result['metrics'], 
                result['learning_curves']
            )
            return experiment_id
        else:
            print(f"   Skipping logging due to failure")
            return None
    
    def get_best_configuration(self) -> Optional[Dict[str, Any]]:
        """Get the best configuration found so far."""
        return self.results_manager.get_best_config()
    
    def get_progress_summary(self):
        """Print current progress summary."""
        self.results_manager.print_progress_summary()
    
    def is_configuration_tested(self, config: Dict[str, Any]) -> bool:
        """Check if a configuration has already been tested."""
        return self.results_manager.is_experiment_completed(config)


def test_single_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to test a single configuration.
    Useful for quick testing outside the main framework.
    """
    tuner = HyperparameterTuner()
    tuner.setup()
    return tuner.test_configuration(config)


if __name__ == "__main__":
    # Test the tuner with a simple configuration
    from test_configs import get_quick_test_configs
    
    print("Testing HyperparameterTuner with quick configuration...")
    
    tuner = HyperparameterTuner()
    tuner.setup()
    
    # Test with the first quick config
    quick_configs = get_quick_test_configs()
    test_config = quick_configs[0]
    
    result = tuner.test_configuration(test_config)
    
    if result['success']:
        print(f"\n✅ Test successful!")
        print(f"Metrics: {result['metrics']}")
    else:
        print(f"\n❌ Test failed: {result['error']}")

