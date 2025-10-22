"""
Results tracking system for hyperparameter testing.
Manages experiment logging, persistence, and resume capability.
"""

import json
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class ResultsManager:
    """
    Manages hyperparameter experiment results with persistence and resume capability.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.json_file = self.results_dir / "experiment_results.json"
        self.csv_file = self.results_dir / "experiment_results.csv"
        self.learning_curves_dir = self.results_dir / "learning_curves"
        self.learning_curves_dir.mkdir(exist_ok=True)
        
        self.results = self._load_existing_results()
        self.best_config = None
        self.best_rmse = float('inf')
    
    def _load_existing_results(self) -> List[Dict]:
        """Load existing results from JSON file."""
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r') as f:
                    results = json.load(f)
                print(f"Loaded {len(results)} existing experiment results")
                return results
            except Exception as e:
                print(f"Error loading existing results: {e}")
                return []
        return []
    
    def _save_results(self):
        """Save results to JSON file."""
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Also save as CSV for easy analysis
            self._save_csv()
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _save_csv(self):
        """Save results as CSV for spreadsheet analysis."""
        if not self.results:
            return
        
        try:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                header = ['experiment_id', 'timestamp', 'rmse', 'mae', 'train_time_seconds']
                # Add config parameters as columns
                config_keys = list(self.results[0]['config'].keys())
                header.extend(config_keys)
                writer.writerow(header)
                
                # Write data rows
                for result in self.results:
                    row = [
                        result['experiment_id'],
                        result['timestamp'],
                        result['metrics']['rmse'],
                        result['metrics']['mae'],
                        result['metrics']['train_time']
                    ]
                    # Add config values
                    for key in config_keys:
                        value = result['config'][key]
                        if isinstance(value, list):
                            value = str(value)  # Convert list to string
                        row.append(value)
                    writer.writerow(row)
                    
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def _generate_experiment_id(self, config: Dict) -> str:
        """Generate a unique experiment ID from configuration."""
        # Create a readable ID from key parameters
        lr = config.get('learning_rate', 0.001)
        patience = config.get('patience', 5)
        emb_dim = config.get('embedding_dim', 50)
        batch_size = config.get('batch_size', 1024)
        
        return f"lr_{lr}_pat_{patience}_emb_{emb_dim}_batch_{batch_size}"
    
    def log_experiment(self, config: Dict, metrics: Dict, learning_curves: Dict) -> str:
        """
        Log a completed experiment.
        
        Args:
            config: Configuration dictionary
            metrics: Dictionary with 'rmse', 'mae', 'train_time'
            learning_curves: Dictionary with 'train' and 'val' loss lists
            
        Returns:
            experiment_id: Unique identifier for this experiment
        """
        experiment_id = self._generate_experiment_id(config)
        
        # Check if this experiment already exists
        existing_ids = [r['experiment_id'] for r in self.results]
        if experiment_id in existing_ids:
            print(f"Experiment {experiment_id} already exists, skipping...")
            return experiment_id
        
        # Create result entry
        result = {
            'experiment_id': experiment_id,
            'config': config.copy(),
            'metrics': metrics.copy(),
            'learning_curves': learning_curves.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to results
        self.results.append(result)
        
        # Update best configuration
        if metrics['rmse'] < self.best_rmse:
            self.best_rmse = metrics['rmse']
            self.best_config = config.copy()
            print(f"*** New best RMSE: {self.best_rmse:.4f}")
        
        # Save results
        self._save_results()
        
        # Save learning curves plot
        self._save_learning_curves(experiment_id, learning_curves)
        
        print(f"[OK] Logged experiment: {experiment_id}")
        return experiment_id
    
    def _save_learning_curves(self, experiment_id: str, learning_curves: Dict):
        """Save learning curves plot to file."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            epochs = range(1, len(learning_curves['train']) + 1)
            plt.plot(epochs, learning_curves['train'], 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, learning_curves['val'], 'r-', label='Validation Loss', linewidth=2)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Learning Curves - {experiment_id}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Find optimal epoch
            optimal_epoch = min(range(len(learning_curves['val'])), 
                              key=lambda i: learning_curves['val'][i]) + 1
            plt.axvline(x=optimal_epoch, color='green', linestyle='--', alpha=0.7,
                       label=f'Optimal Epoch: {optimal_epoch}')
            plt.legend()
            
            filename = self.learning_curves_dir / f"{experiment_id}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            
        except ImportError:
            print("Matplotlib not available, skipping learning curve plots")
        except Exception as e:
            print(f"Error saving learning curves: {e}")
    
    def get_experiment_ids(self) -> List[str]:
        """Get list of all completed experiment IDs."""
        return [r['experiment_id'] for r in self.results]
    
    def is_experiment_completed(self, config: Dict) -> bool:
        """Check if an experiment with this config is already completed."""
        experiment_id = self._generate_experiment_id(config)
        return experiment_id in self.get_experiment_ids()
    
    def get_best_config(self) -> Optional[Dict]:
        """Get the configuration with the best RMSE."""
        if not self.results:
            return None
        
        best_result = min(self.results, key=lambda r: r['metrics']['rmse'])
        return best_result['config']
    
    def get_best_metrics(self) -> Optional[Dict]:
        """Get the metrics for the best configuration."""
        if not self.results:
            return None
        
        best_result = min(self.results, key=lambda r: r['metrics']['rmse'])
        return best_result['metrics']
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of all experiments."""
        if not self.results:
            return {}
        
        rmse_values = [r['metrics']['rmse'] for r in self.results]
        mae_values = [r['metrics']['mae'] for r in self.results]
        train_times = [r['metrics']['train_time'] for r in self.results]
        
        return {
            'total_experiments': len(self.results),
            'best_rmse': min(rmse_values),
            'worst_rmse': max(rmse_values),
            'avg_rmse': sum(rmse_values) / len(rmse_values),
            'best_mae': min(mae_values),
            'avg_mae': sum(mae_values) / len(mae_values),
            'total_train_time': sum(train_times),
            'avg_train_time': sum(train_times) / len(train_times)
        }
    
    def print_progress_summary(self):
        """Print a summary of current progress."""
        if not self.results:
            print("No experiments completed yet.")
            return
        
        stats = self.get_summary_stats()
        print(f"\nProgress Summary:")
        print(f"  Completed experiments: {stats['total_experiments']}")
        print(f"  Best RMSE so far: {stats['best_rmse']:.4f}")
        print(f"  Average RMSE: {stats['avg_rmse']:.4f}")
        print(f"  Total training time: {stats['total_train_time']:.1f} seconds")
        
        if self.best_config:
            print(f"  Best config: {self.best_config}")


