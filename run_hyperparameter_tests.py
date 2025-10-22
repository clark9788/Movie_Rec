"""
Main script for running comprehensive hyperparameter tests.
Supports resume capability, progress tracking, and graceful interruption.
"""

import argparse
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

from test_configs import get_all_test_configs, get_quick_test_configs
from hyperparameter_tuner import HyperparameterTuner
from hyperparameter_results import ResultsManager


def print_progress_bar(current: int, total: int, width: int = 50):
    """Print a progress bar."""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"


def estimate_remaining_time(completed: int, total: int, elapsed_time: float) -> str:
    """Estimate remaining time based on current progress."""
    if completed == 0:
        return "Unknown"
    
    avg_time_per_config = elapsed_time / completed
    remaining_configs = total - completed
    remaining_seconds = remaining_configs * avg_time_per_config
    
    if remaining_seconds < 60:
        return f"{remaining_seconds:.0f}s"
    elif remaining_seconds < 3600:
        return f"{remaining_seconds/60:.1f}m"
    else:
        return f"{remaining_seconds/3600:.1f}h"


def run_hyperparameter_tests(configs: List[Dict[str, Any]], resume: bool = False):
    """
    Run hyperparameter tests with progress tracking and resume support.
    
    Args:
        configs: List of configuration dictionaries to test
        resume: Whether to resume from existing results
    """
    print("Starting Hyperparameter Testing")
    print("=" * 60)
    
    # Initialize components
    results_manager = ResultsManager()
    tuner = HyperparameterTuner(results_manager)
    
    # Setup tuner
    tuner.setup()
    
    # Filter configurations based on resume mode
    if resume:
        untested_configs = [config for config in configs 
                           if not tuner.is_configuration_tested(config)]
        print(f"Resume mode: {len(untested_configs)}/{len(configs)} configurations remaining")
    else:
        untested_configs = configs
        print(f"Starting fresh: {len(configs)} configurations to test")
    
    if not untested_configs:
        print("[OK] All configurations already tested!")
        tuner.get_progress_summary()
        return
    
    # Show configuration summary
    print(f"\nConfiguration breakdown:")
    categories = {}
    for config in untested_configs:
        name = config['name']
        if name == 'baseline':
            categories['Baseline'] = categories.get('Baseline', 0) + 1
        elif name.startswith('lr_'):
            categories['Learning Rate'] = categories.get('Learning Rate', 0) + 1
        elif name.startswith('patience_'):
            categories['Patience'] = categories.get('Patience', 0) + 1
        elif name.startswith('emb_dim_'):
            categories['Embedding Dim'] = categories.get('Embedding Dim', 0) + 1
        elif name.startswith('hidden_'):
            categories['Hidden Layers'] = categories.get('Hidden Layers', 0) + 1
        elif name.startswith('batch_'):
            categories['Batch Size'] = categories.get('Batch Size', 0) + 1
        elif name.startswith('dropout_'):
            categories['Dropout Rate'] = categories.get('Dropout Rate', 0) + 1
        elif name.startswith('combined_'):
            categories['Combined'] = categories.get('Combined', 0) + 1
    
    for category, count in categories.items():
        print(f"  {category}: {count} configs")
    
    print(f"\nEstimated time: {len(untested_configs) * 5:.0f} minutes (assuming 5 min per config)")
    print("=" * 60)
    
    # Start testing
    start_time = time.time()
    completed = 0
    failed = 0
    
    try:
        for i, config in enumerate(untested_configs):
            config_start_time = time.time()
            
            # Print progress
            print(f"\n{print_progress_bar(i, len(untested_configs))}")
            print(f"Testing: {config['name']}")
            print(f"Config: {config}")
            
            # Test configuration
            experiment_id = tuner.test_configuration_with_logging(config)
            
            if experiment_id:
                completed += 1
                config_time = time.time() - config_start_time
                print(f"[OK] Completed in {config_time:.1f}s")
            else:
                failed += 1
                print(f"[FAIL] Failed")
            
            # Show running summary
            elapsed_time = time.time() - start_time
            remaining_time = estimate_remaining_time(completed, len(untested_configs), elapsed_time)
            
            print(f"\n📊 Running Summary:")
            print(f"  Completed: {completed}")
            print(f"  Failed: {failed}")
            print(f"  Remaining: {len(untested_configs) - i - 1}")
            print(f"  Elapsed: {elapsed_time/60:.1f}m")
            print(f"  ETA: {remaining_time}")
            
            # Show best result so far
            best_config = tuner.get_best_configuration()
            if best_config:
                best_metrics = results_manager.get_best_metrics()
                print(f"\nRunning Summary:")
                print(f"  Best RMSE so far: {best_metrics['rmse']:.4f}")
            
            print("-" * 60)
    
    except KeyboardInterrupt:
        print(f"\n\n[WARN] Testing interrupted by user")
        print(f"Progress saved. You can resume with --resume flag")
        
    except Exception as e:
        print(f"\n\n[WARN] Testing interrupted by user")
        print(f"\n\n[ERROR] Unexpected error: {e}")
        print(f"Progress saved. You can resume with --resume flag")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🏁 Testing Complete!")
    print(f"=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/(completed+failed)*100:.1f}%")
    
    # Show best results
    tuner.get_progress_summary()
    
    if completed > 0:
        print(f"\n🎯 Next steps:")
        print(f"1. Run: python analyze_results.py")
        print(f"2. Update optimal config: python update_optimal_config.py --select best_rmse")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run hyperparameter tests for MovieLens Recommender')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume testing from existing results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with reduced configurations')
    parser.add_argument('--configs', type=int, default=None,
                       help='Limit number of configurations to test')
    
    args = parser.parse_args()
    
    # Get configurations
    if args.quick:
        configs = get_quick_test_configs()
        print("Quick test mode - reduced configurations")
    else:
        configs = get_all_test_configs()
        print("Full test mode - all configurations")
    
    # Limit configurations if requested
    if args.configs:
        configs = configs[:args.configs]
        print(f"Limited to {len(configs)} configurations")
    
    # Run tests
    run_hyperparameter_tests(configs, resume=args.resume)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)
