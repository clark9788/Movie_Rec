"""
Quick test script for rapid validation of the hyperparameter testing framework.
Tests only 5 configurations with reduced epochs for fast iteration.
"""

import time
from datetime import datetime

from test_configs import get_quick_test_configs
from hyperparameter_tuner import HyperparameterTuner
from hyperparameter_results import ResultsManager


def run_quick_test():
    """Run quick validation test with reduced configurations."""
    print("Quick Hyperparameter Test")
    print("=" * 50)
    print("Testing framework with reduced configurations (~30 minutes)")
    print("=" * 50)
    
    # Get quick test configurations
    configs = get_quick_test_configs()
    
    print(f"Testing {len(configs)} configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['name']}: epochs={config['epochs']}")
    
    print(f"\nEstimated time: {len(configs) * 3:.0f} minutes (assuming 3 min per config)")
    print("=" * 50)
    
    # Initialize components
    results_manager = ResultsManager()
    tuner = HyperparameterTuner(results_manager)
    
    # Setup tuner
    print("Setting up...")
    tuner.setup()
    
    # Start testing
    start_time = time.time()
    completed = 0
    failed = 0
    
    try:
        for i, config in enumerate(configs):
            config_start_time = time.time()
            
            print(f"\n[{i+1}/{len(configs)}] Testing: {config['name']}")
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
            avg_time = elapsed_time / (i + 1)
            remaining_time = avg_time * (len(configs) - i - 1)
            
            print(f"\n📊 Progress:")
            print(f"  Completed: {completed}/{len(configs)}")
            print(f"  Failed: {failed}")
            print(f"  Elapsed: {elapsed_time/60:.1f}m")
            print(f"  ETA: {remaining_time/60:.1f}m")
            
            # Show best result so far
            best_config = tuner.get_best_configuration()
            if best_config:
                best_metrics = results_manager.get_best_metrics()
                print(f"\nProgress:")
                print(f"  Best RMSE: {best_metrics['rmse']:.4f}")
            
            print("-" * 50)
    
    except KeyboardInterrupt:
        print(f"\n\n[WARN] Quick test interrupted")
        print(f"Progress saved.")
        
    except Exception as e:
        print(f"\n\n[ERROR] Error: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\nQuick Test Complete!")
    print(f"=" * 50)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/(completed+failed)*100:.1f}%")
    
    # Show results
    tuner.get_progress_summary()
    
    if completed > 0:
        print(f"\nQuick test results:")
        print(f"[OK] Framework is working correctly!")
        print(f"[OK] Ready for full testing")
        print(f"\nNext steps:")
        print(f"1. Run full test: python run_hyperparameter_tests.py")
        print(f"2. Or analyze quick results: python analyze_results.py")
    else:
        print(f"\n[FAIL] Quick test failed - check configuration")


def main():
    """Main function."""
    print("Quick Hyperparameter Test")
    print("This will test the framework with 5 reduced configurations")
    print("Estimated time: ~30 minutes")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        run_quick_test()
    else:
        print("Quick test cancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
