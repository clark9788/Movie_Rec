"""
Script to update config.py with the optimal configuration from hyperparameter testing.
Supports different selection criteria and backup functionality.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from hyperparameter_results import ResultsManager


def backup_config_file():
    """Create a backup of the current config.py file."""
    config_file = Path("config.py")
    if config_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Path(f"config_backup_{timestamp}.py")
        shutil.copy2(config_file, backup_file)
        print(f"Backup created: {backup_file}")
        return backup_file
    return None


def update_config_file(optimal_config: Dict[str, Any]):
    """Update config.py with the optimal configuration."""
    config_file = Path("config.py")
    
    if not config_file.exists():
        print(f"[ERROR] config.py not found!")
        return False
    
    # Read current config file
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Find and replace OPTIMAL_CONFIG
    lines = content.split('\n')
    new_lines = []
    in_optimal_config = False
    indent_level = 0
    
    for line in lines:
        if 'OPTIMAL_CONFIG = {' in line:
            in_optimal_config = True
            indent_level = len(line) - len(line.lstrip())
            new_lines.append(line)
            continue
        
        if in_optimal_config:
            if line.strip() == '}':
                # End of OPTIMAL_CONFIG, add new values
                for key, value in optimal_config.items():
                    if key != 'name':  # Skip name field
                        indent = ' ' * (indent_level + 4)
                        new_lines.append(f"{indent}'{key}': {repr(value)},")
                new_lines.append(' ' * indent_level + '}')
                in_optimal_config = False
                continue
            else:
                # Skip old values
                continue
        
        new_lines.append(line)
    
    # Write updated content
    new_content = '\n'.join(new_lines)
    with open(config_file, 'w') as f:
        f.write(new_content)
    
    print(f"[OK] Updated config.py with optimal configuration")
    return True


def show_config_comparison(old_config: Dict[str, Any], new_config: Dict[str, Any]):
    """Show a comparison between old and new configurations."""
    print(f"\nConfiguration Comparison:")
    print(f"=" * 50)
    
    all_keys = set(old_config.keys()) | set(new_config.keys())
    
    for key in sorted(all_keys):
        old_val = old_config.get(key, "N/A")
        new_val = new_config.get(key, "N/A")
        
        if old_val != new_val:
            print(f"  {key}: {old_val} → {new_val}")
        else:
            print(f"  {key}: {old_val} (unchanged)")


def select_best_config(results_manager: ResultsManager, selection_criteria: str) -> Optional[Dict[str, Any]]:
    """Select the best configuration based on criteria."""
    if not results_manager.results:
        print("[ERROR] No results found!")
        return None
    
    if selection_criteria == "best_rmse":
        best_result = min(results_manager.results, key=lambda r: r['metrics']['rmse'])
        print(f"[WINNER] Selected configuration with best RMSE: {best_result['metrics']['rmse']:.4f}")
        return best_result['config']
    
    elif selection_criteria == "best_balanced":
        # Find configuration with good RMSE and reasonable training time
        results = results_manager.results
        rmse_values = [r['metrics']['rmse'] for r in results]
        time_values = [r['metrics']['train_time'] for r in results]
        
        # Normalize scores (lower is better)
        rmse_scores = [(rmse - min(rmse_values)) / (max(rmse_values) - min(rmse_values)) for rmse in rmse_values]
        time_scores = [(t - min(time_values)) / (max(time_values) - min(time_values)) for t in time_values]
        
        # Combined score (equal weight)
        combined_scores = [rmse_scores[i] + time_scores[i] for i in range(len(results))]
        best_idx = min(range(len(combined_scores)), key=lambda i: combined_scores[i])
        
        best_result = results[best_idx]
        print(f"[WINNER] Selected balanced configuration:")
        print(f"   RMSE: {best_result['metrics']['rmse']:.4f}")
        print(f"   Training time: {best_result['metrics']['train_time']:.1f}s")
        return best_result['config']
    
    elif selection_criteria == "fastest":
        best_result = min(results_manager.results, key=lambda r: r['metrics']['train_time'])
        print(f"[WINNER] Selected fastest configuration: {best_result['metrics']['train_time']:.1f}s")
        return best_result['config']
    
    else:
        print(f"[ERROR] Unknown selection criteria: {selection_criteria}")
        return None


def find_config_by_experiment_id(results_manager: ResultsManager, experiment_id: str) -> Optional[Dict[str, Any]]:
    """Find configuration by experiment ID."""
    for result in results_manager.results:
        if result['experiment_id'] == experiment_id:
            print(f"[WINNER] Found configuration: {experiment_id}")
            print(f"   RMSE: {result['metrics']['rmse']:.4f}")
            print(f"   Training time: {result['metrics']['train_time']:.1f}s")
            return result['config']
    
    print(f"[ERROR] Experiment ID not found: {experiment_id}")
    return None


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Update config.py with optimal hyperparameters')
    parser.add_argument('--select', choices=['best_rmse', 'best_balanced', 'fastest'], 
                       default='best_rmse',
                       help='Selection criteria for best configuration')
    parser.add_argument('--experiment-id', type=str,
                       help='Specific experiment ID to use')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup of current config')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without updating')
    
    args = parser.parse_args()
    
    print("Updating Optimal Configuration")
    print("=" * 40)
    
    # Initialize results manager
    results_manager = ResultsManager()
    
    if not results_manager.results:
        print("[ERROR] No hyperparameter testing results found!")
        print("   Run hyperparameter tests first: python run_hyperparameter_tests.py")
        return
    
    # Select configuration
    if args.experiment_id:
        optimal_config = find_config_by_experiment_id(results_manager, args.experiment_id)
    else:
        optimal_config = select_best_config(results_manager, args.select)
    
    if not optimal_config:
        print("❌ Could not select optimal configuration")
        return
    
    # Show current optimal config
    try:
        from config import OPTIMAL_CONFIG
        current_config = OPTIMAL_CONFIG.copy()
    except ImportError:
        print("[ERROR] Could not import current OPTIMAL_CONFIG")
        current_config = {}
    
    # Show comparison
    show_config_comparison(current_config, optimal_config)
    
    if args.dry_run:
        print(f"\n[DRY] Dry run - no changes made")
        return
    
    # Confirm update
        print(f"\n[WARN] This will update config.py with the selected configuration")
    response = input("Proceed? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Update cancelled.")
        return
    
    # Create backup
    if not args.no_backup:
        backup_file = backup_config_file()
    
    # Update config file
    if update_config_file(optimal_config):
        print(f"\n[OK] Configuration updated successfully!")
        print(f"New optimal configuration:")
        for key, value in optimal_config.items():
            if key != 'name':
                print(f"   {key}: {value}")
        
        print(f"\nNext steps:")
        print(f"1. Test the new configuration: python recommender.py")
        print(f"2. Or use in code: recommender.train_with_config(ratings_df)")
    else:
        print(f"[FAIL] Failed to update configuration")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUpdate cancelled")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
