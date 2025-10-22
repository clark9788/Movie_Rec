"""
Analysis and reporting module for hyperparameter testing results.
Generates comprehensive reports with visualizations and recommendations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from hyperparameter_results import ResultsManager


class HyperparameterAnalyzer:
    """
    Analyzes hyperparameter testing results and generates reports.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_manager = ResultsManager(results_dir)
        self.results = self.results_manager.results
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        if not self.results:
            return {"error": "No results found"}
        
        stats = self.results_manager.get_summary_stats()
        
        # Add additional analysis
        rmse_values = [r['metrics']['rmse'] for r in self.results]
        mae_values = [r['metrics']['mae'] for r in self.results]
        train_times = [r['metrics']['train_time'] for r in self.results]
        
        stats.update({
            'rmse_std': np.std(rmse_values),
            'mae_std': np.std(mae_values),
            'train_time_std': np.std(train_times),
            'rmse_range': max(rmse_values) - min(rmse_values),
            'mae_range': max(mae_values) - min(mae_values),
            'time_range': max(train_times) - min(train_times)
        })
        
        return stats
    
    def get_best_configurations(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top K configurations by RMSE."""
        if not self.results:
            return []
        
        sorted_results = sorted(self.results, key=lambda r: r['metrics']['rmse'])
        return sorted_results[:top_k]
    
    def get_fastest_configurations(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top K configurations by training speed."""
        if not self.results:
            return []
        
        sorted_results = sorted(self.results, key=lambda r: r['metrics']['train_time'])
        return sorted_results[:top_k]
    
    def analyze_parameter_impact(self) -> Dict[str, Dict[str, Any]]:
        """Analyze the impact of each parameter on performance."""
        if not self.results:
            return {}
        
        parameter_analysis = {}
        
        # Parameters to analyze
        parameters = ['learning_rate', 'patience', 'batch_size', 'embedding_dim', 'dropout_rate']
        
        for param in parameters:
            if param == 'hidden_dims':
                continue  # Skip complex parameter for now
            
            param_values = []
            rmse_values = []
            
            for result in self.results:
                if param in result['config']:
                    param_values.append(result['config'][param])
                    rmse_values.append(result['metrics']['rmse'])
            
            if param_values:
                # Group by parameter value and calculate statistics
                param_df = pd.DataFrame({param: param_values, 'rmse': rmse_values})
                grouped = param_df.groupby(param)['rmse'].agg(['mean', 'std', 'count']).reset_index()
                
                parameter_analysis[param] = {
                    'values': grouped[param].tolist(),
                    'mean_rmse': grouped['mean'].tolist(),
                    'std_rmse': grouped['std'].tolist(),
                    'count': grouped['count'].tolist(),
                    'best_value': grouped.loc[grouped['mean'].idxmin(), param],
                    'best_rmse': grouped['mean'].min()
                }
        
        return parameter_analysis
    
    def generate_text_report(self) -> str:
        """Generate a comprehensive text report."""
        if not self.results:
            return "No results found for analysis."
        
        report = []
        report.append("# Hyperparameter Testing Results Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Summary statistics
        stats = self.get_summary_stats()
        report.append(f"\n## Summary Statistics")
        report.append(f"Total experiments: {stats['total_experiments']}")
        report.append(f"Best RMSE: {stats['best_rmse']:.4f}")
        report.append(f"Worst RMSE: {stats['worst_rmse']:.4f}")
        report.append(f"Average RMSE: {stats['avg_rmse']:.4f} ± {stats['rmse_std']:.4f}")
        report.append(f"RMSE range: {stats['rmse_range']:.4f}")
        report.append(f"")
        report.append(f"Best MAE: {stats['best_mae']:.4f}")
        report.append(f"Average MAE: {stats['avg_mae']:.4f} ± {stats['mae_std']:.4f}")
        report.append(f"")
        report.append(f"Total training time: {stats['total_train_time']:.1f} seconds ({stats['total_train_time']/60:.1f} minutes)")
        report.append(f"Average training time: {stats['avg_train_time']:.1f} seconds")
        
        # Best configurations
        best_configs = self.get_best_configurations(5)
        report.append(f"\n## Top 5 Configurations by RMSE")
        for i, config_result in enumerate(best_configs, 1):
            config = config_result['config']
            metrics = config_result['metrics']
            report.append(f"\n{i}. **{config_result['experiment_id']}**")
            report.append(f"   RMSE: {metrics['rmse']:.4f}")
            report.append(f"   MAE: {metrics['mae']:.4f}")
            report.append(f"   Training time: {metrics['train_time']:.1f}s")
            report.append(f"   Config: {config}")
        
        # Fastest configurations
        fastest_configs = self.get_fastest_configurations(3)
        report.append(f"\n## Top 3 Fastest Configurations")
        for i, config_result in enumerate(fastest_configs, 1):
            config = config_result['config']
            metrics = config_result['metrics']
            report.append(f"\n{i}. **{config_result['experiment_id']}**")
            report.append(f"   Training time: {metrics['train_time']:.1f}s")
            report.append(f"   RMSE: {metrics['rmse']:.4f}")
            report.append(f"   Config: {config}")
        
        # Parameter impact analysis
        param_analysis = self.analyze_parameter_impact()
        report.append(f"\n## Parameter Impact Analysis")
        for param, analysis in param_analysis.items():
            report.append(f"\n### {param.replace('_', ' ').title()}")
            report.append(f"Best value: {analysis['best_value']}")
            report.append(f"Best RMSE: {analysis['best_rmse']:.4f}")
            
            # Show all values tested
            for val, mean_rmse, std_rmse, count in zip(analysis['values'], analysis['mean_rmse'], analysis['std_rmse'], analysis['count']):
                report.append(f"  {val}: RMSE {mean_rmse:.4f} ± {std_rmse:.4f} (n={count})")
        
        # Recommendations
        report.append(f"\n## Recommendations")
        best_config = best_configs[0]['config'] if best_configs else None
        if best_config:
            report.append(f"**Best Overall Configuration:**")
            report.append(f"```python")
            report.append(f"OPTIMAL_CONFIG = {{")
            for key, value in best_config.items():
                if key != 'name':
                    report.append(f"    '{key}': {value},")
            report.append(f"}}")
            report.append(f"```")
            
            report.append(f"\n**Performance:**")
            best_metrics = best_configs[0]['metrics']
            report.append(f"- RMSE: {best_metrics['rmse']:.4f}")
            report.append(f"- MAE: {best_metrics['mae']:.4f}")
            report.append(f"- Training time: {best_metrics['train_time']:.1f}s")
        
        return "\n".join(report)
    
    def create_visualizations(self, save_plots: bool = True):
        """Create visualization plots."""
        if not self.results:
            print("No results found for visualization.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Hyperparameter Testing Results Analysis', fontsize=16)
            
            # 1. RMSE comparison bar chart
            ax1 = axes[0, 0]
            experiment_ids = [r['experiment_id'] for r in self.results]
            rmse_values = [r['metrics']['rmse'] for r in self.results]
            
            # Truncate long experiment IDs for display
            display_ids = [eid[:20] + '...' if len(eid) > 20 else eid for eid in experiment_ids]
            
            bars = ax1.bar(range(len(rmse_values)), rmse_values)
            ax1.set_xlabel('Experiment ID')
            ax1.set_ylabel('RMSE')
            ax1.set_title('RMSE by Configuration')
            ax1.set_xticks(range(len(display_ids)))
            ax1.set_xticklabels(display_ids, rotation=45, ha='right')
            
            # Highlight best result
            best_idx = np.argmin(rmse_values)
            bars[best_idx].set_color('red')
            bars[best_idx].set_alpha(0.7)
            
            # 2. Training time vs RMSE scatter plot
            ax2 = axes[0, 1]
            train_times = [r['metrics']['train_time'] for r in self.results]
            ax2.scatter(train_times, rmse_values, alpha=0.7)
            ax2.set_xlabel('Training Time (seconds)')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Training Time vs RMSE')
            
            # Add trend line
            z = np.polyfit(train_times, rmse_values, 1)
            p = np.poly1d(z)
            ax2.plot(train_times, p(train_times), "r--", alpha=0.8)
            
            # 3. Parameter impact analysis
            ax3 = axes[1, 0]
            param_analysis = self.analyze_parameter_impact()
            
            if param_analysis:
                # Plot learning rate impact
                if 'learning_rate' in param_analysis:
                    lr_data = param_analysis['learning_rate']
                    ax3.errorbar(lr_data['values'], lr_data['mean_rmse'], 
                               yerr=lr_data['std_rmse'], marker='o', capsize=5)
                    ax3.set_xlabel('Learning Rate')
                    ax3.set_ylabel('RMSE')
                    ax3.set_title('Learning Rate Impact')
                    ax3.set_xscale('log')
                else:
                    ax3.text(0.5, 0.5, 'No parameter analysis available', 
                           ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Parameter Impact')
            else:
                ax3.text(0.5, 0.5, 'No parameter analysis available', 
                       ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Parameter Impact')
            
            # 4. Learning curves overlay (top 5)
            ax4 = axes[1, 1]
            best_configs = self.get_best_configurations(5)
            
            for i, config_result in enumerate(best_configs):
                if 'learning_curves' in config_result and config_result['learning_curves']['val']:
                    val_losses = config_result['learning_curves']['val']
                    epochs = range(1, len(val_losses) + 1)
                    ax4.plot(epochs, val_losses, label=f"{config_result['experiment_id'][:15]}...", alpha=0.8)
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Validation Loss')
            ax4.set_title('Learning Curves - Top 5 Configurations')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plot_file = self.results_manager.results_dir / "analysis_plots.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"Analysis plots saved to: {plot_file}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def save_report(self, filename: str = None):
        """Save the text report to file."""
        if filename is None:
            filename = self.results_manager.results_dir / "final_report.md"
        
        report = self.generate_text_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {filename}")
    
    def print_summary(self):
        """Print a quick summary to console."""
        if not self.results:
            print("No results found.")
            return
        
        stats = self.get_summary_stats()
        best_configs = self.get_best_configurations(3)
        
        print(f"\n📊 Hyperparameter Testing Summary")
        print(f"=" * 50)
        print(f"Total experiments: {stats['total_experiments']}")
        print(f"Best RMSE: {stats['best_rmse']:.4f}")
        print(f"Average RMSE: {stats['avg_rmse']:.4f} ± {stats['rmse_std']:.4f}")
        print(f"RMSE improvement: {stats['worst_rmse'] - stats['best_rmse']:.4f}")
        print(f"")
        print(f"Top 3 configurations:")
        for i, config_result in enumerate(best_configs, 1):
            metrics = config_result['metrics']
            print(f"  {i}. {config_result['experiment_id']}: RMSE {metrics['rmse']:.4f}")


def main():
    """Main function for running analysis."""
    analyzer = HyperparameterAnalyzer()
    
    if not analyzer.results:
        print("[ERROR] No results found. Run hyperparameter tests first.")
        return
    
    print("Analyzing hyperparameter testing results...")
    
    # Print summary
    analyzer.print_summary()
    
    # Generate and save report
    analyzer.save_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: results/final_report.md")
    print(f"Plots saved to: results/analysis_plots.png")


if __name__ == "__main__":
    main()
