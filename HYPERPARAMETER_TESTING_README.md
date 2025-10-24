# Hyperparameter Testing Framework

This framework provides comprehensive automated testing for the MovieLens Recommender System hyperparameters.

## Quick Start

### 1. Quick Test (30 minutes)
```bash
python quick_test.py
```
Tests 5 configurations with reduced epochs to validate the framework.

### 2. Full Test (2+ hours)
```bash
python run_hyperparameter_tests.py
```
Tests all 25+ configurations systematically.

### 3. Resume Interrupted Test
```bash
python run_hyperparameter_tests.py --resume
```
Continues from where you left off.

### 4. Analyze Results
```bash
python analyze_results.py
```
Generates comprehensive reports and visualizations.

### 5. Update Optimal Configuration
```bash
python update_optimal_config.py --select best_rmse
```
Updates `config.py` with the best configuration found.

## Files Created

- `config.py` - Configuration management
- `hyperparameter_results.py` - Results tracking and persistence
- `test_configs.py` - Test configuration definitions
- `hyperparameter_tuner.py` - Testing framework
- `run_hyperparameter_tests.py` - Main testing script
- `quick_test.py` - Quick validation script
- `analyze_results.py` - Analysis and reporting
- `update_optimal_config.py` - Configuration update script
- `results/` - Directory for all test results

## Usage Examples

```bash
# Test specific number of configurations
python run_hyperparameter_tests.py --configs 10

# Quick test mode
python run_hyperparameter_tests.py --quick

# Update with balanced configuration (speed vs accuracy)
python update_optimal_config.py --select best_balanced

# Update with specific experiment
python update_optimal_config.py --experiment-id "lr_0.002_patience_7_emb_100"
```

## Results

All results are saved in the `results/` directory:
- `experiment_results.json` - Complete results data
- `experiment_results.csv` - Spreadsheet-friendly format
- `learning_curves/` - Individual learning curve plots
- `final_report.md` - Comprehensive analysis report
- `analysis_plots.png` - Visualization charts

## Configuration Parameters Tested

- **Learning Rate**: 0.0001, 0.0005, 0.001, 0.002, 0.005
- **Patience**: 3, 5, 7, 10 epochs
- **Batch Size**: 512, 1024, 2048
- **Embedding Dimension**: 25, 50, 100
- **Hidden Layers**: [64,32], [128,64,32], [256,128,64]
- **Dropout Rate**: 0.1, 0.2, 0.3
- **Combined Configurations**: 5 optimized combinations

Total: ~25 configurations tested systematically.


