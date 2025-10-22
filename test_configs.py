"""
Test configurations for hyperparameter testing.
Defines all 25+ configurations to test systematically.
"""

from config import DEFAULT_CONFIG


def get_all_test_configs():
    """
    Get all test configurations for hyperparameter testing.
    Returns a list of configuration dictionaries.
    """
    configs = []
    
    # 1. Baseline configuration
    baseline = DEFAULT_CONFIG.copy()
    baseline['name'] = 'baseline'
    configs.append(baseline)
    
    # 2. Learning Rate Sweep (5 configs)
    learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    for lr in learning_rates:
        config = DEFAULT_CONFIG.copy()
        config['learning_rate'] = lr
        config['name'] = f'lr_{lr}'
        configs.append(config)
    
    # 3. Patience Sweep (4 configs)
    patience_values = [3, 5, 7, 10]
    for patience in patience_values:
        config = DEFAULT_CONFIG.copy()
        config['patience'] = patience
        config['name'] = f'patience_{patience}'
        configs.append(config)
    
    # 4. Architecture Sweep - Embedding Dimensions (3 configs)
    embedding_dims = [25, 50, 100]
    for emb_dim in embedding_dims:
        config = DEFAULT_CONFIG.copy()
        config['embedding_dim'] = emb_dim
        config['name'] = f'emb_dim_{emb_dim}'
        configs.append(config)
    
    # 5. Architecture Sweep - Hidden Layer Configurations (3 configs)
    hidden_configs = [
        [64, 32],
        [128, 64, 32],
        [256, 128, 64]
    ]
    for i, hidden_dims in enumerate(hidden_configs):
        config = DEFAULT_CONFIG.copy()
        config['hidden_dims'] = hidden_dims
        config['name'] = f'hidden_{i+1}'
        configs.append(config)
    
    # 6. Batch Size Testing (2 configs)
    batch_sizes = [512, 2048]
    for batch_size in batch_sizes:
        config = DEFAULT_CONFIG.copy()
        config['batch_size'] = batch_size
        config['name'] = f'batch_{batch_size}'
        configs.append(config)
    
    # 7. Dropout Rate Testing (2 configs)
    dropout_rates = [0.1, 0.3]
    for dropout_rate in dropout_rates:
        config = DEFAULT_CONFIG.copy()
        config['dropout_rate'] = dropout_rate
        config['name'] = f'dropout_{dropout_rate}'
        configs.append(config)
    
    # 8. Combined Optimization Configurations (5 configs)
    # These test combinations of the best individual parameters
    
    # Best learning rate + different architectures
    combined_configs = [
        {'learning_rate': 0.001, 'embedding_dim': 100, 'hidden_dims': [256, 128, 64]},
        {'learning_rate': 0.002, 'embedding_dim': 50, 'hidden_dims': [128, 64, 32]},
        {'learning_rate': 0.0005, 'embedding_dim': 75, 'hidden_dims': [128, 64]},
        {'learning_rate': 0.001, 'patience': 7, 'batch_size': 1536},
        {'learning_rate': 0.001, 'dropout_rate': 0.15, 'embedding_dim': 60}
    ]
    
    for i, combo_config in enumerate(combined_configs):
        config = DEFAULT_CONFIG.copy()
        config.update(combo_config)
        config['name'] = f'combined_{i+1}'
        configs.append(config)
    
    return configs


def get_quick_test_configs():
    """
    Get a smaller set of configurations for quick testing (~30 mins).
    Useful for validating the testing framework.
    """
    configs = []
    
    # Baseline
    baseline = DEFAULT_CONFIG.copy()
    baseline['name'] = 'baseline'
    baseline['epochs'] = 20  # Reduced for speed
    configs.append(baseline)
    
    # Quick learning rate test
    config = DEFAULT_CONFIG.copy()
    config['learning_rate'] = 0.002
    config['epochs'] = 20
    config['name'] = 'lr_0.002'
    configs.append(config)
    
    # Quick architecture test
    config = DEFAULT_CONFIG.copy()
    config['embedding_dim'] = 100
    config['epochs'] = 20
    config['name'] = 'emb_100'
    configs.append(config)
    
    # Quick patience test
    config = DEFAULT_CONFIG.copy()
    config['patience'] = 3
    config['epochs'] = 20
    config['name'] = 'patience_3'
    configs.append(config)
    
    # Quick combined test
    config = DEFAULT_CONFIG.copy()
    config['learning_rate'] = 0.001
    config['embedding_dim'] = 75
    config['patience'] = 7
    config['epochs'] = 20
    config['name'] = 'quick_combined'
    configs.append(config)
    
    return configs


def print_config_summary(configs):
    """Print a summary of all configurations."""
    print(f"\nTest Configuration Summary:")
    print(f"Total configurations: {len(configs)}")
    print("\nConfiguration breakdown:")
    
    categories = {}
    for config in configs:
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
    
    print(f"\nEstimated total time: {len(configs) * 5:.0f} minutes (assuming 5 min per config)")


if __name__ == "__main__":
    # Test the configuration generation
    all_configs = get_all_test_configs()
    print_config_summary(all_configs)
    
    print(f"\nFirst few configurations:")
    for i, config in enumerate(all_configs[:5]):
        print(f"{i+1}. {config['name']}: {config}")
    
    print(f"\nQuick test configurations:")
    quick_configs = get_quick_test_configs()
    for i, config in enumerate(quick_configs):
        print(f"{i+1}. {config['name']}: epochs={config['epochs']}")

