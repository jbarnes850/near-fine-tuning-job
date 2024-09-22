import yaml
import os

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config):
    """Validate configuration parameters."""
    required_keys = [
        'github', 'articles', 'cache', 'fine_tuning', 'openai',
        'logging', 'data_processing', 'example_generation'
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Validate nested keys
    if 'repos' not in config['github'] or not config['github']['repos']:
        raise ValueError("GitHub repositories are not specified in the configuration.")
    if 'model' not in config['openai']:
        raise ValueError("OpenAI model not specified in the configuration.")
    if 'model' not in config['fine_tuning']:
        raise ValueError("Fine-tuning model not specified in the configuration.")

    # Validate numeric parameters
    numeric_params = {
        'n_epochs': config['fine_tuning']['n_epochs'],
        'target_examples': config['fine_tuning']['target_examples'],
        'max_tokens': config['fine_tuning']['max_tokens']
    }
    for param_name, param_value in numeric_params.items():
        if param_value <= 0:
            raise ValueError(f"{param_name} must be a positive integer.")

    # Validate logging level
    valid_logging_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config['logging']['level'].upper() not in valid_logging_levels:
        raise ValueError(f"Invalid logging level: {config['logging']['level']}. Choose from {valid_logging_levels}.")