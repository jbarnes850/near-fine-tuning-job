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
        'logging', 'rate_limit', 'data_processing', 'example_generation',
        'evaluation', 'deployment', 'security', 'content_filtering'
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Add specific validations as needed
    if 'repos' not in config['github']:
        raise ValueError("GitHub repositories are not specified in the configuration.")
    if not config['github']['repos']:
        raise ValueError("GitHub repositories list is empty.")
    if 'api_key' not in config['openai']:
        raise ValueError("OpenAI API key not specified in the configuration.")
    if 'model' not in config['fine_tuning']:
        raise ValueError("Fine-tuning model not specified in the configuration.")

    # Validate numeric parameters
    if config['fine_tuning']['n_epochs'] <= 0:
        raise ValueError("Number of epochs must be a positive integer.")
    if config['fine_tuning']['target_examples'] <= 0:
        raise ValueError("Target examples must be a positive integer.")
    if config['fine_tuning']['max_tokens'] <= 0:
        raise ValueError("Max tokens must be a positive integer.")

    # Validate logging level
    valid_logging_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config['logging']['level'].upper() not in valid_logging_levels:
        raise ValueError(f"Invalid logging level: {config['logging']['level']}. Choose from {valid_logging_levels}.")