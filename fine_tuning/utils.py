import logging
from functools import wraps
import time
import tiktoken

def setup_logging(config):
    """Set up logging based on the configuration."""
    logging_level = getattr(logging, config['logging'].get('level', 'INFO').upper(), logging.INFO)
    logging_format = config['logging'].get('format', '%(asctime)s - %(levelname)s - %(message)s')
    logging_file = config['logging'].get('file')

    handlers = [logging.StreamHandler()]
    if logging_file:
        handlers.append(logging.FileHandler(logging_file))

    logging.basicConfig(level=logging_level, format=logging_format, handlers=handlers)

def error_handler(func):
    """Decorator to catch and log exceptions in functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def estimate_cost(total_tokens, cost_per_1k_tokens=0.03):
    """Estimate the cost of fine-tuning based on total tokens."""
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    return estimated_cost

def num_tokens_from_string(string, encoding_name='cl100k_base'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model="gpt-4o-2024-08-06"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Each message requires 4 tokens
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # For assistant's reply
    return num_tokens

def split_list(lst, n):
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def retry_on_exception(exceptions, max_retries=3, delay=5):
    """Decorator to retry a function on specified exceptions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    logging.warning(f"Retrying {func.__name__} due to {e}. Attempt {retries}/{max_retries}")
                    time.sleep(delay)
            logging.error(f"Failed after {max_retries} retries.")
            raise
        return wrapper
    return decorator