import logging
from functools import wraps
import time
import tiktoken
import openai

def setup_logging(config):
    """Set up logging based on the configuration.

    Args:
        config (dict): Configuration dictionary containing logging settings.
    """
    logging_level = getattr(logging, config['logging'].get('level', 'INFO').upper(), logging.INFO)
    logging_format = config['logging'].get('format', '%(asctime)s - %(levelname)s - %(message)s')
    logging_file = config['logging'].get('file')

    handlers = [logging.StreamHandler()]
    if logging_file:
        handlers.append(logging.FileHandler(logging_file))

    logging.basicConfig(level=logging_level, format=logging_format, handlers=handlers)

def error_handler(func):
    """Decorator to catch and log exceptions in functions.

    Args:
        func (function): The function to wrap with error handling.

    Returns:
        function: The wrapped function with error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise e
    return wrapper

def estimate_cost(total_tokens, cost_per_1k_tokens=0.03):
    """Estimate the cost of fine-tuning based on total tokens.

    Args:
        total_tokens (int): The total number of tokens.
        cost_per_1k_tokens (float): The cost per 1,000 tokens.

    Returns:
        float: The estimated cost of fine-tuning.
    """
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    return estimated_cost

def num_tokens_from_string(string, encoding_name='cl100k_base'):
    """Returns the number of tokens in a text string.

    Args:
        string (str): The input text string.
        encoding_name (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model="gpt-4o-2024-08-06"):
    """Returns the number of tokens used by a list of messages.

    Args:
        messages (list): A list of message dictionaries.
        model (str): The model name to use for encoding.

    Returns:
        int: The total number of tokens used by the messages.
    """
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
    """Split list into chunks of size n.

    Args:
        lst (list): The list to split.
        n (int): The size of each chunk.

    Yields:
        list: Chunks of the original list.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def retry_on_exception(exceptions, max_retries=3, delay=5):
    """Decorator to retry a function on specified exceptions.

    Args:
        exceptions (tuple): A tuple of exception classes to catch.
        max_retries (int): The maximum number of retries.
        delay (int): The delay between retries in seconds.

    Returns:
        function: The wrapped function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    retries += 1
                    logging.warning(f"Retrying {func.__name__} due to {e}. Attempt {retries}/{max_retries}")
                    time.sleep(delay)
            logging.error(f"Failed after {max_retries} retries.")
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

def validate_openai_api_key():
    """Validate the OpenAI API key by making a simple API call.

    Raises:
        openai.error.AuthenticationError: If the API key is invalid.
        Exception: If any other error occurs during validation.
    """
    try:
        openai.Engine.list()
        logging.info("OpenAI API key validated successfully.")
    except openai.error.AuthenticationError:
        logging.error("Invalid OpenAI API key.")
        raise
    except Exception as e:
        logging.error(f"Failed to validate OpenAI API key: {e}")
        raise