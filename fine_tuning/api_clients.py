import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from github import Github

load_dotenv()

def get_github_client():
    """Initialize and return a GitHub client."""
    github_api_key = os.getenv("GITHUB_API_KEY")
    if not github_api_key:
        raise ValueError("GITHUB_API_KEY is not set in the environment variables.")
    return Github(github_api_key)

def initialize_openai():
    """Initialize OpenAI by setting the API key."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    logging.info("OpenAI client initialized.")
    return client

def validate_openai_api_key(client):
    """Validate the OpenAI API key by making a test API call."""
    from fine_tuning.utils import error_handler  # Move import here
    
    @error_handler
    def _validate():
        try:
            client.models.list()
        except Exception as e:
            raise ValueError(f"Invalid OpenAI API key: {str(e)}")
        else:
            logging.info("OpenAI API key validated successfully.")
    
    _validate()