import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from github import Github
from dotenv import load_dotenv
from fine_tuning.utils import error_handler
import logging

load_dotenv()

def get_github_client():
    """Initialize and return a GitHub client."""
    load_dotenv()
    github_api_key = os.getenv("GITHUB_API_KEY")
    if not github_api_key:
        raise ValueError("GITHUB_API_KEY is not set in the environment variables.")
    return Github(github_api_key)

def initialize_openai():
    """Initialize OpenAI by setting the API key."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    logging.info("OpenAI client initialized.")

@error_handler
def validate_openai_api_key():
    """Validate the OpenAI API key by making a test API call."""
    try:
        client.models.list()
    except Exception as e:
        raise ValueError(f"Invalid OpenAI API key: {str(e)}")
    else:
        logging.info("OpenAI API key validated successfully.")