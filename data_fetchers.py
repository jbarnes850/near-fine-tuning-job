import os
import logging
from utils import error_handler, retry_on_exception
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, github_client, config):
        self.github_client = github_client
        self.config = config
        self.cache_dir = self.config['cache']['dir']

    @error_handler
    def fetch_repo_data(self, repo_name):
        """Fetch data from a GitHub repository."""
        logging.info(f"Fetching repository data: {repo_name}")
        cached_data = self.get_cached_data(repo_name, is_repo=True)
        if cached_data:
            logging.info(f"Using cached data for repository: {repo_name}")
            return cached_data

        repo = self.github_client.get_repo(repo_name)
        contents = repo.get_contents("")
        repo_data = []

        def fetch_file_content(file_content):
            files = []
            if file_content.type == 'dir':
                dir_contents = repo.get_contents(file_content.path)
                for item in dir_contents:
                    files.extend(fetch_file_content(item))
            else:
                file_path = file_content.path
                if any(file_path.endswith(ext) for ext in self.config['data_processing']['extensions']):
                    file_data = repo.get_contents(file_path)
                    content = file_data.decoded_content.decode('utf-8')
                    files.append((file_path, content))
            return files

        repo_data.extend(fetch_file_content(contents[0]))
        self.save_cached_data(repo_name, repo_data, is_repo=True)
        logging.info(f"Repository data fetched: {repo_name}")
        return repo_data

    @staticmethod
    @error_handler
    @retry_on_exception(exceptions=(requests.RequestException,))
    def fetch_article_data(url):
        """Fetch and process article data from a given URL."""
        logging.info(f"Fetching article data from: {url}")
        cache_data = DataFetcher.get_cached_data(url, is_repo=False)
        if cache_data:
            logging.info(f"Using cached data for article: {url}")
            return cache_data

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if not content:
            content = soup.body

        article_text = content.get_text(separator='\n', strip=True)
        # Basic cleaning
        article_text = ' '.join(article_text.split())
        DataFetcher.save_cached_data(url, article_text, is_repo=False)
        logging.info(f"Successfully fetched article: {url}")
        return article_text

    @staticmethod
    def get_cached_data(identifier, is_repo=False):
        """Retrieve cached data if available."""
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, f"{'repo' if is_repo else 'article'}_{identifier.replace('/', '_').replace(':', '_')}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if datetime.now() - cached_data['timestamp'] < timedelta(days=7):  # Adjust expiry as needed
                return cached_data['data']
        return None

    @staticmethod
    def save_cached_data(identifier, data, is_repo=False):
        """Save data to cache."""
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, f"{'repo' if is_repo else 'article'}_{identifier.replace('/', '_').replace(':', '_')}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({'timestamp': datetime.now(), 'data': data}, f)