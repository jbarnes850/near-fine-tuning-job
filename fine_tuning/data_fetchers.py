import os
import logging
from fine_tuning.utils import error_handler, retry_on_exception
import requests
from bs4 import BeautifulSoup
import pickle
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, github_client, config):
        self.github_client = github_client
        self.config = config
        self.cache_dir = self.config['cache']['dir']
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    @error_handler
    def fetch_repo_data(self, repo_name):
        """Fetch data from a GitHub repository."""
        logging.info(f"Fetching repository data: {repo_name}")
        cached_data = self.get_cached_data(repo_name, is_repo=True)
        if cached_data:
            logging.info(f"Using cached data for repository: {repo_name}")
            return cached_data

        try:
            repo = self.github_client.get_repo(repo_name)
        except Exception as e:
            logging.error(f"Error fetching repository {repo_name}: {e}")
            return []

        repo_data = []
        contents = repo.get_contents("")
        files_to_process = []

        while contents:
            file_content = contents.pop(0)
            if file_content.type == 'dir':
                contents.extend(repo.get_contents(file_content.path))
            else:
                file_path = file_content.path
                if any(file_path.endswith(ext) for ext in self.config['data_processing']['extensions']):
                    files_to_process.append(file_content)

        for file_content in files_to_process:
            try:
                file_data = repo.get_contents(file_content.path)
                content = file_data.decoded_content.decode('utf-8', errors='ignore')
                repo_data.append((file_content.path, content))
            except Exception as e:
                logging.warning(f"Could not fetch content for file {file_content.path}: {e}")

        self.save_cached_data(repo_name, repo_data, is_repo=True)
        logging.info(f"Repository data fetched: {repo_name}")
        return repo_data

    @error_handler
    @retry_on_exception(exceptions=(requests.RequestException,))
    def fetch_article_data(self, url):
        """Fetch and process article data from a given URL."""
        logging.info(f"Fetching article data from: {url}")
        cached_data = self.get_cached_data(url, is_repo=False)
        if cached_data:
            logging.info(f"Using cached data for article: {url}")
            return cached_data

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if not content:
            content = soup.body

        if content:
            article_text = content.get_text(separator='\n', strip=True)
            article_text = ' '.join(article_text.split())
            self.save_cached_data(url, article_text, is_repo=False)
            logging.info(f"Successfully fetched article: {url}")
            return article_text
        else:
            logging.warning(f"Could not find content in article: {url}")
            return ""

    def get_cached_data(self, identifier, is_repo=False):
        """Retrieve cached data if available."""
        cache_file = os.path.join(self.cache_dir, f"{'repo' if is_repo else 'article'}_{identifier.replace('/', '_').replace(':', '_')}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if datetime.now() - cached_data['timestamp'] < timedelta(days=self.config['cache'].get('expiry_days', 7)):
                return cached_data['data']
        return None

    def save_cached_data(self, identifier, data, is_repo=False):
        """Save data to cache."""
        cache_file = os.path.join(self.cache_dir, f"{'repo' if is_repo else 'article'}_{identifier.replace('/', '_').replace(':', '_')}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({'timestamp': datetime.now(), 'data': data}, f)