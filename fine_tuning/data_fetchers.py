import os
import logging
from fine_tuning.utils import error_handler, retry_on_exception
import requests
from bs4 import BeautifulSoup
import pickle
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from io import BytesIO

class DataFetcher:
    def __init__(self, github_client, config):
        self.github_client = github_client
        self.config = config
        self.cache_dir = config['cache']['dir']
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    @error_handler
    @retry_on_exception(exceptions=(requests.RequestException,))
    def fetch_repo_data(self, repo_name):
        """Fetch and process repository data from a given GitHub repository."""
        logging.info(f"Fetching repository data: {repo_name}")
        cached_data = self.get_cached_data(repo_name, is_repo=True)
        if cached_data:
            logging.info(f"Using cached data for repository: {repo_name}")
            return cached_data

        repo = self.github_client.get_repo(repo_name)
        contents = repo.get_contents("")
        repo_data = self._process_contents(contents, repo)
        self.save_cached_data(repo_name, repo_data, is_repo=True)
        logging.info(f"Successfully fetched repository: {repo_name}")
        return repo_data

    def _process_contents(self, contents, repo):
        """Recursively process repository contents."""
        repo_data = []
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                file_data = repo.get_contents(file_content.path).decoded_content.decode("utf-8")
                repo_data.append((file_content.path, file_data))
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

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                logging.error(f"Access forbidden for URL: {url}")
                return ""
            else:
                raise e

        if 'application/pdf' in response.headers.get('Content-Type', ''):
            article_text = self._extract_text_from_pdf(response.content)
        else:
            article_text = self._extract_text_from_html(response.content)

        if article_text:
            self.save_cached_data(url, article_text, is_repo=False)
            logging.info(f"Successfully fetched article: {url}")
            return article_text
        else:
            logging.warning(f"Could not find content in article: {url}")
            return ""

    def _extract_text_from_html(self, html_content):
        """Extract text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        if content:
            article_text = content.get_text(separator='\n', strip=True)
            return ' '.join(article_text.split())
        return ""

    def _extract_text_from_pdf(self, pdf_content):
        """Extract text from PDF content."""
        try:
            pdf_reader = PdfReader(BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}")
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