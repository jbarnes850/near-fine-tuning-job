import datetime
from openai import OpenAI
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from github import Github
from dotenv import load_dotenv
import tiktoken
import logging
from functools import wraps
import concurrent.futures
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import random

# Load environment variables
load_dotenv()
print("Environment variables loaded.")

github_api_key = os.getenv("GITHUB_API_KEY")
if not github_api_key:
    raise ValueError("GITHUB_API_KEY environment variable is not set. Please set it in your .env file.")
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

# Initialize GitHub API
github_client = Github(github_api_key)
print("GitHub client initialized.")

# Authenticate using GitHub and OpenAI API keys
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("OpenAI client initialized.")

# List of repositories to pull from
repos = [
    "near/docs",
    "near/neps",
    "near/node-docs",
    "near/nearcore",
    "near-examples/near-multichain",
    "near/create-near-app",
    "near/near-cli",
    "near/near-cli-rs",
    "near/wallet-selector",
    "near/fast-auth-signer",
    "near/mpc",
    "near/near-api-js",
    "fastnear/fastnear-api-server-rs",
    "near/near-sdk-js",
    "near/near-sdk-rs",
    "near/near-workspaces-js",
    "near/near-workspaces-rs",
    "near/near-lake-indexer",
    "Mintbase/near-ca",
    "Mintbase/make-agent",
    "Mintbase/mintbase-js",
]

# List of additional articles
articles = [
    "https://pages.near.org/blog/nightshade-2-launches-on-near-mainnet-introducing-stateless-validation/",
    "https://pages.near.org/papers/the-official-near-white-paper/",
    "https://discovery-domain.org/papers/nightshade.pdf",
    "https://pages.near.org/blog/user-owned-ai-is-near/",
    "https://pages.near.org/blog/near-foundation-and-delphi-labs-partner-on-ai-x-web3-accelerator/",
    "https://pages.near.org/blog/near-one-shares-q3-near-protocol-roadmap-update/",
    "https://pages.near.org/blog/ecosystem-update-announcing-near-one-chain-abstraction-spinouts/",
    "https://pages.near.org/blog/chain-signatures-mainnet-launches/",
    "https://pages.near.org/blog/getting-started-with-chain-signatures/",
    "https://docs.near.ai/",
    "https://wiki.near.org/development/tools-infrastructure",
    "https://wiki.near.org/development/best-practices",
    "https://docs.near.org/tutorials/auction/introduction",
    "https://pages.near.org/blog/near-foundation-launches-ai-incubation-program/"
]

CACHE_DIR = "repo_cache"
CACHE_EXPIRY_DAYS = 7

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                if attempt == max_retries - 1:
                    print(f"Max retries reached for {func.__name__}. Skipping.")
                    return None
                time.sleep(5 * (attempt + 1))
    return wrapper

def get_cached_data(repo_name):
    cache_file = os.path.join(CACHE_DIR, f"{repo_name.replace('/', '_')}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        if datetime.now() - cached_data['timestamp'] < timedelta(days=CACHE_EXPIRY_DAYS):
            return cached_data['data']
    return None

def save_cached_data(repo_name, data):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f"{repo_name.replace('/', '_')}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump({'timestamp': datetime.now(), 'data': data}, f)

@error_handler
def fetch_repo_data(repo_name):
    """Fetch content of markdown and code files from a repository."""
    print(f"Fetching data from repository: {repo_name}")
    
    cached_data = get_cached_data(repo_name)
    if cached_data:
        print(f"Using cached data for {repo_name}")
        return cached_data

    check_rate_limit()  # Add this line to check rate limit before making API calls

    repo = github_client.get_repo(repo_name)
    branches = ["main", "master"]
    repo_data = {}

    for branch in branches:
        try:
            contents = repo.get_contents("", ref=branch)
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path, ref=branch))
                elif file_content.name.endswith(('.md', '.rs', '.js', '.ts', '.py', '.yaml', '.json')):
                    try:
                        repo_data[file_content.path] = file_content.decoded_content.decode('utf-8')
                    except UnicodeDecodeError:
                        repo_data[file_content.path] = file_content.decoded_content.decode('iso-8859-1')
            print(f"Fetched data from {repo_name} using {branch} branch")
            save_cached_data(repo_name, repo_data)
            return repo_data
        except Exception as e:
            if "No commit found for the ref" not in str(e):
                raise e
    print(f"Warning: No valid branch found for {repo_name}")
    return {}

def fetch_all_repo_data(repos):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_repo = {executor.submit(fetch_repo_data, repo): repo for repo in repos}
        all_repo_data = {}
        for future in tqdm(concurrent.futures.as_completed(future_to_repo), total=len(repos), desc="Fetching repository data"):
            repo = future_to_repo[future]
            try:
                data = future.result()
                if data:
                    all_repo_data[repo] = data
            except Exception as exc:
                print(f'{repo} generated an exception: {exc}')
    return all_repo_data

@error_handler
def fetch_article_content(url):
    """Fetch content from a web page."""
    print(f"Fetching content from article: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('main') or soup.find('article') or soup.find('body')
    return content.get_text() if content else ""

def fetch_article_data(articles):
    article_data = {}
    for article in tqdm(articles, desc="Fetching article data"):
        content = fetch_article_content(article)
        if content:
            article_data[article] = content
    return article_data

@error_handler
def openai_api_call(mode, **kwargs):
    """Unified function for OpenAI API calls."""
    time.sleep(1)  # Add a small delay to avoid rate limiting
    if mode == "refine":
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": """You are an AI assistant specializing in NEAR Protocol and blockchain technology. Your task is to refine prompts and completions for a fine-tuning dataset. Follow these guidelines:
                1. Ensure the prompt is clear, specific, and encourages detailed responses about NEAR Protocol.
                2. Make sure the completion is comprehensive, accurate, and provides in-depth information about NEAR.
                3. Include technical details, code examples, or comparisons with other blockchain platforms where relevant.
                4. Address common misconceptions or clarify complex concepts related to NEAR.
                5. Incorporate recent developments or updates in the NEAR ecosystem.
                6. For code-related prompts, focus on best practices, performance optimization, and security considerations.
                7. Create multi-turn conversations that dive deeper into topics when appropriate.
                8. Ensure the language is professional and suitable for developers and blockchain enthusiasts."""},
                {"role": "user", "content": f"Refine the following prompt and completion pair for NEAR Protocol fine-tuning:\n\nPrompt: {kwargs['prompt']}\n\nCompletion: {kwargs['completion']}\n\nProvide a refined version of both the prompt and completion, ensuring they meet the guidelines."}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        refined = response.choices[0].message.content.split('\n\n')
        return refined[0].replace("Prompt: ", ""), refined[1].replace("Completion: ", "")
    elif mode == "fine_tune":
        return client.fine_tuning.create(**kwargs)
    elif mode == "retrieve":
        return client.fine_tuning.retrieve(kwargs['job_id'])

def generate_diverse_prompts(file_path, content, repo_name):
    prompts = []
    if file_path.endswith('.md'):
        prompts.extend([
            (f"Summarize the key points from this NEAR Protocol documentation: {file_path}", f"Here's a summary of the key points from the NEAR Protocol documentation {file_path}:"),
            (f"What are the main concepts introduced in {file_path}? How do they relate to other aspects of NEAR Protocol?", f"The main concepts introduced in {file_path} are:"),
            (f"Compare and contrast the approach described in {file_path} with similar concepts in other blockchain platforms.", f"Comparing the approach in {file_path} with other blockchain platforms:"),
            (f"What are some common misconceptions about the topic covered in {file_path}? How would you clarify them?", f"Common misconceptions about the topic in {file_path} and their clarifications:"),
        ])
    else:
        prompts.extend([
            (f"Explain the purpose and key components of this NEAR Protocol code file: {file_path}", f"This NEAR Protocol code file {file_path} serves the following purpose:"),
            (f"How would you implement the functionality in {file_path} using a different programming language? What challenges might you face?", f"Implementing the functionality of {file_path} in a different language:"),
            (f"Identify potential performance bottlenecks in the code from {file_path}. How could they be optimized?", f"Potential performance bottlenecks in {file_path} and optimization strategies:"),
            (f"What are some best practices for error handling and security that should be applied to the code in {file_path}?", f"Best practices for error handling and security in {file_path}:"),
        ])
    
    # Add a multi-turn conversation
    prompts.append((
        f"I'm new to NEAR Protocol and I'm looking at the file {file_path} in the {repo_name} repository. Can you explain its main purpose?\n\nUser: Great, thanks! Now, how does this file interact with other parts of the NEAR ecosystem?\n\nUser: I see. Could you provide a simple example of how a developer might use or extend this code in a real-world application?",
        f"Certainly! Let me explain the main purpose of {file_path} in the {repo_name} repository:\n\nAssistant: Of course! Here's how {file_path} interacts with other parts of the NEAR ecosystem:\n\nAssistant: Absolutely! Here's a simple example of how a developer might use or extend the code from {file_path} in a real-world application:"
    ))
    
    return prompts

def generate_article_prompts(url, content):
    prompts = [
        (f"Summarize the key points from this NEAR Protocol article: {url}", f"Here's a summary of the key points from the NEAR Protocol article {url}:"),
        (f"What are the main implications of the developments described in the article at {url} for NEAR developers?", f"The main implications for NEAR developers based on the article at {url} are:"),
        (f"How do the concepts discussed in {url} relate to other recent developments in the NEAR ecosystem?", f"The concepts in {url} relate to other recent NEAR developments as follows:"),
        (f"Based on the article at {url}, what are some potential use cases or applications that developers could build?", f"Potential use cases or applications based on the article at {url} include:"),
    ]
    
    # Add a multi-turn conversation
    prompts.append((
        f"I just read the article at {url} and I'm intrigued by the concepts it introduces. Can you explain how these might impact existing NEAR applications?\n\nUser: Interesting! Are there any potential challenges or limitations to implementing these ideas?\n\nUser: Given these considerations, what advice would you give to a developer looking to incorporate these concepts into their NEAR project?",
        f"Certainly! The concepts introduced in the article at {url} could impact existing NEAR applications in the following ways:\n\nAssistant: Great question! There are indeed some potential challenges and limitations to consider:\n\nAssistant: Based on these considerations, here's my advice for a developer looking to incorporate these concepts into their NEAR project:"
    ))
    
    return prompts

def split_content(content, max_length=1000):
    """Split content into chunks of maximum length while preserving word boundaries."""
    words = content.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_fine_tuning_data(repo_data, article_data):
    """Create fine-tuning data in the required format."""
    print("Creating fine-tuning data...")
    fine_tuning_data = []

    # Process repository data
    for repo_name, files in repo_data.items():
        print(f"Processing data from repository: {repo_name}")
        for file_path, content in files.items():
            content_chunks = split_content(content)
            for chunk in content_chunks:
                prompts = generate_diverse_prompts(file_path, chunk, repo_name)
                for prompt, completion in prompts:
                    refined_prompt, refined_completion = openai_api_call("refine", prompt=prompt, completion=completion)
                    fine_tuning_data.append({"messages": [{"role": "user", "content": refined_prompt}, {"role": "assistant", "content": refined_completion}]})

    # Process article data
    for url, content in article_data.items():
        print(f"Processing article: {url}")
        content_chunks = split_content(content)
        for chunk in content_chunks:
            prompts = generate_article_prompts(url, chunk)
            for prompt, completion in prompts:
                refined_prompt, refined_completion = openai_api_call("refine", prompt=prompt, completion=completion)
                fine_tuning_data.append({"messages": [{"role": "user", "content": refined_prompt}, {"role": "assistant", "content": refined_completion}]})

    # Add a step to validate and clean the data
    fine_tuning_data = validate_and_clean_data(fine_tuning_data)

    print(f"Fine-tuning data creation complete. Total examples: {len(fine_tuning_data)}")
    return fine_tuning_data

def validate_and_clean_data(data):
    """Validate and clean the fine-tuning data."""
    return [item for item in data if len(item['messages']) == 2 and 
            all(key in item['messages'][0] for key in ['role', 'content']) and 
            all(key in item['messages'][1] for key in ['role', 'content']) and
            len(item['messages'][0]['content']) >= 10 and  # Ensure minimum content length
            len(item['messages'][1]['content']) >= 20]

@error_handler
def save_as_jsonl(data, output_file="near_finetune_data.jsonl"):
    """Save data to a JSONL file."""
    print(f"Saving data to JSONL file: {output_file}")
    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Data saved to {output_file}")

@error_handler
def upload_training_file(file_path):
    """Upload the training file to OpenAI."""
    print(f"Uploading training file: {file_path}")
    with open(file_path, "rb") as file:
        response = client.files.create(file=file, purpose='fine-tune')
    return response.id

@error_handler
def create_fine_tune_job(training_file_id):
    return openai_api_call("fine_tune", training_file=training_file_id, model="gpt-4o-2024-08-06", hyperparameters={"n_epochs": 3})

def monitor_fine_tune_job(job_id):
    """Monitor the status of the fine-tuning job with a progress bar."""
    print(f"Monitoring fine-tuning job: {job_id}")
    pbar = tqdm(total=100, desc="Fine-tuning progress", unit="%")
    last_progress = 0
    while True:
        job = openai_api_call("retrieve", job_id=job_id)
        logging.info(f"Fine-tuning status: {job.status}")
        if job.status == "succeeded":
            pbar.update(100 - last_progress)
            pbar.close()
            logging.info(f"Fine-tuning complete! Fine-tuned model ID: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status == "failed":
            pbar.close()
            logging.error("Fine-tuning job failed. Please check the OpenAI dashboard for more details.")
            return None
        elif hasattr(job, 'fine_tuned_model') and job.fine_tuned_model is not None:
            current_progress = min(job.trained_tokens / job.trained_tokens_n * 100, 99)
            pbar.update(current_progress - last_progress)
            last_progress = current_progress
        time.sleep(60)  # Check status every minute

def check_rate_limit():
    rate_limit = github_client.get_rate_limit()
    core_rate = rate_limit.core
    print(f"GitHub API Rate Limit: {core_rate.remaining}/{core_rate.limit}")
    if core_rate.remaining < 100:
        reset_time = core_rate.reset.replace(tzinfo=None)
        current_time = datetime.now()
        time_until_reset = reset_time - current_time
        print(f"Warning: Low on API calls. Resets in {time_until_reset}")
        
        # If the reset time is in the future, start a countdown
        if reset_time > current_time:
            print("Waiting for rate limit to reset...")
            with tqdm(total=int(time_until_reset.total_seconds()), desc="Time until reset", unit="s") as pbar:
                while datetime.now() < reset_time:
                    time.sleep(1)
                    pbar.update(1)
            print("Rate limit has been reset. Continuing with execution.")
        else:
            print("Rate limit should have already reset. Continuing with execution.")
    else:
        print("Sufficient API calls remaining. Continuing with execution.")

def num_tokens_from_messages(messages, model="gpt-4o-2024-08-06"):
    """Calculate the number of tokens in the messages."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    return num_tokens

def estimate_total_tokens(fine_tuning_data):
    """Estimate the total number of tokens in the fine-tuning data."""
    total_tokens = sum(num_tokens_from_messages(item['messages']) for item in fine_tuning_data)
    print(f"Estimated total tokens: {total_tokens}")
    return total_tokens

def estimate_fine_tuning_cost(total_tokens):
    # Assuming a rate of $0.03 per 1K tokens (this may vary, please check the latest pricing)
    estimated_cost = (total_tokens / 1000) * 0.03
    print(f"Estimated fine-tuning cost: ${estimated_cost:.2f}")
    return estimated_cost

def validate_openai_api_key():
    try:
        client.models.list()
        logging.info("OpenAI API key is valid.")
    except Exception as e:
        logging.error(f"Error validating OpenAI API key: {str(e)}")
        raise ValueError("Invalid OpenAI API key. Please check your .env file.")

def openai_api_call_with_retry(func, max_retries=3, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Max retries reached for OpenAI API call: {str(e)}")
                raise
            logging.warning(f"OpenAI API call failed, retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)  # Exponential backoff

# Main execution
if __name__ == "__main__":
    try:
        print("Starting NEAR fine-tuning data preparation...")

        # Check GitHub API key
        if not os.getenv("GITHUB_API_KEY"):
            raise ValueError("GITHUB_API_KEY environment variable is not set. Please set it in your .env file.")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

        # Check GitHub API rate limit
        check_rate_limit()

        # Fetch data from all repositories
        print("Fetching data from repositories...")
        all_repo_data = fetch_all_repo_data(repos)
        print(f"Repository data fetched. Total repositories processed: {len(all_repo_data)}")

        # Fetch data from articles
        print("Fetching data from articles...")
        article_data = fetch_article_data(articles)
        print(f"Article data fetched. Total articles processed: {len(article_data)}")

        # Create fine-tuning data and save to JSONL
        fine_tuning_data = create_fine_tuning_data(all_repo_data, article_data)
        print(f"Number of examples generated: {len(fine_tuning_data)}")
        total_tokens = estimate_total_tokens(fine_tuning_data)
        estimated_cost = estimate_fine_tuning_cost(total_tokens)

        if not fine_tuning_data:
            print("No data generated. Exiting.")
            exit()

        jsonl_file = "near_finetune_data.jsonl"
        save_as_jsonl(fine_tuning_data, jsonl_file)

        # After saving the JSONL file
        print(f"Fine-tuning data saved to {jsonl_file}")
        confirmation = input(f"The estimated cost of fine-tuning is ${estimated_cost:.2f}. Do you want to proceed? (y/n): ")
        if confirmation.lower() != 'y':
            print("Fine-tuning job cancelled.")
            exit()

        # Upload training file
        print("Uploading training file to OpenAI...")
        training_file_id = upload_training_file(jsonl_file)
        print(f"Training file uploaded. File ID: {training_file_id}")

        # Create and monitor fine-tuning job
        print("Creating fine-tuning job...")
        job_id = create_fine_tune_job(training_file_id)
        print(f"Fine-tuning job created. Job ID: {job_id}")

        print("Starting job monitoring...")
        fine_tuned_model_id = monitor_fine_tune_job(job_id)

        if fine_tuned_model_id:
            print(f"\nFine-tuning process complete.")
            print(f"Your fine-tuned model ID is: {fine_tuned_model_id}")
            print("You can use this model by specifying this ID when making OpenAI API requests.")
            print("Example usage:")
            print(f"response = client.chat.completions.create(model='{fine_tuned_model_id}', messages=[...])")
        else:
            print("\nFine-tuning process failed. Please check the OpenAI dashboard for more information.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        print("An error occurred during execution. Please check the logs for details.")