import datetime
import openai
import os
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from github import Github
from dotenv import load_dotenv
import tiktoken
import logging

# Load environment variables
load_dotenv()
print("Environment variables loaded.")

# Authenticate using GitHub and OpenAI API keys
github_client = Github(os.getenv("GITHUB_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
print("GitHub and OpenAI clients initialized.")

# List of repositories to pull from
repos = [
    "near/docs",
    "near/neps",
    "near/node-docs",
    "near/nearcore",
    "Nuffle-Labs/data-availability",
    "near-examples/near-multichain",
    "near/create-near-app",
    "near/near-vscode",
    "near/near-cli",
    "near/near-cli-rs",
    "near/wallet-selector",
    "blocknative/web3-onboard",
    "near/fast-auth-signer",
    "near/mpc",
    "near/near-api-js",
    "fastnear/fastnear-api-server-rs",
    "near/queryapi",
    "near/near-sdk-js",
    "near/near-sdk-rs",
    "near/near-workspaces-js",
    "near/near-workspaces-rs",
    "near/near-lake-indexer",
    "Mintbase/near-ca",
    "Mintbase/make-agent"
]

# List of additional articles
articles = [
    "https://pages.near.org/blog/nightshade-2-launches-on-near-mainnet-introducing-stateless-validation/",
    "https://pages.near.org/blog/user-owned-ai-is-near/",
    "https://pages.near.org/blog/near-foundation-and-delphi-labs-partner-on-ai-x-web3-accelerator/",
    "https://pages.near.org/blog/near-one-shares-q3-near-protocol-roadmap-update/",
    "https://pages.near.org/blog/ecosystem-update-announcing-near-one-chain-abstraction-spinouts/",
    "https://pages.near.org/blog/chain-signatures-mainnet-launches/",
    "https://pages.near.org/blog/getting-started-with-chain-signatures/",
    "https://docs.near.ai/",
    "https://pages.near.org/blog/near-foundation-launches-ai-incubation-program/"
]

def fetch_repo_data(repo_name):
    """Fetch content of markdown and code files from a repository."""
    print(f"Fetching data from repository: {repo_name}")
    max_retries = 5
    base_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
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
                            if file_content.decoded_content is None:
                                print(f"Warning: No content found for {file_content.path} in {repo_name}. Skipping this file.")
                                continue
                            try:
                                # Try UTF-8 encoding first
                                repo_data[file_content.path] = file_content.decoded_content.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    # If UTF-8 fails, try ISO-8859-1 encoding
                                    repo_data[file_content.path] = file_content.decoded_content.decode('iso-8859-1')
                                except UnicodeDecodeError:
                                    print(f"Warning: Unable to decode {file_content.path} in {repo_name}. Skipping this file.")
                    print(f"Fetched data from {repo_name} using {branch} branch")
                    return repo_data
                except Exception as e:
                    if "No commit found for the ref" in str(e):
                        continue
                    else:
                        raise e
            
            print(f"Warning: No valid branch found for {repo_name}")
            return {}
        except Exception as e:
            if "403" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Encountered 403 error for {repo_name}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to fetch data from {repo_name}: {str(e)}")
                return {}
    
    print(f"Max retries reached for {repo_name}. Skipping this repository.")
    return {}

def fetch_article_content(url):
    """Fetch content from a web page."""
    print(f"Fetching content from article: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('main') or soup.find('article') or soup.find('body')
    return content.get_text() if content else ""

def gpt4_refine(prompt, completion):
    """Use GPT-4o to refine the prompt and completion."""
    print("Refining prompt and completion with GPT-4o")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06", 
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in NEAR Protocol and blockchain technology. Your task is to refine prompts and completions for a fine-tuning dataset."},
                {"role": "user", "content": f"Refine the following prompt and completion pair for NEAR Protocol fine-tuning:\n\nPrompt: {prompt}\n\nCompletion: {completion}\n\nProvide a refined version of both the prompt and completion."}
            ],
            temperature=0.7
        )
        refined = response.choices[0].message['content'].split('\n\n')
        return refined[0].replace("Prompt: ", ""), refined[1].replace("Completion: ", "")
    except Exception as e:
        print(f"Error in GPT-4o refinement: {e}")
        return prompt, completion

def create_fine_tuning_data(repo_data, article_data):
    """Create fine-tuning data in the required format."""
    print("Creating fine-tuning data...")
    fine_tuning_data = []
    
    # Process repository data
    for repo_name, files in repo_data.items():
        print(f"Processing data from repository: {repo_name}")
        for file_path, content in files.items():
            print(f"  Processing file: {file_path}")
            if file_path.endswith('.md'):
                prompt = f"Summarize the following NEAR Protocol documentation: {file_path}\n\nContent:\n{content[:1000]}"
                completion = f"Here's a summary of the NEAR Protocol documentation {file_path}:"
            else:
                prompt = f"Explain the purpose and key components of this NEAR Protocol code file: {file_path}\n\nCode:\n{content[:1000]}"
                completion = f"This NEAR Protocol code file {file_path} serves the following purpose:"
            
            # Refine prompt and completion using GPT-4o
            refined_prompt, refined_completion = gpt4_refine(prompt, completion)
            fine_tuning_data.append({"messages": [{"role": "user", "content": refined_prompt}, {"role": "assistant", "content": refined_completion}]})
    
    # Process article data
    for url, content in article_data.items():
        print(f"Processing article: {url}")
        prompt = f"Summarize the key points from this NEAR Protocol article: {url}\n\nContent:\n{content[:1000]}"
        completion = f"Here's a summary of the key points from the NEAR Protocol article {url}:"
        
        # Refine prompt and completion using GPT-4o
        refined_prompt, refined_completion = gpt4_refine(prompt, completion)
        fine_tuning_data.append({"messages": [{"role": "user", "content": refined_prompt}, {"role": "assistant", "content": refined_completion}]})
    
    # Add a step to validate and clean the data
    fine_tuning_data = validate_and_clean_data(fine_tuning_data)
    
    print(f"Fine-tuning data creation complete. Total examples: {len(fine_tuning_data)}")
    return fine_tuning_data

def validate_and_clean_data(data):
    cleaned_data = []
    for item in data:
        if len(item['messages']) == 2 and all(key in item['messages'][0] for key in ['role', 'content']) and all(key in item['messages'][1] for key in ['role', 'content']):
            cleaned_data.append(item)
    return cleaned_data

def save_as_jsonl(data, output_file="near_finetune_data.jsonl"):
    """Save data to a JSONL file."""
    print(f"Saving data to JSONL file: {output_file}")
    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Data saved to {output_file}")

def upload_training_file(file_path):
    """Upload the training file to OpenAI."""
    print(f"Uploading training file: {file_path}")
    with open(file_path, "rb") as file:
        response = openai.File.create(file=file, purpose='fine-tune')
    return response.id

def create_fine_tune_job(training_file_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = openai.FineTuningJob.create(
                training_file=training_file_id,
                model="gpt-4o-2024-08-06", 
                hyperparameters={
                    "n_epochs": 3,
                }
            )
            return response.id
        except openai.error.OpenAIError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Error creating fine-tune job: {e}. Retrying...")
            time.sleep(5 * (attempt + 1))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def monitor_fine_tune_job(job_id):
    """Monitor the status of the fine-tuning job."""
    print(f"Monitoring fine-tuning job: {job_id}")
    while True:
        job = openai.FineTuningJob.retrieve(job_id)
        logging.info(f"Fine-tuning status: {job.status}")
        if job.status == "succeeded":
            logging.info(f"Fine-tuning complete! Fine-tuned model ID: {job.fine_tuned_model}")
            logging.info("You can now use this model ID to make requests to your fine-tuned model.")
            return job.fine_tuned_model
        elif job.status == "failed":
            logging.error("Fine-tuning job failed. Please check the OpenAI dashboard for more details.")
            return None
        time.sleep(60)  # Check status every minute

def check_rate_limit():
    rate_limit = github_client.get_rate_limit()
    core_rate = rate_limit.core
    print(f"GitHub API Rate Limit: {core_rate.remaining}/{core_rate.limit}")
    if core_rate.remaining < 100:
        reset_time = core_rate.reset.replace(tzinfo=None) - datetime.datetime.now()
        print(f"Warning: Low on API calls. Resets in {reset_time}")

def num_tokens_from_messages(messages, model="gpt-4o-2024-08-06"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

# Main execution
if __name__ == "__main__":
    print("Starting NEAR fine-tuning data preparation...")

    # Check GitHub API rate limit
    check_rate_limit()

    # Fetch data from all repositories
    print("Fetching data from repositories...")
    all_repo_data = {}
    for repo in repos:
        repo_data = fetch_repo_data(repo)
        if repo_data:
            all_repo_data[repo] = repo_data
    print(f"Repository data fetched. Total repositories processed: {len(all_repo_data)}")

    # Fetch data from articles
    print("Fetching data from articles...")
    article_data = {}
    for article in articles:
        try:
            content = fetch_article_content(article)
            if content:
                article_data[article] = content
                print(f"Fetched content from {article}")
            else:
                print(f"No content found for {article}")
        except Exception as e:
            print(f"Failed to fetch content from {article}: {e}")
    print(f"Article data fetched. Total articles processed: {len(article_data)}")

    # Create fine-tuning data and save to JSONL
    fine_tuning_data = create_fine_tuning_data(all_repo_data, article_data)
    print(f"Number of examples generated: {len(fine_tuning_data)}")
    
    if len(fine_tuning_data) == 0:
        print("No data generated. Exiting.")
        exit()

    jsonl_file = "near_finetune_data.jsonl"
    save_as_jsonl(fine_tuning_data, jsonl_file)

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
        print(f"response = openai.ChatCompletion.create(model='{fine_tuned_model_id}', messages=[...])")
    else:
        print("\nFine-tuning process failed. Please check the OpenAI dashboard for more information.")
