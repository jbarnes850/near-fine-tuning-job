import logging
import sys
from fine_tuning.config import load_config, validate_config
from fine_tuning.utils import setup_logging, estimate_cost, num_tokens_from_messages, error_handler
from fine_tuning.api_clients import get_github_client, initialize_openai, validate_openai_api_key
from fine_tuning.data_fetchers import DataFetcher
from fine_tuning.data_processors import DataProcessor
from fine_tuning.fine_tuning import FineTuner

@error_handler
def main():
    # Load and validate configuration
    config = load_config()
    validate_config(config)
    setup_logging(config)
    logging.info("Configuration loaded and validated.")

    # Initialize API clients
    github_client = get_github_client()
    openai_client = initialize_openai()
    validate_openai_api_key(openai_client)
    logging.info("API clients initialized.")

    # Initialize components
    data_fetcher = DataFetcher(github_client, config)
    data_processor = DataProcessor(openai_client, config)
    fine_tuner = FineTuner(config)

    # Fetch data from GitHub repositories
    logging.info("Fetching data from GitHub repositories...")
    all_repo_data = {}
    for repo_name in config['github']['repos']:
        repo_data = data_fetcher.fetch_repo_data(repo_name)
        if repo_data:
            all_repo_data[repo_name] = repo_data

    # Fetch data from articles
    logging.info("Fetching data from articles...")
    all_article_data = {}
    for url in config['articles']['urls']:
        article_data = data_fetcher.fetch_article_data(url)
        if article_data:
            all_article_data[url] = article_data

    # Process fetched data
    logging.info("Processing fetched data...")
    processed_data = []
    for repo_name, repo_data in all_repo_data.items():
        processed = data_processor.process_repo_data(repo_data)
        processed_data.extend(processed)

    for url, article_text in all_article_data.items():
        processed = data_processor.process_article_data(article_text)
        processed_data.extend(processed)

    # Generate refined examples using OpenAI API
    logging.info("Generating refined examples...")
    refined_examples = data_processor.generate_refined_examples(processed_data)

    # Create fine-tuning data
    logging.info("Creating fine-tuning data...")
    fine_tuning_data = data_processor.create_fine_tuning_data(refined_examples)
    data_processor.save_as_jsonl(fine_tuning_data, output_file="fine_tuning_data.jsonl")

    # Estimate cost
    total_tokens = sum(num_tokens_from_messages(example['messages']) for example in fine_tuning_data)
    estimated_cost = estimate_cost(total_tokens, cost_per_1k_tokens=0.03)  # Adjust cost per 1K tokens as needed
    logging.info(f"Estimated fine-tuning cost: ${estimated_cost:.2f}")

    confirmation = input(f"The estimated cost is ${estimated_cost:.2f}. Proceed with fine-tuning? (y/n): ")
    if confirmation.lower() != 'y':
        logging.info("Fine-tuning process cancelled.")
        sys.exit()

    # Fine-tuning process
    logging.info("Starting fine-tuning process...")
    try:
        training_file_id = fine_tuner.upload_training_file("fine_tuning_data.jsonl")
    except Exception as e:
        logging.error(f"Training file upload failed: {str(e)}")
        sys.exit(1)

    try:
        job_id = fine_tuner.create_fine_tune_job(training_file_id)
    except Exception as e:
        logging.error(f"Fine-tuning job creation failed: {str(e)}")
        sys.exit(1)

    model_id = fine_tuner.monitor_fine_tune_job(job_id)

    if model_id:
        logging.info(f"Fine-tuning completed successfully. Model ID: {model_id}")
        logging.info(f"Example usage: response = openai.ChatCompletion.create(model='{model_id}', messages=[...])")
    else:
        logging.error("Fine-tuning failed.")

if __name__ == "__main__":
    main()