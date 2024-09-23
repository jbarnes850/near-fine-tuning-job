# Fine-Tuning Guide for the NEAR Ecosystem Model

*Leverage the power of AI to enhance your projects, experiment with new ideas, and deepen your understanding of Large Language Models (LLMs) by fine-tuning a language model tailored to the NEAR ecosystem.*

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Understanding Fine-Tuning](#understanding-fine-tuning)
4. [Setup and Installation](#setup-and-installation)
5. [Configuring the Project](#configuring-the-project)
6. [Data Collection](#data-collection)
7. [Data Processing](#data-processing)
8. [Generating Refined Examples](#generating-refined-examples)
9. [Preparing the Training Data](#preparing-the-training-data)
10. [Fine-Tuning the Model](#fine-tuning-the-model)
11. [Using the Fine-Tuned Model](#using-the-fine-tuned-model)
12. [Troubleshooting Common Errors](#troubleshooting-common-errors)
13. [Evaluating the Model](#evaluating-the-model)
14. [Conclusion](#conclusion)
15. [References](#references)
16. [Appendix: Understanding the Codebase](#appendix-understanding-the-codebase)

---

## Introduction

Welcome to this comprehensive guide on fine-tuning a language model specifically for the NEAR ecosystem. This tutorial is designed to educate developers who are eager to delve into the world of fine-tuning Large Language Models (LLMs). By the end of this guide, you'll have a deep understanding of the fine-tuning process, from data collection to deploying your customized model.

## Prerequisites

Before diving in, ensure you have the following:

- **Programming Knowledge**: Basic understanding of Python programming.
- **Familiarity with LLMs**: General knowledge of Large Language Models and their applications.
- **Development Environment**:
  - A computer with internet connectivity.
  - Permissions to install software.
  - Python 3.8 or higher installed.
- **Accounts and API Keys**:
  - [OpenAI account](https://platform.openai.com/signup) with API access.
  - [GitHub account](https://github.com/signup) (optional, for accessing private repositories).
- **Environment Setup**:
  - Familiarity with using virtual environments in Python.

## Understanding Fine-Tuning

### What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained language model and further training it on a custom dataset to specialize it for specific tasks or domains. This allows the model to generate more relevant and accurate responses in the desired context.

### Why Fine-Tune a Model?

- **Domain Specificity**: Tailor the model to understand and generate content related to the NEAR codebase.
- **Improved Performance**: Enhance the accuracy and relevance of responses for NEAR-related queries.
- **Customization**: Implement specific styles, terminologies, or formats required by your application.

### OpenAI's Approach to Fine-Tuning

OpenAI provides APIs to fine-tune models like `gpt-4o`, enabling developers to customize models for their specific needs while leveraging the robust capabilities of large pre-trained models.

---

## Setup and Installation

### 1. Clone the Repository

Start by cloning the repository containing the fine-tuning codebase:

```bash
git clone https://github.com/jbarnes850/near-fine-tuning-job.git
cd near-fine-tuning-job
```

### 2. Set Up a Virtual Environment

Creating a virtual environment is a best practice to manage dependencies:

```bash
python -m venv near_env
source near_env/bin/activate  # On Windows, use `near_env\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries, including OpenAI's Python SDK, `tiktoken`, and others.

## Configuring the Project

### 1. Set Up Environment Variables

Create a `.env` file in the project root directory and add your API keys:

```plaintext
OPENAI_API_KEY=your_openai_api_key
GITHUB_API_KEY=your_github_api_key  # Required if accessing private repositories
```

Alternatively, you can export them in your shell:

```bash
export OPENAI_API_KEY='your_openai_api_key'
export GITHUB_API_KEY='your_github_api_key'
```

### 2. Update the Configuration File

Modify `config.yaml` to customize the fine-tuning process. Key sections include:

#### **Repositories to Fetch Data From**

```yaml
github:
  repos:
    - "near/docs"
    - "near/neps"
    # Add more repositories as needed
```

#### **Articles to Fetch Data From**

```yaml
articles:
  urls:
    - "https://near.org/blog/near-protocol-economics/"
    - "https://near.org/blog/understanding-nears-nightshade-sharding-design/"
    # Add more articles as needed
```

#### **OpenAI API Settings**

```yaml
openai:
  model: "gpt-4o-mini-2024-07-18"  # Ensure this is a model that supports fine-tuning
  temperature: 0.7
  max_tokens: 1000
  system_prompt: "You are an AI assistant specializing in NEAR Protocol and blockchain technology."
```

#### **Fine-Tuning Settings**

```yaml
fine_tuning:
  model: "gpt-4o-2024-08-06"
  n_epochs: 4
  suffix: "NEAR_Ecosystem_Model"
  monitoring_interval: 60  # In seconds
```

---

## Data Collection

The first step in fine-tuning is collecting relevant data.

### 1. Collecting GitHub Data

We fetch code and documentation from specified GitHub repositories.

- **File**: `fine_tuning/data_fetchers.py`
- **Function**: `fetch_repo_data`

```python
def fetch_repo_data(self, repo_name):
    """Fetch data from a GitHub repository."""
    # Check if cached data exists
    if self.use_cache and self.is_data_cached(repo_name, is_repo=True):
        logging.info(f"Using cached data for repository: {repo_name}")
        data = self.load_cached_data(repo_name, is_repo=True)
    else:
        logging.info(f"Fetching repository data: {repo_name}")
        # Fetch data from GitHub
        data = self._fetch_repo_contents(repo_name)
        self.cache_data(repo_name, data, is_repo=True)
    return data
```

### 2. Collecting Article Data

We scrape content from specified web articles.

- **File**: `fine_tuning/data_fetchers.py`
- **Function**: `fetch_article_data`

```python
def fetch_article_data(self, url):
    """Fetch data from a web article."""
    # Check if cached data exists
    if self.use_cache and self.is_data_cached(url, is_repo=False):
        logging.info(f"Using cached data for article: {url}")
        data = self.load_cached_data(url, is_repo=False)
    else:
        logging.info(f"Fetching article data from: {url}")
        # Fetch data from the web
        data = self._fetch_web_content(url)
        self.cache_data(url, data, is_repo=False)
    return data
```

---

## Data Processing

After collecting data, we need to process it into a suitable format for fine-tuning.

### 1. Processing Repository Data

We split code files into manageable chunks.

- **File**: `fine_tuning/data_processors.py`
- **Function**: `process_repo_data`

```python
def process_repo_data(self, all_repo_data):
    """Process data from multiple repositories into prompts."""
    processed_data = []
    for repo_name, repo_files in all_repo_data.items():
        for file_path, content in repo_files.items():
            # Skip binary files or files that are too large
            if self.is_binary_file(content) or self.is_large_file(content):
                continue
            splits = self.split_content(content, self.config['openai']['max_tokens'])
            for split_content in splits:
                prompt = f"Explain the following code snippet from `{file_path}` in the `{repo_name}` repository:\n```{split_content}```"
                processed_data.append({'prompt': prompt})
    return processed_data
```

### 2. Processing Article Data

We split articles into sections.

- **File**: `fine_tuning/data_processors.py`
- **Function**: `process_article_data`

```python
def process_article_data(self, all_article_data):
    """Process data from multiple articles into prompts."""
    processed_data = []
    for url, content in all_article_data.items():
        splits = self.split_content(content, self.config['openai']['max_tokens'])
        for split_content in splits:
            prompt = f"Summarize the following section from the article at {url}:\n{split_content}"
            processed_data.append({'prompt': prompt})
    return processed_data
```

### 3. Splitting Content

We ensure that content chunks do not exceed token limits.

```python
def split_content(self, content, max_tokens):
    """Split content into chunks based on token limits."""
    encoding = get_encoding('cl100k_base')
    tokens = encoding.encode(content)
    splits = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        split_tokens = tokens[start:end]
        split_content = encoding.decode(split_tokens)
        splits.append(split_content)
        start = end
    return splits
```

## Generating Refined Examples

We use an OpenAI model to generate refined examples.

- **Purpose**: Create high-quality question-answer pairs for fine-tuning.
- **Process**:
  - Use prompts from processed data.
  - Generate assistant responses.

### Code Snippet

- **File**: `fine_tuning/data_processors.py`
- **Function**: `generate_refined_examples`

```python
def generate_refined_examples(self, processed_data):
    """Generate assistant responses for each prompt using OpenAI API."""
    refined_examples = []
    for data in tqdm(processed_data, desc="Generating refined examples"):
        messages = [
            {"role": "system", "content": self.config['openai']['system_prompt']},
            {"role": "user", "content": data['prompt']}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=self.config['openai']['model'],
                messages=messages,
                temperature=self.config['openai']['temperature'],
                max_tokens=self.config['openai']['max_tokens']
            )
            assistant_message = response.choices[0].message.content
            refined_examples.append({
                "messages": [
                    {"role": "user", "content": data['prompt']},
                    {"role": "assistant", "content": assistant_message}
                ]
            })
        except Exception as e:
            logging.error(f"Failed to generate response for prompt: {data['prompt']}\nError: {e}")
    return refined_examples
```

**Note**: Ensure that you have correctly imported and initialized the `openai` module:

```python
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
```

## Preparing the Training Data

We prepare the data in the format required by OpenAI's fine-tuning API.

### 1. Validating Examples

Ensure each example meets the required format.

```python
def validate_example(self, example):
    """Validate the structure of a training example."""
    required_keys = {'messages'}
    if not isinstance(example, dict):
        return False
    if not required_keys.issubset(example.keys()):
        return False
    if not isinstance(example['messages'], list):
        return False
    for message in example['messages']:
        if 'role' not in message or 'content' not in message:
            return False
        if message['role'] not in ['system', 'user', 'assistant']:
            return False
        if not isinstance(message['content'], str) or not message['content'].strip():
            return False
    return True
```

### 2. Saving Data as JSONL

We save the validated examples in a `.jsonl` file.

```python
def save_as_jsonl(self, data, output_file="fine_tuning_data.jsonl"):
    """Save data to a JSONL file with UTF-8 encoding and proper escaping."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    logging.info(f"Fine-tuning data saved to {output_file}")
```

## Fine-Tuning the Model

We upload the training data and start the fine-tuning job.

### 1. Uploading the Training File

- **File**: `fine_tuning/fine_tuning.py`
- **Function**: `upload_training_file`

```python
def upload_training_file(self, file_path):
    """Upload the training file to OpenAI with validation."""
    with open(file_path, 'rb') as f:
        response = openai.File.create(
            file=f,
            purpose='fine-tune'
        )
    file_id = response['id']
    logging.info(f"Training file uploaded successfully. File ID: {file_id}")
    return file_id
```

### 2. Creating the Fine-Tuning Job

- **File**: `fine_tuning/fine_tuning.py`
- **Function**: `create_fine_tune_job`

```python
def create_fine_tune_job(self, training_file_id):
    """Create a fine-tuning job in OpenAI."""
    response = openai.FineTune.create(
        training_file=training_file_id,
        model=self.config['fine_tuning']['model'],
        n_epochs=self.config['fine_tuning']['n_epochs'],
        suffix=self.config['fine_tuning'].get('suffix', '')
    )
    job_id = response['id']
    logging.info(f"Fine-tuning job created successfully. Job ID: {job_id}")
    return job_id
```

### 3. Monitoring the Fine-Tuning Job

- **File**: `fine_tuning/fine_tuning.py`
- **Function**: `monitor_fine_tune_job`

```python
def monitor_fine_tune_job(self, job_id):
    """Monitor the fine-tuning job until completion."""
    logging.info(f"Monitoring fine-tuning job: {job_id}")
    while True:
        try:
            response = openai.FineTune.retrieve(job_id)
            status = response['status']
            logging.info(f"Job status: {status}")
            if status == 'succeeded':
                model_id = response['fine_tuned_model']
                logging.info(f"Fine-tuning succeeded. Fine-tuned model ID: {model_id}")
                return model_id
            elif status in ['failed', 'cancelled']:
                error_message = response.get('status_details', 'No details provided.')
                logging.error(f"Fine-tuning {status}. Reason: {error_message}")
                return None
            else:
                time.sleep(self.config['fine_tuning'].get('monitoring_interval', 60))
        except openai.error.OpenAIError as e:
            logging.error(f"Error while checking fine-tuning job status: {e}")
            time.sleep(self.config['fine_tuning'].get('monitoring_interval', 60))
```

### 4. Running the Fine-Tuning Script

- **File**: `fine_tuning/main.py`

```python
if __name__ == "__main__":
    main()
```

Run the script:

```bash
python -m fine_tuning.main
```

## Using the Fine-Tuned Model

Once fine-tuning is complete and you have your `model_id`, you can use the fine-tuned model:

```python
import openai

openai.api_key = 'your_openai_api_key'

response = openai.ChatCompletion.create(
    model='your_fine_tuned_model_id',
    messages=[
        {"role": "system", "content": "You are a NEAR Protocol expert."},
        {"role": "user", "content": "Explain NEAR's sharding mechanism."}
    ]
)

print(response.choices[0].message.content)
```

## Troubleshooting Common Errors

Fine-tuning language models can present various challenges. Below are some common errors you might encounter during the fine-tuning process and their solutions.

### 1. Authentication Errors

**Error Message**: `AuthenticationError: Incorrect API key provided`

**Cause**: This error occurs when the OpenAI API key is missing, incorrect, or improperly configured.

**Solution**:

- **Check API Key**: Ensure that your `OPENAI_API_KEY` is correctly set in your environment variables or in the `.env` file.
- **Correct Usage**: Verify you are accessing the API key without quotes if set in the environment variables.
- **Update Configuration**: Make sure the API key is properly loaded in your script (e.g., using `os.getenv("OPENAI_API_KEY")`).

### 2. Insufficient Quota

**Error Message**: `RateLimitError: You exceeded your current quota, please check your plan and billing details.`

**Cause**: This indicates you've exceeded your allocated usage quota for the OpenAI API.

**Solution**:

- **Check Usage Dashboard**: Visit the [OpenAI Usage Dashboard](https://platform.openai.com/account/usage) to monitor your usage.
- **Upgrade Plan**: Consider upgrading your subscription plan for higher quotas.
- **Optimize Requests**: Reduce the number of API calls or optimize your code to make efficient use of the API.

### 3. Invalid Request Error

**Error Message**: `InvalidRequestError: This model does not support fine-tuning.`

**Cause**: Attempting to fine-tune a model that doesn't support fine-tuning.

**Solution**:

- **Supported Models**: Ensure you're using a model that supports fine-tuning, such as `gpt-3.5-turbo`.
- **Update Configuration**: Modify the `model` parameter in your `config.yaml` and code to use a fine-tune-compatible model.

### 4. File Format Issues

**Error Message**: `InvalidRequestError: The file is not formatted correctly.`

**Cause**: The training file is not in the required JSONL format or contains invalid data.

**Solution**:

- **Validate JSONL File**: Check the training file for proper JSON Lines formatting.
- **Use Validators**: Utilize JSONL validators or linters to detect issues in the file.
- **Correct Data Structure**: Ensure each line in the file is a valid JSON object with the required fields.

### 5. Network Errors

**Error Message**: `APIConnectionError: Error communicating with OpenAI`

**Cause**: Network connectivity issues between your environment and the OpenAI API servers.

**Solution**:

- **Check Internet Connection**: Ensure your network connection is stable.
- **Retry Logic**: Implement retry logic with exponential backoff in your API calls.
- **Firewall Settings**: Verify that your firewall or proxy settings are not blocking API requests.

## Evaluating the Model

After fine-tuning, it's essential to evaluate your model to ensure it meets your performance expectations.

### 1. Testing with Domain-Specific Questions

Assess the model's ability to answer questions related to the NEAR Protocol.

**Example**:

```python
import openai

openai.api_key = 'your_openai_api_key'

def ask_near_question(question):
    response = openai.ChatCompletion.create(
        model='your_fine_tuned_model_id',
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Test the model
question = "How does NEAR's consensus mechanism work?"
answer = ask_near_question(question)
print(f"Q: {question}\nA: {answer}")
```

### 2. Measuring Performance Metrics

Consider quantitative metrics to evaluate your model:

- **Response Accuracy**: Manually review responses for correctness.
- **Relevance Score**: Rate how relevant the responses are to the questions asked.
- **Completeness**: Check if the model provides comprehensive answers.

### 3. User Feedback

Gather feedback from end-users or testers:

- **Surveys and Questionnaires**: Collect user opinions on the model's performance.
- **Error Reporting**: Encourage reporting of any incorrect or unsatisfactory responses.

### 4. Benchmarking

Compare the fine-tuned model against the base model:

- **Baseline Comparison**: Use the original `gpt-3.5-turbo` model to answer the same set of questions.
- **Evaluate Improvements**: Identify areas where the fine-tuned model performs better.

## Conclusion

Congratulations on fine-tuning your custom NEAR Protocol language model! This tailored model should provide more accurate and relevant responses for NEAR-related queries, enhancing your applications and user experience.

**Key Takeaways**:

- Fine-tuning allows you to specialize a general-purpose model for specific domains.
- Proper data collection and processing are critical for effective fine-tuning.
- Always evaluate and iterate on your model to maintain and improve performance.

## References

- **OpenAI Fine-Tuning Documentation**: [Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- **NEAR Official Site**: [NEAR.org](https://near.org/)
- **NEAR Documentation**: [NEAR Docs](https://docs.near.org/)
- **GitHub API Documentation**: [GitHub REST API](https://docs.github.com/en/rest)

## Appendix: Understanding the Codebase

For a deeper understanding of the project's structure and components, review the following sections.

### Project Structure

```plaintext
near-fine-tuned-model/
├── fine_tuning/
│   ├── __init__.py
│   ├── data_fetchers.py
│   ├── api_clients.py
│   ├── config.py
│   ├── data_processors.py
│   ├── fine_tuning.py
│   ├── main.py
│   ├── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_file_upload.py
│   ├── test_fine_tuning_creation.py
│   ├── test_job_monitoring.py
│   └── test_model_evaluation.py
├── config.yaml
├── config_template.yaml
├── requirements.txt
├── README.md
├── model_card.md
├── LICENSE
└── .env.example
```

### Module Overview

- **data_fetchers.py**: Retrieves data from GitHub repositories and web articles.
- **data_processors.py**: Processes and cleans the fetched data, prepares prompts.
- **fine_tuning.py**: Handles interactions with the OpenAI API for file upload and fine-tuning jobs.
- **main.py**: The primary script that orchestrates data fetching, processing, and fine-tuning.

### Configuration (`config.yaml`)

The `config.yaml` file contains all configurable parameters:

- **GitHub Repositories**: Specify repositories to fetch data from.
- **Articles**: List of article URLs to include in the dataset.
- **OpenAI Settings**: Model choice, temperature, max tokens, and system prompts.
- **Fine-Tuning Parameters**: Epochs, model suffix, and monitoring intervals.

### Environment Variables (`.env`)

Store sensitive information like API keys:

```plaintext
OPENAI_API_KEY=your_openai_api_key
GITHUB_API_KEY=your_github_api_key
```

**Security Tip**: Never commit the `.env` file to version control systems.

## Additional Resources

- **NEAR Developer Community**: Engage with other developers in the [NEAR Discord Channel](https://discord.gg/nearprotocol).
- **OpenAI Community Forum**: Discuss and seek assistance at the [OpenAI Community](https://community.openai.com/).

---

*This guide is intended to empower developers to harness the capabilities of fine-tuned language models within the NEAR ecosystem. Continue exploring and innovating!*
