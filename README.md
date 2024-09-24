# NEAR Ecosystem Fine-Tuned Model

We're open sourcing a fine-tuned language model that's deeply versed in the NEAR ecosystem, designed specifically for developers like you.

## Key Features and Use Cases

This model is your go-to tool for building AI and Web3 applications within the NEAR ecosystem. Whether you're working on onchain AI agents or generating code, this model has you covered.

### Key Features

- **This model is meticulously fine-tuned for the NEAR ecosystem**. Utilizing the GitHub API, we directly extracted data from NEAR codebases and fine-tuned the model on 50 million tokens and 5,000 example prompts over 4 epochs. This rigorous training process ensures a highly performant and specialized model adept at understanding the intricacies of NEAR.

- **The model excels in generating accurate and efficient code in Rust, TypeScript, and JavaScript**. It adheres to NEAR's coding standards and best practices, making it a valuable tool for developers seeking to produce high-quality code.

- **This model is optimized to enhance the memory and performance of onchain agents**. It aids agents in comprehending and navigating the complexities of the NEAR ecosystem.

- **A comprehensive test suite is included in the repository to benchmark the model's performance**. This ensures the reliability of the fine-tuning components and the model's exceptional performance in real-world scenarios.

- **The model is designed for a wide range of applications, from building AI and Web3 applications to generating code and enhancing onchain agents**.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jbarnes850/near-fine-tuned-model.git
   cd near-fine-tuned-model
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**

   - Create a `.env` file in the project root.
   - Add your GitHub and OpenAI API keys:

     ```plaintext
     GITHUB_API_KEY=your_github_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

To use the NEAR Ecosystem Fine-Tuned Model, follow these steps:

1. **Ensure you have completed the installation steps above.**

2. **Run the fine-tuning script:**

   ```bash
   python -m fine_tuning.main
   ```

   This script will:

   - Fetch data from specified NEAR repositories and articles.
   - Process and refine the data using GPT-4o.
   - Create a JSONL file with the training data.
   - Upload the training file to OpenAI.
   - Start a fine-tuning job.
   - Monitor the job until completion.

3. **Once the fine-tuning is complete, you will receive a fine-tuned model ID.** You can use this ID to make API requests to your specialized NEAR ecosystem model.

4. **To use the fine-tuned model in your applications, use the OpenAI API with the provided model ID:**

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

## Testing

We have included a comprehensive test suite to verify each component of the fine-tuning process and to evaluate the performance of the fine-tuned model against the base model.

### Running Tests

1. **Install Testing Dependencies (if not already installed):**

   ```bash
   pip install -r requirements.txt
   ```

2. **Navigate to the `tests` directory:**

   ```bash
   cd tests
   ```

3. **Run all tests:**

   ```bash
   python -m unittest discover
   ```

   This command will discover and run all test cases in the `tests` directory.

### Test Descriptions

- **File Upload and Processing Tests (`test_file_upload.py`):**

  - Verifies that the training file is uploaded successfully to OpenAI and processed correctly.
  - Tests handling of upload failures and exceptions.

- **Fine-Tuning Job Creation Tests (`test_fine_tune_creation.py`):**

  - Ensures that fine-tuning jobs are created correctly with validated training files.
  - Tests behavior with unprocessed or invalid training files.

- **Job Monitoring Tests (`test_job_monitoring.py`):**

  - Checks that the fine-tuning job status is monitored accurately until completion.
  - Tests handling of different job outcomes (success, failure, cancellation).

- **Model Evaluation Test (`test_model_evaluation.py`):**

  - Compares responses from the fine-tuned model and the base model using a set of evaluation prompts.
  - Helps assess the quality and improvements of the fine-tuned model.

#### Running the Model Evaluation Test

1. **Set Your OpenAI API Key:**

   Ensure your OpenAI API key is set in your environment variables:

   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   ```

2. **Replace the Fine-Tuned Model ID:**

   In `tests/test_model_evaluation.py`, replace `'your_fine_tuned_model_id'` with the actual model ID obtained after fine-tuning.

3. **Run the Model Evaluation Test:**

   ```bash
   python test_model_evaluation.py
   ```

   The script will output the prompts and the corresponding responses from both the base model and the fine-tuned model for manual comparison.

## Data Sources

The NEAR Ecosystem Fine-Tuned Model uses a variety of data sources to ensure comprehensive coverage of the NEAR Protocol ecosystem:

1. **GitHub Repositories:**

   - NEAR Protocol documentation
   - NEAR Enhancement Proposals (NEPs)
   - NEAR node documentation
   - NEAR core implementation
   - NEAR examples and SDKs
   - NEAR tools and utilities

2. **Web Articles:**

   - Official NEAR blog posts
   - Technical updates and announcements
   - Ecosystem news and developments

**The full list of repositories and articles can be found in the `config.yaml` file.**

## Fine-Tuning Process

The fine-tuning process consists of several steps:

1. **Data Collection:**

   - Fetches markdown and code files from specified GitHub repositories.
   - Retrieves content from selected web articles.

2. **Data Processing:**

   - Splits content into manageable chunks.
   - Generates diverse prompts for both repository files and articles.
   - Uses GPT-4o-mini to refine prompts and completions for each data point.

3. **Training Data Creation:**

   - Generates a JSONL file with processed data in the required format for fine-tuning

4. **Fine-Tuning:**

   - Uploads the training data to OpenAI
   - Initiates a fine-tuning job on the GPT-4o model
   - Monitors the progress of the fine-tuning job

5. **Model Deployment:**

   - Upon successful completion, provides a fine-tuned model ID for use in applications

## Model Hosting

The fine-tuned model is hosted on OpenAI's servers and can be accessed through their API. In the future, we plan to make this model available on Hugging Face for easier access and integration.

To use the hosted model, follow the usage instructions provided above.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Jarrod Barnes - <jarrod.barnes@near.foundation>

Project Link: [https://github.com/jbarnes850/near-fine-tuned-model](https://github.com/jbarnes850/near-fine-tuned-model)

For any questions, suggestions, or concerns, please open an issue in the GitHub repository or contact the maintainers directly.
