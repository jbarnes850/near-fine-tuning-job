# NEAR Ecosystem Fine-Tuned Model

We're open sourcing a fine-tuned language model that's deeply versed in the NEAR ecosystem, designed specifically for developers like you.

## Key Features

- Incorporates up-to-date data from NEAR repositories and ecosystem updates
- Optimized for understanding and generating NEAR-related code and content
- Uses content splitting for better processing of large documents
- Generates diverse prompts for both repository files and articles
- Leverages GPT-4o and GPT-4o-mini for data refinement and fine-tuning

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

   - Create a `.env` file in the project root
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
   python main.py
   ```

   This script will:

   - Fetch data from specified NEAR repositories and articles
   - Process and refine the data using GPT-4o
   - Create a JSONL file with the training data
   - Upload the training file to OpenAI
   - Start a fine-tuning job
   - Monitor the job until completion

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

   - Fetches markdown and code files from specified GitHub repositories
   - Retrieves content from selected web articles

2. **Data Processing:**

   - Splits content into manageable chunks
   - Generates diverse prompts for both repository files and articles
   - Uses GPT-4o-mini to refine prompts and completions for each data point

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
