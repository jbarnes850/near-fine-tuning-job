import logging
from fine_tuning.utils import error_handler, num_tokens_from_messages, split_list
from tqdm import tqdm
import random
import json
from tiktoken import get_encoding

class DataProcessor:
    def __init__(self, openai_client, config):
        self.client = openai_client
        self.config = config

    def process_repo_data(self, repo_data):
        """Process repository data into prompts."""
        processed_data = []
        for file_path, content in repo_data:
            splits = self.split_content(content, self.config['data_processing']['max_tokens'])
            for split_content in splits:
                prompt = f"Explain the following code snippet from NEAR repository file `{file_path}`:\n```{split_content}```"
                processed_data.append({'prompt': prompt, 'completion': ''})
        return processed_data

    def process_article_data(self, article_text):
        """Process article data into prompts."""
        splits = self.split_content(article_text, self.config['data_processing']['max_tokens'])
        processed_data = []
        for split_content in splits:
            prompt = f"Summarize the following section of a NEAR Protocol article:\n{split_content}"
            processed_data.append({'prompt': prompt, 'completion': ''})
        return processed_data

    def split_content(self, content, max_tokens):
        """Split content into chunks no longer than max_tokens."""
        encoding = get_encoding('cl100k_base')
        tokens = encoding.encode(content)
        splits = []
        current_chunk = []
        for token in tokens:
            current_chunk.append(token)
            if len(current_chunk) >= max_tokens:
                chunk_text = encoding.decode(current_chunk)
                splits.append(chunk_text)
                current_chunk = []
        if current_chunk:
            chunk_text = encoding.decode(current_chunk)
            splits.append(chunk_text)
        return splits

    @error_handler
    def generate_refined_examples(self, processed_data):
        """Use GPT-4o-mini to refine prompts and generate completions."""
        refined_examples = []
        batch_size = self.config['example_generation']['batch_size']
        batches = list(split_list(processed_data, batch_size))
        for batch in tqdm(batches, desc="Generating refined examples"):
            messages = [
                {"role": "system", "content": "You are an AI assistant specializing in NEAR Protocol and blockchain technology. Your task is to refine prompts and generate detailed completions for fine-tuning a language model."}
            ]
            for item in batch:
                messages.append({"role": "user", "content": item['prompt']})

            response = self.client.create_chat_completion(
                model=self.config['openai']['model'],
                messages=messages,
                temperature=self.config['openai']['temperature'],
                max_tokens=self.config['openai']['max_tokens']
            )
            assistant_message = response['choices'][0]['message']['content']

            # Process assistant's response
            # Assuming the assistant returns structured data
            prompts_and_completions = self.parse_assistant_response(assistant_message)
            refined_examples.extend(prompts_and_completions)
        return refined_examples

    def parse_assistant_response(self, response_content):
        """Parse the assistant's response into prompts and completions."""
        examples = []
        pairs = response_content.strip().split('\n\n')
        for pair in pairs:
            if 'Prompt:' in pair and 'Completion:' in pair:
                prompt_part, completion_part = pair.split('Completion:')
                prompt = prompt_part.replace('Prompt:', '').strip()
                completion = completion_part.strip()
                examples.append({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion}
                    ]
                })
        return examples

    def create_fine_tuning_data(self, refined_examples):
        """Prepare the fine-tuning data."""
        fine_tuning_data = []
        total_tokens = 0
        target_examples = self.config['fine_tuning']['target_examples']
        max_tokens = self.config['fine_tuning']['max_tokens']
        for example in refined_examples:
            num_tokens = num_tokens_from_messages(example['messages'])
            if total_tokens + num_tokens > max_tokens:
                break
            # Validate the example
            if not self.validate_example(example):
                logging.warning("Invalid example detected and skipped.")
                continue
            fine_tuning_data.append(example)
            total_tokens += num_tokens
            if len(fine_tuning_data) >= target_examples:
                break
        logging.info(f"Total examples for fine-tuning: {len(fine_tuning_data)}")
        logging.info(f"Total tokens: {total_tokens}")
        return fine_tuning_data

    def validate_example(self, example):
        """Validate a single example to ensure it meets OpenAI's requirements."""
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

    def save_as_jsonl(self, data, output_file="fine_tuning_data.jsonl"):
        """Save data to a JSONL file with UTF-8 encoding and proper escaping."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
        logging.info(f"Fine-tuning data saved to {output_file}")