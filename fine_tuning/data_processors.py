import logging
import os
from fine_tuning.utils import error_handler, num_tokens_from_messages, split_list
from tqdm import tqdm
import random
import json
from tiktoken import get_encoding
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        """Generate assistant responses for each prompt using OpenAI API."""
        refined_examples = []
        random.shuffle(processed_data)  # Shuffle the order of the prompts

        def process_prompt(data):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a highly knowledgeable and helpful assistant specialized in NEAR Protocol development, "
                        "blockchain architecture, and AI technologies within the NEAR ecosystem. "
                        "Your primary tasks include generating accurate and efficient code examples, providing detailed explanations of NEAR's architecture, "
                        "and assisting developers with technical guidance. "
                        "When generating code, use appropriate programming languages such as Rust, TypeScript, and JavaScript. "
                        "Adhere to NEAR's coding standards and best practices, and include comprehensive comments. "
                        "Ensure all code is functional, secure, and optimized for performance. "
                        "Provide clear, concise, and informative responses to enhance developers' understanding and implementation of NEAR technologies. "
                        "NEAR Protocol is a blockchain platform that allows developers to build and deploy smart contracts and decentralized applications (dApps) on its blockchain network. "
                        "Write for a technical audience and prioritize clarity and accuracy in your responses. Developers are your primary users, so ensure your explanations are comprehensive and easy to understand. "
                        "This fine-tuning data will be used to improve the assistant's ability to understand and generate code and explanations related to NEAR. "
                        "Focus on creating concise and informative responses that are both technically accurate and easy to understand. "
                        "Use markdown code blocks to format your responses, and include inline comments to explain your code. "
                        "When providing answers, ensure they are structured in a way that is easy to follow and implement. "
                        "Include examples where applicable to illustrate your points effectively."
                    ),
                },
                {"role": "user", "content": data['prompt']}
            ]
            try:
                response = self.client.chat.completions.create(
                    model=self.config['openai']['model'],
                    messages=messages,
                    temperature=self.config['openai']['temperature'],
                    max_tokens=self.config['openai']['max_tokens']
                )
                assistant_message = response.choices[0].message['content']
                return {
                    "messages": [
                        {"role": "user", "content": data['prompt']},
                        {"role": "assistant", "content": assistant_message}
                    ]
                }
            except Exception as e:
                logging.error(f"Failed to generate response for prompt: {data['prompt']}\nError: {e}")
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_prompt, data) for data in processed_data]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating refined examples"):
                result = future.result()
                if result:
                    refined_examples.append(result)

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