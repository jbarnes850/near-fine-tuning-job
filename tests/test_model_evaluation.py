import os
import unittest
import openai
from config import load_config

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # Load configuration and API keys
        self.config = load_config('config.yaml')
        openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure the API key is securely loaded

        self.base_model = self.config['openai']['model']
        self.fine_tuned_model = 'your_fine_tuned_model_id'  # Replace with your actual fine-tuned model ID

        # Define evaluation prompts relevant to NEAR Protocol
        self.prompts = [
            "Explain NEAR Protocol's sharding mechanism.",
            "How does NEAR handle transaction fees?",
            "Describe the role of validators in NEAR.",
            "What is the NEAR Rainbow Bridge?",
            "How does NEAR differ from other blockchain platforms?",
            # Add more prompts as needed
        ]

    def test_model_responses(self):
        for prompt in self.prompts:
            base_response = self.get_response(prompt, self.base_model)
            fine_tuned_response = self.get_response(prompt, self.fine_tuned_model)

            # For simplicity, we will print out the responses for manual evaluation
            print(f"Prompt: {prompt}\n")
            print(f"Base Model Response:\n{base_response}\n")
            print(f"Fine-tuned Model Response:\n{fine_tuned_response}\n")
            print("=" * 80 + "\n")

            # Optionally, implement automated evaluation metrics here
            # For example, compare responses to expected answers

    def get_response(self, prompt, model):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a NEAR Protocol expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['openai'].get('temperature', 0.7),
                max_tokens=self.config['openai'].get('max_tokens', 512),
                top_p=self.config['openai'].get('top_p', 1),
                frequency_penalty=self.config['openai'].get('frequency_penalty', 0),
                presence_penalty=self.config['openai'].get('presence_penalty', 0)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

if __name__ == '__main__':
    unittest.main()