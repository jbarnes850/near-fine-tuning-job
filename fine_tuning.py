import logging
import os
import time
import openai
from utils import error_handler

class FineTuner:
    def __init__(self, config):
        self.config = config

    @error_handler
    def upload_training_file(self, file_path):
        """Upload the training file to OpenAI with validation."""
        logging.info(f"Uploading training file: {file_path}")

        # Validate the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file {file_path} does not exist.")

        # Validate the file size (less than 1GB per OpenAI limits)
        file_size = os.path.getsize(file_path)
        if file_size > 1e9:
            raise ValueError(f"Training file {file_path} exceeds the 1GB size limit.")

        # Attempt to upload the file
        try:
            with open(file_path, 'rb') as f:
                response = openai.File.create(
                    file=f,
                    purpose='fine-tune'
                )
            file_id = response['id']
            logging.info(f"Training file uploaded successfully. File ID: {file_id}")
        except openai.error.OpenAIError as e:
            logging.error(f"Failed to upload training file: {e}")
            raise

        # Wait until the file status is 'processed'
        max_wait_time = 600  # Maximum wait time in seconds (10 minutes)
        wait_interval = 10   # Wait interval in seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            file_info = openai.File.retrieve(file_id)
            status = file_info['status']
            if status == 'processed':
                logging.info(f"Training file {file_id} is processed and ready.")
                return file_id
            elif status == 'failed':
                error_message = file_info.get('status_details', 'Unknown error')
                raise ValueError(f"File processing failed: {error_message}")
            else:
                logging.info(f"Waiting for training file {file_id} to be processed... Status: {status}")
                time.sleep(wait_interval)
                elapsed_time += wait_interval

        raise TimeoutError(f"Training file {file_id} was not processed within the expected time.")

    @error_handler
    def create_fine_tune_job(self, training_file_id):
        """Create a fine-tuning job using the specified model with validation."""
        logging.info("Creating fine-tuning job...")

        # Validate that the training_file_id is valid and processed
        file_info = openai.File.retrieve(training_file_id)
        if file_info['status'] != 'processed':
            raise ValueError(f"Training file {training_file_id} is not ready. Status: {file_info['status']}")

        # Check if the model is available for fine-tuning
        # As of now, OpenAI allows fine-tuning on specific models
        allowed_models = ["davinci", "curie", "babbage", "ada", "gpt-3.5-turbo"]
        model = self.config['fine_tuning']['model']
        if model not in allowed_models:
            raise ValueError(f"The model '{model}' is not available for fine-tuning. Allowed models: {allowed_models}")

        # Create the fine-tuning job
        try:
            response = openai.FineTune.create(
                training_file=training_file_id,
                model=model,
                n_epochs=self.config['fine_tuning']['n_epochs'],
                learning_rate_multiplier=self.config['fine_tuning']['learning_rate_multiplier'],
                prompt_loss_weight=self.config['fine_tuning']['prompt_loss_weight'],
                suffix=self.config['fine_tuning'].get('suffix', '')
            )
            job_id = response['id']
            logging.info(f"Fine-tuning job created successfully. Job ID: {job_id}")
            return job_id
        except openai.error.OpenAIError as e:
            logging.error(f"Failed to create fine-tuning job: {e}")
            raise

    @error_handler
    def monitor_fine_tune_job(self, job_id):
        """Monitor the fine-tuning job until completion."""
        logging.info(f"Monitoring fine-tuning job: {job_id}")
        status = ''
        while status not in ['succeeded', 'failed', 'cancelled']:
            try:
                response = openai.FineTune.retrieve(id=job_id)
                status = response['status']
                logging.info(f"Fine-tuning status: {status}")
                if status == 'succeeded':
                    model_id = response['fine_tuned_model']
                    logging.info(f"Fine-tuning succeeded. Fine-tuned model ID: {model_id}")
                    return model_id
                elif status == 'failed':
                    error_message = response.get('status_details', 'Unknown error')
                    logging.error(f"Fine-tuning failed. Reason: {error_message}")
                    return None
                elif status == 'cancelled':
                    logging.error("Fine-tuning was cancelled.")
                    return None
                else:
                    time.sleep(self.config['fine_tuning'].get('monitoring_interval', 60))
            except openai.error.OpenAIError as e:
                logging.error(f"Error while checking fine-tuning job status: {e}")
                time.sleep(self.config['fine_tuning'].get('monitoring_interval', 60))
        return None