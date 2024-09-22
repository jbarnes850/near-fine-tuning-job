import logging
from utils import error_handler
import time
import openai

class FineTuner:
    def __init__(self, openai_client, config):
        self.client = openai_client
        self.config = config

    @error_handler
    def upload_training_file(self, file_path):
        """Upload the training file to OpenAI."""
        logging.info(f"Uploading training file: {file_path}")
        with open(file_path, 'rb') as f:
            response = openai.File.create(
                file=f,
                purpose='fine-tune'
            )
        file_id = response['id']
        logging.info(f"Training file uploaded. File ID: {file_id}")
        return file_id

    @error_handler
    def create_fine_tune_job(self, training_file_id):
        """Create a fine-tuning job using the specified model."""
        logging.info("Creating fine-tuning job...")
        response = openai.FineTuningJob.create(
            training_file=training_file_id,
            model=self.config['fine_tuning']['model'],
            n_epochs=self.config['fine_tuning']['n_epochs'],
            learning_rate_multiplier=self.config['fine_tuning']['learning_rate_multiplier'],
            prompt_loss_weight=self.config['fine_tuning']['prompt_loss_weight']
        )
        job_id = response['id']
        logging.info(f"Fine-tuning job created. Job ID: {job_id}")
        return job_id

    @error_handler
    def monitor_fine_tune_job(self, job_id):
        """Monitor the fine-tuning job until completion."""
        logging.info(f"Monitoring fine-tuning job: {job_id}")
        status = ''
        while status not in ['succeeded', 'failed', 'cancelled']:
            response = openai.FineTuningJob.retrieve(id=job_id)
            status = response['status']
            logging.info(f"Fine-tuning status: {status}")
            if status == 'succeeded':
                model_id = response['fine_tuned_model']
                logging.info(f"Fine-tuning succeeded. Model ID: {model_id}")
                return model_id
            elif status == 'failed':
                logging.error("Fine-tuning failed.")
                return None
            elif status == 'cancelled':
                logging.error("Fine-tuning was cancelled.")
                return None
            else:
                time.sleep(self.config['fine_tuning'].get('monitoring_interval', 60))
        return None