import unittest
from unittest.mock import patch, MagicMock
import openai
from fine_tuning.fine_tuning import FineTuner
from fine_tuning.config import load_config

class TestFineTuneCreation(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config.yaml')
        self.fine_tuner = FineTuner(self.config)
        self.training_file_id = 'file-abc123'

    @patch('openai.File.retrieve')
    @patch('openai.FineTune.create')
    def test_create_fine_tune_job(self, mock_fine_tune_create, mock_file_retrieve):
        # Mock the file retrieve response to indicate 'processed'
        mock_file_retrieve.return_value = {'id': self.training_file_id, 'status': 'processed'}

        # Mock the fine-tuning job creation response
        mock_fine_tune_create.return_value = {'id': 'ft-job-123', 'status': 'created'}

        # Test the create_fine_tune_job method
        job_id = self.fine_tuner.create_fine_tune_job(self.training_file_id)

        # Assert that the job_id is correct
        self.assertEqual(job_id, 'ft-job-123')

        # Assert that OpenAI FineTune.create was called with correct parameters
        mock_fine_tune_create.assert_called_once()
        # Assert that OpenAI File.retrieve was called to check file status
        mock_file_retrieve.assert_called_once_with(self.training_file_id)

    @patch('openai.File.retrieve')
    def test_create_fine_tune_job_with_unprocessed_file(self, mock_file_retrieve):
        # Mock the file retrieve response to indicate file is not processed yet
        mock_file_retrieve.return_value = {'id': self.training_file_id, 'status': 'uploaded'}

        with self.assertRaises(ValueError):
            self.fine_tuner.create_fine_tune_job(self.training_file_id)

if __name__ == '__main__':
    unittest.main()