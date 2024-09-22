import unittest
from unittest.mock import patch, MagicMock
import openai
from fine_tuning.fine_tuning import FineTuner
from fine_tuning.config import load_config

class TestJobMonitoring(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config.yaml')
        self.fine_tuner = FineTuner(self.config)
        self.job_id = 'ft-job-123'

    @patch('time.sleep', return_value=None)
    @patch('openai.resources.FineTunes.retrieve')
    def test_monitor_fine_tune_job_success(self, mock_fine_tune_retrieve, mock_sleep):
        # Mock the sequence of status responses
        mock_fine_tune_retrieve.side_effect = [
            {'id': self.job_id, 'status': 'pending'},
            {'id': self.job_id, 'status': 'running'},
            {'id': self.job_id, 'status': 'succeeded', 'fine_tuned_model': 'ft-model-abc123'}
        ]

        model_id = self.fine_tuner.monitor_fine_tune_job(self.job_id)

        # Assert that the returned model_id is correct
        self.assertEqual(model_id, 'ft-model-abc123')

    @patch('time.sleep', return_value=None)
    @patch('openai.resources.FineTunes.retrieve')
    def test_monitor_fine_tune_job_failure(self, mock_fine_tune_retrieve, mock_sleep):
        # Mock the sequence to immediately return a failed status
        mock_fine_tune_retrieve.return_value = {'id': self.job_id, 'status': 'failed', 'status_details': 'An error occurred'}

        model_id = self.fine_tuner.monitor_fine_tune_job(self.job_id)

        # Assert that the model_id is None due to failure
        self.assertIsNone(model_id)

    @patch('time.sleep', return_value=None)
    @patch('openai.resources.FineTunes.retrieve')
    def test_monitor_fine_tune_job_cancelled(self, mock_fine_tune_retrieve, mock_sleep):
        # Mock the sequence to return a cancelled status
        mock_fine_tune_retrieve.return_value = {'id': self.job_id, 'status': 'cancelled'}

        model_id = self.fine_tuner.monitor_fine_tune_job(self.job_id)

        # Assert that the model_id is None due to cancellation
        self.assertIsNone(model_id)

if __name__ == '__main__':
    unittest.main()