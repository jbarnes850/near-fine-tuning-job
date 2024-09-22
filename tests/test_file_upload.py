import unittest
from unittest.mock import patch, MagicMock
import openai
from fine_tuning import FineTuner
from config import load_config

class TestFileUpload(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config.yaml')
        self.fine_tuner = FineTuner(self.config)
        self.test_file_path = 'fine_tuning_data.jsonl'

    @patch('openai.File.create')
    @patch('openai.File.retrieve')
    def test_upload_training_file(self, mock_file_retrieve, mock_file_create):
        # Mock the file upload response
        mock_file_create.return_value = {'id': 'file-abc123', 'object': 'file'}
        # Mock the file status to 'processed'
        mock_file_retrieve.return_value = {'id': 'file-abc123', 'status': 'processed'}

        # Test the upload_training_file method
        file_id = self.fine_tuner.upload_training_file(self.test_file_path)

        # Assert that the file_id is correct
        self.assertEqual(file_id, 'file-abc123')

        # Assert that the OpenAI File.create was called with correct parameters
        mock_file_create.assert_called_once()
        # Assert that OpenAI File.retrieve was called to check status
        mock_file_retrieve.assert_called()

    @patch('openai.File.create')
    def test_upload_training_file_failure(self, mock_file_create):
        # Mock the file upload to raise an OpenAIError
        mock_file_create.side_effect = openai.error.OpenAIError('Upload failed')

        with self.assertRaises(openai.error.OpenAIError):
            self.fine_tuner.upload_training_file(self.test_file_path)

if __name__ == '__main__':
    unittest.main()