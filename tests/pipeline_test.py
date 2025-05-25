import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import argparse
from io import StringIO
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pipeline

class TestPipeline(unittest.TestCase):
    """
    Test suite for the pipeline module that handles dataset creation and model training.
    Each test case corresponds to a specific parameter combination from the test frame.
    """
    
    def setUp(self):
        """Set up test environment before each test case."""
        # Common mock objects used across tests
        self.mock_model_selection = MagicMock()
        self.mock_dataset_creation = MagicMock()
        self.mock_model_training = MagicMock()
        
        # This is the key fix: we need to make sure the selection method returns the expected tuple
        self.mock_placeholder_manager = MagicMock()
        self.mock_placeholder_manager.selection.return_value = ("placeholder_user", False)
        
        self.mock_logger_report = MagicMock()
        self.mock_logger_user = MagicMock()
        self.mock_logger_user_model_history = MagicMock()
        
        # Set up common return values
        self.mock_model_selection.model = "mock_model"
        self.mock_dataset_creation.dataFrame = "mock_dataframe"
        self.mock_dataset_creation.run = MagicMock()
        
        # Save original sys.argv
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up after each test case."""
        # Restore original sys.argv
        sys.argv = self.original_argv

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_pc1_default_output_name_valid_config(self, mock_path_exists,
                                                mock_logger_history, mock_logger_user, 
                                                mock_logger_report, mock_placeholder, 
                                                mock_training, mock_dataset, mock_model):
        """
        -- TEST PC1 --
        Test pipeline with default output name, valid config file, no retrain,
        with or without alias file, with or without refactor.
        """
        # Setup mocks
        mock_path_exists.return_value = True
        mock_model.return_value = self.mock_model_selection
        mock_dataset.return_value = self.mock_dataset_creation
        mock_training.return_value = self.mock_model_training
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        mock_logger_history.append_model_user = MagicMock()
        
        # Set command line arguments - must be done after the parser is created
        with patch('sys.argv', ['pipeline.py', '-c', 'valid_config.json']):
            # Run the pipeline
            pipeline.create_dataset_and_train_model()
        
        # Assert that the pipeline components were called with expected parameters
        mock_dataset.assert_called_once()
        mock_training.assert_called_once()
        # Check if retrain is False in the ModelTraining call
        self.assertFalse(mock_training.call_args[0][4])  # retrain should be False

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_pc2_custom_output_name_valid_config(self, mock_path_exists,
                                               mock_logger_history, mock_logger_user, 
                                               mock_logger_report, mock_placeholder, 
                                               mock_training, mock_dataset, mock_model):
        """
        -- TEST PC2 --
        Test pipeline with custom output name "custom_model.joblib", valid config file,
        no retrain, with or without alias file, with or without refactor.
        """
        # Setup mocks
        mock_path_exists.return_value = True
        mock_model.return_value = self.mock_model_selection
        mock_dataset.return_value = self.mock_dataset_creation
        mock_training.return_value = self.mock_model_training
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        mock_logger_history.append_model_user = MagicMock()
        
        # Set command line arguments
        with patch('sys.argv', ['pipeline.py', '-oN', 'custom_model.joblib', '-c', 'valid_config.json']):
            # Run the pipeline
            pipeline.create_dataset_and_train_model()
        
        # Assert that the pipeline components were called with expected parameters
        mock_dataset.assert_called_once()
        mock_training.assert_called_once()
        
        # Check output name is properly passed to ModelTraining
        output_name_arg = mock_training.call_args[0][0]
        self.assertEqual(output_name_arg, 'custom_model.joblib')
        self.assertFalse(mock_training.call_args[0][4])  # retrain should be False

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_pc3_default_output_name_retrain_enabled(self, mock_path_exists,
                                                   mock_logger_history, mock_logger_user, 
                                                   mock_logger_report, mock_placeholder, 
                                                   mock_training, mock_dataset, mock_model):
        """
        -- TEST PC3 --
        Test pipeline with default output name, valid config file, retrain enabled,
        with or without alias file, with or without refactor.
        """
        # Setup mocks
        mock_path_exists.return_value = True
        mock_model.return_value = self.mock_model_selection
        mock_dataset.return_value = self.mock_dataset_creation
        mock_training.return_value = self.mock_model_training
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        mock_logger_history.append_model_user = MagicMock()
        
        # Set command line arguments
        with patch('sys.argv', ['pipeline.py', '-c', 'valid_config.json', '-r']):
            # Run the pipeline
            pipeline.create_dataset_and_train_model()
        
        # Assert that the pipeline components were called with expected parameters
        mock_dataset.assert_called_once()
        mock_training.assert_called_once()
        # Check if retrain is True in the ModelTraining call
        self.assertTrue(mock_training.call_args[0][4])  # retrain should be True

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_pc4_custom_output_name_retrain_enabled(self, mock_path_exists,
                                                  mock_logger_history, mock_logger_user, 
                                                  mock_logger_report, mock_placeholder, 
                                                  mock_training, mock_dataset, mock_model):
        """
        -- TEST PC4 --
        Test pipeline with custom output name "custom_model.joblib", valid config file,
        retrain enabled, with or without alias file, with or without refactor.
        """
        # Setup mocks
        mock_path_exists.return_value = True
        mock_model.return_value = self.mock_model_selection
        mock_dataset.return_value = self.mock_dataset_creation
        mock_training.return_value = self.mock_model_training
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        mock_logger_history.append_model_user = MagicMock()
        
        # Set command line arguments
        with patch('sys.argv', ['pipeline.py', '-oN', 'custom_model.joblib', '-c', 'valid_config.json', '-r']):
            # Run the pipeline
            pipeline.create_dataset_and_train_model()
        
        # Assert that the pipeline components were called with expected parameters
        mock_dataset.assert_called_once()
        mock_training.assert_called_once()
        self.assertEqual(mock_training.call_args[0][0], 'custom_model.joblib')
        self.assertTrue(mock_training.call_args[0][4])  # retrain should be True

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_pc5_invalid_config_file(self, mock_path_exists,
                                    mock_logger_history, mock_logger_user, 
                                    mock_logger_report, mock_placeholder, 
                                    mock_training, mock_dataset, mock_model):
        """
        -- TEST PC5 --
        Test pipeline with default output name, invalid config file path,
        no retrain, with or without alias file, with or without refactor.
        """

        self.expected_output = "File di configurazione non trovato"

        # Setup mocks
        mock_path_exists.return_value = False  # Config file doesn't exist
        mock_model.return_value = self.mock_model_selection
        mock_dataset.side_effect = FileNotFoundError("File di configurazione non trovato")
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        
        # Set command line arguments
        with patch('sys.argv', ['pipeline.py', '-c', 'invalid_config.json']):
            # Run the pipeline and check for exception
            with self.assertRaises(FileNotFoundError) as context:
                pipeline.create_dataset_and_train_model()
            
            self.assertEqual(str(context.exception), "File di configurazione non trovato")

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_pc6_internal_module_error(self, mock_path_exists,
                                      mock_logger_history, mock_logger_user, 
                                      mock_logger_report, mock_placeholder, 
                                      mock_training, mock_dataset, mock_model):
        """
        -- TEST PC6 --
        Test pipeline with default output name, valid config file, no retrain,
        with or without alias file, with or without refactor, but with an internal
        error occurring in one of the modules.
        """

        self.expected_output = "Errore interno nel modulo datasetCreation"

        # Setup mocks
        mock_path_exists.return_value = True
        mock_model.return_value = self.mock_model_selection
        
        # Create mock for dataset with an error in the run method
        dataset_instance = MagicMock()
        dataset_instance.run.side_effect = RuntimeError("Errore interno nel modulo datasetCreation")
        mock_dataset.return_value = dataset_instance
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        
        # Set command line arguments
        with patch('sys.argv', ['pipeline.py', '-c', 'valid_config.json']):
            # Run the pipeline and check for exception
            with self.assertRaises(RuntimeError) as context:
                pipeline.create_dataset_and_train_model()
            
            self.assertEqual(str(context.exception), "Errore interno nel modulo datasetCreation")

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_alias_file_presence(self, mock_path_exists,
                               mock_logger_history, mock_logger_user, 
                               mock_logger_report, mock_placeholder, 
                               mock_training, mock_dataset, mock_model):
        """
        Additional test to verify alias file parameter is properly passed.
        """
        # Setup mocks
        mock_path_exists.return_value = True
        mock_model.return_value = self.mock_model_selection
        mock_dataset.return_value = self.mock_dataset_creation
        mock_training.return_value = self.mock_model_training
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        mock_logger_history.append_model_user = MagicMock()
        
        # Set command line arguments
        with patch('sys.argv', ['pipeline.py', '-c', 'valid_config.json', '-a', 'aliases.json']):
            # Run the pipeline
            pipeline.create_dataset_and_train_model()
        
        # Assert that the PlaceholderUserManager was called with alias file
        mock_placeholder.assert_called_once_with('aliases.json')

    @patch('pipeline.ModelSelection')
    @patch('pipeline.datasetCreation')
    @patch('pipeline.ModelTraining')
    @patch('pipeline.PlaceholderUserManager')
    @patch('pipeline.LoggerReport')
    @patch('pipeline.LoggerUser')
    @patch('pipeline.LoggerUserModelHistory')
    @patch('os.path.exists')
    def test_refactor_option(self, mock_path_exists,
                           mock_logger_history, mock_logger_user, 
                           mock_logger_report, mock_placeholder, 
                           mock_training, mock_dataset, mock_model):
        """
        Additional test to verify refactor option is properly passed.
        """
        # Setup mocks
        mock_path_exists.return_value = True
        mock_model.return_value = self.mock_model_selection
        mock_dataset.return_value = self.mock_dataset_creation
        mock_training.return_value = self.mock_model_training
        
        # Fix: Configure the mock_placeholder to return an instance with a selection method
        placeholder_instance = MagicMock()
        placeholder_instance.selection.return_value = ("placeholder_user", False)
        mock_placeholder.return_value = placeholder_instance
        
        mock_logger_report.return_value = self.mock_logger_report
        mock_logger_user.open = MagicMock()
        mock_logger_user.close = MagicMock()
        mock_logger_history.append_model_user = MagicMock()
        
        # Set command line arguments
        with patch('sys.argv', ['pipeline.py', '-c', 'valid_config.json', '-ref']):
            # Run the pipeline
            pipeline.create_dataset_and_train_model()
        
        # Assert that datasetCreation was called with refactor=True
        called_args = mock_dataset.call_args[0]
        # The refactor parameter should be at index 5
        self.assertTrue(called_args[5])  # refactor parameter should be True

if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("Pipeline.csv"))