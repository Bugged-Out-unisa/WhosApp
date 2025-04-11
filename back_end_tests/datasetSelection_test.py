import unittest
from unittest import mock
import os
import pandas as pd
from io import StringIO
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.exceptions import ExtensionError
from utility.cmdlineManagement.datasetSelection import DatasetSelection

class TestDatasetSelection(unittest.TestCase):
    """Test cases for the DatasetSelection class."""

    @mock.patch('os.listdir')
    @mock.patch('os.path.getctime')
    @mock.patch('pandas.read_parquet')
    @mock.patch('inquirer.prompt')
    @mock.patch('logging.info')
    def test_DF1_successful_dataset_selection(self, mock_logging, mock_prompt, mock_read_parquet, 
                                              mock_getctime, mock_listdir):
        """Test case DF1: Directory contains .parquet files and user selects one."""

        mock_listdir.return_value = ['dataset1.parquet', 'dataset2.parquet']
        mock_getctime.side_effect = lambda path: {
            os.path.join(DatasetSelection.DATASET_PATH, 'dataset1.parquet'): 123456789,
            os.path.join(DatasetSelection.DATASET_PATH, 'dataset2.parquet'): 123456700
        }[path]
        mock_prompt.return_value = {'dataset': 'dataset1.parquet'}
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_parquet.return_value = mock_df
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        dataset_selection = DatasetSelection()
        result_df = dataset_selection.dataset
        result_name = dataset_selection.dataset_name
        
        sys.stdout = sys.__stdout__
        
        # Assert
        mock_listdir.assert_called_once_with(DatasetSelection.DATASET_PATH)
        mock_prompt.assert_called_once()
        mock_read_parquet.assert_called_once_with(f"{DatasetSelection.DATASET_PATH}dataset1.parquet")
        self.assertEqual(result_df.equals(mock_df), True)
        self.assertEqual(result_name, 'dataset1.parquet')
        self.assertIn("Dataset selezionato: dataset1.parquet", captured_output.getvalue())
        mock_logging.assert_called_once_with("Dataset usato per il training: dataset1.parquet")

    @mock.patch('os.listdir')
    @mock.patch('inquirer.prompt')  # Add this mock to prevent actual prompt
    def test_DF2_no_datasets_available(self, mock_prompt, mock_listdir):
        """Test case DF2: Directory is empty, no datasets available."""

        mock_listdir.return_value = []

        
        with self.assertRaises(Exception) as context:
            DatasetSelection()
        
        mock_listdir.assert_called_once_with(DatasetSelection.DATASET_PATH)


    def test_DF3_extension_error_direct(self):
        """Test case DF3: Testing ExtensionError by directly calling __load_dataset."""

        with mock.patch.object(DatasetSelection, '__init__', return_value=None):
            selection = DatasetSelection()
            
            test_datasets = ['dataset1.parquet', 'dataset2.txt']
            
            with self.assertRaises(ExtensionError) as context:
                selection._DatasetSelection__load_dataset(1, test_datasets)
            
            self.assertIn("Il dataset deve essere in formato .parquet", str(context.exception))

if __name__ == '__main__':
    unittest.main()