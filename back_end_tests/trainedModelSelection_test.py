import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.cmdlineManagement.trainedModelSelection import TrainedModelSelection

class TestTrainedModelSelection(unittest.TestCase):

    # SMA1: MF1 – SU1: Valid folder with at least one .joblib file and user selects a valid file.
    @patch('inquirer.prompt', return_value={'model': 'model1.joblib'})
    @patch('utility.cmdlineManagement.trainedModelSelection.load', return_value='DummyModel')
    def test_valid_model_selection(self, mock_load, mock_prompt):
        # Simulate a folder with valid model files
        fake_files = ['model1.joblib', 'model2.onnx']
        # Patch os.listdir to return fake_files
        with patch('os.listdir', return_value=fake_files):
            # Patch os.path.isfile to return True for any file in fake_files
            with patch('os.path.isfile', return_value=True):
                # Patch os.path.getctime to return a fixed value (simulate creation time)
                with patch('os.path.getctime', return_value=1000):
                    selection = TrainedModelSelection()
                    # __load_model returns a tuple (selected_model, loaded_model)
                    self.assertEqual(selection.model, ('model1.joblib', 'DummyModel'))

    # SMA2: MF2 – Folder is empty: should raise an error (e.g., because no model is available).
    @patch('inquirer.prompt', return_value=None)
    def test_empty_folder(self, mock_prompt):
        # Simulate an empty folder by returning an empty list from os.listdir.
        with patch('os.listdir', return_value=[]):
            with patch('os.path.isfile', return_value=False):
                # Expect that accessing model["model"] will cause an error
                with self.assertRaises(TypeError):
                    TrainedModelSelection()

    # SMA3: MF3 – SU2: Folder contains only non-.joblib files, so if the user selects one, an ExtensionError is raised.
    @patch('inquirer.prompt', return_value={'model': 'model1.onnx'})
    def test_invalid_model_extension(self, mock_prompt):
        # Simulate a folder with only files not ending with '.joblib'
        fake_files = ['model1.onnx', 'another_model.onnx']
        with patch('os.listdir', return_value=fake_files):
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.getctime', return_value=1000):
                    with self.assertRaises(Exception) as context:
                        TrainedModelSelection()
                    self.assertEqual(str(context.exception), "Il modello deve essere in formato .joblib")

if __name__ == '__main__':
    unittest.main()
