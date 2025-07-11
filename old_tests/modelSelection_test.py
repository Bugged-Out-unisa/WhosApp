import os
import sys
import unittest
from unittest.mock import patch
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.model.model_list import models
from utility.cmdlineManagement.modelSelection import ModelSelection

class TestModelSelection(unittest.TestCase):
    # Test for valid model selection
    @patch('inquirer.prompt', return_value={'model': 'valid_model'})
    def test_model_selection_valid(self, mock_prompt):
        # Patch the 'models' dictionary within the model_selection module
        with patch('utility.cmdlineManagement.modelSelection.models', {"valid_model": "ModelObject"}):
            # Patch the model_names attribute to reflect the patched models dictionary
            with patch.object(ModelSelection, 'model_names', ["valid_model"]):
                selection = ModelSelection()
                self.assertEqual(selection.model, "ModelObject")

    # Test for invalid model selection: user selects an invalid model
    @patch('inquirer.prompt', return_value={'model': 'invalid_model'})
    def test_model_selection_invalid(self, mock_prompt):
        self.expected_output = "Modello non valido."

        # Patch the 'models' dictionary within the model_selection module
        with patch('utility.cmdlineManagement.modelSelection.models', {"valid_model": "ModelObject"}):
            # Patch the model_names attribute to reflect the patched models dictionary
            with patch.object(ModelSelection, 'model_names', ["valid_model"]):
                with self.assertRaises(ValueError) as context:
                    ModelSelection()
                self.assertEqual(str(context.exception), "Modello non valido.")

if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("ModelSelection.csv"))