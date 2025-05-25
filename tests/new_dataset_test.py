import sys
import os
import runpy
import types
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from test_logger import TableTestRunner

# Path to the new_dataset.py script (adjust the path if needed)
SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/new_dataset.py"))

def inject_dummy_modules():
    # Create a dummy module for datasetCreation
    dummy_datasetCreation = types.ModuleType("utility.dataset.datasetCreation")
    dummy_datasetCreation.datasetCreation = MagicMock()
    sys.modules["utility.dataset.datasetCreation"] = dummy_datasetCreation

    # Create a dummy module for PlaceholderUserManager
    dummy_placeholder = types.ModuleType("utility.cmdlineManagement.PlaceholderUserManager")
    dummy_placeholder.PlaceholderUserManager = MagicMock()
    sys.modules["utility.cmdlineManagement.PlaceholderUserManager"] = dummy_placeholder

    # Create a dummy module for logging (LoggerReport and LoggerUser)
    dummy_logging = types.ModuleType("utility.logging")
    dummy_logging.LoggerReport = MagicMock()
    dummy_logging.LoggerUser = MagicMock()
    sys.modules["utility.logging"] = dummy_logging

    # Return the dummy modules so tests can configure them
    return dummy_datasetCreation, dummy_placeholder, dummy_logging

class TestNewDataset(unittest.TestCase):
    def setUp(self):
        # Inject our dummy modules into sys.modules.
        self.dummy_datasetCreation, self.dummy_placeholder, self.dummy_logging = inject_dummy_modules()

    def tearDown(self):
        # Optionally remove our dummy modules from sys.modules after each test.
        for key in ["utility.dataset.datasetCreation",
                    "utility.cmdlineManagement.PlaceholderUserManager",
                    "utility.logging"]:
            if key in sys.modules:
                del sys.modules[key]

    def test_icd1_success(self):
        # Configure the dummy PlaceholderUserManager to return dummy values.
        ph_mock = self.dummy_placeholder.PlaceholderUserManager.return_value
        ph_mock.selection.return_value = ("dummy_user", False)

        # Create a fake datasetCreation instance that simulates a successful run.
        ds_instance = MagicMock()
        ds_instance.run.return_value = None
        ds_instance.dataFrame = pd.DataFrame({'col': [1, 2, 3]})
        self.dummy_datasetCreation.datasetCreation.return_value = ds_instance

        # Prepare command-line arguments simulating DN1, C1, A1, and R2.
        test_args = ['new_dataset.py', '-dN', 'default', '-c', 'config.json', '-a', 'aliases.json']
        with patch.object(sys, 'argv', test_args):
            runpy.run_path(SCRIPT_PATH, run_name="__main__")
        
        # Check that the fake datasetCreation instance was used.
        ds_instance.run.assert_called_once()
        ph_mock.selection.assert_called_once()

    def test_icd2_failure(self):
        # Configure the dummy PlaceholderUserManager to return dummy values.

        self.expected_output = "Test induced error"

        ph_mock = self.dummy_placeholder.PlaceholderUserManager.return_value
        ph_mock.selection.return_value = ("dummy_user", False)

        # Create a fake datasetCreation instance that raises an exception when run.
        ds_instance = MagicMock()
        ds_instance.run.side_effect = Exception("Test induced error")
        self.dummy_datasetCreation.datasetCreation.return_value = ds_instance

        test_args = ['new_dataset.py', '-dN', 'default', '-c', 'config.json', '-a', 'aliases.json']
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(Exception) as context:
                runpy.run_path(SCRIPT_PATH, run_name="__main__")
            self.assertEqual(str(context.exception), "Test induced error")
            ph_mock.selection.assert_called_once()

if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("NewDataset.csv"))
