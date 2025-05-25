import sys
import os
import runpy
import types
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Path to the new_training.py script (adjust the path if needed)
SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/new_training.py"))

def inject_dummy_modules_new_training():
    # --- Dummy for utility.logging ---
    dummy_logging = types.ModuleType("utility.logging")
    # Dummy LoggerReport: add the required TRAINING_LOGGING_PATH attribute.
    class DummyLoggerReport:
        TRAINING_LOGGING_PATH = "dummy_path"
        def __init__(self, name, start_message, path):
            self.name = name
            self.start_message = start_message
            self.path = path
        def run(self):
            pass
    dummy_logging.LoggerReport = DummyLoggerReport
    # LoggerUserModelHistory: record calls.
    dummy_logging.LoggerUserModelHistory = MagicMock()
    sys.modules["utility.logging"] = dummy_logging

    # --- Dummy for utility.cmdlineManagement.datasetSelection ---
    dummy_datasetSelection = types.ModuleType("utility.cmdlineManagement.datasetSelection")
    class DummyDatasetSelection:
        def __init__(self):
            self.dataset = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
            self.dataset_name = "dummy_dataset"
    dummy_datasetSelection.DatasetSelection = DummyDatasetSelection
    sys.modules["utility.cmdlineManagement.datasetSelection"] = dummy_datasetSelection

    # --- Dummy for utility.cmdlineManagement.modelSelection ---
    dummy_modelSelection = types.ModuleType("utility.cmdlineManagement.modelSelection")
    class DummyModelSelection:
        def __init__(self):
            self.model = MagicMock(name="dummy_model")
    dummy_modelSelection.ModelSelection = DummyModelSelection
    sys.modules["utility.cmdlineManagement.modelSelection"] = dummy_modelSelection

    # --- Dummy for utility.model.modelTraining ---
    dummy_modelTraining = types.ModuleType("utility.model.modelTraining")
    class DummyModelTraining:
        last_instance = None  # Class attribute to store the last created instance.
        # Create a class-level mock for the run method that can be easily configured in tests
        run_mock = MagicMock(return_value=0.85)
        
        def __init__(self, outputName, model, dataFrame, configFile, retrain):
            self.outputName = outputName  # Store the passed outputName.
            self.model = model
            self.dataFrame = dataFrame
            self.configFile = configFile
            self.retrain = retrain
            DummyModelTraining.last_instance = self  # Store this instance for later retrieval.
        
        def run(self):
            # This delegates to the class-level mock so we can control its behavior in tests
            return self.__class__.run_mock()
            
    dummy_modelTraining.ModelTraining = DummyModelTraining
    sys.modules["utility.model.modelTraining"] = dummy_modelTraining

    return dummy_logging, dummy_datasetSelection, dummy_modelSelection, dummy_modelTraining

class TestNewTraining(unittest.TestCase):
    def setUp(self):
        # Inject dummy modules.
        self.dummy_logging, self.dummy_datasetSelection, self.dummy_modelSelection, self.dummy_modelTraining = inject_dummy_modules_new_training()

    def tearDown(self):
        # Reset any mocks to their default state
        if hasattr(self, 'dummy_modelTraining') and hasattr(self.dummy_modelTraining.ModelTraining, 'run_mock'):
            self.dummy_modelTraining.ModelTraining.run_mock.reset_mock()
            self.dummy_modelTraining.ModelTraining.run_mock.side_effect = None
            self.dummy_modelTraining.ModelTraining.run_mock.return_value = 0.85
        
        # Remove injected modules.
        for key in ["utility.logging",
                    "utility.cmdlineManagement.datasetSelection",
                    "utility.cmdlineManagement.modelSelection",
                    "utility.model.modelTraining"]:
            if key in sys.modules:
                del sys.modules[key]

    def test_IT1_default_output_valid_config_no_retrain(self):
        # IT1: ON1 - CF1 - R1: default output, valid config ("config.json"), retrain False.
        test_args = ['new_training.py', '-c', 'config.json']
        with patch.object(sys, 'argv', test_args):
            runpy.run_path(SCRIPT_PATH, run_name="__main__")
        # Retrieve the DummyModelTraining instance created.
        instance = self.dummy_modelTraining.ModelTraining.last_instance
        # Default output should not equal "custom_model.joblib"
        self.assertNotEqual(instance.outputName, "custom_model.joblib")
        # Verify LoggerUserModelHistory.append_model_user was called with dummy dataset and the output name.
        self.dummy_logging.LoggerUserModelHistory.append_model_user.assert_called_once_with("dummy_dataset", instance.outputName)

    def test_IT2_custom_output_valid_config_no_retrain(self):
        # IT2: ON2 - CF1 - R1: custom output "custom_model.joblib", valid config, retrain False.
        test_args = ['new_training.py', '-oN', 'custom_model.joblib', '-c', 'config.json']
        with patch.object(sys, 'argv', test_args):
            runpy.run_path(SCRIPT_PATH, run_name="__main__")
        instance = self.dummy_modelTraining.ModelTraining.last_instance
        self.assertEqual(instance.outputName, "custom_model.joblib")
        self.dummy_logging.LoggerUserModelHistory.append_model_user.assert_called_once_with("dummy_dataset", "custom_model.joblib")

    def test_IT3_default_output_valid_config_retrain_true(self):
        # IT3: ON1 - CF1 - R2: default output, valid config, retrain True.
        test_args = ['new_training.py', '-c', 'config.json', '-r']
        with patch.object(sys, 'argv', test_args):
            runpy.run_path(SCRIPT_PATH, run_name="__main__")
        instance = self.dummy_modelTraining.ModelTraining.last_instance
        self.assertNotEqual(instance.outputName, "custom_model.joblib")
        self.assertTrue(instance.retrain)
        self.dummy_logging.LoggerUserModelHistory.append_model_user.assert_called_once_with("dummy_dataset", instance.outputName)

    def test_IT4_custom_output_valid_config_retrain_true(self):
        # IT4: ON2 - CF1 - R2: custom output, valid config, retrain True.
        test_args = ['new_training.py', '-oN', 'custom_model.joblib', '-c', 'config.json', '-r']
        with patch.object(sys, 'argv', test_args):
            runpy.run_path(SCRIPT_PATH, run_name="__main__")
        instance = self.dummy_modelTraining.ModelTraining.last_instance
        self.assertEqual(instance.outputName, "custom_model.joblib")
        self.assertTrue(instance.retrain)
        self.dummy_logging.LoggerUserModelHistory.append_model_user.assert_called_once_with("dummy_dataset", "custom_model.joblib")

    def test_IT5_invalid_config_file(self):
        # IT5: ON1 - CF2 - R1: default output, invalid config file ("non_esiste.json"), retrain False.
        # We'll simulate this by patching DummyModelTraining.__init__ to raise FileNotFoundError.

        self.expected_output = "File di configurazione non trovato"

        original_init = self.dummy_modelTraining.ModelTraining.__init__
        def init_raise(self, outputName, model, dataFrame, configFile, retrain):
            if configFile == "non_esiste.json":
                raise FileNotFoundError("File di configurazione non trovato")
            original_init(self, outputName, model, dataFrame, configFile, retrain)
        with patch.object(self.dummy_modelTraining.ModelTraining, "__init__", new=init_raise):
            test_args = ['new_training.py', '-c', 'non_esiste.json']
            with patch.object(sys, 'argv', test_args):
                with self.assertRaises(FileNotFoundError) as context:
                    runpy.run_path(SCRIPT_PATH, run_name="__main__")
                self.assertEqual(str(context.exception), "File di configurazione non trovato")

    def test_IT6_internal_error_propagation(self):
        # IT6: Simulate internal error propagation in ModelTraining

        self.expected_output = "Internal error"

        test_args = ['new_training.py', '-c', 'config.json']

        # Configure the mock to raise an exception
        self.dummy_modelTraining.ModelTraining.run_mock.side_effect = Exception("Internal error")
        
        try:
            with patch.object(sys, 'argv', test_args):
                with self.assertRaises(Exception) as context:
                    runpy.run_path(SCRIPT_PATH, run_name="__main__")
                    
                self.assertEqual(str(context.exception), "Internal error")
        finally:
            # Reset the mock for other tests
            self.dummy_modelTraining.ModelTraining.run_mock.side_effect = None


if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("NewTraining.csv"))