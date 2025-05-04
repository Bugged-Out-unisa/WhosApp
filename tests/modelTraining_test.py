import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import numpy as np
from joblib import dump
from test_logger import TableTestRunner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the ModelTraining class from the correct namespace.
from utility.model.modelTraining import ModelTraining
from utility.exceptions import ModelNotFoundError, DatasetNotFoundError

# Dummy configuration JSON (valid config)
dummy_config = json.dumps({
    "feature1": True,
    "feature2": False,
    "bag_of_words": False
})

# Create a larger dummy DataFrame with 20 rows and balanced binary classes.
dummy_df = pd.DataFrame({
    "user": [0, 1] * 10,  # alternating 0 and 1 for 20 samples
    "feature1": np.linspace(0.1, 1.0, 20)
})

# Dummy model factory simulating a scikit-learn estimator.
def get_dummy_model():
    dummy_model = MagicMock()
    # Simulate feature importances for one feature.
    dummy_model.feature_importances_ = np.array([0.3, 0.7])
    # Define two classes for binary classification.
    dummy_model.classes_ = np.array([0, 1])
    # fit: simply returns self.
    dummy_model.fit.return_value = dummy_model
    # predict: return all zeros (so accuracy might be 0, but that's acceptable for testing)
    dummy_model.predict.side_effect = lambda X: [0] * len(X)
    # predict_proba: always return two probabilities per sample.
    dummy_model.predict_proba.side_effect = lambda X: np.array([[0.8, 0.2]] * len(X))
    return dummy_model

# Helper to simulate os.path.exists.
def fake_os_path_exists(path):
    # For config files.
    if "../configs/config.json" in path or "../configs/custom_config.json" in path:
        return True
    if "non_esiste.json" in path:
        return False
    # For model files.
    if "../models/" in path:
        return fake_os_path_exists.exists
    return False

# Default: no model file exists.
fake_os_path_exists.exists = False

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Reset the fake existence flag.
        fake_os_path_exists.exists = False

    @patch("utility.model.modelTraining.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM1_default_output_valid_model_valid_df_default_config_no_retrain(self, mock_dump, mock_file, mock_makedirs, mock_exists):
        # TM1: Default output name, valid model, valid DataFrame, default config, retrain False.
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName=None,
            model=dummy_model,
            dataFrame=dummy_df,
            configFile=None,  # Should use default "config.json"
            retrain=False
        )
        accuracy = mt.run()
        self.assertTrue(mock_dump.called)
        self.assertIsNotNone(accuracy)

    @patch("utility.model.modelTraining.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM2_default_output_valid_model_valid_df_custom_config_no_retrain(self, mock_dump, mock_file, mock_makedirs, mock_exists):
        # TM2: Default output, valid model, valid DataFrame, custom config ("custom_config.json"), retrain False.
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName=None,
            model=dummy_model,
            dataFrame=dummy_df,
            configFile="custom_config.json",
            retrain=False
        )
        accuracy = mt.run()
        self.assertTrue(mock_dump.called)
        self.assertIsNotNone(accuracy)

    @patch("utility.model.modelTraining.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM3_custom_output_valid_model_valid_df_default_config_no_retrain(self, mock_dump, mock_file, mock_makedirs, mock_exists):
        # TM3: Custom output name, valid model, valid DataFrame, default config, retrain False.
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName="custom_model.joblib",
            model=dummy_model,
            dataFrame=dummy_df,
            configFile=None,
            retrain=False
        )
        accuracy = mt.run()
        self.assertIn("custom_model.joblib", mt._ModelTraining__outputName)
        self.assertTrue(mock_dump.called)

    @patch("utility.model.modelTraining.input", return_value="y")
    @patch("utility.model.modelTraining.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_os_path_exists(path))
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM4_retrain_true_existing_file(self, mock_dump, mock_file, mock_makedirs, mock_exists, mock_input):
        # TM4: Default output, valid model, valid DataFrame, default config, retrain True (simulate existing file).
        fake_os_path_exists.exists = True  # Simulate that model file exists.
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName=None,
            model=dummy_model,
            dataFrame=dummy_df,
            configFile=None,
            retrain=True
        )
        accuracy = mt.run()
        self.assertTrue(mock_dump.called)
        self.assertIsNotNone(accuracy)

    @patch("utility.model.modelTraining.input", return_value="n")
    @patch("utility.model.modelTraining.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_os_path_exists(path))
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM5_existing_file_no_retrain_user_declines(self, mock_dump, mock_file, mock_makedirs, mock_exists, mock_input):
        # TM5: Default output, valid model, valid DataFrame, default config, retrain False with existing file and user declining overwrite.
        fake_os_path_exists.exists = True
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName=None,
            model=dummy_model,
            dataFrame=dummy_df,
            configFile=None,
            retrain=False
        )
        with patch("builtins.print") as mock_print:
            accuracy = mt.run()
            printed_messages = [args[0] for args, _ in mock_print.call_args_list]
            self.assertTrue(any("Operazione di Training annullata" in m for m in printed_messages))
        self.assertFalse(mock_dump.called)

    @patch("utility.model.modelTraining.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM6_custom_output_valid_model_valid_df_custom_config_no_retrain(self, mock_dump, mock_file, mock_makedirs, mock_exists):
        # TM6: Custom output name, valid model, valid DataFrame, custom config, retrain False.
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName="custom_model.joblib",
            model=dummy_model,
            dataFrame=dummy_df,
            configFile="custom_config.json",
            retrain=False
        )
        # Only initialization is checked.
        self.assertIn("custom_model.joblib", mt._ModelTraining__outputName)

    @patch("utility.model.modelTraining.input", return_value="y")
    @patch("utility.model.modelTraining.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_os_path_exists(path))
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM7_custom_config_retrain_true(self, mock_dump, mock_file, mock_makedirs, mock_exists, mock_input):
        # TM7: Default output, valid model, valid DataFrame, custom config, retrain True.
        fake_os_path_exists.exists = True
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName=None,
            model=dummy_model,
            dataFrame=dummy_df,
            configFile="custom_config.json",
            retrain=True
        )
        accuracy = mt.run()
        self.assertTrue(mock_dump.called)
        self.assertIsNotNone(accuracy)

    @patch("utility.model.modelTraining.input", return_value="y")
    @patch("utility.model.modelTraining.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_os_path_exists(path))
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path: None)
    @patch("utility.model.modelTraining.open", new_callable=mock_open, read_data=dummy_config)
    @patch("utility.model.modelTraining.dump")
    @patch("utility.model.modelTraining.plt.show", lambda: None)
    def test_TM8_custom_output_custom_config_retrain_true(self, mock_dump, mock_file, mock_makedirs, mock_exists, mock_input):
        # TM8: Custom output, valid model, valid DataFrame, custom config, retrain True.
        fake_os_path_exists.exists = True
        dummy_model = get_dummy_model()
        mt = ModelTraining(
            outputName="custom_model.joblib",
            model=dummy_model,
            dataFrame=dummy_df,
            configFile="custom_config.json",
            retrain=True
        )
        accuracy = mt.run()
        self.assertIn("custom_model.joblib", mt._ModelTraining__outputName)
        self.assertTrue(mock_dump.called)
        self.assertIsNotNone(accuracy)

    def test_TM9_no_model_provided(self):
        # TM9: Missing model should raise ModelNotFoundError.

        self.expected_output = "Model not found"

        with self.assertRaises(ModelNotFoundError):
            ModelTraining(
                outputName=None,
                model=None,
                dataFrame=dummy_df,
                configFile=None,
                retrain=False
            )

    def test_TM10_no_dataframe_provided(self):
        # TM10: Missing DataFrame should raise DatasetNotFoundError.

        self.expected_output = "Dataset not found"

        dummy_model = get_dummy_model()
        with self.assertRaises(DatasetNotFoundError):
            ModelTraining(
                outputName=None,
                model=dummy_model,
                dataFrame=None,
                configFile=None,
                retrain=False
            )

    @patch("utility.model.modelTraining.dump", return_value=None)
    @patch("utility.model.modelTraining.os.path.exists", side_effect=lambda path: False)
    @patch("utility.model.modelTraining.os.makedirs", side_effect=lambda path, exist_ok=False: None)
    def test_TM11_invalid_config_file(self, mock_makedirs, mock_exists, mock_dump):
        # TM11: Invalid config file should raise FileNotFoundError.

        self.expected_output = "Config file not found"

        dummy_model = get_dummy_model()

        with self.assertRaises(FileNotFoundError):
            mt = ModelTraining(
                outputName=None,
                model=dummy_model,
                dataFrame=dummy_df,
                configFile="non_esiste.json",
                retrain=False
            )
            mt.run()

if __name__ == "__main__":
    unittest.main(testRunner=TableTestRunner("ModelTraining.csv"))