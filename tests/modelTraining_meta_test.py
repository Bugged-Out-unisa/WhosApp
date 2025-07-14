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

# Import the MetaLearner class from the correct namespace
from utility.model.modelTraining_meta import MetaLearner

# Create a dummy DataFrame with balanced binary classes for meta-learning
dummy_meta_df = pd.DataFrame({
    "user": [0, 1] * 50,  # 100 samples with balanced classes
    "feature1": np.random.uniform(0.1, 1.0, 100),
    "feature2": np.random.uniform(0.2, 0.8, 100),
    "feature3": np.random.uniform(0.0, 1.0, 100),
    "feature4": np.random.uniform(0.3, 0.9, 100),
    "feature5": np.random.uniform(0.1, 0.7, 100)
})

# Dummy XGBoost model factory
def get_dummy_xgb_model():
    dummy_model = MagicMock()
    dummy_model.fit.return_value = dummy_model
    dummy_model.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
    dummy_model.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
    dummy_model.score.side_effect = lambda X, y: 0.85  # Mock validation score
    return dummy_model

# Dummy GASearchCV mock
def get_dummy_ga_search():
    dummy_ga = MagicMock()
    dummy_ga.fit.return_value = dummy_ga
    dummy_ga.best_estimator_ = get_dummy_xgb_model()
    dummy_ga.best_score_ = 0.87
    return dummy_ga

# Helper to simulate os.path.exists for meta-learner
def fake_meta_os_path_exists(path):
    if "../models/" in path:
        return fake_meta_os_path_exists.exists
    return False

# Default: no model file exists
fake_meta_os_path_exists.exists = False

class TestMetaLearner(unittest.TestCase):

    def setUp(self):
        # Reset the fake existence flag
        fake_meta_os_path_exists.exists = False

    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML1_default_output_valid_dataset_no_retrain(self, mock_dump, mock_makedirs, mock_exists):
        """TML1: Default output name, valid dataset, no retrain."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            output_name=None,
            retrain=False
        )
        
        # Test data preparation
        X_train, y_train, X_val, y_val, X_test, y_test = ml.prepare_data(test_size=0.2)
        
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(X_val)
        self.assertIsNotNone(y_val)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)
        
        # Check data shapes
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(dummy_meta_df))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_val), len(X_val))
        self.assertEqual(len(y_test), len(X_test))

    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML2_custom_output_valid_dataset_no_retrain(self, mock_dump, mock_makedirs, mock_exists):
        """TML2: Custom output name, valid dataset, no retrain."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            output_name="custom_meta_model.joblib",
            retrain=False
        )
        
        self.assertIn("custom_meta_model.joblib", ml.output_name)

    @patch("utility.model.modelTraining_meta.GASearchCV", return_value=get_dummy_ga_search())
    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML3_train_with_genetic_algorithm(self, mock_dump, mock_makedirs, mock_exists, mock_ga):
        """TML3: Train meta-learner with genetic algorithm optimization."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            retrain=False
        )
        
        X_train, y_train, X_val, y_val, _, _ = ml.prepare_data(test_size=0.2)
        
        # Mock the GASearchCV to return our dummy
        mock_ga_instance = get_dummy_ga_search()
        mock_ga.return_value = mock_ga_instance
        
        accuracy = ml.train(X_train, y_train, X_val, y_val)
        
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    @patch("utility.model.modelTraining_meta.metrics.accuracy_score", return_value=0.90)
    @patch("utility.model.modelTraining_meta.metrics.classification_report", return_value={'0': {'precision': 0.89, 'recall': 0.91, 'f1-score': 0.90}, '1': {'precision': 0.91, 'recall': 0.89, 'f1-score': 0.90}})
    @patch("utility.model.modelTraining_meta.metrics.confusion_matrix", return_value=np.array([[45, 5], [3, 47]]))
    @patch("utility.model.modelTraining_meta.metrics.roc_auc_score", return_value=0.92)
    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML4_evaluate_model_performance(self, mock_dump, mock_makedirs, mock_exists, mock_roc_auc, mock_confusion_matrix, mock_classification_report, mock_accuracy):
        """TML4: Evaluate meta-learner model performance."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            retrain=False
        )
        
        # Mock the model to have predict and predict_proba methods
        ml.model = get_dummy_xgb_model()
        
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        accuracy, classification_report, roc_auc, confusion_matrix = ml.evaluate(X_test, y_test)
        
        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(classification_report)
        self.assertIsNotNone(roc_auc)
        self.assertIsNotNone(confusion_matrix)
        
        # Verify the mocked return values
        self.assertEqual(accuracy, 0.90)
        self.assertEqual(roc_auc, 0.92)

    @patch("utility.model.modelTraining_meta.input", return_value="y")
    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_meta_os_path_exists(path))
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML5_retrain_true_existing_file(self, mock_dump, mock_makedirs, mock_exists, mock_input):
        """TML5: Retrain with existing model file."""
        fake_meta_os_path_exists.exists = True
        
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            retrain=True
        )
        
        # Should not raise an exception since retrain=True
        self.assertIsNotNone(ml)

    @patch("utility.model.modelTraining_meta.input", return_value="n")
    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_meta_os_path_exists(path))
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML6_existing_file_no_retrain_user_declines(self, mock_dump, mock_makedirs, mock_exists, mock_input):
        """TML6: Existing file, no retrain, user declines overwrite."""
        fake_meta_os_path_exists.exists = True
        
        with self.assertRaises(ValueError):
            ml = MetaLearner(
                dataset=dummy_meta_df,
                label_column='user',
                retrain=False
            )

    @patch("utility.model.modelTraining_meta.GASearchCV", return_value=get_dummy_ga_search())
    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML7_prepare_and_train_workflow(self, mock_dump, mock_makedirs, mock_exists, mock_ga):
        """TML7: Test complete prepare_and_train workflow."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            retrain=False
        )
        
        mock_ga_instance = get_dummy_ga_search()
        mock_ga.return_value = mock_ga_instance
        
        accuracy = ml.prepare_and_train()
        
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy, 0.0)

    @patch("utility.model.modelTraining_meta.GASearchCV", return_value=get_dummy_ga_search())
    @patch("utility.model.modelTraining_meta.metrics.accuracy_score", return_value=0.88)
    @patch("utility.model.modelTraining_meta.metrics.classification_report", return_value={'0': {'precision': 0.87, 'recall': 0.89, 'f1-score': 0.88}, '1': {'precision': 0.89, 'recall': 0.87, 'f1-score': 0.88}})
    @patch("utility.model.modelTraining_meta.metrics.confusion_matrix", return_value=np.array([[42, 8], [5, 45]]))
    @patch("utility.model.modelTraining_meta.metrics.roc_auc_score", return_value=0.89)
    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML8_complete_train_and_evaluate_workflow(self, mock_dump, mock_makedirs, mock_exists, mock_roc_auc, mock_confusion_matrix, mock_classification_report, mock_accuracy, mock_ga):
        """TML8: Test complete train_and_evaluate workflow."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            retrain=False
        )
        
        mock_ga_instance = get_dummy_ga_search()
        mock_ga.return_value = mock_ga_instance
        
        results = ml.train_and_evaluate(plot_results=False)
        
        self.assertIsNotNone(results)
        self.assertIn('accuracy', results)
        self.assertIn('classification_report', results)
        self.assertIn('roc_auc', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('best_params', results)

    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML9_save_model_functionality(self, mock_dump, mock_makedirs, mock_exists):
        """TML9: Test model saving functionality."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            retrain=False
        )
        
        ml.model = get_dummy_xgb_model()
        
        with patch("builtins.print") as mock_print:
            ml.save_model()
            
        self.assertTrue(mock_dump.called)
        # Verify print was called with save confirmation
        mock_print.assert_called()

    def test_TML10_no_dataset_provided(self):
        """TML10: Missing dataset should raise ValueError."""
        self.expected_output = "Dataset not found"
        
        with self.assertRaises(ValueError):
            MetaLearner(
                dataset=None,
                label_column='user',
                retrain=False
            )

    def test_TML11_no_label_column_provided(self):
        """TML11: Missing label column should raise ValueError."""
        self.expected_output = "Label column not found"
        
        with self.assertRaises(ValueError):
            MetaLearner(
                dataset=dummy_meta_df,
                label_column=None,
                retrain=False
            )

    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML12_custom_label_column(self, mock_dump, mock_makedirs, mock_exists):
        """TML12: Test with custom label column name."""
        # Create dataset with different label column name
        custom_df = dummy_meta_df.copy()
        custom_df = custom_df.rename(columns={'user': 'target'})
        
        ml = MetaLearner(
            dataset=custom_df,
            label_column='target',
            retrain=False
        )
        
        self.assertEqual(ml.label_column, 'target')

    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML13_custom_model_path(self, mock_dump, mock_makedirs, mock_exists):
        """TML13: Test with custom model path."""
        custom_path = "../custom_models/"
        
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            model_path=custom_path,
            retrain=False
        )
        
        self.assertEqual(ml.model_path, custom_path)

    @patch("utility.model.modelTraining_meta.os.path.exists", side_effect=fake_meta_os_path_exists)
    @patch("utility.model.modelTraining_meta.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("joblib.dump")
    @patch("utility.model.modelTraining_meta.plt.show", lambda: None)
    def test_TML14_data_preparation_no_test_split(self, mock_dump, mock_makedirs, mock_exists):
        """TML14: Test data preparation without test split."""
        ml = MetaLearner(
            dataset=dummy_meta_df,
            label_column='user',
            retrain=False
        )
        
        X_train, y_train, X_val, y_val, X_test, y_test = ml.prepare_data(test_size=0)
        
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(X_val)
        self.assertIsNotNone(y_val)
        self.assertIsNone(X_test)
        self.assertIsNone(y_test)

if __name__ == "__main__":
    unittest.main(testRunner=TableTestRunner("MetaLearner.csv"))