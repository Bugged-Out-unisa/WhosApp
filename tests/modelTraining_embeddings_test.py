import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from test_logger import TableTestRunner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the CNN1D class
from utility.model.modelTraining_embeddings import CNN1D

# Create dummy embedding DataFrame with sufficient samples for train/val/test split
np.random.seed(42)
torch.manual_seed(42)

# Create a balanced dataset with 3 classes and 150 samples total
dummy_embedding_df = pd.DataFrame({
    'user': ['user_a'] * 50 + ['user_b'] * 50 + ['user_c'] * 50,
    **{f'embed_{i}': np.random.randn(150) for i in range(768)}  # 768-dimensional embeddings
})

# Smaller dataset for edge case testing
small_embedding_df = pd.DataFrame({
    'user': ['user_a'] * 10 + ['user_b'] * 10,
    **{f'embed_{i}': np.random.randn(20) for i in range(384)}  # 384-dimensional embeddings
})

# Helper to simulate os.path.exists
def fake_os_path_exists(path):
    if "../models/" in path:
        return fake_os_path_exists.exists
    return True

# Default: no model file exists
fake_os_path_exists.exists = False

class TestCNN1D(unittest.TestCase):

    def setUp(self):
        # Reset the fake existence flag
        fake_os_path_exists.exists = False
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC1_default_params_valid_df_no_retrain(self, mock_save, mock_makedirs, mock_exists):
        """TC1: Default parameters, valid DataFrame, no retrain"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Verify model initialization
        self.assertEqual(model.embedding_dim, 768)
        self.assertEqual(model.num_classes, 3)
        self.assertEqual(len(model.class_names), 3)
        self.assertIn('user_a', model.class_names)
        self.assertIn('user_b', model.class_names)
        self.assertIn('user_c', model.class_names)

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC2_custom_output_name_valid_df_no_retrain(self, mock_save, mock_makedirs, mock_exists):
        """TC2: Custom output name, valid DataFrame, no retrain"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            output_name="custom_cnn_model.pth",
            retrain=False
        )
        
        # Verify custom output name is set
        self.assertIn("custom_cnn_model.pth", model.output_name)

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC3_custom_dropout_rate_valid_df_no_retrain(self, mock_save, mock_makedirs, mock_exists):
        """TC3: Custom dropout rate, valid DataFrame, no retrain"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            dropout_rate=0.5,
            retrain=False
        )
        
        # Verify model was created successfully
        self.assertIsNotNone(model)
        self.assertEqual(model.embedding_dim, 768)

    @patch("utility.model.modelTraining_embeddings.input", return_value="y")
    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_os_path_exists(path))
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC4_retrain_true_existing_file(self, mock_save, mock_makedirs, mock_exists, mock_input):
        """TC4: Retrain true with existing file"""
        fake_os_path_exists.exists = True
        
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=True
        )
        
        # Verify model was created successfully even with existing file
        self.assertIsNotNone(model)

    @patch("utility.model.modelTraining_embeddings.input", return_value="n")
    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=lambda path: True if "../models/" in path else fake_os_path_exists(path))
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    def test_TC5_existing_file_no_retrain_user_declines(self, mock_makedirs, mock_exists, mock_input):
        """TC5: Existing file, no retrain, user declines overwrite"""
        fake_os_path_exists.exists = True
        
        with self.assertRaises(ValueError):
            CNN1D(
                embedding_input=dummy_embedding_df,
                label_column='user',
                retrain=False
            )

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC6_custom_embedding_prefix_valid_df(self, mock_save, mock_makedirs, mock_exists):
        """TC6: Custom embedding prefix with valid DataFrame"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            embedding_prefix='embed_',
            retrain=False
        )
        
        # Verify embedding columns are correctly identified
        self.assertEqual(len(model.embedding_columns), 768)
        self.assertTrue(all(col.startswith('embed_') for col in model.embedding_columns))

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC7_small_dataset_valid_initialization(self, mock_save, mock_makedirs, mock_exists):
        """TC7: Small dataset with valid initialization"""
        model = CNN1D(
            embedding_input=small_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Verify model works with smaller dataset
        self.assertEqual(model.embedding_dim, 384)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(len(model.class_names), 2)

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC8_prepare_data_functionality(self, mock_save, mock_makedirs, mock_exists):
        """TC8: Test prepare_data functionality"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Test data preparation
        train_loader, val_loader, test_loader = model.prepare_data(
            test_size=0.2,
            val_size=0.2,
            batch_size=16,
            random_state=42
        )
        
        # Verify data loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Verify batch size
        train_batch = next(iter(train_loader))
        self.assertEqual(len(train_batch), 2)  # embeddings and labels
        self.assertTrue(train_batch[0].shape[1] == 768)  # embedding dimension

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC9_forward_pass_functionality(self, mock_save, mock_makedirs, mock_exists):
        """TC9: Test forward pass functionality"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Create dummy input
        dummy_input = torch.randn(4, 768)  # batch_size=4, embedding_dim=768
        
        # Test forward pass
        output = model.forward(dummy_input)
        
        # Verify output shape
        self.assertEqual(output.shape, (4, 3))  # batch_size=4, num_classes=3

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC10_predict_functionality(self, mock_save, mock_makedirs, mock_exists):
        """TC10: Test predict functionality"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Create dummy input
        dummy_input = np.random.randn(5, 768)
        
        # Test predict
        predictions = model.predict(dummy_input)
        
        # Verify predictions shape and type
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in predictions))
        self.assertTrue(all(0 <= p < 3 for p in predictions))

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC11_predict_proba_functionality(self, mock_save, mock_makedirs, mock_exists):
        """TC11: Test predict_proba functionality"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Create dummy input
        dummy_input = np.random.randn(3, 768)
        
        # Test predict_proba
        probabilities = model.predict_proba(dummy_input)
        
        # Verify probabilities shape and properties
        self.assertEqual(probabilities.shape, (3, 3))  # 3 samples, 3 classes
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))  # Probabilities sum to 1
        self.assertTrue(np.all(probabilities >= 0))  # All probabilities non-negative

    def test_TC12_no_embedding_input_provided(self):
        """TC12: No embedding input provided should raise ValueError"""
        with self.assertRaises(TypeError):
            CNN1D(
                embedding_input=None,
                label_column='user',
                retrain=False
            )

    def test_TC13_invalid_label_column_dataframe_input(self):
        """TC13: Invalid label column with DataFrame input should raise ValueError"""
        with self.assertRaises(KeyError):
            CNN1D(
                embedding_input=dummy_embedding_df,
                label_column='nonexistent_column',
                retrain=False
            )

    def test_TC14_missing_label_column_dataframe_input(self):
        """TC14: Missing label column with DataFrame input should raise ValueError"""
        with self.assertRaises(ValueError):
            CNN1D(
                embedding_input=dummy_embedding_df,
                label_column=None,
                retrain=False
            )

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC15_prepare_data_without_dataframe(self, mock_save, mock_makedirs, mock_exists):
        """TC15: Prepare data without DataFrame should raise ValueError"""
        model = CNN1D(
            embedding_input=768,
            num_classes=3,
            retrain=False
        )
        
        with self.assertRaises(ValueError):
            model.prepare_data()

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC16_save_model_functionality(self, mock_save, mock_makedirs, mock_exists):
        """TC16: Test save_model functionality"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Test save_model
        model.save_model()
        
        # Verify torch.save was called
        self.assertTrue(mock_save.called)
        
        # Verify the save call includes the expected structure
        save_args = mock_save.call_args[0][0]
        self.assertIn('model_state_dict', save_args)
        self.assertIn('class_names', save_args)

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC17_predict_proba_batch_functionality(self, mock_save, mock_makedirs, mock_exists):
        """TC17: Test predict_proba_batch functionality"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Create large dummy input
        dummy_input = np.random.randn(100, 768)
        
        # Test predict_proba_batch
        probabilities = model.predict_proba_batch(dummy_input, batch_size=32)
        
        # Verify probabilities shape and properties
        self.assertEqual(probabilities.shape, (100, 3))  # 100 samples, 3 classes
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))  # Probabilities sum to 1
        self.assertTrue(np.all(probabilities >= 0))  # All probabilities non-negative

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    @patch("utility.model.modelTraining_embeddings.plt.show", lambda: None)
    def test_TC18_train_and_evaluate_complete_pipeline(self, mock_save, mock_makedirs, mock_exists):
        """TC18: Test complete train_and_evaluate pipeline"""
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            retrain=False
        )
        
        # Test complete pipeline with minimal epochs
        results = model.train_and_evaluate(
            test_size=0.2,
            val_size=0.2,
            batch_size=16,
            num_epochs=1,  # Minimal epochs for testing
            learning_rate=0.01,
            plot_results=False  # Disable plotting for testing
        )
        
        # Verify results structure
        self.assertIn('model', results)
        self.assertIn('history', results)
        self.assertIn('report', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('predictions', results)
        self.assertIn('true_labels', results)
        self.assertIn('data_loaders', results)
        
        # Verify history contains expected keys
        history = results['history']
        self.assertIn('train_losses', history)
        self.assertIn('val_losses', history)
        self.assertIn('train_accs', history)
        self.assertIn('val_accs', history)

    @patch("utility.model.modelTraining_embeddings.os.path.exists", side_effect=fake_os_path_exists)
    @patch("utility.model.modelTraining_embeddings.os.makedirs", side_effect=lambda path, **kwargs: None)
    @patch("torch.save")
    def test_TC19_custom_model_path_initialization(self, mock_save, mock_makedirs, mock_exists):
        """TC19: Custom model path initialization"""
        custom_path = "../custom_models/"
        
        model = CNN1D(
            embedding_input=dummy_embedding_df,
            label_column='user',
            model_path=custom_path,
            retrain=True
        )
        
        # Verify custom model path is set
        self.assertEqual(model.model_path, custom_path)

if __name__ == "__main__":
    unittest.main(testRunner=TableTestRunner("CNN1D.csv"))