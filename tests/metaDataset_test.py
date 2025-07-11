import unittest
import pandas as pd
import numpy as np
import os
import sys
import warnings
from unittest.mock import patch, Mock, MagicMock
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.dataset.metaDataset import MetaDataset
from utility.model.model_list import models

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MockModelTraining:
    """Mock ModelTraining class for testing."""
    
    def __init__(self, output_name, model_choice, data, config, retrain):
        self.output_name = output_name
        self.model_choice = model_choice
        self.data = data
        self.config = config
        self.retrain = retrain
        self.mock_model = Mock()
        
    def train(self, plot_results=False):
        """Mock training method."""
        pass
        
    def get_model(self):
        """Return mock model with predict_proba method."""
        # Create a mock model that returns probabilities based on input size
        def mock_predict_proba(X):
            n_samples = len(X)
            n_classes = 3  # Mock 3 classes
            # Generate mock probabilities that sum to 1
            return np.random.dirichlet(np.ones(n_classes), size=n_samples)
        
        self.mock_model.predict_proba = mock_predict_proba
        return self.mock_model

class MockCNN1D:
    """Mock CNN1D class for testing."""
    
    def __init__(self, data, output_name, retrain):
        self.data = data
        self.output_name = output_name
        self.retrain = retrain
        
    def train_and_evaluate(self, criterion=None, plot_results=False):
        """Mock training method."""
        pass
        
    def predict_proba_batch(self, data):
        """Mock prediction method."""
        n_samples = len(data)
        n_classes = 3  # Mock 3 classes
        
        # Generate mock probabilities that sum to 1
        mock_probabilities = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        return mock_probabilities

class MockFocalLoss:
    """Mock FocalLoss class for testing."""
    
    def __init__(self, alpha=0.5, gamma=4):
        self.alpha = alpha
        self.gamma = gamma

class TestMetaDataset(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock feature training dataset
        self.feature_train_dataset = pd.DataFrame({
            'message_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'user': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'feature3': np.random.randn(10)
        })
        
        # Create mock embeddings training dataset
        self.embeddings_train_dataset = pd.DataFrame({
            'message_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'user': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        })
        
        # Add mock embedding columns
        for i in range(768):  # BERT-like embeddings
            self.embeddings_train_dataset[f'embed_{i}'] = np.random.randn(10)
            
        # Mock probabilities for testing - ensure they're well-bounded for logit function
        # Use fixed seed for reproducible tests
        np.random.seed(42)
        self.mock_probs_feature = np.random.dirichlet(np.ones(3) * 2, size=5)  # More balanced probabilities
        self.mock_probs_cnn = np.random.dirichlet(np.ones(3) * 2, size=5)
        
        # Ensure probabilities are well within (0,1) range to avoid logit issues
        self.mock_probs_feature = np.clip(self.mock_probs_feature, 0.01, 0.99)
        self.mock_probs_cnn = np.clip(self.mock_probs_cnn, 0.01, 0.99)
        
        # Re-normalize after clipping
        self.mock_probs_feature = self.mock_probs_feature / self.mock_probs_feature.sum(axis=1, keepdims=True)
        self.mock_probs_cnn = self.mock_probs_cnn / self.mock_probs_cnn.sum(axis=1, keepdims=True)
        
        # Test parameters
        self.output_name = "test_model"
        self.model_choice = models["random_forest"]
        self.config = "config.json"
        self.n_folds = 3

    def test_build_single_message_meta_valid_input(self):
        """Test build_single_message_meta with valid probability arrays."""
        probs_feature = self.mock_probs_feature
        probs_cnn = self.mock_probs_cnn
        
        # Mock the logit function to avoid PyTorch tensor issues
        with patch('utility.dataset.metaDataset.logit') as mock_logit:
            # Create a mock logit function that works with numpy arrays
            def mock_logit_func(x):
                # Convert to numpy if it's not already, and apply logit transformation
                x_np = np.array(x) if not isinstance(x, np.ndarray) else x
                return np.log(x_np / (1 - x_np))
            
            mock_logit.side_effect = mock_logit_func
            
            result = MetaDataset.build_single_message_meta(probs_feature, probs_cnn)
        
        # Verify the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check basic structure
        self.assertEqual(len(result), 5)  # Same number of samples
        
        # Check that original probability columns are present
        expected_feature_cols = [f"probs_feature{i}" for i in range(3)]
        expected_cnn_cols = [f"probs_cnn{i}" for i in range(3)]
        
        for col in expected_feature_cols + expected_cnn_cols:
            self.assertIn(col, result.columns)
        
        # Check that enhanced features are present
        enhanced_features = ['mean_prob_feature', 'mean_prob_cnn', 'model_disagreement']
        for feature in enhanced_features:
            self.assertIn(feature, result.columns)

    def test_build_single_message_meta_empty_input(self):
        """Test build_single_message_meta with empty probability arrays."""
        self.expected_output = "IndexError"
        
        probs_feature = np.array([]).reshape(0, 3)
        probs_cnn = np.array([]).reshape(0, 3)
        
        # Mock the logit function to avoid PyTorch tensor issues
        with patch('utility.dataset.metaDataset.logit') as mock_logit:
            # Create a mock logit function that works with numpy arrays
            def mock_logit_func(x):
                # Convert to numpy if it's not already, and apply logit transformation
                x_np = np.array(x) if not isinstance(x, np.ndarray) else x
                return np.log(x_np / (1 - x_np))
            
            mock_logit.side_effect = mock_logit_func
            
            result = MetaDataset.build_single_message_meta(probs_feature, probs_cnn)

        # Verify the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that it has no rows
        self.assertEqual(len(result), 0)

    def test_build_single_message_meta_mismatched_dimensions(self):
        """Test build_single_message_meta with mismatched probability array dimensions."""
        self.expected_output = "ValueError"
        
        # Create properly bounded probabilities but with mismatched dimensions
        probs_feature = np.random.dirichlet(np.ones(3) * 2, size=5)
        probs_cnn = np.random.dirichlet(np.ones(4) * 2, size=3)  # Different dimensions
        
        # Ensure probabilities are well within (0,1) range
        probs_feature = np.clip(probs_feature, 0.01, 0.99)
        probs_cnn = np.clip(probs_cnn, 0.01, 0.99)
        
        # Re-normalize after clipping
        probs_feature = probs_feature / probs_feature.sum(axis=1, keepdims=True)
        probs_cnn = probs_cnn / probs_cnn.sum(axis=1, keepdims=True)
    
        # Mock the logit function to avoid PyTorch tensor issues
        with patch('utility.dataset.metaDataset.logit') as mock_logit:
            # Create a mock logit function that works with numpy arrays
            def mock_logit_func(x):
                # Convert to numpy if it's not already, and apply logit transformation
                x_np = np.array(x) if not isinstance(x, np.ndarray) else x
                return np.log(x_np / (1 - x_np))
            
            mock_logit.side_effect = mock_logit_func
            
            with self.assertRaises(ValueError):
                result = MetaDataset.build_single_message_meta(probs_feature, probs_cnn)

    @patch('utility.dataset.metaDataset.CNN1D')
    @patch('utility.dataset.metaDataset.ModelTraining')
    @patch('utility.dataset.metaDataset.FocalLoss')
    @patch('sklearn.model_selection.KFold')
    def test_build_simple_meta_dataset_valid_input(self, mock_kfold, mock_focal_loss, 
                                                   mock_model_training, mock_cnn1d):
        """Test build_simple_meta_dataset with valid inputs."""
        # Mock KFold splits
        mock_kfold_instance = Mock()
        mock_kfold.return_value = mock_kfold_instance
        
        # Create mock fold splits
        train_indices = [np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])]
        val_indices = [np.array([5, 6, 7, 8, 9]), np.array([0, 1, 2, 3, 4])]
        mock_splits = list(zip(train_indices, val_indices))
        mock_kfold_instance.split.return_value = mock_splits
        
        # Mock ModelTraining
        mock_model_training.return_value = MockModelTraining("test", "rf", pd.DataFrame(), {}, True)
        
        # Mock CNN1D - return instance directly
        mock_cnn1d.return_value = MockCNN1D(pd.DataFrame(), "test", True)
        
        # Mock FocalLoss
        mock_focal_loss.return_value = MockFocalLoss()
        
        result = MetaDataset.build_simple_meta_dataset(
            self.output_name,
            self.model_choice,
            self.config,
            self.feature_train_dataset,
            self.embeddings_train_dataset,
            n_folds=2  # Use 2 folds for testing
        )
        
        # Verify the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that it has the expected number of samples
        self.assertEqual(len(result), 10)  # All samples should be present
        
        # Check that probability columns are present
        prob_cols = [col for col in result.columns if col.startswith('probs_')]
        self.assertTrue(len(prob_cols) > 0)
        
        # Check that user column is present
        self.assertIn('user', result.columns)

    @patch('utility.dataset.metaDataset.CNN1D')
    @patch('utility.dataset.metaDataset.ModelTraining')
    @patch('utility.dataset.metaDataset.FocalLoss')
    @patch('sklearn.model_selection.KFold')
    def test_build_simple_meta_dataset_empty_dataset(self, mock_kfold, mock_focal_loss, 
                                                     mock_model_training, mock_cnn1d):
        """Test build_simple_meta_dataset with empty datasets."""
        self.expected_output = "ValueError"
        
        empty_feature_dataset = pd.DataFrame()
        empty_embeddings_dataset = pd.DataFrame()
        
        # Mock KFold to raise error for empty dataset
        mock_kfold_instance = Mock()
        mock_kfold.return_value = mock_kfold_instance
        mock_kfold_instance.split.side_effect = ValueError("Empty dataset")
        
        with self.assertRaises(ValueError):
            MetaDataset.build_simple_meta_dataset(
                self.output_name,
                self.model_choice,
                self.config,
                empty_feature_dataset,
                empty_embeddings_dataset,
                self.n_folds
            )

    @patch('utility.dataset.metaDataset.CNN1D')
    @patch('utility.dataset.metaDataset.ModelTraining')
    @patch('utility.dataset.metaDataset.FocalLoss')
    @patch('sklearn.model_selection.KFold')
    def test_build_simple_meta_dataset_missing_columns(self, mock_kfold, mock_focal_loss, 
                                                       mock_model_training, mock_cnn1d):
        """Test build_simple_meta_dataset with datasets missing required columns."""
        self.expected_output = "KeyError"
        
        # Dataset missing 'user' column
        invalid_feature_dataset = pd.DataFrame({
            'message_id': [1, 2, 3],
            'feature1': [1, 2, 3]
        })
        
        invalid_embeddings_dataset = pd.DataFrame({
            'message_id': [1, 2, 3],
            'embed_0': [1, 2, 3]
        })
        
        # Mock KFold
        mock_kfold_instance = Mock()
        mock_kfold.return_value = mock_kfold_instance
        mock_kfold_instance.split.return_value = [(np.array([0, 1]), np.array([2]))]
        
        # Mock ModelTraining
        mock_model_training.return_value = MockModelTraining("test", "rf", pd.DataFrame(), {}, True)
        
        # Mock CNN1D
        mock_cnn1d.return_value = MockCNN1D(pd.DataFrame(), "test", True)
        
        with self.assertRaises(KeyError):
            MetaDataset.build_simple_meta_dataset(
                self.output_name,
                self.model_choice,
                self.config,
                invalid_feature_dataset,
                invalid_embeddings_dataset,
                self.n_folds
            )

    def test_enhance_meta_dataset_valid_input(self):
        """Test enhance_meta_dataset with valid DataFrame input."""
        # Create a simple meta dataset
        df_meta = pd.DataFrame({
            'probs_feature0': [0.7, 0.3, 0.5],
            'probs_feature1': [0.2, 0.4, 0.3],
            'probs_feature2': [0.1, 0.3, 0.2],
            'probs_cnn0': [0.6, 0.2, 0.4],
            'probs_cnn1': [0.3, 0.5, 0.4],
            'probs_cnn2': [0.1, 0.3, 0.2]
        })
        
        # Mock the logit function to avoid PyTorch tensor issues
        with patch('utility.dataset.metaDataset.logit') as mock_logit:
            # Create a mock logit function that works with numpy arrays
            def mock_logit_func(x):
                # Convert to numpy if it's not already, and apply logit transformation
                x_np = np.array(x) if not isinstance(x, np.ndarray) else x
                return np.log(x_np / (1 - x_np))
            
            mock_logit.side_effect = mock_logit_func
            
            result = MetaDataset.enhance_meta_dataset(df_meta)
        
        # Verify the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that original columns are preserved
        for col in df_meta.columns:
            self.assertIn(col, result.columns)
        
        # Check that enhanced features are added
        expected_features = [
            'mean_prob_feature', 'mean_prob_cnn', 'std_prob_feature', 'std_prob_cnn',
            'model_disagreement', 'margin_feature', 'margin_cnn', 'entropy_feature',
            'entropy_cnn', 'confidence_feature', 'confidence_cnn'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check that values are computed correctly
        self.assertTrue(np.allclose(result['mean_prob_feature'], 
                                   df_meta[['probs_feature0', 'probs_feature1', 'probs_feature2']].mean(axis=1)))

    def test_enhance_meta_dataset_empty_input(self):
        """Test enhance_meta_dataset with empty DataFrame input."""
        self.expected_output = "IndexError"
        
        df_meta = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            MetaDataset.enhance_meta_dataset(df_meta)

    def test_enhance_meta_dataset_missing_probability_columns(self):
        """Test enhance_meta_dataset with DataFrame missing probability columns."""
        self.expected_output = "IndexError"
        
        df_meta = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': [4, 5, 6]
        })
        
        with self.assertRaises(ValueError):
            MetaDataset.enhance_meta_dataset(df_meta)

if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("MetaDataset.csv"))