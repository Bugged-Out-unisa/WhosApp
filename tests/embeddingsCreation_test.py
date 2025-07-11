import unittest
import pandas as pd
import numpy as np
import torch
import os
import sys
from unittest.mock import patch, Mock, MagicMock
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.dataset.embeddingsCreation import EmbeddingsCreation

class MockBertModel:
    """Mock BERT model for testing."""
    def __init__(self):
        self.device = 'cpu'
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        pass
    
    def __call__(self, input_ids, attention_mask, output_hidden_states=True):
        batch_size = input_ids.shape[0]
        hidden_size = 768  # BERT base hidden size
        seq_length = input_ids.shape[1]
        
        # Create mock outputs
        last_hidden_state = torch.randn(batch_size, seq_length, hidden_size)
        
        # Mock outputs object
        outputs = Mock()
        outputs.last_hidden_state = last_hidden_state
        
        return outputs

class MockBertTokenizer:
    """Mock BERT tokenizer for testing."""
    def __init__(self):
        pass
    
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()
    
    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        batch_size = len(texts)
        seq_length = 10  # Fixed sequence length for testing
        
        # Create mock tokenizer outputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class TestEmbeddingsCreation(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # DF1: Valid dataframe with required columns
        self.df_valid = pd.DataFrame({
            "user": ["Alice", "Bob", "Charlie"],
            "message": ["Hello world", "Hi there", "Good morning"],
            "responsiveness": [0, 5, 3]
        })
        
        # DF2: Invalid dataframe (missing required 'message' column)
        self.df_missing_message = pd.DataFrame({
            "user": ["Alice", "Bob"],
            "responsiveness": [0, 5]
        })
        
        # DF3: DataFrame is None
        self.df_none = None
        
        # DF4: Empty dataframe
        self.df_empty = pd.DataFrame()
        
        # Dataset paths
        self.valid_dataset_path = "./"
        self.invalid_dataset_path = "/invalid/path/that/does/not/exist"
        
        # Embedding strategies
        self.valid_strategies = ["cls", "mean", "mixed"]
        self.invalid_strategy = "invalid_strategy"

    def _setup_bert_mocks(self):
        """Helper method to set up BERT-related mocks."""
        # Mock BERT model and tokenizer
        mock_tokenizer = MockBertTokenizer()
        mock_model = MockBertModel()
        
        return mock_tokenizer, mock_model

    # Test Case 1: Valid dataframe with CLS strategy and save enabled
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_valid_dataframe_cls_strategy_with_save(self, mock_makedirs, mock_exists, 
                                                    mock_tokenizer_pretrained, mock_model_pretrained, 
                                                    mock_cuda_available):
        """Test valid dataframe with CLS strategy and save enabled."""
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            embeddings_creator = EmbeddingsCreation(
                self.df_valid, 
                self.valid_dataset_path, 
                saveDataFrame=True,
                embeddings_strategy='cls'
            )
            
            result_df = embeddings_creator.get_dataframe()
            
            # Verify the dataframe structure
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(result_df), 3)  # Same number of rows as input
            self.assertNotIn('message', result_df.columns)  # Message column should be removed
            self.assertNotIn('responsiveness', result_df.columns)  # Responsiveness column should be removed
            
            # Check that CLS embeddings columns are present
            cls_columns = [col for col in result_df.columns if col.startswith('cls_embed_')]
            self.assertEqual(len(cls_columns), 768)  # BERT base has 768 dimensions
            
            # Verify to_parquet was called
            mock_to_parquet.assert_called_once()

    # Test Case 2: Valid dataframe with MEAN strategy and save disabled
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_valid_dataframe_mean_strategy_no_save(self, mock_makedirs, mock_exists, 
                                                   mock_tokenizer_pretrained, mock_model_pretrained, 
                                                   mock_cuda_available):
        """Test valid dataframe with MEAN strategy and save disabled."""
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            embeddings_creator = EmbeddingsCreation(
                self.df_valid, 
                self.valid_dataset_path, 
                saveDataFrame=False,
                embeddings_strategy='mean'
            )
            
            result_df = embeddings_creator.get_dataframe()
            
            # Verify the dataframe structure
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(result_df), 3)
            
            # Check that MEAN embeddings columns are present
            mean_columns = [col for col in result_df.columns if col.startswith('mean_embed_')]
            self.assertEqual(len(mean_columns), 768)
            
            # Verify to_parquet was NOT called
            mock_to_parquet.assert_not_called()

    # Test Case 3: Valid dataframe with MIXED strategy
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_valid_dataframe_mixed_strategy(self, mock_makedirs, mock_exists, 
                                           mock_tokenizer_pretrained, mock_model_pretrained, 
                                           mock_cuda_available):
        """Test valid dataframe with MIXED strategy."""
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            embeddings_creator = EmbeddingsCreation(
                self.df_valid, 
                self.valid_dataset_path, 
                saveDataFrame=False,
                embeddings_strategy='mixed'
            )
            
            result_df = embeddings_creator.get_dataframe()
            
            # Verify the dataframe structure
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(result_df), 3)
            
            # Check that MIXED embeddings columns are present (should be 768 * 2 = 1536)
            mixed_columns = [col for col in result_df.columns if col.startswith('mixed_embed_')]
            self.assertEqual(len(mixed_columns), 1536)  # CLS + MEAN embeddings concatenated

    # Test Case 4: Invalid embeddings strategy
    def test_invalid_embeddings_strategy(self):
        """Test that invalid embeddings strategy raises ValueError."""
        self.expected_output = "ValueError"
        
        with self.assertRaises(ValueError) as context:
            EmbeddingsCreation(
                self.df_valid, 
                self.valid_dataset_path, 
                saveDataFrame=False,
                embeddings_strategy=self.invalid_strategy
            )
        
        self.assertIn("embeddings_strategy can only have values", str(context.exception))

    # Test Case 5: Invalid input dataframe (missing message column)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_invalid_dataframe_missing_message(self, mock_makedirs, mock_exists, 
                                              mock_tokenizer_pretrained, mock_model_pretrained, 
                                              mock_cuda_available):
        """Test that dataframe missing message column raises KeyError."""
        self.expected_output = "KeyError"
        
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        with self.assertRaises(KeyError):
            EmbeddingsCreation(
                self.df_missing_message, 
                self.valid_dataset_path, 
                saveDataFrame=False,
                embeddings_strategy='cls'
            )

    # Test Case 6: None dataframe
    def test_none_dataframe(self):
        """Test that None dataframe raises TypeError."""
        self.expected_output = "TypeError"
        
        with self.assertRaises(ValueError):
            EmbeddingsCreation(
                self.df_none, 
                self.valid_dataset_path, 
                saveDataFrame=False,
                embeddings_strategy='cls'
            )

    # Test Case 7: Empty dataframe
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_empty_dataframe(self, mock_makedirs, mock_exists, 
                            mock_tokenizer_pretrained, mock_model_pretrained, 
                            mock_cuda_available):
        """Test that empty dataframe raises KeyError."""
        self.expected_output = "KeyError"
        
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        with self.assertRaises(KeyError):
            EmbeddingsCreation(
                self.df_empty, 
                self.valid_dataset_path, 
                saveDataFrame=False,
                embeddings_strategy='cls'
            )

    # Test Case 8: Invalid dataset path causing save failure
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_invalid_dataset_path_save_failure(self, mock_makedirs, mock_exists, 
                                              mock_tokenizer_pretrained, mock_model_pretrained, 
                                              mock_cuda_available):
        """Test that invalid dataset path causes save failure."""
        self.expected_output = "OSError"
        
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        with patch.object(pd.DataFrame, 'to_parquet', side_effect=OSError("Invalid path")):
            with self.assertRaises(OSError):
                EmbeddingsCreation(
                    self.df_valid, 
                    self.invalid_dataset_path, 
                    saveDataFrame=True,
                    embeddings_strategy='cls'
                )

    # Test Case 9: Test dataset path processing
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_dataset_path_processing(self, mock_makedirs, mock_exists, 
                                    mock_tokenizer_pretrained, mock_model_pretrained, 
                                    mock_cuda_available):
        """Test that dataset path is correctly processed with embeddings_ prefix."""
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        test_path = "test_dataset.parquet"
        
        with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            embeddings_creator = EmbeddingsCreation(
                self.df_valid, 
                test_path, 
                saveDataFrame=True,
                embeddings_strategy='cls'
            )
            
            # Check that the path was processed correctly
            expected_path = "embeddings_test_dataset.parquet"
            mock_to_parquet.assert_called_once_with(expected_path, index=False)

    # Test Case 10: Test with CUDA available
    @patch('torch.cuda.is_available', return_value=True)
    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizerFast.from_pretrained')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_cuda_available(self, mock_makedirs, mock_exists, 
                           mock_tokenizer_pretrained, mock_model_pretrained, 
                           mock_cuda_available):
        """Test that CUDA is properly handled when available."""
        mock_tokenizer, mock_model = self._setup_bert_mocks()
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        mock_model_pretrained.return_value = mock_model
        
        # Mock the tensor.to() method to verify CUDA usage
        with patch.object(torch.Tensor, 'to') as mock_tensor_to:
            mock_tensor_to.return_value = torch.randn(1, 10)  # Return dummy tensor
            
            embeddings_creator = EmbeddingsCreation(
                self.df_valid, 
                self.valid_dataset_path, 
                saveDataFrame=False,
                embeddings_strategy='cls'
            )
            
            result_df = embeddings_creator.get_dataframe()
            self.assertIsInstance(result_df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("EmbeddingsCreation.csv"))