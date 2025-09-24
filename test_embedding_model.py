"""
Test cases for embedding model with fallback mechanism.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import tempfile
import shutil
import logging

from embedding_model import IndustrialEmbeddingModel, create_embedding_model

# Suppress logs during testing
logging.getLogger('embedding_model').setLevel(logging.ERROR)


class TestIndustrialEmbeddingModel(unittest.TestCase):
    """Test cases for IndustrialEmbeddingModel with memory management and caching."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_sentences = [
            "This is a test sentence.",
            "Another example for testing.",
            "Testing the embedding model."
        ]
        self.temp_cache_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_cache_dir'):
            shutil.rmtree(self.temp_cache_dir, ignore_errors=True)
    
    @patch('embedding_model.SentenceTransformer')
    def test_memory_managed_encoding(self, mock_sentence_transformer):
        """Test memory-managed encoding with torch.no_grad()."""
        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(3, 768, dtype=torch.float32)
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize model with memory management
        embedding_model = IndustrialEmbeddingModel(
            preferred_model='primary_large',
            memory_threshold=0.8
        )
        
        # Test encoding
        embeddings = embedding_model.encode(self.test_sentences)
        
        # Verify output format
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertEqual(embeddings.shape, (3, 768))
    
    @patch('embedding_model.SentenceTransformer')
    def test_disk_cache_functionality(self, mock_sentence_transformer):
        """Test disk caching with torch.save/load."""
        # Mock model  
        mock_model = MagicMock()
        test_embeddings = torch.randn(2, 768, dtype=torch.float32)
        mock_model.encode.return_value = test_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize model with disk caching
        embedding_model = IndustrialEmbeddingModel(
            enable_disk_cache=True,
            cache_size=100
        )
        
        # Override cache directory for testing
        embedding_model.embedding_cache.cache_dir = self.temp_cache_dir
        
        # First encoding - should cache to disk
        test_texts = ["First text", "Second text"]
        embeddings1 = embedding_model.encode(test_texts)
        
        # Verify cache file was created
        cache_files = list(embedding_model.embedding_cache.cache_dir.glob("*.pt"))
        self.assertTrue(len(cache_files) > 0)
        
        # Second encoding - should hit cache
        mock_model.encode.reset_mock()
        embeddings2 = embedding_model.encode(test_texts)
        
        # Verify cache was used (model.encode not called again)
        self.assertFalse(mock_model.encode.called)
        np.testing.assert_array_equal(embeddings1, embeddings2)
    
    @patch('embedding_model.SentenceTransformer')
    @patch('psutil.virtual_memory')
    def test_memory_adaptive_batch_size(self, mock_memory, mock_sentence_transformer):
        """Test adaptive batch sizing based on memory constraints."""
        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(10, 768, dtype=torch.float32)
        mock_sentence_transformer.return_value = mock_model
        
        # Mock high memory usage scenario
        mock_memory_info = MagicMock()
        mock_memory_info.percent = 85.0  # High memory usage
        mock_memory_info.available = 1024**3  # 1GB available
        mock_memory.return_value = mock_memory_info
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel(preferred_model='primary_large')
        
        # Test batch size calculation
        test_texts = ["text"] * 100
        batch_size = embedding_model._calculate_optimal_batch_size(len(test_texts))
        
        # Should be reduced due to high memory usage
        self.assertLess(batch_size, embedding_model.model_config.batch_size)
    
    @patch('embedding_model.SentenceTransformer')
    def test_chunked_processing(self, mock_sentence_transformer):
        """Test chunked processing for large datasets."""
        # Mock model
        mock_model = MagicMock()
        
        def mock_encode(texts, **kwargs):
            # Return appropriately sized tensor for each chunk
            return torch.randn(len(texts), 768, dtype=torch.float32)
        
        mock_model.encode.side_effect = mock_encode
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize model with small batch size to force chunking
        embedding_model = IndustrialEmbeddingModel(preferred_model='fallback_fast')
        
        # Process large dataset
        large_texts = ["text"] * 200
        embeddings = embedding_model.encode(large_texts, batch_size=32)
        
        # Verify output shape
        self.assertEqual(embeddings.shape, (200, 384))  # fallback_fast has 384 dims
        
        # Verify multiple encode calls were made (chunking happened)
        self.assertGreater(mock_model.encode.call_count, 1)
    
    @patch('embedding_model.SentenceTransformer') 
    def test_torch_tensor_dtype_consistency(self, mock_sentence_transformer):
        """Test that all tensors use float32 dtype."""
        # Mock model with different dtype
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(2, 768, dtype=torch.float64)  # Wrong dtype
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel(preferred_model='primary_large')
        
        # Test encoding
        embeddings = embedding_model.encode(["test1", "test2"])
        
        # Verify output is float32
        self.assertEqual(embeddings.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()