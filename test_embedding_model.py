"""
Test cases for embedding model with fallback mechanism.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import logging

from embedding_model import EmbeddingModel, create_embedding_model

# Suppress logs during testing
logging.getLogger('embedding_model').setLevel(logging.ERROR)


class TestEmbeddingModel(unittest.TestCase):
    """Test cases for EmbeddingModel with fallback mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_sentences = [
            "This is a test sentence.",
            "Another example for testing.",
            "Testing the embedding model."
        ]
    
    @patch('embedding_model.SentenceTransformer')
    def test_primary_model_success(self, mock_sentence_transformer):
        """Test successful loading of primary MPNet model."""
        # Mock successful primary model loading
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize model
        embedding_model = EmbeddingModel()
        
        # Verify primary model was loaded
        self.assertEqual(embedding_model.model_name, EmbeddingModel.PRIMARY_MODEL)
        self.assertEqual(embedding_model.embedding_dimension, 768)
        self.assertFalse(embedding_model.get_model_info()['is_fallback'])
    
    @patch('embedding_model.SentenceTransformer')
    def test_fallback_mechanism(self, mock_sentence_transformer):
        """Test fallback to MiniLM when MPNet fails."""
        # Mock primary model failure and successful fallback
        def side_effect(model_name):
            if model_name == EmbeddingModel.PRIMARY_MODEL:
                raise Exception("Primary model failed to load")
            else:
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                return mock_model
        
        mock_sentence_transformer.side_effect = side_effect
        
        # Initialize model (should fallback)
        embedding_model = EmbeddingModel()
        
        # Verify fallback model was loaded
        self.assertEqual(embedding_model.model_name, EmbeddingModel.FALLBACK_MODEL)
        self.assertEqual(embedding_model.embedding_dimension, 384)
        self.assertTrue(embedding_model.get_model_info()['is_fallback'])
    
    @patch('embedding_model.SentenceTransformer')
    def test_both_models_fail(self, mock_sentence_transformer):
        """Test exception when both models fail to load."""
        # Mock both models failing
        mock_sentence_transformer.side_effect = Exception("All models failed")
        
        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            EmbeddingModel()
        
        self.assertIn("Both primary and fallback models failed", str(context.exception))
    
    @patch('embedding_model.SentenceTransformer')
    def test_force_fallback(self, mock_sentence_transformer):
        """Test forcing fallback to MiniLM."""
        # Mock successful fallback model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Force fallback
        embedding_model = EmbeddingModel(force_fallback=True)
        
        # Verify only fallback model was attempted
        mock_sentence_transformer.assert_called_once_with(EmbeddingModel.FALLBACK_MODEL)
        self.assertEqual(embedding_model.model_name, EmbeddingModel.FALLBACK_MODEL)
        self.assertTrue(embedding_model.get_model_info()['is_fallback'])
    
    @patch('embedding_model.SentenceTransformer')
    def test_encode_functionality(self, mock_sentence_transformer):
        """Test encoding functionality."""
        # Mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize and test encoding
        embedding_model = EmbeddingModel()
        result = embedding_model.encode(self.test_sentences)
        
        # Verify encoding was called correctly
        mock_model.encode.assert_called_once()
        np.testing.assert_array_equal(result, mock_embeddings)
    
    @patch('embedding_model.SentenceTransformer')
    def test_batch_size_optimization(self, mock_sentence_transformer):
        """Test batch size optimization based on model type."""
        # Test with MPNet (primary)
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        embedding_model = EmbeddingModel()
        embedding_model.model_name = EmbeddingModel.PRIMARY_MODEL
        
        batch_size = embedding_model._get_optimal_batch_size()
        self.assertEqual(batch_size, 16)
        
        # Test with MiniLM (fallback)
        embedding_model.model_name = EmbeddingModel.FALLBACK_MODEL
        batch_size = embedding_model._get_optimal_batch_size()
        self.assertEqual(batch_size, 32)
    
    @patch('embedding_model.SentenceTransformer')
    def test_model_info(self, mock_sentence_transformer):
        """Test model info retrieval."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedding_model = EmbeddingModel(force_fallback=True)
        info = embedding_model.get_model_info()
        
        expected_info = {
            "model_name": EmbeddingModel.FALLBACK_MODEL,
            "embedding_dimension": 384,
            "is_fallback": True,
            "primary_model": EmbeddingModel.PRIMARY_MODEL,
            "fallback_model": EmbeddingModel.FALLBACK_MODEL
        }
        
        self.assertEqual(info, expected_info)
    
    @patch('embedding_model.SentenceTransformer')
    def test_factory_function(self, mock_sentence_transformer):
        """Test create_embedding_model factory function."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        # Test normal creation
        model1 = create_embedding_model()
        self.assertIsInstance(model1, EmbeddingModel)
        
        # Test with force_fallback
        model2 = create_embedding_model(force_fallback=True)
        self.assertIsInstance(model2, EmbeddingModel)
    
    @patch('embedding_model.SentenceTransformer')
    def test_similarity_calculation(self, mock_sentence_transformer):
        """Test similarity calculation between embeddings."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedding_model = EmbeddingModel()
        
        # Create mock embeddings
        embeddings1 = np.random.rand(2, 384)
        embeddings2 = np.random.rand(2, 384)
        
        # Test similarity calculation
        similarity_scores = embedding_model.similarity(embeddings1, embeddings2)
        
        # Verify output shape
        self.assertEqual(similarity_scores.shape, (2, 2))
        
        # Verify similarity scores are in valid range [-1, 1]
        self.assertTrue(np.all(similarity_scores >= -1))
        self.assertTrue(np.all(similarity_scores <= 1))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)