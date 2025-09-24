"""
Test cases for embedding model with fallback mechanism.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import logging

from embedding_model import IndustrialEmbeddingModel, create_industrial_embedding_model

# Suppress logs during testing
logging.getLogger('embedding_model').setLevel(logging.ERROR)


class TestEmbeddingModel(unittest.TestCase):
    """Test cases for IndustrialEmbeddingModel with fallback mechanism."""
    
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
        # Mock the validation embedding
        mock_model.encode.return_value = np.random.rand(1, 768)
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel()
        
        # Verify primary model was loaded
        self.assertEqual(embedding_model.model_config.name, "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(embedding_model.model_config.dimension, 768)
        diagnostics = embedding_model.get_comprehensive_diagnostics()
        self.assertTrue(diagnostics['system_status']['model_loaded'])
    
    @patch('embedding_model.SentenceTransformer')
    def test_fallback_mechanism(self, mock_sentence_transformer):
        """Test fallback to MiniLM when MPNet fails."""
        # Mock primary model failure and successful fallback
        def side_effect(model_name):
            if "all-mpnet-base-v2" in model_name:
                raise Exception("Primary model failed to load")
            else:
                mock_model = MagicMock()
                mock_model.encode.return_value = np.random.rand(1, 384)
                return mock_model
        
        mock_sentence_transformer.side_effect = side_effect
        
        # Initialize model (should fallback)
        embedding_model = IndustrialEmbeddingModel()
        
        # Verify fallback model was loaded
        self.assertIn("all-MiniLM-L6-v2", embedding_model.model_config.name)
        self.assertEqual(embedding_model.model_config.dimension, 384)
        self.assertTrue(embedding_model.quality_metrics['model_switches'] >= 1)
    
    @patch('embedding_model.SentenceTransformer')
    def test_both_models_fail(self, mock_sentence_transformer):
        """Test exception when both models fail to load."""
        # Mock both models failing
        mock_sentence_transformer.side_effect = Exception("All models failed")
        
        # Should raise ModelInitializationError
        with self.assertRaises(Exception) as context:
            IndustrialEmbeddingModel()
        
        self.assertIn("Failed to initialize any embedding model", str(context.exception))
    
    @patch('embedding_model.SentenceTransformer')
    def test_force_fallback(self, mock_sentence_transformer):
        """Test forcing fallback to MiniLM."""
        # Mock successful fallback model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer.return_value = mock_model
        
        # Force fallback by using secondary_efficient model
        embedding_model = IndustrialEmbeddingModel(preferred_model='secondary_efficient')
        
        # Verify secondary model was loaded
        self.assertIn("all-MiniLM-L6-v2", embedding_model.model_config.name)
        self.assertEqual(embedding_model.model_config.dimension, 384)
    
    @patch('embedding_model.SentenceTransformer')
    def test_encode_functionality(self, mock_sentence_transformer):
        """Test encoding functionality."""
        # Mock model
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize and test encoding
        embedding_model = IndustrialEmbeddingModel()
        result = embedding_model.encode(self.test_sentences)
        
        # Verify encoding was called and returns correct shape
        mock_model.encode.assert_called()
        self.assertEqual(result.shape, (3, 384))
    
    @patch('embedding_model.SentenceTransformer')
    def test_batch_size_optimization(self, mock_sentence_transformer):
        """Test batch size optimization based on model type."""
        # Test with MPNet (primary)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768)
        mock_sentence_transformer.return_value = mock_model
        
        embedding_model = IndustrialEmbeddingModel(preferred_model='primary_large')
        
        # Test with primary model
        batch_size = embedding_model._calculate_optimal_batch_size(50)
        self.assertGreaterEqual(batch_size, 1)
        
        # Test with secondary model
        embedding_model = IndustrialEmbeddingModel(preferred_model='secondary_efficient')
        batch_size = embedding_model._calculate_optimal_batch_size(50)
        self.assertGreaterEqual(batch_size, 1)
    
    @patch('embedding_model.SentenceTransformer')
    def test_model_info(self, mock_sentence_transformer):
        """Test model diagnostics retrieval."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer.return_value = mock_model
        
        embedding_model = IndustrialEmbeddingModel(preferred_model='secondary_efficient')
        diagnostics = embedding_model.get_comprehensive_diagnostics()
        
        # Verify diagnostics structure
        self.assertIn('model_info', diagnostics)
        self.assertIn('performance_metrics', diagnostics)
        self.assertIn('system_status', diagnostics)
        
        # Verify model info
        model_info = diagnostics['model_info']
        self.assertIn('name', model_info)
        self.assertIn('dimension', model_info)
        self.assertIn('quality_tier', model_info)
    
    @patch('embedding_model.SentenceTransformer')
    def test_factory_function(self, mock_sentence_transformer):
        """Test create_industrial_embedding_model factory function."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768)
        mock_sentence_transformer.return_value = mock_model
        
        # Test normal creation
        model1 = create_industrial_embedding_model()
        self.assertIsInstance(model1, IndustrialEmbeddingModel)
        
        # Test with different tier
        model2 = create_industrial_embedding_model(model_tier="basic")
        self.assertIsInstance(model2, IndustrialEmbeddingModel)
    
    @patch('embedding_model.SentenceTransformer')
    def test_similarity_calculation(self, mock_sentence_transformer):
        """Test similarity calculation between embeddings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer.return_value = mock_model
        
        embedding_model = IndustrialEmbeddingModel()
        
        # Create mock embeddings
        embeddings1 = np.random.rand(2, 384)
        embeddings2 = np.random.rand(2, 384)
        
        # Test similarity calculation
        similarity_scores = embedding_model.compute_similarity(embeddings1, embeddings2)
        
        # Verify output shape
        self.assertEqual(similarity_scores.shape, (2, 2))
        
        # Verify similarity scores are in valid range [-1, 1]
        self.assertTrue(np.all(similarity_scores >= -1))
        self.assertTrue(np.all(similarity_scores <= 1))

    @patch('embedding_model.SentenceTransformer')
    def test_semantic_search_basic(self, mock_sentence_transformer):
        """Test basic semantic search functionality."""
        # Mock model setup
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Mock embeddings - create query and document embeddings
        query_embedding = np.random.rand(384).astype(np.float32)
        doc_embeddings = np.random.rand(3, 384).astype(np.float32)
        
        # Mock encode calls: first for query, then for documents
        def encode_side_effect(texts, *args, **kwargs):
            if len(texts) == 1:  # Query
                return query_embedding.reshape(1, -1)
            else:  # Documents
                return doc_embeddings
                
        mock_model.encode.side_effect = encode_side_effect
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel()
        
        # Test semantic search
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = embedding_model.semantic_search("test query", documents, k=2)
        
        # Verify results structure
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(len(result), 4)  # (index, page, text, score)
            self.assertIsInstance(result[0], int)
            self.assertIsInstance(result[1], str)
            self.assertIsInstance(result[2], str)
            self.assertIsInstance(result[3], float)

    @patch('embedding_model.SentenceTransformer')
    def test_semantic_search_with_pages(self, mock_sentence_transformer):
        """Test semantic search with custom page identifiers."""
        # Mock model setup
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Mock embeddings
        query_embedding = np.random.rand(384).astype(np.float32)
        doc_embeddings = np.random.rand(3, 384).astype(np.float32)
        
        def encode_side_effect(texts, *args, **kwargs):
            if len(texts) == 1:
                return query_embedding.reshape(1, -1)
            else:
                return doc_embeddings
                
        mock_model.encode.side_effect = encode_side_effect
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel()
        
        # Test with custom pages
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        pages = ["Page A", "Page B", "Page C"]
        results = embedding_model.semantic_search("test query", documents, pages=pages, k=3, return_scores=False)
        
        # Verify results structure without scores
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(len(result), 3)  # (index, page, text)
            self.assertIn(result[1], pages)  # Page should be from our custom list

    @patch('embedding_model.SentenceTransformer')
    def test_torch_topk_integration(self, mock_sentence_transformer):
        """Test that torch.topk is used correctly for efficient retrieval."""
        # Mock model setup
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Create predictable normalized embeddings for testing
        # Query embedding: first dimension = 1, rest = 0, then normalized
        query_embedding = np.array([1.0] + [0.0] * 383).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Document embeddings with decreasing similarity to query
        doc_embeddings = np.array([
            [0.9] + [0.1] * 383,   # High similarity (more orthogonal components)
            [0.5] + [0.5] * 383,   # Medium similarity
            [0.1] + [0.9] * 383,   # Low similarity
        ]).astype(np.float32)
        
        # Normalize document embeddings
        for i in range(len(doc_embeddings)):
            doc_embeddings[i] = doc_embeddings[i] / np.linalg.norm(doc_embeddings[i])
        
        def encode_side_effect(texts, *args, **kwargs):
            if len(texts) == 1:
                return query_embedding.reshape(1, -1)
            else:
                return doc_embeddings
                
        mock_model.encode.side_effect = encode_side_effect
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel()
        
        # Test semantic search ordering
        documents = ["High sim doc", "Medium sim doc", "Low sim doc"]
        results = embedding_model.semantic_search("test query", documents, k=3)
        
        # Verify results are sorted by similarity (highest first)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 0)  # Index 0 should have highest score
        self.assertEqual(results[1][0], 1)  # Index 1 should have medium score  
        self.assertEqual(results[2][0], 2)  # Index 2 should have lowest score
        
        # Verify scores are in descending order
        self.assertGreater(results[0][3], results[1][3])
        self.assertGreater(results[1][3], results[2][3])

    @patch('embedding_model.SentenceTransformer')
    def test_semantic_search_empty_documents(self, mock_sentence_transformer):
        """Test semantic search with empty document list."""
        # Mock model setup
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel()
        
        # Test with empty documents
        results = embedding_model.semantic_search("test query", [], k=5)
        
        # Should return empty list
        self.assertEqual(len(results), 0)
        self.assertEqual(results, [])

    @patch('embedding_model.SentenceTransformer')
    def test_semantic_search_k_larger_than_documents(self, mock_sentence_transformer):
        """Test semantic search when k is larger than number of documents."""
        # Mock model setup
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # Mock embeddings
        query_embedding = np.random.rand(384).astype(np.float32)
        doc_embeddings = np.random.rand(2, 384).astype(np.float32)
        
        def encode_side_effect(texts, *args, **kwargs):
            if len(texts) == 1:
                return query_embedding.reshape(1, -1)
            else:
                return doc_embeddings
                
        mock_model.encode.side_effect = encode_side_effect
        
        # Initialize model
        embedding_model = IndustrialEmbeddingModel()
        
        # Test with k larger than number of documents
        documents = ["Doc 1", "Doc 2"]
        results = embedding_model.semantic_search("test query", documents, k=10)
        
        # Should return only available documents
        self.assertEqual(len(results), 2)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)