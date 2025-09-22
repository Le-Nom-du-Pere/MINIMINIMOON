"""
Embedding model with fallback mechanism.
Attempts to load MPNet model first, falls back to MiniLM if loading fails.
"""

import logging
import warnings
from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Embedding model with automatic fallback mechanism.
    Tries MPNet first, falls back to MiniLM if MPNet fails to load.
    """
    
    # Model configurations
    PRIMARY_MODEL = "sentence-transformers/all-mpnet-base-v2"
    FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, force_fallback: bool = False):
        """
        Initialize embedding model with fallback mechanism.
        
        Args:
            force_fallback: If True, skip MPNet and use MiniLM directly
        """
        self.model = None
        self.model_name = None
        self.embedding_dimension = None
        self._initialize_model(force_fallback)
    
    def _initialize_model(self, force_fallback: bool = False) -> None:
        """Initialize the embedding model with fallback logic."""
        
        if not force_fallback:
            # First attempt: Load MPNet model
            try:
                logger.info(f"Attempting to load primary model: {self.PRIMARY_MODEL}")
                self.model = SentenceTransformer(self.PRIMARY_MODEL)
                self.model_name = self.PRIMARY_MODEL
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Successfully loaded {self.PRIMARY_MODEL} (dimension: {self.embedding_dimension})")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load primary model {self.PRIMARY_MODEL}: {str(e)}")
                logger.info("Attempting fallback to MiniLM model...")
        
        # Fallback: Load MiniLM model
        try:
            logger.info(f"Loading fallback model: {self.FALLBACK_MODEL}")
            self.model = SentenceTransformer(self.FALLBACK_MODEL)
            self.model_name = self.FALLBACK_MODEL
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Successfully loaded fallback model {self.FALLBACK_MODEL} (dimension: {self.embedding_dimension})")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model {self.FALLBACK_MODEL}: {str(e)}")
            raise RuntimeError(f"Both primary and fallback models failed to load. Last error: {str(e)}")
    
    def encode(self, 
               sentences: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: Single sentence or list of sentences to encode
            batch_size: Batch size for encoding (auto-adjusted based on model)
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            NumPy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Auto-adjust batch size based on model type
        if batch_size is None:
            batch_size = self._get_optimal_batch_size()
        
        try:
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode sentences: {str(e)}")
            raise
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on current model."""
        if self.model_name == self.PRIMARY_MODEL:
            return 16  # MPNet can handle larger batches
        else:
            return 32  # MiniLM is more efficient with smaller batches
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        return self.embedding_dimension
    
    def get_model_info(self) -> dict:
        """Get information about the currently loaded model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "is_fallback": self.model_name == self.FALLBACK_MODEL,
            "primary_model": self.PRIMARY_MODEL,
            "fallback_model": self.FALLBACK_MODEL
        }
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity scores
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)


def create_embedding_model(force_fallback: bool = False) -> EmbeddingModel:
    """
    Factory function to create embedding model instance.
    
    Args:
        force_fallback: If True, skip MPNet and use MiniLM directly
        
    Returns:
        Initialized EmbeddingModel instance
    """
    return EmbeddingModel(force_fallback=force_fallback)


# Example usage and testing
if __name__ == "__main__":
    # Test the embedding model with fallback
    print("Testing embedding model with fallback mechanism...")
    
    try:
        # Initialize model (will try MPNet first, fallback to MiniLM if needed)
        model = create_embedding_model()
        
        # Print model info
        info = model.get_model_info()
        print(f"Loaded model: {info['model_name']}")
        print(f"Embedding dimension: {info['embedding_dimension']}")
        print(f"Using fallback: {info['is_fallback']}")
        
        # Test encoding
        test_sentences = [
            "This is a test sentence.",
            "Another example sentence for testing.",
            "The embedding model works with fallback."
        ]
        
        print(f"\nEncoding {len(test_sentences)} test sentences...")
        embeddings = model.encode(test_sentences)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Test similarity
        if len(embeddings) > 1:
            similarity_scores = model.similarity(embeddings[0:1], embeddings[1:2])
            print(f"Similarity between first two sentences: {similarity_scores[0][0]:.4f}")
        
        print("✓ Embedding model test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        raise