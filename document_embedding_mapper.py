# coding=utf-8
"""
Document Embedding Mapper with Parallel Arrays
=============================================
Text-to-page-embedding mapping system that eliminates the use of .index() method
by maintaining parallel arrays where the embedding array index directly corresponds 
to the text segment and page number arrays at the same position.
"""

import logging
import time
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
from embedding_model import create_industrial_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class DocumentSegment:
    """Represents a text segment with its page information."""
    text: str
    page_number: int
    segment_id: int
    
    def __hash__(self):
        return hash((self.text, self.page_number))
    
    def __eq__(self, other):
        if not isinstance(other, DocumentSegment):
            return False
        return self.text == other.text and self.page_number == other.page_number


class DocumentEmbeddingMapper:
    """
    Text-to-page-embedding mapping system using parallel arrays.
    
    Maintains three parallel arrays:
    - text_segments: List of text strings
    - page_numbers: List of page numbers corresponding to each text segment
    - embeddings: NumPy array of embeddings corresponding to each text segment
    
    The index in each array corresponds to the same document segment.
    """
    
    def __init__(self, model=None):
        """
        Initialize the mapper with an embedding model.
        
        Args:
            model: Optional embedding model. If None, creates a default model.
        """
        self.model = model or create_industrial_embedding_model()
        
        # Parallel arrays - indices correspond across all arrays
        self.text_segments: List[str] = []
        self.page_numbers: List[int] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Tracking for duplicate handling
        self._segment_count: Dict[Tuple[str, int], List[int]] = {}
        self._next_segment_id = 0
        
        logger.info("DocumentEmbeddingMapper initialized")
    
    def add_document_segments(self, segments: List[Tuple[str, int]]) -> List[int]:
        """
        Add document segments and compute their embeddings.
        
        Args:
            segments: List of (text, page_number) tuples
            
        Returns:
            List of indices where each segment was stored
        """
        if not segments:
            return []
        
        start_time = time.time()
        texts = [text for text, _ in segments]
        pages = [page for _, page in segments]
        
        # Compute embeddings for all texts
        new_embeddings = self.model.encode(texts)
        
        # Store current size to calculate new indices
        start_index = len(self.text_segments)
        
        # Add to parallel arrays
        self.text_segments.extend(texts)
        self.page_numbers.extend(pages)
        
        # Handle embeddings array
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update duplicate tracking
        new_indices = []
        for i, (text, page) in enumerate(segments):
            current_index = start_index + i
            segment_key = (text, page)
            
            if segment_key not in self._segment_count:
                self._segment_count[segment_key] = []
            self._segment_count[segment_key].append(current_index)
            new_indices.append(current_index)
        
        duration = time.time() - start_time
        logger.info(f"Added {len(segments)} segments in {duration:.2f}s")
        
        return new_indices
    
    def get_segment_info(self, index: int) -> Tuple[str, int]:
        """
        Get text and page number for a given index.
        
        Args:
            index: Index in the parallel arrays
            
        Returns:
            Tuple of (text, page_number)
        """
        if index < 0 or index >= len(self.text_segments):
            raise IndexError(f"Index {index} out of range [0, {len(self.text_segments)})")
        
        return self.text_segments[index], self.page_numbers[index]
    
    def get_embedding(self, index: int) -> np.ndarray:
        """
        Get embedding for a given index.
        
        Args:
            index: Index in the parallel arrays
            
        Returns:
            Embedding vector
        """
        if self.embeddings is None:
            raise ValueError("No embeddings stored")
        
        if index < 0 or index >= len(self.embeddings):
            raise IndexError(f"Index {index} out of range [0, {len(self.embeddings)})")
        
        return self.embeddings[index]
    
    def find_duplicate_indices(self, text: str, page: int) -> List[int]:
        """
        Find all indices where a specific text-page combination appears.
        
        Args:
            text: Text to search for
            page: Page number to search for
            
        Returns:
            List of indices where this combination appears
        """
        segment_key = (text, page)
        return self._segment_count.get(segment_key, [])
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[int, str, int, float]]:
        """
        Perform top-k similarity search using direct indexing.
        
        Args:
            query: Query text
            k: Number of top results to return
            
        Returns:
            List of tuples: (index, text, page_number, similarity_score)
        """
        if self.embeddings is None or len(self.text_segments) == 0:
            return []
        
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Compute similarities with all embeddings
        similarities = self.model.compute_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices using numpy argsort (no .index() method)
        k = min(k, len(similarities))
        top_indices = np.argsort(-similarities)[:k]
        
        # Build results using direct indexing
        results = []
        for idx in top_indices:
            idx = int(idx)  # Convert numpy int to Python int
            text = self.text_segments[idx]
            page = self.page_numbers[idx]
            score = float(similarities[idx])
            results.append((idx, text, page, score))
        
        duration = time.time() - start_time
        logger.info(f"Similarity search completed in {duration:.2f}s")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the stored data.
        
        Returns:
            Dictionary with statistics
        """
        if not self.text_segments:
            return {
                "total_segments": 0,
                "unique_segments": 0,
                "duplicate_segments": 0,
                "pages_covered": 0,
                "embedding_dimension": 0
            }
        
        # Count unique segments
        unique_segments = len(self._segment_count)
        duplicate_segments = sum(len(indices) - 1 for indices in self._segment_count.values() if len(indices) > 1)
        
        return {
            "total_segments": len(self.text_segments),
            "unique_segments": unique_segments,
            "duplicate_segments": duplicate_segments,
            "pages_covered": len(set(self.page_numbers)),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "max_duplicates_for_segment": max(len(indices) for indices in self._segment_count.values()) if self._segment_count else 0
        }
    
    def verify_parallel_arrays(self) -> bool:
        """
        Verify that all parallel arrays have consistent lengths.
        
        Returns:
            True if arrays are consistent, False otherwise
        """
        text_len = len(self.text_segments)
        page_len = len(self.page_numbers)
        embed_len = len(self.embeddings) if self.embeddings is not None else 0
        
        if text_len != page_len:
            logger.error(f"Length mismatch: texts={text_len}, pages={page_len}")
            return False
        
        if self.embeddings is not None and text_len != embed_len:
            logger.error(f"Length mismatch: texts={text_len}, embeddings={embed_len}")
            return False
        
        return True
    
    def batch_similarity_search(self, queries: List[str], k: int = 5) -> List[List[Tuple[int, str, int, float]]]:
        """
        Perform batch similarity search for multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of top results per query
            
        Returns:
            List of result lists, one per query
        """
        if self.embeddings is None or len(self.text_segments) == 0:
            return [[] for _ in queries]
        
        start_time = time.time()
        
        # Encode all queries at once
        query_embeddings = self.model.encode(queries)
        
        # Compute similarities between all queries and all documents
        similarities = self.model.compute_similarity(query_embeddings, self.embeddings)
        
        # Process results for each query
        all_results = []
        k = min(k, len(self.text_segments))
        
        for i, query_similarities in enumerate(similarities):
            # Get top-k indices for this query
            top_indices = np.argsort(-query_similarities)[:k]
            
            # Build results using direct indexing
            query_results = []
            for idx in top_indices:
                idx = int(idx)
                text = self.text_segments[idx]
                page = self.page_numbers[idx]
                score = float(query_similarities[idx])
                query_results.append((idx, text, page, score))
            
            all_results.append(query_results)
        
        duration = time.time() - start_time
        logger.info(f"Batch similarity search for {len(queries)} queries completed in {duration:.2f}s")
        
        return all_results