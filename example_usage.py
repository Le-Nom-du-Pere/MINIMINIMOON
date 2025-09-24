"""
Example usage of the industrial embedding model with LRU cache functionality.
"""

from embedding_model import create_industrial_embedding_model
import numpy as np

def main():
    """Demonstrate the industrial embedding model with LRU cache."""
    
    print("=== Industrial Embedding Model with LRU Cache Demo ===\n")
    
    # Create embedding model 
    print("1. Initializing industrial embedding model...")
    try:
        with create_industrial_embedding_model(model_tier="standard") as model:
            print("✓ Model initialized successfully!")
            
            # Display model information
            stats = model.get_embedding_statistics()
            info = stats.get('model_information', {})
            print(f"   Model: {info.get('name', 'Unknown')}")
            print(f"   Dimension: {info.get('dimension', 'Unknown')}")
            print(f"   Quality tier: {info.get('quality_tier', 'Unknown')}")
            
            # Sample documents for demonstration
            documents = [
                "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process natural language.",
        "Embedding vectors represent semantic meaning of text.",
        "This is a completely different topic about cooking.",
        "Neural networks learn patterns from data."
    ]
    
    try:
        # Encode sentences
        embeddings = model.encode(sentences)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        
        # Calculate and display similarities
        print("\n3. Computing similarity matrix...")
        similarity_matrix = model.compute_similarity(embeddings, embeddings)
        
        print("\nSimilarity Matrix:")
        print("                    ", end="")
        for i in range(len(sentences)):
            print(f"Sent{i+1:2d}", end=" ")
        print()
        
        for i, sentence in enumerate(sentences):
            print(f"Sentence {i+1:2d}: ", end="")
            for j in range(len(sentences)):
                print(f"{similarity_matrix[i][j]:5.2f}", end=" ")
            print(f" | {sentence[:40]}...")
        
        # Find most similar pair (excluding self-similarity)
        print("\n4. Finding most similar sentence pairs...")
        max_similarity = 0
        best_pair = (0, 0)
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if similarity_matrix[i][j] > max_similarity:
                    max_similarity = similarity_matrix[i][j]
                    best_pair = (i, j)
        
        print(f"Most similar pair (similarity: {max_similarity:.4f}):")
        print(f"  Sentence {best_pair[0]+1}: {sentences[best_pair[0]]}")
        print(f"  Sentence {best_pair[1]+1}: {sentences[best_pair[1]]}")
        
=======
        with create_industrial_embedding_model(model_tier="standard") as model:
            print("✓ Model initialized successfully!")
            
            # Display model information
            stats = model.get_embedding_statistics()
            info = stats.get('model_information', {})
            print(f"   Model: {info.get('name', 'Unknown')}")
            print(f"   Dimension: {info.get('dimension', 'Unknown')}")
            print(f"   Quality tier: {info.get('quality_tier', 'Unknown')}")
            
            # Sample documents for demonstration
            documents = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models can process natural language efficiently.",
                "Embedding vectors represent semantic meaning of text content.",
                "This is a completely different topic about cooking recipes.",
                "Neural networks learn complex patterns from training data.",
                "Financial reports show quarterly revenue growth of 15%.",
                "The weather today is sunny with temperatures around 25°C.",
                "Deep learning techniques advance artificial intelligence research."
            ]
            
            print(f"\n2. Setting up document embeddings cache...")
            # Set up document embeddings for reuse
            model.set_document_embeddings(documents, cache_key="demo_docs")
            print(f"✓ Cached embeddings for {len(documents)} documents")
            
            print(f"\n3. Performing semantic searches with query caching...")
            
            # Test queries - some repeated to demonstrate LRU cache
            test_queries = [
                "machine learning and artificial intelligence",
                "cooking and food preparation", 
                "financial performance and revenue",
                "machine learning and artificial intelligence",  # Repeat to test cache
                "weather conditions and temperature",
                "financial performance and revenue",  # Another repeat
                "neural networks and deep learning"
            ]
            
            # Perform searches and track cache performance
            print("\nSearch Results:")
            for i, query in enumerate(test_queries):
                print(f"\nQuery {i+1}: '{query}'")
                
                # Search using cached embeddings
                results = model.search_documents(query, k=2)
                
                # Display top results
                for rank, (doc_idx, score) in enumerate(results):
                    print(f"  {rank+1}. Score: {score:.3f} - {documents[doc_idx][:60]}...")
            
            print(f"\n4. Cache Performance Statistics:")
            # Get final cache statistics
            final_stats = model.get_embedding_statistics()
            
            lru_stats = final_stats.get('query_lru_cache', {})
            print(f"   Query Cache Hits: {lru_stats.get('hits', 0)}")
            print(f"   Query Cache Misses: {lru_stats.get('misses', 0)}")
            print(f"   Query Cache Hit Rate: {lru_stats.get('hit_rate', 0):.1%}")
            print(f"   Query Cache Size: {lru_stats.get('current_size', 0)}/{lru_stats.get('max_size', 0)}")
            
            doc_cache = final_stats.get('document_cache', {})
            print(f"   Document Cache: {doc_cache.get('cached_document_count', 0)} documents")
            print(f"   Document Cache Key: {doc_cache.get('cache_key', 'None')}")
            
            # Additional cache stats
            quality_stats = final_stats.get('quality_metrics', {})
            print(f"   Total Embeddings Generated: {quality_stats.get('total_embeddings', 0)}")
            print(f"   Adaptive Cache Hits: {quality_stats.get('cache_hits', 0)}")
            
            print(f"\n5. Testing different query variations...")
            
            # Test similar queries to show nuanced caching behavior
            similar_queries = [
                "artificial intelligence and machine learning",  # Similar to cached query
                "AI and ML",  # Abbreviation
                "machine learning and artificial intelligence"   # Exact repeat
            ]
            
            for query in similar_queries:
                results = model.search_documents(query, k=1)
                cache_info = model._encode_query_cached.cache_info()
                print(f"   '{query}' -> Cache hits: {cache_info.hits}, misses: {cache_info.misses}")
                
>>>>>>> 4872831 (Implement LRU cache for query embeddings to optimize semantic search performance)
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        return
    
    print("\n=== LRU Cache Demo completed successfully! ===")


def demonstrate_cache_efficiency():
    """Demonstrate cache efficiency with repeated queries."""
    
    print("\n=== Cache Efficiency Demonstration ===")
    
    try:
        with create_industrial_embedding_model(model_tier="basic") as model:
            documents = [f"Document {i} with sample content about topic {i%5}" for i in range(20)]
            model.set_document_embeddings(documents, "efficiency_test")
            
            # Clear cache for clean test
        
        # Create page identifiers
        pages = [f"doc_{i+1}.txt" for i in range(len(documents))]
        
        # Test queries
        queries = [
            "Tell me about pets and animals",
            "What is artificial intelligence and machine learning?",
            "How is the weather forecast?"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            # Perform semantic search with torch.topk
            results = model.semantic_search(query, documents, pages=pages, k=3)
            
            for i, (idx, page, text, score) in enumerate(results, 1):
                print(f"{i}. [{page}] (Similarity: {score:.4f})")
                print(f"   {text}")
                print()
        
        # Test semantic search without scores
        print(f"Testing search without scores:")
        print("-" * 40)
        results_no_scores = model.semantic_search(
            "artificial intelligence", 
            documents, 
            k=2, 
            return_scores=False
        )
        
        for idx, page, text in results_no_scores:
            print(f"[{page}] {text}")
        
        print("✓ Semantic search tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Semantic search test failed: {e}")


def test_fallback_scenario():
    """Demonstrate forced fallback scenario."""
=======
def demonstrate_cache_efficiency():
    """Demonstrate cache efficiency with repeated queries."""
>>>>>>> 4872831 (Implement LRU cache for query embeddings to optimize semantic search performance)
    
    print("\n=== Cache Efficiency Demonstration ===")
    
    try:
        with create_industrial_embedding_model(model_tier="basic") as model:
            documents = [f"Document {i} with sample content about topic {i%5}" for i in range(20)]
            model.set_document_embeddings(documents, "efficiency_test")
            
            # Clear cache for clean test
            model._encode_query_cached.cache_clear()
            
            # Test query repeated many times
            repeated_query = "sample content and topics"
            num_repeats = 10
            
            print(f"\nTesting query '{repeated_query}' repeated {num_repeats} times:")
            
            for i in range(num_repeats):
                _ = model.search_documents(repeated_query, k=3)
                cache_info = model._encode_query_cached.cache_info()
                if i == 0:
                    print(f"   First search: {cache_info.misses} miss, {cache_info.hits} hits")
                elif i == num_repeats - 1:
                    print(f"   Final search: {cache_info.misses} misses, {cache_info.hits} hits")
            
            print(f"✓ Cache efficiency: {cache_info.hits}/{cache_info.hits + cache_info.misses} hits ({cache_info.hits/(cache_info.hits + cache_info.misses)*100:.1f}%)")
            
    except Exception as e:
        print(f"✗ Cache efficiency test failed: {e}")


if __name__ == "__main__":
    main()
    demonstrate_cache_efficiency()
