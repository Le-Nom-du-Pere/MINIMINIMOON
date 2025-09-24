"""
Example usage of the embedding model with fallback mechanism.
"""

from embedding_model import create_industrial_embedding_model
import numpy as np

def main():
    """Demonstrate the embedding model with fallback mechanism."""
    
    print("=== Embedding Model with Fallback Demo ===\n")
    
    # Create embedding model (will try MPNet first, fallback to MiniLM if needed)
    print("1. Initializing embedding model...")
    try:
        model = create_industrial_embedding_model()
        print("✓ Model initialized successfully!")
        
        # Display model information
        diagnostics = model.get_comprehensive_diagnostics()
        info = diagnostics['model_info']
        print(f"   Model: {info['name']}")
        print(f"   Dimension: {info['dimension']}")
        print(f"   Quality Tier: {info['quality_tier']}")
        
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return
    
    print("\n2. Encoding sample sentences...")
    
    # Sample sentences for demonstration
    sentences = [
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
        
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        return
    
    print("\n5. Testing with different batch sizes...")
    
    # Test different configurations
    test_sentences = ["Test sentence " + str(i) for i in range(50)]
    
    try:
        # Test with default batch size
        embeddings_default = model.encode(test_sentences)
        print(f"✓ Default batch encoding: {embeddings_default.shape}")
        
        # Test with custom batch size
        embeddings_custom = model.encode(test_sentences, batch_size=8)
        print(f"✓ Custom batch encoding: {embeddings_custom.shape}")
        
        # Verify results are similar
        difference = np.mean(np.abs(embeddings_default - embeddings_custom))
        print(f"✓ Batch size difference (should be ~0): {difference:.6f}")
        
    except Exception as e:
        print(f"✗ Error during batch testing: {e}")
    
    print("\n=== Demo completed successfully! ===")


def test_semantic_search():
    """Demonstrate semantic search functionality."""
    
    print("\n=== Testing Semantic Search ===")
    
    try:
        model = create_industrial_embedding_model()
        
        # Create a document corpus
        documents = [
            "Cats are independent and graceful animals that make wonderful pets.",
            "Dogs are loyal and friendly companions that love to play with their owners.",
            "Machine learning algorithms can recognize patterns in large datasets.",
            "Deep learning uses neural networks to solve complex computational problems.",
            "Natural language processing helps computers understand and generate human text.",
            "Computer vision enables machines to interpret and analyze visual information.",
            "The weather is sunny and warm today, perfect for outdoor activities.",
            "Rain is expected tomorrow afternoon, so bring an umbrella."
        ]
        
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
    
    print("\n=== Testing Fallback Scenario ===")
    
    try:
        # Force fallback to MiniLM
        model = create_industrial_embedding_model(model_tier="basic")
        
        diagnostics = model.get_comprehensive_diagnostics()
        info = diagnostics['model_info']
        print(f"✓ Basic tier model loaded successfully!")
        print(f"   Model: {info['name']}")
        print(f"   Dimension: {info['dimension']}")
        print(f"   Quality Tier: {info['quality_tier']}")
        
        # Test encoding with fallback model
        test_sentence = "Testing fallback model functionality."
        embedding = model.encode(test_sentence)
        print(f"✓ Fallback encoding successful: shape {embedding.shape}")
        
    except Exception as e:
        print(f"✗ Fallback test failed: {e}")


if __name__ == "__main__":
    main()
    test_semantic_search()
    test_fallback_scenario()