# coding=utf-8
"""
Example usage of the embedding model with fallback mechanism.
"""

from embedding_model import create_embedding_model
import numpy as np

def main():
    """Demonstrate the embedding model with fallback mechanism."""
    
    print("=== Embedding Model with Fallback Demo ===\n")
    
    # Create embedding model (will try MPNet first, fallback to MiniLM if needed)
    print("1. Initializing embedding model...")
    try:
        model = create_embedding_model()
        print("✓ Model initialized successfully!")
        
        # Display model information
        info = model.get_model_info()
        print(f"   Model: {info['model_name']}")
        print(f"   Dimension: {info['embedding_dimension']}")
        print(f"   Using fallback: {'Yes' if info['is_fallback'] else 'No'}")
        
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
        embeddings = model.encode(sentences, show_progress_bar=True)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        
        # Calculate and display similarities
        print("\n3. Computing similarity matrix...")
        similarity_matrix = model.similarity(embeddings, embeddings)
        
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


def test_fallback_scenario():
    """Demonstrate forced fallback scenario."""
    
    print("\n=== Testing Fallback Scenario ===")
    
    try:
        # Force fallback to MiniLM
        model = create_embedding_model(force_fallback=True)
        
        info = model.get_model_info()
        print(f"✓ Forced fallback successful!")
        print(f"   Model: {info['model_name']}")
        print(f"   Dimension: {info['embedding_dimension']}")
        print(f"   Using fallback: {info['is_fallback']}")
        
        # Test encoding with fallback model
        test_sentence = "Testing fallback model functionality."
        embedding = model.encode(test_sentence)
        print(f"✓ Fallback encoding successful: shape {embedding.shape}")
        
    except Exception as e:
        print(f"✗ Fallback test failed: {e}")


if __name__ == "__main__":
    main()
    test_fallback_scenario()