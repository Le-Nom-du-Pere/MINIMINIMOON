#!/usr/bin/env python3
"""
Example usage of industrial embedding model with memory management and caching.
This demonstrates advanced features including torch memory optimization and disk caching.
"""

from embedding_model import IndustrialEmbeddingModel, create_embedding_model
import numpy as np
import torch
import time
import gc


def demo_memory_managed_embedding():
    """Demonstrate memory-managed embedding with torch optimizations."""
    print("=== Memory-Managed Embedding Demo ===")
    
    # Create embedding model with memory management
    model = IndustrialEmbeddingModel(
        preferred_model='primary_large',
        memory_threshold=0.8,
        enable_disk_cache=True
    )
    
    # Show initial memory usage
    if hasattr(model, 'memory_manager'):
        print(f"Initial memory usage: {model.memory_manager.get_memory_usage():.1%}")
        print(f"Available memory: {model.memory_manager.get_available_memory_gb():.2f} GB")
    
    # Test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",  
        "Machine learning transforms how we process data.",
        "Natural language processing enables human-computer interaction.",
        "Deep learning models require significant computational resources.",
        "Efficient memory management is crucial for large-scale processing."
    ]
    
    # Generate embeddings with memory monitoring
    print("\nGenerating embeddings with memory optimization...")
    start_time = time.time()
    
    with torch.no_grad():  # Demonstrate explicit no_grad usage
        embeddings = model.encode(sentences, enable_caching=True)
    
    end_time = time.time()
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Processing time: {end_time - start_time:.3f} seconds")
    
    # Show final memory usage
    if hasattr(model, 'memory_manager'):
        print(f"Final memory usage: {model.memory_manager.get_memory_usage():.1%}")
    
    # Show cache statistics
    if hasattr(model.embedding_cache, 'stats'):
        cache_stats = model.embedding_cache.stats()
        print(f"Cache size: {cache_stats.get('size', 'N/A')}")
    
    print()
    return embeddings


def demo_chunked_processing():
    """Demonstrate chunked processing for large datasets."""
    print("=== Chunked Processing Demo ===")
    
    model = IndustrialEmbeddingModel(
        preferred_model='secondary_efficient',
        memory_threshold=0.7
    )
    
    # Create large dataset
    large_dataset = [f"Document {i}: This is a test document for chunked processing." for i in range(100)]
    
    print(f"Processing {len(large_dataset)} documents...")
    start_time = time.time()
    
    # Process with automatic chunking based on memory
    embeddings = model.encode(large_dataset, batch_size=16)
    
    end_time = time.time()
    
    print(f"Processed {len(embeddings)} embeddings")
    print(f"Shape: {embeddings.shape}")
    print(f"Processing time: {end_time - start_time:.3f} seconds")
    print(f"Rate: {len(large_dataset)/(end_time - start_time):.1f} docs/second")
    
    # Show memory cleanup stats
    if hasattr(model, 'quality_metrics'):
        cleanups = model.quality_metrics.get('memory_cleanups', 0)
        print(f"Memory cleanups performed: {cleanups}")
    
    print()


def demo_disk_caching():
    """Demonstrate disk caching with torch.save/load."""
    print("=== Disk Caching Demo ===")
    
    model = IndustrialEmbeddingModel(
        enable_disk_cache=True,
        cache_size=1000
    )
    
    test_texts = [
        "This text will be cached to disk",
        "Another cached text example",
        "Disk caching improves performance"
    ]
    
    # First encoding - will cache to disk
    print("First encoding (caching)...")
    start_time = time.time()
    embeddings1 = model.encode(test_texts)
    first_time = time.time() - start_time
    
    # Second encoding - should hit cache
    print("Second encoding (from cache)...")
    start_time = time.time()
    embeddings2 = model.encode(test_texts) 
    second_time = time.time() - start_time
    
    print(f"First encoding time: {first_time:.4f}s")
    print(f"Second encoding time: {second_time:.4f}s")
    print(f"Cache speedup: {first_time/second_time:.1f}x")
    
    # Verify identical results
    np.testing.assert_array_equal(embeddings1, embeddings2)
    print("Cache consistency verified ✓")
    
    print()


def demo_adaptive_batch_sizing():
    """Demonstrate adaptive batch sizing based on memory."""
    print("=== Adaptive Batch Sizing Demo ===")
    
    model = IndustrialEmbeddingModel(preferred_model='primary_large')
    
    # Test different memory scenarios
    test_sizes = [10, 50, 100, 200, 500]
    
    print("Size | Batch | Time  | Rate")
    print("-" * 30)
    
    for size in test_sizes:
        texts = [f"Test document {i}" for i in range(size)]
        
        # Calculate optimal batch size
        optimal_batch = model._calculate_optimal_batch_size(size)
        
        start_time = time.time()
        embeddings = model.encode(texts, batch_size=optimal_batch)
        end_time = time.time()
        
        rate = size / (end_time - start_time)
        print(f"{size:4d} | {optimal_batch:5d} | {end_time-start_time:.3f}s | {rate:4.0f}/s")
    
    print()


def demo_instruction_learning():
    """Demonstrate instruction-based transformations."""
    print("=== Instruction Learning Demo ===")
    
    model = IndustrialEmbeddingModel(
        enable_instruction_learning=True,
        preferred_model='primary_large'
    )
    
    texts = [
        "The stock market reached new highs",
        "Company profits increased significantly",
        "Economic indicators show positive trends"
    ]
    
    instruction = "Focus on financial and economic concepts"
    
    print("Encoding with instruction transformation...")
    
    # Encode with instruction
    embeddings_with_instruction = model.encode(
        texts,
        instruction=instruction,
        instruction_strength=0.6
    )
    
    # Encode without instruction for comparison
    embeddings_without = model.encode(texts)
    
    # Compare similarity to instruction
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Encode instruction itself
    instruction_embedding = model.encode([instruction])
    
    # Calculate similarities
    sim_with = cosine_similarity(embeddings_with_instruction, instruction_embedding).mean()
    sim_without = cosine_similarity(embeddings_without, instruction_embedding).mean()
    
    print(f"Similarity to instruction with transformation: {sim_with:.4f}")
    print(f"Similarity to instruction without transformation: {sim_without:.4f}")
    print(f"Instruction effectiveness: {sim_with - sim_without:.4f}")
    
    print()


if __name__ == "__main__":
    print("Industrial Embedding Model - Advanced Usage Demo")
    print("=" * 60)
    print()
    
    try:
        demo_memory_managed_embedding()
        demo_chunked_processing()
        demo_disk_caching()
        demo_adaptive_batch_sizing()
        demo_instruction_learning()
        
        print("All advanced demos completed successfully!")
        
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()