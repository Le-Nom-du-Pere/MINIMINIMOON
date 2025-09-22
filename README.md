# Multi-Component Python Suite

This project combines multiple complementary components for text processing, graph validation, and embedding models:

1. **Embedding Model with Fallback Mechanism** - Robust embedding model with automatic MPNet->MiniLM fallback
2. **Unicode Normalization for Regex Matching** - Text processing with Unicode normalization (if available)
3. **Deterministic Monte Carlo DAG Validation** - Statistical validation of causal graphs (if available)

## Embedding Model Component

A robust embedding model implementation that automatically falls back from MPNet to MiniLM if the primary model fails to load, ensuring uninterrupted operation.

### Files
- **embedding_model.py** - Core embedding model with fallback mechanism
- **test_embedding_model.py** - Comprehensive test suite for embedding model
- **example_usage.py** - Demo and usage examples

### Features
- **Automatic Fallback**: Tries MPNet (768-dim) first, falls back to MiniLM (384-dim) if loading fails
- **Exception Handling**: Comprehensive error handling around model initialization
- **Model-Dependent Configuration**: Automatically adjusts batch sizes and parameters based on the loaded model
- **Similarity Calculations**: Built-in cosine similarity computation
- **Factory Pattern**: Easy model instantiation with `create_embedding_model()`

### Usage
```python
from embedding_model import create_embedding_model

# Create model (tries MPNet first, falls back to MiniLM if needed)
model = create_embedding_model()

# Check which model was loaded
info = model.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Using fallback: {info['is_fallback']}")

# Encode sentences
embeddings = model.encode(["Hello world", "Embedding test"])
```

## Text Processing Component (if available)

Implements Unicode normalization using `unicodedata.normalize("NFKC", text)` before applying regex patterns to ensure consistent character representation.

### Usage
```python
from text_processor import normalize_unicode, count_words
from utils import TextAnalyzer

# Basic normalization
normalized = normalize_unicode("café résumé")
count = count_words("Text with—em dashes")
```

## DAG Validation Component (if available)

A Python implementation for validating Directed Acyclic Graphs (DAGs) using deterministic Monte Carlo sampling, specifically designed for causal graph analysis.

### Usage
```python
from dag_validation import DAGValidator, create_sample_causal_graph

# Create validator and test acyclicity
validator = create_sample_causal_graph()
result = validator.calculate_acyclicity_pvalue("plan_name", iterations=1000)
```

## Installation

```bash
<<<<<<< HEAD
pip install -r requirements.txt
```

## Testing

Run individual test suites:
```bash
python3 test_unicode_normalization.py  # Text processing tests
python3 test_dag_validation.py         # DAG validation tests
```

Run complete validation:
```bash
python3 validate.py  # Full validation suite
```

## Statistical Interpretation (DAG Validation)

⚠️ **Important**: DAG validation tests structural acyclicity, not causal validity.

- **P-value**: Probability of observing acyclic structure in random subgraphs under the null hypothesis
- **Lower p-values**: Suggest the observed acyclicity is unlikely to occur by chance alone
- **Higher p-values**: Indicate the structure could reasonably arise from random processes

### Limitations
1. **Not a causal test**: Validates graph structure, not causal relationships
2. **Domain expertise required**: Statistical significance ≠ causal validity
3. **Interpretation context**: Results must be interpreted within domain knowledge
4. **Subgraph sampling**: Tests random subsets, not the full graph structure

## Example Outputs

### Text Processing Demo
```
Decomposed characters:
Original text: café vs café
Text length: 13 characters
Normalized:   café vs café  
Normalized length: 12 characters
Character differences detected:
  Position 11: 'e' (U+0065) -> 'é' (U+00E9)
```

### DAG Validation Demo
```
Testing reproducibility for plan: teoria_cambio_educacion_2024
Reproducible: True

Monte Carlo Results:
Plan: teoria_cambio_educacion_2024
Seed: 2175693273
Total iterations: 1000
Acyclic count: 1000
P-value: 1.0000
Average subgraph size: 5.4

Graph Statistics:
Total nodes: 8
Total edges: 8
```

## Architecture

Both components use Python standard library only, ensuring minimal dependencies and maximum compatibility. The text processing component handles Unicode normalization challenges, while the DAG validation component provides statistical tools for causal graph analysis with guaranteed reproducibility.
=======
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from embedding_model import create_embedding_model

# Create model (tries MPNet first, falls back to MiniLM if needed)
model = create_embedding_model()

# Check which model was loaded
info = model.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Dimension: {info['embedding_dimension']}")
print(f"Using fallback: {info['is_fallback']}")

# Encode sentences
sentences = ["Hello world", "Embedding test"]
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")
```

## API Reference

### EmbeddingModel Class

#### Initialization
- `EmbeddingModel(force_fallback=False)`: Initialize with optional fallback forcing

#### Methods
- `encode(sentences, batch_size=None, show_progress_bar=False, normalize_embeddings=True)`: Encode text to embeddings
- `similarity(embeddings1, embeddings2)`: Calculate cosine similarity
- `get_model_info()`: Get current model information
- `get_embedding_dimension()`: Get embedding dimension

#### Factory Function
- `create_embedding_model(force_fallback=False)`: Create model instance

## Model Configuration

### Primary Model (MPNet)
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Dimensions**: 768
- **Optimal Batch Size**: 16

### Fallback Model (MiniLM)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` 
- **Dimensions**: 384
- **Optimal Batch Size**: 32

## Exception Handling

The fallback mechanism handles various loading failures:
- Network connectivity issues
- Insufficient disk space
- Corrupted model files
- CUDA/device compatibility problems
- Memory constraints

## Examples

### Basic Usage
```python
model = create_embedding_model()
embeddings = model.encode(["Sample text"])
```

### Force Fallback
```python
model = create_embedding_model(force_fallback=True)  # Skip MPNet, use MiniLM
```

### Batch Processing
```python
large_text_list = ["Text " + str(i) for i in range(1000)]
embeddings = model.encode(large_text_list, batch_size=32, show_progress_bar=True)
```

### Similarity Calculation
```python
text1_embeddings = model.encode(["First text"])
text2_embeddings = model.encode(["Second text"]) 
similarity = model.similarity(text1_embeddings, text2_embeddings)
```

## Testing

Run the comprehensive test suite:

```bash
python3 -m pytest test_embedding_model.py -v
```

The tests cover:
- Primary model loading success
- Fallback mechanism triggering
- Both models failing scenario
- Force fallback functionality
- Encoding operations
- Batch size optimization
- Model information retrieval
- Similarity calculations

## Running Examples

```bash
python3 example_usage.py
```

This demonstrates:
- Model initialization with fallback
- Sentence encoding
- Similarity matrix computation
- Batch size testing
- Fallback scenario testing
>>>>>>> fa1edfd (Add fallback mechanism from MPNet to MiniLM model for embedding initialization)
