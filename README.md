# Multi-Component Python Suite

This project combines multiple complementary components for text processing, graph validation, embedding models, and Spanish pattern detection:

1. **Embedding Model with Fallback Mechanism** - Robust embedding model with automatic MPNet->MiniLM fallback
2. **Factibilidad Scoring Module** - Spanish pattern detection for baseline, target, and timeframe indicators
3. **Unicode Normalization for Regex Matching** - Text processing with Unicode normalization (if available)
4. **Deterministic Monte Carlo DAG Validation** - Statistical validation of causal graphs (if available)

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

## Factibilidad Scoring Component

A Python module for detecting and scoring text segments based on the presence of baseline, target, and timeframe patterns in Spanish text.

### Files
- **factibilidad/pattern_detector.py** - Core pattern detection logic
- **factibilidad/scoring.py** - Scoring algorithms and analysis
- **test_factibilidad.py** - Test cases and examples

### Features
- **Pattern Detection**: Identifies three types of patterns in Spanish text:
  - Baseline indicators (línea base, situación inicial, punto de partida, etc.)
  - Target indicators (meta, objetivo, alcanzar, lograr, etc.)
  - Timeframe indicators (dates, quarters, relative time expressions)
- **Proximity-Based Clustering**: Groups patterns that appear within a configurable distance window
- **Factibilidad Scoring**: Calculates scores based on individual pattern presence, complete pattern clusters, proximity bonuses, and pattern density bonuses

### Usage
```python
from factibilidad import PatternDetector, FactibilidadScorer

# Basic pattern detection
detector = PatternDetector()
matches = detector.detect_patterns(text)

# Calculate factibilidad scores
scorer = FactibilidadScorer(proximity_window=500)
result = scorer.score_text(text)

print(f"Score: {result['total_score']}")
print(f"Clusters found: {result['cluster_scores']['count']}")
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
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Testing

Run individual test suites:
```bash
python3 -m pytest test_embedding_model.py -v  # Embedding model tests
python3 test_factibilidad.py                  # Factibilidad pattern tests
python3 test_unicode_normalization.py 2>/dev/null || echo "Text processing tests not available"
python3 test_dag_validation.py 2>/dev/null || echo "DAG validation tests not available"
```

Run complete validation:
```bash
python3 validate.py 2>/dev/null || echo "Full validation suite not available"
```

## Quick Start Examples

### Embedding Model
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

### Factibilidad Scoring
```python
from factibilidad import FactibilidadScorer

scorer = FactibilidadScorer()
text = "La línea base actual muestra 100 usuarios. Nuestro objetivo es alcanzar 500 usuarios para diciembre de 2024."
result = scorer.score_text(text)
print(f"Factibilidad Score: {result['total_score']:.1f}")
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

## Architecture

The suite combines multiple components with minimal dependencies. The embedding model requires sentence-transformers and scikit-learn, while the factibilidad scoring module uses only Python standard library. Text processing and DAG validation components (if available) also use standard library only, ensuring maximum compatibility.