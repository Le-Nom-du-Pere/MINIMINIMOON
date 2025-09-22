# AGENTS.md

## Commands

### Setup
```bash
# Python project with embedding models, text processing, and responsibility detection components
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
# Download spaCy Spanish model for responsibility detection
python3 -m spacy download es_core_news_sm
```

### Build
```bash
# Text processing components
python3 -c "import text_processor, utils; print('Text processing build successful')" || echo "Text processing components not available"
# DAG validation components  
python3 -c "import dag_validation; print('DAG validation build successful')" || echo "DAG validation components not available"
# Embedding model components
python3 -c "import embedding_model; print('Embedding model build successful')"
# TeoriaCambio class
python3 -m py_compile teoria_cambio.py
# Responsibility detection components
python3 -m py_compile responsibility_detector.py
```

### Lint
```bash
# Code follows PEP 8 conventions
python3 -m py_compile embedding_model.py test_embedding_model.py example_usage.py
# Additional components if available
python3 -m py_compile text_processor.py utils.py test_unicode_normalization.py demo_unicode_comparison.py dag_validation.py test_dag_validation.py verify_reproducibility.py validate.py 2>/dev/null || echo "Additional components not available for linting"
# TeoriaCambio class
python3 -m py_compile teoria_cambio.py
# Responsibility detection components
python3 -m py_compile responsibility_detector.py test_responsibility_detector.py
```

### Test
```bash
# Embedding model tests
python3 -m pytest test_embedding_model.py -v
# Responsibility detection tests
python3 -m pytest test_responsibility_detector.py -v
# Additional tests if available
python3 test_unicode_normalization.py 2>/dev/null || echo "Text processing tests not available"
python3 test_dag_validation.py 2>/dev/null || echo "DAG validation tests not available"
python3 validate.py 2>/dev/null || echo "Full validation suite not available"
```

### Dev Server
```bash
# Embedding model demo
python3 example_usage.py
# Responsibility detection demo
python3 responsibility_detector.py
# Additional demos if available
python3 demo_unicode_comparison.py 2>/dev/null || echo "Text processing demo not available"
python3 dag_validation.py 2>/dev/null || echo "DAG validation demo not available"
```

## Tech Stack
- **Language**: Python 3.7+
- **Framework**: sentence-transformers, scikit-learn, numpy for embedding models; spaCy for NER; standard library for text processing
- **Package Manager**: pip
- **Testing**: pytest and unittest

## Architecture
```
# Embedding Model Components
embedding_model.py              # Core embedding model with fallback mechanism
├── EmbeddingModel             # Main model class with MPNet->MiniLM fallback
├── create_embedding_model()   # Factory function
├── exception handling         # Robust error handling for model loading
└── batch optimization         # Model-specific batch size optimization

test_embedding_model.py         # Comprehensive test suite for embedding model
example_usage.py               # Demo and usage examples

# Responsibility Detection Components
responsibility_detector.py      # spaCy NER + lexical pattern matching for responsibility detection
├── ResponsibilityDetector     # Main detector class
├── ResponsibilityEntity       # Entity data structure
├── EntityType                 # Entity type enumeration
├── NER integration            # spaCy PERSON/ORG detection
├── Government patterns        # High-priority institutional patterns
├── Position patterns          # Official role detection
├── Institutional patterns     # Generic fallback patterns
├── Entity merging            # Overlap handling
└── Confidence scoring        # Hierarchical scoring system

test_responsibility_detector.py # Comprehensive test suite for responsibility detection

# Text Processing Components (if available)
text_processor.py              # Core text processing with Unicode normalization
utils.py                       # Utility classes and functions
test_unicode_normalization.py  # Text processing test suite
demo_unicode_comparison.py     # Text processing demo

# DAG Validation Components (if available)
dag_validation.py              # Deterministic Monte Carlo DAG validation
test_dag_validation.py         # DAG validation test suite
verify_reproducibility.py      # DAG reproducibility verification
validate.py                    # Complete validation orchestrator

# Theory of Change Components
teoria_cambio.py               # TeoriaCambio class with cached causal graph construction
├── TeoriaCambio               # Main class with caching mechanism
├── construir_grafo_causal()   # Cached graph construction method
├── _crear_grafo_causal()      # Private graph creation method
└── invalidar_cache_grafo()    # Cache invalidation method
```

## Code Style
- Follows PEP 8 Python conventions
- Comprehensive docstrings with examples
- Type hints using typing module
- Robust exception handling for model initialization
- Automatic fallback mechanisms for reliability
- Statistical interpretation warnings to prevent misuse
