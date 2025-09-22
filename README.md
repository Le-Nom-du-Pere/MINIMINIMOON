# Text Processing & DAG Validation Suite

This project combines two complementary components:

1. **Unicode Normalization for Regex Matching** - Text processing with Unicode normalization
2. **Deterministic Monte Carlo DAG Validation** - Statistical validation of causal graphs

## Text Processing Component

Implements Unicode normalization using `unicodedata.normalize("NFKC", text)` before applying regex patterns to ensure consistent character representation and prevent overcounting issues.

### Files
- **text_processor.py** - Core text processing functions with Unicode normalization
- **utils.py** - Utility classes and functions for text analysis
- **test_unicode_normalization.py** - Text processing test suite
- **demo_unicode_comparison.py** - Before/after normalization demo

### Features
- Unicode normalization functions (quotes, words, emails, etc.)
- TextAnalyzer class with pattern matching
- Consistent character representation across Unicode variants
- Comprehensive test coverage

### Usage
```python
from text_processor import normalize_unicode, count_words
from utils import TextAnalyzer

# Basic normalization
normalized = normalize_unicode("café résumé")
count = count_words("Text with—em dashes")

# Pattern analysis
analyzer = TextAnalyzer()
emails = analyzer.find_pattern_matches(text, 'email')
```

## DAG Validation Component

A Python implementation for validating Directed Acyclic Graphs (DAGs) using deterministic Monte Carlo sampling, specifically designed for causal graph analysis in "teoria de cambio" (theory of change) models.

### Files
- **dag_validation.py** - Core DAG validation with Monte Carlo sampling
- **test_dag_validation.py** - DAG validation test suite
- **verify_reproducibility.py** - Reproducibility verification script
- **validate.py** - Complete validation orchestrator

### Features
- **Deterministic Seeding**: Creates reproducible random sequences from plan names using SHA-256 hashing
- **Monte Carlo Sampling**: Statistical testing of acyclicity on random subgraphs
- **P-value Calculation**: Quantifies the probability of observing acyclic structure by chance
- **Reproducibility Verification**: Ensures identical results across multiple executions
- **Graph Statistics**: Provides basic metrics about graph structure

### Usage
```python
from dag_validation import DAGValidator, create_sample_causal_graph

# Create validator with sample causal graph
validator = create_sample_causal_graph()

# Calculate p-value for acyclicity
plan_name = "teoria_cambio_educacion_2024"
result = validator.calculate_acyclicity_pvalue(plan_name, iterations=1000)

print(f"P-value: {result.p_value:.4f}")
print(f"Acyclic subgraphs: {result.acyclic_count}/{result.total_iterations}")

# Verify reproducibility
is_reproducible = validator.verify_reproducibility("mi_plan", 100)
```

## Installation

```bash
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
