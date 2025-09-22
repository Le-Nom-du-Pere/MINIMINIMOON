# AGENTS.md

## Commands

### Setup
```bash
# Python project - no additional dependencies required beyond standard library
# Optional: install pytest for advanced testing
pip install pytest
```

### Build
```bash
python3 -c "import feasibility_scorer; print('Build successful')"
```

### Lint
```bash
# No linting tools configured - code follows PEP 8 conventions
python3 -c "import py_compile; py_compile.compile('feasibility_scorer.py', doraise=True); print('Lint successful')"
```

### Test
```bash
python3 run_tests.py
# Alternative with pytest (if installed):
# pytest test_feasibility_scorer.py -v
```

### Dev Server
```bash
# Demo script for interactive testing
python3 demo.py
```

## Tech Stack
- **Language**: Python 3.7+
- **Framework**: Standard library (regex, dataclasses, enum)
- **Package Manager**: pip (minimal requirements)
- **Testing**: Custom test runner + pytest support

## Architecture
```
feasibility_scorer.py     # Main scorer implementation
├── FeasibilityScorer    # Main class with regex patterns
├── ComponentType        # Enum for component types  
├── DetectionResult      # Dataclass for matches
└── IndicatorScore       # Dataclass for final results

test_feasibility_scorer.py  # Comprehensive pytest test suite
run_tests.py                 # Simple test runner (no dependencies)
demo.py                      # Interactive demonstration
```

## Code Style
- Follows PEP 8 Python conventions
- Comprehensive docstrings with examples
- Type hints using typing module
- Dataclasses for structured data
- Regex patterns organized by component type
- Spanish and English multilingual support