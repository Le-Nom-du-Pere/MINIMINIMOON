# AGENTS.md

## Commands

### Setup
```bash
# Python project with Unicode normalization - standard library only
# Optional: install pytest for advanced testing
pip install pytest
```

### Build
```bash
python3 -c "import text_processor, utils; print('Build successful')"
```

### Lint
```bash
# Code follows PEP 8 conventions
python3 -m py_compile text_processor.py utils.py test_unicode_normalization.py demo_unicode_comparison.py
```

### Test
```bash
python3 test_unicode_normalization.py
# Alternative demo:
python3 demo_unicode_comparison.py
```

### Dev Server
```bash
# Demo script for interactive testing
python3 demo_unicode_comparison.py
```

## Tech Stack
- **Language**: Python 3.7+
- **Framework**: Standard library (regex, unicodedata, typing)
- **Package Manager**: pip (minimal requirements)
- **Testing**: unittest with custom test cases

## Architecture
```
text_processor.py           # Core text processing with Unicode normalization
├── normalize_unicode()    # NFKC normalization function
├── find_quotes()          # Quote detection with normalization
├── count_words()          # Word counting with normalization
└── extract_emails()       # Email extraction with normalization

utils.py                    # Utility classes and functions
├── TextAnalyzer           # Main analysis class
├── normalize_text()       # Normalization wrapper
└── pattern matching       # Various regex patterns with normalization

test_unicode_normalization.py  # Comprehensive test suite
demo_unicode_comparison.py     # Before/after normalization demo
```

## Code Style
- Follows PEP 8 Python conventions
- Comprehensive docstrings with examples
- Type hints using typing module
- Unicode normalization applied before all regex operations
- Consistent NFKC normalization prevents overcounting issues
