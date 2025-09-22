# Unicode Normalization for Regex Matching

This project implements Unicode normalization using `unicodedata.normalize("NFKC", text)` before applying regex patterns to ensure consistent character representation and prevent overcounting issues, extending the existing feasibility scorer system.

## Files

- **text_processor.py** - Core text processing functions with Unicode normalization
- **utils.py** - Utility classes and functions for text analysis with normalization
- **test_unicode_normalization.py** - Comprehensive test suite
- **demo_unicode_comparison.py** - Demo showing before/after normalization effects

## Features

### Unicode Normalization Functions
- `normalize_unicode()` - Normalizes text using NFKC normalization
- `find_quotes()` - Finds quote characters with normalization
- `count_words()` - Counts words with normalization
- `extract_emails()` - Extracts email addresses with normalization
- `replace_special_chars()` - Replaces special characters with normalization
- `split_sentences()` - Splits text into sentences with normalization
- `search_pattern()` - Pattern search with normalization
- `match_phone_numbers()` - Phone number matching with normalization
- `highlight_keywords()` - Keyword highlighting with normalization

### TextAnalyzer Class
- Pattern matching for emails, URLs, phone numbers, hashtags, mentions
- Text cleaning and tokenization
- Quoted text extraction
- Unicode punctuation replacement
- Whitespace normalization

## Usage

```python
from text_processor import normalize_unicode, count_words, find_quotes
from utils import TextAnalyzer

# Basic normalization
text = "café résumé"
normalized = normalize_unicode(text)

# Word counting with normalization
count = count_words("Text with—em dashes")

# Using TextAnalyzer
analyzer = TextAnalyzer()
emails = analyzer.find_pattern_matches(text, 'email')
```

## Testing

Run the test suite:
```bash
python3 test_unicode_normalization.py
```

Run the demo:
```bash
python3 demo_unicode_comparison.py
```

## Benefits

1. **Consistent Character Representation** - Different Unicode encodings of the same visual character are normalized
2. **Prevents Overcounting** - Composed vs decomposed characters are treated identically
3. **Reliable Pattern Matching** - Regex patterns work consistently across different Unicode representations
4. **Comprehensive Coverage** - Handles quotes, accents, dashes, and other Unicode variants

## Example Output

The demo shows normalization effects:

```
Decomposed characters:
Original text: café vs café
Text length: 13 characters
Normalized:   café vs café  
Normalized length: 12 characters
Character differences detected:
  Position 11: 'e' (U+0065) -> 'é' (U+00E9)
```