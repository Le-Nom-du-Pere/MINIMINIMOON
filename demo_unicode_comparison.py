#!/usr/bin/env python3
# coding=utf-8
"""
Demo script showing Unicode normalization effects on regex matching.
Compares regex match counts before and after normalization.
"""

import re
import unicodedata
from text_processor import normalize_unicode, find_quotes, count_words
from utils import TextAnalyzer


def compare_regex_results_before_after_normalization():
    """Compare regex results before and after Unicode normalization."""
    
    # Sample texts with Unicode variations
    test_texts = {
        "Smart quotes": '"Hello" vs "Hello"',
        "Accented characters": "café résumé naïve vs café résumé naïve", 
        "Em dashes": "Text—with—dashes vs Text-with-dashes",
        "Mixed Unicode": 'Mixed "quote" styles',
        "Decomposed characters": "café vs cafe\u0301",  # Different Unicode forms
    }
    
    print("Unicode Normalization Comparison")
    print("=" * 60)
    
    analyzer = TextAnalyzer()
    
    for description, text in test_texts.items():
        print(f"\n{description}:")
        print(f"Original text: {text}")
        print(f"Text length: {len(text)} characters")
        
        # Before normalization - direct regex on original text
        quotes_before = len(re.findall(r'[""''"\']', text))
        words_before = len(re.findall(r'\b\w+\b', text))
        
        # After normalization - using our normalized functions
        quotes_after = len(find_quotes(text))
        words_after = count_words(text)
        
        # Show normalized version
        normalized = normalize_unicode(text)
        print(f"Normalized:   {normalized}")
        print(f"Normalized length: {len(normalized)} characters")
        
        print(f"Quotes found - Before: {quotes_before}, After: {quotes_after}")
        print(f"Words found  - Before: {words_before}, After: {words_after}")
        
        # Show character-by-character differences if any
        if text != normalized:
            print("Character differences detected:")
            for i, (orig, norm) in enumerate(zip(text, normalized)):
                if orig != norm:
                    print(f"  Position {i}: '{orig}' (U+{ord(orig):04X}) -> '{norm}' (U+{ord(norm):04X})")
    
    # Demonstrate overcounting prevention
    print(f"\n{'Overcounting Prevention Demo'}")
    print("=" * 40)
    
    # Same visual character in different Unicode forms
    composed = "é"      # Single composed character
    decomposed = "e\u0301"  # Base + combining accent
    
    print(f"Composed form: '{composed}' (length: {len(composed)})")
    print(f"Decomposed form: '{decomposed}' (length: {len(decomposed)})")
    print(f"Are they equal? {composed == decomposed}")
    
    # Without normalization - different counts
    text1 = f"The word {composed} appears here"
    text2 = f"The word {decomposed} appears here"
    
    count1_before = len(re.findall(r'\w+', text1))
    count2_before = len(re.findall(r'\w+', text2))
    
    # With normalization - same counts
    count1_after = count_words(text1)
    count2_after = count_words(text2)
    
    print(f"\nWord counting without normalization:")
    print(f"Text with composed é: {count1_before} words")
    print(f"Text with decomposed é: {count2_before} words")
    
    print(f"\nWord counting with normalization:")
    print(f"Text with composed é: {count1_after} words")
    print(f"Text with decomposed é: {count2_after} words")
    print(f"Counts are equal: {count1_after == count2_after}")


if __name__ == "__main__":
    compare_regex_results_before_after_normalization()