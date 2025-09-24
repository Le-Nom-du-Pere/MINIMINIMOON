# coding=utf-8
import re
import unicodedata
from typing import List, Optional, Match, Pattern


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text using NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def find_quotes(text: str) -> List[Match[str]]:
    """Find all quote characters in text with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    pattern = r'[""''"\']'
    return re.findall(pattern, normalized_text)


def count_words(text: str) -> int:
    """Count words in text with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    pattern = r'\b\w+\b'
    matches = re.findall(pattern, normalized_text)
    return len(matches)


def extract_emails(text: str) -> List[str]:
    """Extract email addresses with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    # Updated pattern to handle Unicode characters in email addresses
    pattern = r'\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b'
    return re.findall(pattern, normalized_text)


def replace_special_chars(text: str, replacement: str = " ") -> str:
    """Replace special characters with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    pattern = r'[^\w\s]'
    return re.sub(pattern, replacement, normalized_text)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    pattern = r'[.!?]+\s*'
    return re.split(pattern, normalized_text)


def search_pattern(text: str, pattern: str) -> Optional[Match[str]]:
    """Search for a pattern with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    normalized_pattern = normalize_unicode(pattern)
    return re.search(normalized_pattern, normalized_text)


def match_phone_numbers(text: str) -> List[str]:
    """Match phone numbers with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    pattern = r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}'
    return re.findall(pattern, normalized_text)


def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords with Unicode normalization."""
    normalized_text = normalize_unicode(text)
    for keyword in keywords:
        normalized_keyword = normalize_unicode(keyword)
        pattern = re.escape(normalized_keyword)
        normalized_text = re.sub(pattern, f"**{normalized_keyword}**", normalized_text, flags=re.IGNORECASE)
    return normalized_text