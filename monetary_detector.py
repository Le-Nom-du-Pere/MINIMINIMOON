import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class MonetaryType(Enum):
    """Enumeration of monetary value types."""
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    NUMERIC = "numeric"


@dataclass
class MonetaryMatch:
    """Represents a detected monetary expression."""
    text: str
    value: float
    type: MonetaryType
    currency: Optional[str] = None
    original_match: str = ""
    start_pos: int = -1
    end_pos: int = -1


class MonetaryDetector:
    """
    Detects and normalizes monetary amounts, percentages, and numeric values in Spanish text.
    
    Conventions for ambiguous abbreviations:
    - M = million (1,000,000) - as per standard Spanish financial notation
    - MM = million (1,000,000) - alternative notation, treated same as M
    - K/k = thousand (1,000)
    - B = billion (1,000,000,000) - mil millones in Spanish
    
    Currency handling:
    - $ without explicit currency code defaults to generic dollar
    - COP = Colombian Peso
    - USD = US Dollar
    - EUR = Euro
    
    Decimal separators:
    - Comma (,) as decimal separator (Spanish convention)
    - Period (.) as thousands separator or decimal (context-dependent)
    """
    
    def __init__(self):
        self._setup_patterns()
        self._setup_scale_multipliers()
        self._setup_currency_symbols()
    
    def _setup_patterns(self):
        """Initialize regex patterns for detection."""
        # Currency symbols and codes
        currency_symbols = r'[\$€£¥]|USD|COP|EUR|GBP|JPY|CHF|CAD|AUD|MXN|PEN|CLP|ARS|BRL'
        
        # Scale indicators (Spanish terms)
        scale_terms = r'(?:millones?|miles?|mil|millón|billones?|billón|trillones?|trillón|[MmKkBb]|MM)'
        
        # Number patterns - supports both comma and period as decimal separators
        # Handles formats like: 1.234.567,89 or 1,234,567.89 or 1234567.89
        number_core = r'(?:\d+(?:[.,]\d{3})*(?:[.,]\d{1,3})?|\d+(?:[.,]\d+)?)'
        
        # Monetary pattern: [currency]?[number][scale]?[currency]?
        # Examples: $1.5 millones, 2.3M COP, USD 500K, 1,200.5 miles EUR
        self.monetary_pattern = re.compile(
            rf'(?P<pre_currency>{currency_symbols})?\s*'
            rf'(?P<number>{number_core})\s*'
            rf'(?P<scale>{scale_terms})?\s*'
            rf'(?P<post_currency>{currency_symbols})?',
            re.IGNORECASE
        )
        
        # Percentage pattern: number%
        # Examples: 15%, 12.5%, 0,75%
        self.percentage_pattern = re.compile(
            rf'(?P<number>{number_core})\s*%',
            re.IGNORECASE
        )
        
        # Pure numeric pattern (no currency or percentage)
        # Examples: 1.234.567, 12,5, 1000
        self.numeric_pattern = re.compile(
            rf'\b(?P<number>{number_core})\s*(?P<scale>{scale_terms})?\b',
            re.IGNORECASE
        )
    
    def _setup_scale_multipliers(self):
        """Setup scale multiplier mappings."""
        self.scale_multipliers = {
            # Spanish terms
            'mil': 1000,
            'miles': 1000,
            'millón': 1000000,
            'millones': 1000000,
            'billón': 1000000000,  # Spanish billion = 10^9 (US billion)
            'billones': 1000000000,
            'trillón': 1000000000000,
            'trillones': 1000000000000,
            # Abbreviations
            'k': 1000,
            'K': 1000,
            'm': 1000000,
            'M': 1000000,
            'MM': 1000000,  # Alternative million notation
            'b': 1000000000,
            'B': 1000000000,
        }
    
    def _setup_currency_symbols(self):
        """Setup currency symbol mappings."""
        self.currency_symbols = {
            '$': 'USD',  # Default assumption
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            'USD': 'USD',
            'COP': 'COP',
            'EUR': 'EUR',
            'GBP': 'GBP',
            'JPY': 'JPY',
            'CHF': 'CHF',
            'CAD': 'CAD',
            'AUD': 'AUD',
            'MXN': 'MXN',
            'PEN': 'PEN',
            'CLP': 'CLP',
            'ARS': 'ARS',
            'BRL': 'BRL',
        }
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode text using NFKC normalization."""
        return unicodedata.normalize("NFKC", text)
    
    @staticmethod
    def _parse_number(number_str: str) -> float:
        """
        Parse a number string handling Spanish decimal conventions.
        
        Spanish conventions:
        - Comma (,) as decimal separator
        - Period (.) as thousands separator
        - Both can be mixed: 1.234.567,89
        
        Ambiguous cases are resolved by position:
        - Last separator is decimal if followed by 1-2 digits
        - All others are thousands separators
        """
        # Remove whitespace
        number_str = number_str.strip()
        
        # Find all separators and their positions
        separators = []
        for i, char in enumerate(number_str):
            if char in '.,':
                separators.append((i, char))
        
        if not separators:
            # No separators, simple integer
            return float(number_str)
        
        if len(separators) == 1:
            pos, sep = separators[0]
            # Check if it's likely a decimal separator
            digits_after = len(number_str) - pos - 1
            if digits_after <= 2 and sep == ',':
                # Spanish decimal separator
                return float(number_str.replace(',', '.'))
            elif digits_after <= 2 and sep == '.' and digits_after > 0:
                # Could be decimal
                return float(number_str)
            else:
                # Thousands separator
                return float(number_str.replace(sep, ''))
        
        # Multiple separators - last one might be decimal
        last_pos, last_sep = separators[-1]
        digits_after_last = len(number_str) - last_pos - 1
        
        if digits_after_last <= 2:
            # Last separator is likely decimal
            if last_sep == ',':
                # Spanish format: last comma is decimal
                before_decimal = number_str[:last_pos]
                after_decimal = number_str[last_pos + 1:]
                clean_integer = re.sub(r'[.,]', '', before_decimal)
                return float(f"{clean_integer}.{after_decimal}")
            elif last_sep == '.' and digits_after_last == 2:
                # English format: last period with exactly 2 digits is decimal
                before_decimal = number_str[:last_pos]
                after_decimal = number_str[last_pos + 1:]
                clean_integer = re.sub(r'[.,]', '', before_decimal)
                return float(f"{clean_integer}.{after_decimal}")
        
        # All separators are thousands separators
        clean_number = re.sub(r'[.,]', '', number_str)
        return float(clean_number)
    
    def _extract_currency(self, pre_currency: Optional[str], post_currency: Optional[str]) -> Optional[str]:
        """Extract currency from pre or post position."""
        currency = pre_currency or post_currency
        if currency:
            return self.currency_symbols.get(currency.upper(), currency.upper())
        return None
    
    def _apply_scale(self, value: float, scale_str: Optional[str]) -> float:
        """Apply scale multiplier to value."""
        if not scale_str:
            return value
        
        # Handle both upper and lower case
        scale_key = scale_str if scale_str in self.scale_multipliers else scale_str.lower()
        multiplier = self.scale_multipliers.get(scale_key, 1.0)
        return value * multiplier
    
    def detect_monetary_expressions(self, text: str) -> List[MonetaryMatch]:
        """
        Detect all monetary expressions in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of MonetaryMatch objects with detected expressions
        """
        normalized_text = self.normalize_unicode(text)
        results = []
        
        # Track processed positions to avoid overlapping matches
        processed_positions = set()
        
        # Find monetary amounts with currencies
        for match in self.monetary_pattern.finditer(normalized_text):
            start, end = match.span()
            if any(pos in processed_positions for pos in range(start, end)):
                continue
                
            number_str = match.group('number')
            if not number_str:
                continue
                
            try:
                base_value = self._parse_number(number_str)
                scale_str = match.group('scale')
                final_value = self._apply_scale(base_value, scale_str)
                
                currency = self._extract_currency(
                    match.group('pre_currency'),
                    match.group('post_currency')
                )
                
                # Only add if it has currency (to distinguish from pure numbers)
                if currency:
                    results.append(MonetaryMatch(
                        text=match.group(0).strip(),
                        value=final_value,
                        type=MonetaryType.CURRENCY,
                        currency=currency,
                        original_match=match.group(0),
                        start_pos=start,
                        end_pos=end
                    ))
                    
                    # Mark positions as processed
                    processed_positions.update(range(start, end))
                    
            except ValueError:
                continue
        
        # Find percentages
        for match in self.percentage_pattern.finditer(normalized_text):
            start, end = match.span()
            if any(pos in processed_positions for pos in range(start, end)):
                continue
                
            number_str = match.group('number')
            try:
                percentage_value = self._parse_number(number_str)
                # Convert percentage to decimal (50% -> 0.5)
                decimal_value = percentage_value / 100.0
                
                results.append(MonetaryMatch(
                    text=match.group(0).strip(),
                    value=decimal_value,
                    type=MonetaryType.PERCENTAGE,
                    original_match=match.group(0),
                    start_pos=start,
                    end_pos=end
                ))
                
                processed_positions.update(range(start, end))
                
            except ValueError:
                continue
        
        # Find pure numeric values with scales
        for match in self.numeric_pattern.finditer(normalized_text):
            start, end = match.span()
            if any(pos in processed_positions for pos in range(start, end)):
                continue
                
            scale_str = match.group('scale')
            if not scale_str:  # Only process if it has a scale
                continue
                
            number_str = match.group('number')
            try:
                base_value = self._parse_number(number_str)
                final_value = self._apply_scale(base_value, scale_str)
                
                results.append(MonetaryMatch(
                    text=match.group(0).strip(),
                    value=final_value,
                    type=MonetaryType.NUMERIC,
                    original_match=match.group(0),
                    start_pos=start,
                    end_pos=end
                ))
                
                processed_positions.update(range(start, end))
                
            except ValueError:
                continue
        
        # Sort by position in text
        return sorted(results, key=lambda x: x.start_pos)
    
    def normalize_monetary_expression(self, expression: str) -> Optional[float]:
        """
        Normalize a single monetary expression to float value.
        
        Args:
            expression: Monetary expression string
            
        Returns:
            Normalized float value or None if not parseable
        """
        matches = self.detect_monetary_expressions(expression)
        return matches[0].value if matches else None


def create_monetary_detector() -> MonetaryDetector:
    """Factory function to create a MonetaryDetector instance."""
    return MonetaryDetector()