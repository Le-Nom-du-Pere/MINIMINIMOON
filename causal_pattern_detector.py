# coding=utf-8
"""
CAUSAL PATTERN DETECTOR
======================

Detects causal connector patterns in Spanish text using regex-based pattern matching
with Unicode normalization support. Includes semantic strength weighting and 
precision optimization to minimize false positives.

Features:
- Unicode-normalized pattern matching
- Weighted pattern scoring based on semantic strength
- False positive mitigation through context analysis  
- Comprehensive test coverage for Spanish causal connectors
- Integration with existing text processing pipeline
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from text_processor import normalize_unicode


@dataclass
class CausalMatch:
    """Represents a causal pattern match with context and confidence scoring."""
    connector: str
    pattern_type: str
    text: str
    start: int
    end: int
    confidence: float
    context_before: str = ""
    context_after: str = ""
    semantic_strength: float = 1.0


class CausalPatternDetector:
    """
    Detects causal connector patterns in Spanish text with semantic strength weighting
    and false positive reduction through context analysis.
    """

    def __init__(self):
        # Define causal connectors with their semantic strength weights
        # Higher weights indicate stronger causal relationships, lower weights
        # indicate potential for more false positives
        self.causal_connectors = {
            # Strong causal connectors (high confidence, less false positives)
            'porque': 0.95,
            'debido a': 0.90,
            'a causa de': 0.90,
            'por causa de': 0.85,
            'como resultado de': 0.85,
            'como consecuencia de': 0.85,
            
            # Medium strength connectors (moderate confidence)
            'ya que': 0.80,
            'puesto que': 0.80,
            'dado que': 0.75,
            'por lo que': 0.75,
            'de manera que': 0.70,
            'de modo que': 0.70,
            
            # NEW PATTERNS - with adjusted weights based on semantic analysis
            'implica': 0.60,      # Lower weight - can be used in non-causal contexts
            'conduce a': 0.75,    # Medium-high weight - usually indicates causation
            'mediante': 0.50,     # Lower weight - often instrumental rather than causal
            'por medio de': 0.55, # Similar to 'mediante' - instrumental usage
            'tendencia a': 0.45,  # Lowest weight - often correlational rather than causal
            
            # Additional existing patterns
            'genera': 0.70,
            'produce': 0.70,
            'provoca': 0.75,
            'origina': 0.75,
            'conlleva': 0.70,
            'resulta en': 0.80,
            'tiene como resultado': 0.85
        }
        
        # Compile regex patterns with Unicode normalization
        self.compiled_patterns = self._compile_causal_patterns()
        
        # Context patterns that reduce confidence (common false positive contexts)
        self.false_positive_contexts = self._compile_false_positive_patterns()

    def _compile_causal_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for causal connectors with Unicode support."""
        patterns = {}
        
        for connector, weight in self.causal_connectors.items():
            # Normalize the connector for pattern matching
            normalized_connector = normalize_unicode(connector)
            
            # Create flexible patterns that handle:
            # - Word boundaries
            # - Optional articles/prepositions
            # - Accent variations
            # - Case insensitivity
            
            if connector == 'implica':
                # Handle verb conjugations: implica, implican, implicaba, etc.
                pattern = r'\b(?:implica[rns]?|implicab[aa]|impliq[uúü]e?)\b'
            elif connector == 'conduce a':
                # Handle verb conjugations and preposition variations
                # Including infinitive forms like 'conducir a'
                pattern = r'\b(?:conduc(?:e|en|ir)(?:\s+a|\s+hacia|\s+al?))\b'
            elif connector == 'mediante':
                # Simple pattern but with boundary checks
                pattern = r'\b(?:mediante)\b'
            elif connector == 'por medio de':
                # Allow some flexibility in preposition structure
                pattern = r'\b(?:por\s+medio\s+de(?:\s+la|\s+el|\s+los|\s+las)?)\b'
            elif connector == 'tendencia a':
                # Match various forms of tendency expressions
                pattern = r'\b(?:tendencia\s+a|tiend[eao]\s+a)\b'
            else:
                # For existing patterns, escape special regex characters
                escaped = re.escape(normalized_connector)
                pattern = r'\b(?:' + escaped + r')\b'
            
            patterns[connector] = re.compile(pattern, re.IGNORECASE | re.UNICODE)
        
        return patterns

    def _compile_false_positive_patterns(self) -> List[re.Pattern]:
        """Compile patterns that indicate likely false positives."""
        fp_patterns = [
            # Question contexts - reduce confidence for interrogative sentences
            r'[¿?].*?[?¿]',
            
            # Negation contexts - "no implica", "no conduce", etc.
            r'\b(?:no|nunca|jamás)\s+(?:implica|conduce|mediante)',
            
            # Hypothetical contexts - "si implica", "podría conducir"
            r'\b(?:si|cuando|podría|debería)\s+.*?(?:implica|conduce)',
            
            # Mathematical/technical contexts for "implica"
            r'\b(?:ecuación|fórmula|teorema|proposición)\s+.*?implica',
            
            # Instrumental vs causal distinction for "mediante"
            r'\b(?:método|técnica|herramienta|instrumento)\s+.*?mediante',
            
            # Statistical vs causal for "tendencia"
            r'\b(?:estadística|correlación|datos|gráfico)\s+.*?tendencia'
        ]
        
        return [re.compile(p, re.IGNORECASE | re.UNICODE) for p in fp_patterns]

    def detect_causal_patterns(self, text: str, context_window: int = 100) -> List[CausalMatch]:
        """
        Detect causal patterns in text with confidence scoring and context analysis.
        
        Args:
            text: Input text to analyze
            context_window: Character window for context extraction
            
        Returns:
            List of CausalMatch objects with confidence scores
        """
        if not text:
            return []
        
        # Normalize text for consistent pattern matching
        normalized_text = normalize_unicode(text)
        matches = []
        
        for connector, pattern in self.compiled_patterns.items():
            base_confidence = self.causal_connectors[connector]
            
            for match in pattern.finditer(normalized_text):
                start, end = match.span()
                matched_text = match.group()
                
                # Extract context for analysis
                context_start = max(0, start - context_window)
                context_end = min(len(normalized_text), end + context_window)
                context_before = normalized_text[context_start:start]
                context_after = normalized_text[end:context_end]
                full_context = normalized_text[context_start:context_end]
                
                # Calculate adjusted confidence based on context
                adjusted_confidence = self._calculate_context_adjusted_confidence(
                    base_confidence, full_context, connector
                )
                
                # Determine pattern type based on connector characteristics
                pattern_type = self._classify_pattern_type(connector)
                
                causal_match = CausalMatch(
                    connector=connector,
                    pattern_type=pattern_type,
                    text=matched_text,
                    start=start,
                    end=end,
                    confidence=adjusted_confidence,
                    context_before=context_before.strip(),
                    context_after=context_after.strip(),
                    semantic_strength=base_confidence
                )
                
                matches.append(causal_match)
        
        # Remove overlapping matches, preferring higher confidence
        return self._resolve_overlapping_matches(matches)

    def _calculate_context_adjusted_confidence(self, base_confidence: float, 
                                             context: str, connector: str) -> float:
        """
        Adjust confidence based on contextual clues that indicate false positives.
        
        Args:
            base_confidence: Base semantic strength of the connector
            context: Surrounding text context
            connector: The specific connector being analyzed
            
        Returns:
            Adjusted confidence score (0.0 to 1.0)
        """
        adjusted_confidence = base_confidence
        
        # Check for false positive patterns
        for fp_pattern in self.false_positive_contexts:
            if fp_pattern.search(context):
                # Reduce confidence by 20-40% depending on pattern severity
                if 'no ' in context.lower() or 'nunca ' in context.lower():
                    adjusted_confidence *= 0.3  # Strong negation - major reduction
                elif '¿' in context or '?' in context:
                    adjusted_confidence *= 0.7  # Questions - moderate reduction
                else:
                    adjusted_confidence *= 0.6  # Other contexts - moderate reduction
        
        # Check for conditional/hypothetical contexts
        conditional_indicators = ['si ', 'cuando ', 'podría ', 'debería ', 'tal vez ', 'quizás ']
        if any(indicator in context.lower() for indicator in conditional_indicators):
            adjusted_confidence *= 0.6  # Conditional/hypothetical - reduce confidence
        
        # Connector-specific adjustments
        if connector in ['implica']:
            # Check for mathematical/logical contexts
            math_indicators = ['ecuación', 'fórmula', 'teorema', 'lógica', 'matemática']
            if any(indicator in context.lower() for indicator in math_indicators):
                adjusted_confidence *= 0.4  # Likely logical implication, not causal
        
        elif connector in ['mediante', 'por medio de']:
            # Check for instrumental vs causal usage
            instrumental_indicators = ['método', 'técnica', 'herramienta', 'proceso', 'procedimiento']
            if any(indicator in context.lower() for indicator in instrumental_indicators):
                adjusted_confidence *= 0.5  # Likely instrumental, not causal
        
        elif connector == 'tendencia a':
            # Check for statistical vs causal usage
            stats_indicators = ['estadística', 'correlación', 'datos', 'porcentaje', 'gráfico']
            if any(indicator in context.lower() for indicator in stats_indicators):
                adjusted_confidence *= 0.3  # Likely statistical trend, not causal
        
        return max(0.0, min(1.0, adjusted_confidence))

    def _classify_pattern_type(self, connector: str) -> str:
        """Classify the type of causal pattern based on connector characteristics."""
        if connector in ['porque', 'debido a', 'a causa de', 'por causa de']:
            return 'direct_causation'
        elif connector in ['como resultado de', 'como consecuencia de', 'resulta en']:
            return 'result_causation'
        elif connector in ['ya que', 'puesto que', 'dado que']:
            return 'reason_causation'
        elif connector in ['implica', 'conlleva']:
            return 'implication_causation'
        elif connector in ['conduce a', 'genera', 'produce', 'provoca']:
            return 'generative_causation'
        elif connector in ['mediante', 'por medio de']:
            return 'instrumental_causation'
        elif connector == 'tendencia a':
            return 'tendency_causation'
        else:
            return 'general_causation'

    def _resolve_overlapping_matches(self, matches: List[CausalMatch]) -> List[CausalMatch]:
        """
        Remove overlapping matches, preferring those with higher confidence scores.
        
        Args:
            matches: List of causal matches potentially containing overlaps
            
        Returns:
            List of non-overlapping matches
        """
        if not matches:
            return []
        
        # Sort by confidence (descending) then by position
        matches.sort(key=lambda x: (-x.confidence, x.start))
        
        filtered_matches = []
        for match in matches:
            # Check if this match overlaps with any already selected match
            overlaps = any(
                self._matches_overlap(match, existing) 
                for existing in filtered_matches
            )
            
            if not overlaps:
                filtered_matches.append(match)
        
        # Sort final results by position
        filtered_matches.sort(key=lambda x: x.start)
        return filtered_matches

    def _matches_overlap(self, match1: CausalMatch, match2: CausalMatch) -> bool:
        """Check if two matches overlap in text position."""
        return not (match1.end <= match2.start or match2.end <= match1.start)

    def calculate_pattern_statistics(self, text: str) -> Dict[str, any]:
        """
        Calculate comprehensive statistics about causal patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing pattern statistics
        """
        matches = self.detect_causal_patterns(text)
        
        if not matches:
            return {
                'total_matches': 0,
                'pattern_types': {},
                'confidence_distribution': {},
                'average_confidence': 0.0,
                'high_confidence_matches': 0,
                'potential_false_positives': 0
            }
        
        # Pattern type distribution
        pattern_types = {}
        for match in matches:
            pattern_types[match.pattern_type] = pattern_types.get(match.pattern_type, 0) + 1
        
        # Confidence distribution
        high_conf = len([m for m in matches if m.confidence >= 0.8])
        medium_conf = len([m for m in matches if 0.5 <= m.confidence < 0.8])
        low_conf = len([m for m in matches if m.confidence < 0.5])
        
        confidence_distribution = {
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf
        }
        
        avg_confidence = sum(m.confidence for m in matches) / len(matches)
        
        return {
            'total_matches': len(matches),
            'pattern_types': pattern_types,
            'confidence_distribution': confidence_distribution,
            'average_confidence': avg_confidence,
            'high_confidence_matches': high_conf,
            'potential_false_positives': low_conf
        }

    def get_supported_patterns(self) -> Dict[str, float]:
        """Return dictionary of supported causal patterns and their base weights."""
        return self.causal_connectors.copy()