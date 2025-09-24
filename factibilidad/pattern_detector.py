import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PatternMatch:
    """Represents a pattern match with position and type information."""
    pattern_type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0


class PatternDetector:
    """Detects baseline, target, and timeframe patterns in Spanish text.
    
    This class provides comprehensive pattern detection capabilities for identifying
    baseline values, targets/objectives, and temporal indicators in Spanish text
    using compiled regular expressions.
    
    Attributes:
        baseline_patterns: Compiled regex patterns for baseline detection.
        target_patterns: Compiled regex patterns for target detection.
        timeframe_patterns: Compiled regex patterns for timeframe detection.
    """

    def __init__(self):
        """Initialize the pattern detector with compiled regex patterns."""
        self.baseline_patterns = self._compile_baseline_patterns()
        self.target_patterns = self._compile_target_patterns()
        self.timeframe_patterns = self._compile_timeframe_patterns()

    def _compile_baseline_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for baseline indicators.
        
        Creates compiled regular expressions for detecting various forms
        of baseline references in Spanish text.
        
        Returns:
            List of compiled regex pattern objects for baseline detection.
        """

    def _compile_baseline_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for baseline indicators.
        
        Creates compiled regular expressions for detecting various forms
        of baseline references in Spanish text.
        
        Returns:
            List of compiled regex pattern objects for baseline detection.
        """
        patterns = [
            r'\b(?:línea\s+base|linea\s+base|línea\s+de\s+base|linea\s+de\s+base)\b',
            r'\b(?:situación\s+inicial|situacion\s+inicial)\b',
            r'\b(?:punto\s+de\s+partida)\b',
            r'\b(?:estado\s+actual)\b',
            r'\b(?:condición\s+inicial|condicion\s+inicial)\b',
            r'\b(?:nivel\s+base)\b',
            r'\b(?:valor\s+inicial)\b',
            r'\b(?:posición\s+inicial|posicion\s+inicial)\b',
            r'\b(?:baseline)\b',
            r'\b(?:actualmente|en\s+la\s+actualidad)\b',
            r'\b(?:al\s+inicio|inicialmente)\b'
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _compile_target_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for target indicators.
        
        Creates compiled regular expressions for detecting various forms
        of target, goal, and objective references in Spanish text.
        
        Returns:
            List of compiled regex pattern objects for target detection.
        """
        patterns = [
            r'\b(?:meta|metas)\b',
            r'\b(?:objetivo|objetivos)\b',
            r'\b(?:alcanzar|lograr)\b',
            r'\b(?:conseguir|obtener)\b',
            r'\b(?:target|targets)\b',
            r'\b(?:propósito|proposito)\b',
            r'\b(?:finalidad)\b',
            r'\b(?:resultado\s+esperado)\b',
            r'\b(?:expectativa|expectativas)\b',
            r'\b(?:aspiración|aspiracion)\b',
            r'\b(?:pretende|pretender)\b',
            r'\b(?:busca|buscar)\b',
            r'\b(?:se\s+espera)\b',
            r'\b(?:se\s+proyecta)\b'
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _compile_timeframe_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for timeframe indicators.
        
        Creates compiled regular expressions for detecting various forms
        of temporal references including absolute dates, relative time
        expressions, and administrative periods in Spanish text.
        
        Returns:
            List of compiled regex pattern objects for timeframe detection.
        """
        patterns = [
            # Absolute years
            r'\b(?:20\d{2})\b',
            # Relative time expressions
            r'\b(?:al\s+(?:20\d{2}|año\s+20\d{2}))\b',
            r'\b(?:en\s+(?:\d+\s+(?:años?|meses?|días?)))\b',
            r'\b(?:para\s+(?:el\s+)?(?:20\d{2}|fin\s+de\s+año))\b',
            r'\b(?:hasta\s+(?:el\s+)?20\d{2})\b',
            # Quarters and periods
            r'\b(?:[1-4]º?\s*(?:trimestre|cuatrimestre))\b',
            r'\b(?:primer|segundo|tercer|cuarto)\s+(?:trimestre|cuatrimestre)\b',
            r'\b(?:Q[1-4])\b',
            # Months + year
            r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?20\d{2}\b',
            # Relative temporal expressions
            r'\b(?:en\s+los\s+próximos\s+\d+\s+(?:años?|meses?))\b',
            r'\b(?:dentro\s+de\s+\d+\s+(?:años?|meses?))\b',
            r'\b(?:a\s+(?:corto|mediano|largo)\s+plazo)\b',
            r'\b(?:próximo\s+año|proximo\s+año)\b',
            r'\b(?:año\s+(?:que\s+viene|entrante))\b',
            # Date ranges
            r'\b(?:20\d{2}\s*[-–—]\s*20\d{2})\b',
            # Specific date patterns
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b',
            # Vigencias y periodos administrativos
            r'\b(?:vigencia\s+20\d{2})\b',
            r'\b(?:PDD\s*20\d{2}\s*[-–—]\s*20\d{2})\b'
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def detect_patterns(self, text: str) -> Dict[str, List[PatternMatch]]:
        """Detect all pattern types in the given text.
        
        Analyzes input text to identify baseline, target, and timeframe patterns
        using the compiled regular expressions.

        Args:
            text: The text to analyze for pattern detection.

        Returns:
            Dictionary with pattern types as keys and lists of PatternMatch
            objects as values, containing detected patterns with position
            and confidence information.
        """
        return {
            'baseline': PatternDetector._find_matches(text, self.baseline_patterns, 'baseline'),
            'target': PatternDetector._find_matches(text, self.target_patterns, 'target'),
            'timeframe': PatternDetector._find_matches(text, self.timeframe_patterns, 'timeframe')
        }

    def _find_matches(self, text: str, patterns: List[re.Pattern], pattern_type: str) -> List[PatternMatch]:
        """Find all matches for a specific pattern type.
        
        Applies all patterns of a given type to the text and filters out
        overlapping matches, preferring longer matches when conflicts occur.
        
        Args:
            text: Text to search for patterns.
            patterns: List of compiled regex patterns to apply.
            pattern_type: Type identifier for the patterns being applied.
            
        Returns:
            List of non-overlapping PatternMatch objects found in the text.
        """
        matches: List[PatternMatch] = []
        for pattern in patterns:
            for m in pattern.finditer(text):
                matches.append(PatternMatch(
                    pattern_type=pattern_type,
                    text=m.group(),
                    start=m.start(),
                    end=m.end()
                ))

        # Remove overlapping matches, keeping the longest one
        matches.sort(key=lambda x: (x.start, -(x.end - x.start)))
        filtered: List[PatternMatch] = []
        for m in matches:
            overlaps = any(
                (e.start <= m.start < e.end) or
                (e.start < m.end <= e.end) or
                (m.start <= e.start and m.end >= e.end)
                for e in filtered
            )
            if not overlaps:
                filtered.append(m)
        return filtered

    def find_pattern_clusters(self, text: str, proximity_window: int = 500) -> List[Dict]:
        """Find text segments where all three pattern types appear within proximity.
        
        Identifies regions in the text where baseline, target, and timeframe
        patterns co-occur within a specified character distance, indicating
        comprehensive indicator descriptions.

        Args:
            text: The text to analyze for pattern clustering.
            proximity_window: Maximum character distance between patterns
                            for them to be considered part of the same cluster.

        Returns:
            List of dictionaries containing cluster information including
            start/end positions, extracted text, matches by type, and span length.
        """
        all_matches = self.detect_patterns(text)
        clusters: List[Dict] = []

        baseline_matches = all_matches['baseline']
        target_matches = all_matches['target']
        timeframe_matches = all_matches['timeframe']

        for baseline in baseline_matches:
            cluster = {'baseline': [baseline], 'target': [], 'timeframe': []}

            for target in target_matches:
                if PatternDetector._within_proximity(baseline, target, proximity_window):
                    cluster['target'].append(target)

            for timeframe in timeframe_matches:
                if PatternDetector._within_proximity(baseline, timeframe, proximity_window):
                    cluster['timeframe'].append(timeframe)

            if cluster['target'] and cluster['timeframe']:
                all_m = cluster['baseline'] + cluster['target'] + cluster['timeframe']
                start_pos = min(m.start for m in all_m)
                end_pos = max(m.end for m in all_m)

                clusters.append({
                    'start': start_pos,
                    'end': end_pos,
                    'text': text[start_pos:end_pos],
                    'matches': cluster,
                    'span': end_pos - start_pos
                })

        return clusters

    def _within_proximity(self, a: PatternMatch, b: PatternMatch, proximity_window: int) -> bool:
        """Check if two matches are within the specified proximity window.
        
        Calculates the minimum distance between two pattern matches and
        determines if they fall within the specified proximity threshold.
        
        Args:
            a: First pattern match to compare.
            b: Second pattern match to compare.
            proximity_window: Maximum distance in characters for proximity.
            
        Returns:
            True if the matches are within the proximity window, False otherwise.
        """
        distance = min(
            abs(a.start - b.end),
            abs(a.end - b.start),
            abs(a.start - b.start),
            abs(a.end - b.end)
        )
        return distance <= proximity_window
