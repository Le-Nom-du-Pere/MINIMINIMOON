import re
from typing import List, Dict, Tuple, NamedTuple
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
    """Detects baseline, target, and timeframe patterns in Spanish text."""
    
    def __init__(self):
        self.baseline_patterns = self._compile_baseline_patterns()
        self.target_patterns = self._compile_target_patterns()
        self.timeframe_patterns = self._compile_timeframe_patterns()
    
    def _compile_baseline_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for baseline indicators."""
        patterns = [
            r'\b(?:línea\s+base|linea\s+base)\b',
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
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _compile_target_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for target indicators."""
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
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _compile_timeframe_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for timeframe indicators."""
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
            # Months
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
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b'
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def detect_patterns(self, text: str) -> Dict[str, List[PatternMatch]]:
        """
        Detect all pattern types in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with pattern types as keys and lists of matches as values
        """
        matches = {
            'baseline': self._find_matches(text, self.baseline_patterns, 'baseline'),
            'target': self._find_matches(text, self.target_patterns, 'target'),
            'timeframe': self._find_matches(text, self.timeframe_patterns, 'timeframe')
        }
        return matches
    
    def _find_matches(self, text: str, patterns: List[re.Pattern], pattern_type: str) -> List[PatternMatch]:
        """Find all matches for a specific pattern type."""
        matches = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                matches.append(PatternMatch(
                    pattern_type=pattern_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end()
                ))
        
        # Remove overlapping matches, keeping the longest one
        matches.sort(key=lambda x: (x.start, -(x.end - x.start)))
        filtered_matches = []
        for match in matches:
            if not any(existing.start <= match.start < existing.end or 
                      existing.start < match.end <= existing.end 
                      for existing in filtered_matches):
                filtered_matches.append(match)
        
        return filtered_matches
    
    def find_pattern_clusters(self, text: str, proximity_window: int = 500) -> List[Dict]:
        """
        Find text segments where all three pattern types appear within proximity.
        
        Args:
            text: The text to analyze
            proximity_window: Maximum character distance between patterns
            
        Returns:
            List of dictionaries containing cluster information
        """
        all_matches = self.detect_patterns(text)
        clusters = []
        
        # Get all baseline matches
        baseline_matches = all_matches['baseline']
        target_matches = all_matches['target']
        timeframe_matches = all_matches['timeframe']
        
        # For each baseline match, look for nearby target and timeframe matches
        for baseline in baseline_matches:
            cluster_matches = {'baseline': [baseline], 'target': [], 'timeframe': []}
            
            # Find targets within proximity
            for target in target_matches:
                if self._within_proximity(baseline, target, proximity_window):
                    cluster_matches['target'].append(target)
            
            # Find timeframes within proximity
            for timeframe in timeframe_matches:
                if self._within_proximity(baseline, timeframe, proximity_window):
                    cluster_matches['timeframe'].append(timeframe)
            
            # If we have all three types, create a cluster
            if cluster_matches['target'] and cluster_matches['timeframe']:
                all_cluster_matches = (cluster_matches['baseline'] + 
                                     cluster_matches['target'] + 
                                     cluster_matches['timeframe'])
                
                start_pos = min(match.start for match in all_cluster_matches)
                end_pos = max(match.end for match in all_cluster_matches)
                
                clusters.append({
                    'start': start_pos,
                    'end': end_pos,
                    'text': text[start_pos:end_pos],
                    'matches': cluster_matches,
                    'span': end_pos - start_pos
                })
        
        return clusters
    
    def _within_proximity(self, match1: PatternMatch, match2: PatternMatch, 
                         proximity_window: int) -> bool:
        """Check if two matches are within the specified proximity window."""
        distance = min(
            abs(match1.start - match2.end),
            abs(match1.end - match2.start),
            abs(match1.start - match2.start),
            abs(match1.end - match2.end)
        )
        return distance <= proximity_window