# coding=utf-8
import math
from typing import List, Dict, Tuple
from .pattern_detector import PatternDetector, PatternMatch


class FactibilidadScorer:
    """Calculates factibilidad scores based on pattern detection and proximity."""
    
    def __init__(self, proximity_window: int = 500, base_score: float = 0.0):
        """
        Initialize the scorer.
        
        Args:
            proximity_window: Maximum character distance for pattern clustering
            base_score: Base factibilidad score before pattern bonuses
        """
        self.pattern_detector = PatternDetector()
        self.proximity_window = proximity_window
        self.base_score = base_score
        
        # Scoring weights
        self.cluster_bonus = 25.0  # Bonus for finding all three patterns together
        self.proximity_bonus_max = 15.0  # Maximum bonus for close proximity
        self.individual_pattern_bonus = 5.0  # Bonus for each pattern type found
        self.multiple_instance_bonus = 2.0  # Bonus for multiple instances of same type
    
    def score_text(self, text: str) -> Dict:
        """
        Calculate factibilidad score for the given text.
        
        Args:
            text: Text to analyze and score
            
        Returns:
            Dictionary containing score breakdown and analysis
        """
        # Detect all patterns
        all_matches = self.pattern_detector.detect_patterns(text)
        clusters = self.pattern_detector.find_pattern_clusters(text, self.proximity_window)
        
        # Calculate base individual pattern scores
        individual_scores = self._calculate_individual_scores(all_matches)
        
        # Calculate cluster scores
        cluster_scores = self._calculate_cluster_scores(clusters)
        
        # Calculate final score
        total_score = (self.base_score + 
                      individual_scores['total'] + 
                      cluster_scores['total'])
        
        return {
            'total_score': min(total_score, 100.0),  # Cap at 100
            'base_score': self.base_score,
            'individual_pattern_scores': individual_scores,
            'cluster_scores': cluster_scores,
            'pattern_matches': all_matches,
            'clusters': clusters,
            'analysis': self._generate_analysis(all_matches, clusters)
        }
    
    def _calculate_individual_scores(self, matches: Dict[str, List[PatternMatch]]) -> Dict:
        """Calculate scores for individual pattern types."""
        scores = {}
        total = 0.0
        
        for pattern_type, pattern_matches in matches.items():
            if pattern_matches:
                # Base bonus for having this pattern type
                type_score = self.individual_pattern_bonus
                
                # Bonus for multiple instances (diminishing returns)
                if len(pattern_matches) > 1:
                    type_score += self.multiple_instance_bonus * math.log(len(pattern_matches))
                
                scores[pattern_type] = {
                    'score': type_score,
                    'count': len(pattern_matches),
                    'matches': pattern_matches
                }
                total += type_score
            else:
                scores[pattern_type] = {
                    'score': 0.0,
                    'count': 0,
                    'matches': []
                }
        
        scores['total'] = total
        return scores
    
    def _calculate_cluster_scores(self, clusters: List[Dict]) -> Dict:
        """Calculate scores for pattern clusters."""
        cluster_scores = []
        total_score = 0.0
        
        for cluster in clusters:
            # Base cluster bonus
            cluster_score = self.cluster_bonus
            
            # Proximity bonus (inverse relationship with span)
            span = cluster['span']
            proximity_factor = max(0, 1 - (span / self.proximity_window))
            proximity_bonus = self.proximity_bonus_max * proximity_factor
            cluster_score += proximity_bonus
            
            # Density bonus (more patterns in smaller space)
            total_matches = (len(cluster['matches']['baseline']) + 
                           len(cluster['matches']['target']) + 
                           len(cluster['matches']['timeframe']))
            density_bonus = min(5.0, total_matches * 1.5)
            cluster_score += density_bonus
            
            cluster_info = {
                'base_bonus': self.cluster_bonus,
                'proximity_bonus': proximity_bonus,
                'density_bonus': density_bonus,
                'total_score': cluster_score,
                'span': span,
                'match_count': total_matches,
                'cluster': cluster
            }
            
            cluster_scores.append(cluster_info)
            total_score += cluster_score
        
        return {
            'clusters': cluster_scores,
            'total': total_score,
            'count': len(clusters)
        }
    
    def _generate_analysis(self, matches: Dict[str, List[PatternMatch]], 
                          clusters: List[Dict]) -> Dict:
        """Generate human-readable analysis of the scoring."""
        analysis = {
            'has_baseline': len(matches['baseline']) > 0,
            'has_target': len(matches['target']) > 0,
            'has_timeframe': len(matches['timeframe']) > 0,
            'has_complete_cluster': len(clusters) > 0,
            'pattern_counts': {
                'baseline': len(matches['baseline']),
                'target': len(matches['target']),
                'timeframe': len(matches['timeframe'])
            },
            'cluster_count': len(clusters),
            'strengths': [],
            'weaknesses': []
        }
        
        # Identify strengths
        if analysis['has_complete_cluster']:
            analysis['strengths'].append("Contiene patrones completos de línea base, metas y plazos")
        
        if len(clusters) > 1:
            analysis['strengths'].append("Múltiples grupos de patrones completos identificados")
        
        if any(len(matches[pt]) > 2 for pt in matches):
            analysis['strengths'].append("Rica en indicadores específicos")
        
        # Identify weaknesses
        if not analysis['has_baseline']:
            analysis['weaknesses'].append("Falta indicadores de línea base o situación inicial")
        
        if not analysis['has_target']:
            analysis['weaknesses'].append("Falta indicadores de metas u objetivos claros")
        
        if not analysis['has_timeframe']:
            analysis['weaknesses'].append("Falta indicadores temporales o plazos específicos")
        
        if not analysis['has_complete_cluster']:
            analysis['weaknesses'].append("Los patrones no aparecen agrupados en el texto")
        
        return analysis
    
    def score_segments(self, text: str, segment_size: int = 1000, 
                      overlap: int = 200) -> List[Dict]:
        """
        Score text in segments for more granular analysis.
        
        Args:
            text: Text to analyze
            segment_size: Size of each segment in characters
            overlap: Overlap between segments in characters
            
        Returns:
            List of segment scores
        """
        segments = []
        start = 0
        segment_id = 0
        
        while start < len(text):
            end = min(start + segment_size, len(text))
            segment_text = text[start:end]
            
            score_result = self.score_text(segment_text)
            score_result.update({
                'segment_id': segment_id,
                'start_pos': start,
                'end_pos': end,
                'text': segment_text
            })
            
            segments.append(score_result)
            
            start += segment_size - overlap
            segment_id += 1
            
            if end >= len(text):
                break
        
        return segments