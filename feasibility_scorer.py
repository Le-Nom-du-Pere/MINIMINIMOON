"""
Feasibility Scorer for Indicator Quality Assessment

This module implements a weighted quality assessment system that evaluates indicators
based on the presence of baseline values, targets/goals, and time horizons.
"""

import re
import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ComponentType(Enum):
    BASELINE = "baseline"
    TARGET = "target"
    TIME_HORIZON = "time_horizon"
    NUMERICAL = "numerical"
    DATE = "date"


@dataclass
class DetectionResult:
    component_type: ComponentType
    matched_text: str
    confidence: float
    position: int


@dataclass
class IndicatorScore:
    feasibility_score: float
    components_detected: List[ComponentType]
    detailed_matches: List[DetectionResult]
    has_quantitative_baseline: bool
    has_quantitative_target: bool
    quality_tier: str


class FeasibilityScorer:
    """
    Assesses indicator quality by detecting baseline values, targets/goals, and time horizons
    using regex patterns and named entity recognition.
    """
    
    def __init__(self):
        self.detection_patterns = self._initialize_patterns()
        self.weights = {
            ComponentType.BASELINE: 0.4,
            ComponentType.TARGET: 0.4,
            ComponentType.TIME_HORIZON: 0.2,
            ComponentType.NUMERICAL: 0.1,
            ComponentType.DATE: 0.1
        }
        self.quality_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
    
    def _initialize_patterns(self) -> Dict[ComponentType, List[Dict]]:
        """Initialize regex patterns for detecting indicator components in Spanish and English."""
        return {
            ComponentType.BASELINE: [
                {
                    'pattern': r'(?:línea\s+base|baseline|valor\s+inicial|situación\s+inicial|estado\s+actual)',
                    'confidence': 0.9,
                    'language': 'es/en'
                },
                {
                    'pattern': r'(?:punto\s+de\s+partida|referencia\s+inicial|nivel\s+base)',
                    'confidence': 0.8,
                    'language': 'es'
                },
                {
                    'pattern': r'(?:current\s+level|initial\s+value|starting\s+point)',
                    'confidence': 0.8,
                    'language': 'en'
                }
            ],
            ComponentType.TARGET: [
                {
                    'pattern': r'(?:meta|objetivo|target|goal)',
                    'confidence': 0.9,
                    'language': 'es/en'
                },
                {
                    'pattern': r'(?:propósito|finalidad|alcanzar|lograr|hasta)',
                    'confidence': 0.7,
                    'language': 'es'
                },
                {
                    'pattern': r'(?:achieve|reach|attain|aim|to\s)',
                    'confidence': 0.7,
                    'language': 'en'
                }
            ],
            ComponentType.TIME_HORIZON: [
                {
                    'pattern': r'(?:horizonte\s+temporal|plazo|período|periodo|duración)',
                    'confidence': 0.9,
                    'language': 'es'
                },
                {
                    'pattern': r'(?:time\s+horizon|timeline|timeframe|duration|period)',
                    'confidence': 0.9,
                    'language': 'en'
                },
                {
                    'pattern': r'(?:para\s+el\s+año|hasta\s+el|en\s+los\s+próximos|within|by\s+\d{4})',
                    'confidence': 0.8,
                    'language': 'es/en'
                }
            ],
            ComponentType.NUMERICAL: [
                {
                    'pattern': r'\d+(?:[.,]\d+)?(?:\s*%|\s*por\s*ciento|\s*percent)',
                    'confidence': 0.95,
                    'language': 'universal'
                },
                {
                    'pattern': r'\d+(?:[.,]\d+)?\s*(?:millones?|millions?|mil|thousand)',
                    'confidence': 0.9,
                    'language': 'es/en'
                },
                {
                    'pattern': r'(?:incrementar|aumentar|reducir|disminuir|increase|reduce)\s+(?:en\s+|by\s+)?\d+',
                    'confidence': 0.85,
                    'language': 'es/en'
                }
            ],
            ComponentType.DATE: [
                {
                    'pattern': r'\b(?:20\d{2}|19\d{2})\b',
                    'confidence': 0.9,
                    'language': 'universal'
                },
                {
                    'pattern': r'(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?20\d{2}',
                    'confidence': 0.95,
                    'language': 'es'
                },
                {
                    'pattern': r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+20\d{2}',
                    'confidence': 0.95,
                    'language': 'en'
                },
                {
                    'pattern': r'\d{1,2}[-/]\d{1,2}[-/](?:20\d{2}|\d{2})',
                    'confidence': 0.8,
                    'language': 'universal'
                }
            ]
        }
    
    def detect_components(self, text: str) -> List[DetectionResult]:
        """Detect all components in the given text using regex patterns."""
        results = []
        text_lower = text.lower()
        
        for component_type, patterns in self.detection_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                confidence = pattern_info['confidence']
                
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    result = DetectionResult(
                        component_type=component_type,
                        matched_text=match.group(),
                        confidence=confidence,
                        position=match.start()
                    )
                    results.append(result)
        
        return results
    
    def _has_quantitative_component(self, text: str, component_type: ComponentType) -> bool:
        """Check if a component has quantitative elements nearby."""
        text_lower = text.lower()
        
        # Find component mentions
        component_positions = []
        for pattern_info in self.detection_patterns[component_type]:
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            component_positions.extend([match.start() for match in matches])
        
        if not component_positions:
            return False
        
        # Check for numerical values within 30 characters of component mentions
        numerical_patterns = self.detection_patterns[ComponentType.NUMERICAL]
        for pos in component_positions:
            context_start = max(0, pos - 30)
            context_end = min(len(text), pos + 30)
            context = text_lower[context_start:context_end]
            
            for pattern_info in numerical_patterns:
                if re.search(pattern_info['pattern'], context, re.IGNORECASE):
                    return True
        
        return False
    
    def calculate_feasibility_score(self, text: str) -> IndicatorScore:
        """
        Calculate feasibility score based on detected components and their quality.
        
        Requirements for positive score:
        - Must have both baseline and target components
        - Higher scores for quantitative baselines and targets
        - Bonus for time horizons, numerical values, and dates
        """
        detected_components = self.detect_components(text)
        component_types = set(result.component_type for result in detected_components)
        
        # Check mandatory requirements
        has_baseline = ComponentType.BASELINE in component_types
        has_target = ComponentType.TARGET in component_types
        
        if not (has_baseline and has_target):
            return IndicatorScore(
                feasibility_score=0.0,
                components_detected=list(component_types),
                detailed_matches=detected_components,
                has_quantitative_baseline=False,
                has_quantitative_target=False,
                quality_tier='insufficient'
            )
        
        # Calculate base score from mandatory components
        base_score = (
            self.weights[ComponentType.BASELINE] + 
            self.weights[ComponentType.TARGET]
        )
        
        # Check for quantitative components
        has_quantitative_baseline = self._has_quantitative_component(text, ComponentType.BASELINE)
        has_quantitative_target = self._has_quantitative_component(text, ComponentType.TARGET)
        
        # Bonus for quantitative elements
        if has_quantitative_baseline:
            base_score += 0.2
        if has_quantitative_target:
            base_score += 0.2
        
        # Additional component bonuses
        if ComponentType.TIME_HORIZON in component_types:
            base_score += self.weights[ComponentType.TIME_HORIZON]
        
        if ComponentType.NUMERICAL in component_types:
            base_score += self.weights[ComponentType.NUMERICAL]
        
        if ComponentType.DATE in component_types:
            base_score += self.weights[ComponentType.DATE]
        
        # Apply confidence weighting
        avg_confidence = sum(result.confidence for result in detected_components) / len(detected_components)
        final_score = min(1.0, base_score * avg_confidence)
        
        # Determine quality tier
        if final_score >= self.quality_thresholds['high']:
            quality_tier = 'high'
        elif final_score >= self.quality_thresholds['medium']:
            quality_tier = 'medium'
        elif final_score >= self.quality_thresholds['low']:
            quality_tier = 'low'
        else:
            quality_tier = 'poor'
        
        return IndicatorScore(
            feasibility_score=final_score,
            components_detected=list(component_types),
            detailed_matches=detected_components,
            has_quantitative_baseline=has_quantitative_baseline,
            has_quantitative_target=has_quantitative_target,
            quality_tier=quality_tier
        )
    
    def batch_score(self, indicators: List[str]) -> List[IndicatorScore]:
        """Score multiple indicators."""
        return [self.calculate_feasibility_score(indicator) for indicator in indicators]
    
    def get_detection_rules_documentation(self) -> str:
        """Return comprehensive documentation of detection rules."""
        doc = """
# Feasibility Scorer Detection Rules Documentation

## Overview
The feasibility scorer evaluates indicator quality by detecting three core components:
1. **Baseline values** (línea base, baseline, valor inicial)
2. **Targets/goals** (meta, objetivo, target, goal)  
3. **Time horizons** (horizonte temporal, plazo, timeline)

## Scoring Logic
- **Minimum requirement**: Both baseline AND target components must be present for positive score
- **Base score**: 0.8 (0.4 baseline + 0.4 target)
- **Quantitative bonus**: +0.2 each for quantitative baseline and target
- **Component bonuses**: +0.2 time horizon, +0.1 numerical, +0.1 dates
- **Confidence weighting**: Final score multiplied by average pattern confidence

## Spanish Pattern Recognition

### Baseline Patterns
- línea base, valor inicial, situación inicial, estado actual
- punto de partida, referencia inicial, nivel base

### Target Patterns  
- meta, objetivo, propósito, finalidad
- alcanzar, lograr

### Time Horizon Patterns
- horizonte temporal, plazo, período, periodo, duración
- para el año, hasta el, en los próximos

### Numerical Patterns
- Percentages: 25%, 25 por ciento
- Quantities: 1.5 millones, 2 mil
- Change indicators: incrementar en 10, reducir 5%

### Date Patterns
- Years: 2024, 2025
- Spanish months: enero 2024, febrero de 2025
- Date formats: 15/12/2024, 15-12-2024

## Quality Tiers
- **High** (≥0.8): Complete indicators with quantitative elements
- **Medium** (≥0.5): Has baseline/target, some quantitative elements  
- **Low** (≥0.2): Basic baseline/target, limited quantitative data
- **Poor** (<0.2): Missing core components or very low confidence
- **Insufficient** (0.0): Missing baseline or target components

## Examples

### High Quality (Score: 0.95)
"Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025"
- ✓ Baseline: "línea base de 65%"  
- ✓ Target: "meta de 85%"
- ✓ Time horizon: "para el año 2025"
- ✓ Quantitative baseline and target

### Medium Quality (Score: 0.6)  
"Mejorar el objetivo de acceso al agua desde la situación actual hasta alcanzar la meta establecida"
- ✓ Baseline: "situación actual"
- ✓ Target: "meta establecida"  
- ✗ No quantitative elements
- ✗ No time horizon

### Insufficient Quality (Score: 0.0)
"Aumentar el acceso a servicios de salud en la región"
- ✗ No baseline reference
- ✓ Implied target: "aumentar"
- ✗ No quantitative elements
"""
        return doc