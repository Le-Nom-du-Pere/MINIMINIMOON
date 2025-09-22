"""
Feasibility Scorer for Indicator Quality Assessment

This module implements a weighted quality assessment system that evaluates indicators
based on the presence of baseline values, targets/goals, and time horizons.
"""

import os
import re
import datetime
import time
import unicodedata
import uuid
from pathlib import Path
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


@dataclass
class BatchScoreResult:
    scores: List[IndicatorScore]
    total_indicators: int
    execution_time: str
    duracion_segundos: float
    planes_por_minuto: float


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
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text using Unicode NFKC normalization for consistent character representation."""
        return unicodedata.normalize('NFKC', text)
    
    def detect_components(self, text: str) -> List[DetectionResult]:
        """Detect all components in the given text using regex patterns."""
        results = []
        # Apply Unicode normalization before processing
        normalized_text = self._normalize_text(text)
        text_lower = normalized_text.lower()
        
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
        # Apply Unicode normalization before processing
        normalized_text = self._normalize_text(text)
        text_lower = normalized_text.lower()
        
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
        # Apply Unicode normalization at entry point
        normalized_text = self._normalize_text(text)
        detected_components = self.detect_components(normalized_text)
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
        
        # Check for quantitative components (use normalized text)
        has_quantitative_baseline = self._has_quantitative_component(normalized_text, ComponentType.BASELINE)
        has_quantitative_target = self._has_quantitative_component(normalized_text, ComponentType.TARGET)
        
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
    
    def batch_score(self, indicators: List[str], use_parallel: bool = False) -> List[IndicatorScore]:
        """Score multiple indicators with optional parallel processing."""
        if use_parallel and len(indicators) > 1:
            # Set environment variables to prevent thread oversubscription when using joblib
            os.environ.setdefault('OMP_NUM_THREADS', '1')
            os.environ.setdefault('MKL_NUM_THREADS', '1') 
            os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            
            try:
                from joblib import Parallel, delayed
                return Parallel(n_jobs=-1)(
                    delayed(self.calculate_feasibility_score)(indicator) 
                    for indicator in indicators
                )
            except ImportError:
                # Fall back to sequential processing if joblib is not available
                pass
        
        return [self.calculate_feasibility_score(indicator) for indicator in indicators]
    
    def batch_score_with_monitoring(self, indicators: List[str]) -> BatchScoreResult:
        """Score multiple indicators with execution monitoring."""
        start_time = time.time()
        
        scores = []
        for indicator in indicators:
            scores.append(self.calculate_feasibility_score(indicator))
        
        end_time = time.time()
        duracion_segundos = end_time - start_time
        
        # Calculate processing rate (plans per minute)
        if duracion_segundos > 0:
            planes_por_minuto = (len(indicators) / duracion_segundos) * 60
        else:
            planes_por_minuto = 0.0
        
        # Format human-readable time
        if duracion_segundos < 1:
            execution_time = f"{duracion_segundos * 1000:.1f}ms"
        elif duracion_segundos < 60:
            execution_time = f"{duracion_segundos:.2f}s"
        else:
            minutes = int(duracion_segundos // 60)
            seconds = duracion_segundos % 60
            execution_time = f"{minutes}m {seconds:.1f}s"
        
        return BatchScoreResult(
            scores=scores,
            total_indicators=len(indicators),
            execution_time=execution_time,
            duracion_segundos=duracion_segundos,
            planes_por_minuto=planes_por_minuto
        )
    
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
    
    def calcular_calidad_evidencia(self, fragment: str) -> float:
        """
        Calculate evidence quality score for text fragments (0.0 to 1.0).
        
        Higher scores for fragments containing:
        - Numerical values with monetary amounts (COP, $, millones)
        - Dates (YYYY format, quarters Q1-Q4, months)
        - Measurement terminology (baseline, meta, periodicidad)
        
        Penalties applied for:
        - Indicators appearing only in titles/bullet points without values
        - Empty or malformed content
        
        Args:
            fragment (str): Text fragment to analyze
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if not fragment or not fragment.strip():
            return 0.0
        
        # Normalize Unicode text using NFKC
        normalized_text = unicodedata.normalize('NFKC', fragment.strip())
        text_lower = normalized_text.lower()
        
        # Initialize scoring components
        scores = {
            'monetary': 0.0,
            'dates': 0.0,
            'terminology': 0.0,
            'structure_penalty': 0.0
        }
        
        # Weights for different scoring components
        weights = {
            'monetary': 0.35,
            'dates': 0.25,
            'terminology': 0.25,
            'structure_penalty': -0.15
        }
        
        # 1. Monetary amount detection
        scores['monetary'] = self._detect_monetary_values(text_lower)
        
        # 2. Date detection  
        scores['dates'] = self._detect_temporal_indicators(text_lower)
        
        # 3. Measurement terminology detection
        scores['terminology'] = self._detect_measurement_terminology(text_lower)
        
        # 4. Structure penalty for title-only indicators
        scores['structure_penalty'] = self._calculate_structure_penalty(normalized_text)
        
        # Calculate weighted final score
        final_score = sum(scores[component] * weights[component] 
                         for component in scores.keys())
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, final_score))
    
    def _detect_monetary_values(self, text: str) -> float:
        """Detect monetary amounts and return normalized score."""
        monetary_patterns = [
            # Colombian pesos with COP
            r'cop\s*[\$]?\s*[\d,.\s]+(?:millones?|mil|thousands?|millions?)?',
            
            # Dollar amounts with various formats
            r'[\$]\s*[\d,.\s]+(?:millones?|mil|thousands?|millions?)?',
            r'[\d,.\s]+\s*(?:dollars?|dolares?|usd)',
            
            # Millions/thousands indicators in Spanish/English
            r'[\d,.\s]+\s*(?:millones?|millions?)\s*(?:de\s*)?(?:pesos?|cop|[\$])?',
            r'[\d,.\s]+\s*mil(?:es)?\s*(?:pesos?|cop|[\$])?',
            
            # Percentage with monetary context
            r'[\d,.\s]+\s*%\s*(?:del\s*)?(?:presupuesto|budget|recursos?)',
            
            # Investment/cost terminology
            r'(?:inversion|investment|costo|cost|gasto|expense).*?[\d,.\s]+',
            r'[\d,.\s]+.*?(?:inversion|investment|costo|cost)'
        ]
        
        matches = []
        for pattern in monetary_patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return 0.0
        
        # Score based on number and quality of monetary references
        base_score = min(len(matches) * 0.3, 1.0)
        
        # Bonus for high-precision monetary values
        precision_bonus = 0.0
        for match in matches:
            match_text = match.group()
            # Look for decimal places or specific amounts
            if re.search(r'[\d]+[.,][\d]{1,3}', match_text):
                precision_bonus += 0.1
            # Look for currency symbols
            if re.search(r'[\$]|cop|usd', match_text):
                precision_bonus += 0.1
        
        return min(base_score + precision_bonus, 1.0)
    
    def _detect_temporal_indicators(self, text: str) -> float:
        """Detect dates and temporal indicators."""
        temporal_patterns = [
            # Year patterns (YYYY)
            r'\b(?:20[0-9]{2}|19[0-9]{2})\b',
            
            # Quarter patterns (Q1-Q4)
            r'q[1-4](?:\s+20[0-9]{2})?',
            r'(?:trimestre|quarter)\s*[1-4]',
            r'(?:primer|segundo|tercer|cuarto)\s*trimestre',
            
            # Month patterns in Spanish
            r'(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+(?:de\s+)?20[0-9]{2})?',
            
            # Month patterns in English  
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+20[0-9]{2})?',
            
            # Date formats
            r'\b\d{1,2}[-/]\d{1,2}[-/](?:20[0-9]{2}|\d{2})\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            
            # Relative temporal references
            r'(?:periodicidad|periodicity|frequency).*?(?:anual|annual|mensual|monthly|trimestral|quarterly)',
            r'(?:cada|every)\s+(?:\d+\s+)?(?:años?|years?|meses?|months?|trimestres?|quarters?)',
            
            # Time horizons
            r'(?:para|by|hasta|until|en)\s+(?:el\s+)?(?:año\s+)?20[0-9]{2}',
            r'(?:horizon|horizonte).*?(?:20[0-9]{2}|\d+\s+años?)'
        ]
        
        matches = []
        for pattern in temporal_patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return 0.0
        
        # Score based on temporal precision
        score = 0.0
        for match in matches:
            match_text = match.group()
            # Higher score for specific dates
            if re.search(r'20[0-9]{2}', match_text):
                score += 0.4
            elif re.search(r'q[1-4]|trimestre|quarter', match_text):
                score += 0.3
            elif re.search(r'enero|febrero|january|february', match_text):
                score += 0.25
            else:
                score += 0.15
        
        return min(score, 1.0)
    
    def _detect_measurement_terminology(self, text: str) -> float:
        """Detect measurement and evaluation terminology."""
        terminology_patterns = [
            # Baseline terminology
            r'(?:baseline|línea\s+base|valor\s+inicial|situación\s+inicial)',
            r'(?:punto\s+de\s+partida|referencia\s+inicial|estado\s+actual)',
            
            # Target/goal terminology  
            r'(?:meta|objetivo|target|goal|propósito)',
            r'(?:alcanzar|lograr|achieve|reach)',
            
            # Measurement concepts
            r'(?:periodicidad|periodicity|frecuencia|frequency)',
            r'(?:indicador|indicator|métrica|metric|medición|measurement)',
            r'(?:monitoreo|monitoring|seguimiento|tracking)',
            
            # Performance terminology
            r'(?:desempeño|performance|resultado|result|impacto|impact)',
            r'(?:evaluación|evaluation|assessment|valoración)',
            
            # Quantitative terms
            r'(?:incremento|aumento|reducción|mejora|improvement)',
            r'(?:porcentaje|percentage|proporción|proportion|ratio)',
            
            # Comparative terms
            r'(?:comparado\s+con|compared\s+to|respecto\s+a|versus)',
            r'(?:mayor\s+que|menor\s+que|igual\s+a|greater\s+than|less\s+than)'
        ]
        
        matches = []
        for pattern in terminology_patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return 0.0
        
        # Score based on terminology richness
        unique_matches = set(match.group().lower() for match in matches)
        richness_score = min(len(unique_matches) * 0.2, 1.0)
        
        # Bonus for measurement-specific terminology
        measurement_bonus = 0.0
        measurement_terms = ['periodicidad', 'periodicity', 'indicador', 'indicator', 
                           'monitoreo', 'monitoring', 'evaluación', 'evaluation']
        
        for term in measurement_terms:
            if term in text:
                measurement_bonus += 0.15
        
        return min(richness_score + measurement_bonus, 1.0)
    
    def _calculate_structure_penalty(self, text: str) -> float:
        """Calculate penalty for indicators in titles/bullets without values."""
        # Check for title/bullet point patterns
        title_patterns = [
            r'^[-•*]\s+',  # Bullet points
            r'^#{1,6}\s+', # Markdown headers
            r'^[A-Z\s]+:$', # All caps titles with colon
            r'^[^\w]*(?:[A-Z][^.]*[^.]|[A-Z\s]+)$'  # Title-like structure
        ]
        
        is_title_like = any(re.match(pattern, text, re.MULTILINE) 
                           for pattern in title_patterns)
        
        if not is_title_like:
            return 0.0
        
        # Check if title has associated quantitative values
        value_patterns = [
            r'\d+(?:[.,]\d+)?(?:\s*%|\s*millones?|\s*mil)',
            r'[\$][\d,.\s]+',
            r'cop\s*[\d,.\s]+',
            r'\d{4}',  # Years
            r'q[1-4]'  # Quarters
        ]
        
        has_values = any(re.search(pattern, text, re.IGNORECASE) 
                        for pattern in value_patterns)
        
        # Apply penalty if title-like without values
        return 1.0 if not has_values else 0.0
    
    def generate_report(self, indicators: List[str], output_path: str) -> None:
        """
        Generate a comprehensive feasibility report and save it to file using atomic operations.
        
        Uses atomic file operations to prevent corrupted output files if the process is 
        interrupted during report generation. This is achieved by:
        1. Writing the complete report content to a temporary file in the same directory
        2. Using Path.rename() to atomically move the temporary file to the final destination
        
        Note: Atomicity may not be guaranteed on some remote filesystems (NFS, SMB) due to
        their implementation of rename operations. For local filesystems (ext4, NTFS, APFS),
        the rename operation is atomic.
        
        Args:
            indicators: List of indicator texts to analyze
            output_path: Path where the report should be saved
            
        Raises:
            IOError: If file operations fail
            ValueError: If indicators list is empty
        """
        if not indicators:
            raise ValueError("Indicators list cannot be empty")
        
        output_file = Path(output_path)
        
        # Create a unique temporary file in the same directory as the target
        temp_file = output_file.parent / f"{output_file.name}.tmp.{uuid.uuid4().hex[:8]}"
        
        try:
            # Generate the complete report content
            report_content = self._generate_report_content(indicators)
            
            # Write to temporary file first
            with temp_file.open('w', encoding='utf-8') as f:
                f.write(report_content)
                f.flush()  # Ensure all content is written to disk
            
            # Atomically move temporary file to final destination
            temp_file.rename(output_file)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass  # Ignore cleanup errors
            raise IOError(f"Failed to generate report: {e}") from e
    
    def _generate_report_content(self, indicators: List[str]) -> str:
        """Generate the complete report content for the given indicators."""
        results = self.batch_score(indicators)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build report content
        content_parts = []
        content_parts.append("# Feasibility Assessment Report")
        content_parts.append(f"Generated on: {timestamp}")
        content_parts.append(f"Total indicators analyzed: {len(indicators)}")
        content_parts.append("")
        
        # Summary statistics
        scores = [result.feasibility_score for result in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        tier_counts = {}
        for result in results:
            tier = result.quality_tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        content_parts.append("## Summary")
        content_parts.append(f"Average feasibility score: {avg_score:.3f}")
        content_parts.append("Quality tier distribution:")
        for tier, count in sorted(tier_counts.items()):
            percentage = (count / len(results)) * 100
            content_parts.append(f"  - {tier}: {count} ({percentage:.1f}%)")
        content_parts.append("")
        
        # Detailed results
        content_parts.append("## Detailed Analysis")
        content_parts.append("")
        
        # Sort results by score (highest first)
        sorted_results = list(zip(indicators, results))
        sorted_results.sort(key=lambda x: x[1].feasibility_score, reverse=True)
        
        for i, (indicator, result) in enumerate(sorted_results, 1):
            content_parts.append(f"### {i}. Indicator Analysis")
            content_parts.append(f"**Text:** {indicator}")
            content_parts.append(f"**Score:** {result.feasibility_score:.3f}")
            content_parts.append(f"**Quality Tier:** {result.quality_tier}")
            content_parts.append(f"**Quantitative Baseline:** {'Yes' if result.has_quantitative_baseline else 'No'}")
            content_parts.append(f"**Quantitative Target:** {'Yes' if result.has_quantitative_target else 'No'}")
            
            if result.components_detected:
                content_parts.append(f"**Components Detected:** {', '.join(c.value for c in result.components_detected)}")
            
            if result.detailed_matches:
                content_parts.append("**Pattern Matches:**")
                for match in result.detailed_matches:
                    content_parts.append(f"  - {match.component_type.value}: '{match.matched_text}' (confidence: {match.confidence:.2f})")
            
            content_parts.append("")
        
        # Recommendations
        content_parts.append("## Recommendations")
        
        low_quality_count = sum(1 for result in results if result.feasibility_score < 0.5)
        if low_quality_count > 0:
            content_parts.append(f"- {low_quality_count} indicators have scores below 0.5 and need improvement")
            content_parts.append("- Focus on adding quantitative baselines and targets")
            content_parts.append("- Include specific time horizons where missing")
        
        insufficient_count = sum(1 for result in results if result.quality_tier == 'insufficient')
        if insufficient_count > 0:
            content_parts.append(f"- {insufficient_count} indicators are missing core components (baseline or target)")
            content_parts.append("- These require fundamental restructuring to be measurable")
        
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("*Report generated by Feasibility Scorer v1.0*")
        
        return "\n".join(content_parts)
