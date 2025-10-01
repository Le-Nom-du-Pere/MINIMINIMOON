"""
Feasibility Scorer for Indicator Quality Assessment
===================================================

Advanced quality assessment system for policy indicators with comprehensive
detection capabilities and statistical validation.

This module implements a sophisticated weighted quality assessment system that evaluates
policy indicators based on the presence and quality of baseline values, targets/goals,
and time horizons. It uses advanced pattern recognition, Unicode normalization, and
parallel processing for robust indicator analysis in production environments.

The scorer provides comprehensive analysis including:
- Multilingual pattern detection (Spanish/English)
- Unicode normalization for consistent text processing
- Quantitative component detection and validation
- Parallel processing capabilities for large datasets
- Detailed confidence scoring and quality classification
- Comprehensive logging and performance monitoring

Classes:
    ComponentType: Enumeration of detectable component types
    DetectionResult: Individual component detection result
    IndicatorScore: Comprehensive indicator quality score
    BatchScoreResult: Batch processing results with performance metrics
    FeasibilityScorer: Main scoring engine with advanced capabilities

Functions:
    All scoring and detection methods are encapsulated within the FeasibilityScorer class

Example:
    >>> scorer = FeasibilityScorer(enable_parallel=True)
    >>> score = scorer.calculate_feasibility_score(
    ...     "Reducir la tasa de pobreza del 25% actual a 15% para el año 2025"
    ... )
    >>> print(f"Feasibility score: {score.feasibility_score:.2f}")
    >>> print(f"Quality tier: {score.quality_tier}")

Note:
    All text processing includes Unicode normalization to ensure consistent
    handling of accented characters and different text encodings. The system
    is designed for production use with comprehensive error handling.
"""

import argparse
import datetime
import gzip
import json
import logging
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class ComponentType(Enum):
    """
    Enumeration of detectable component types in policy indicators.

    Defines the key components that determine indicator quality and feasibility
    for effective policy evaluation and monitoring.

    Attributes:
        BASELINE: Baseline or current state values
        TARGET: Goals, targets, or desired outcomes
        TIME_HORIZON: Temporal frameworks and deadlines
        NUMERICAL: Quantitative values and metrics
        DATE: Specific dates and temporal references
    """

    BASELINE = "baseline"
    TARGET = "target"
    TIME_HORIZON = "time_horizon"
    NUMERICAL = "numerical"
    DATE = "date"


@dataclass
class DetectionResult:
    """
    Individual component detection result with position and confidence.

    Args:
        component_type (ComponentType): Type of component detected
        matched_text (str): Actual text that matched the pattern
        confidence (float): Confidence level of the match (0.0-1.0)
        position (int): Character position where match was found
    """

    component_type: ComponentType
    matched_text: str
    confidence: float
    position: int


@dataclass
class IndicatorScore:
    """
    Comprehensive indicator quality score with detailed analysis.

    Args:
        feasibility_score (float): Overall feasibility score (0.0-1.0)
        components_detected (List[ComponentType]): List of detected component types
        detailed_matches (List[DetectionResult]): Detailed pattern matches
        has_quantitative_baseline (bool): Whether quantitative baseline was detected
        has_quantitative_target (bool): Whether quantitative target was detected
        quality_tier (str): Quality classification ("high", "medium", "low", "poor", "insufficient")
    """

    feasibility_score: float
    components_detected: List[ComponentType]
    detailed_matches: List[DetectionResult]
    has_quantitative_baseline: bool
    has_quantitative_target: bool
    quality_tier: str


@dataclass
class BatchScoreResult:
    """
    Batch processing results with comprehensive performance metrics.

    Args:
        scores (List[IndicatorScore]): Individual indicator scores
        total_indicators (int): Total number of indicators processed
        execution_time (str): Human-readable execution time
        duracion_segundos (float): Execution duration in seconds
        planes_por_minuto (float): Processing rate (indicators per minute)
    """

    scores: List[IndicatorScore]
    total_indicators: int
    execution_time: str
    duracion_segundos: float
    planes_por_minuto: float


class FeasibilityScorer:
    """
    Advanced feasibility scorer for policy indicator quality assessment.

    Comprehensive assessment engine that evaluates policy indicators using advanced
    pattern recognition, multilingual support, and statistical validation methods.
    Includes parallel processing capabilities for large-scale analysis.

    The scorer uses a weighted scoring system based on the presence and quality of:
    - Baseline values (40% weight)
    - Targets/goals (40% weight)
    - Time horizons (20% weight)
    - Quantitative components (10% bonus each)

    Args:
        enable_parallel (bool, optional): Enable parallel processing. Defaults to True.
        n_jobs (int, optional): Number of parallel jobs. Defaults to None (auto-detect).
        backend (str, optional): Parallel backend ('loky', 'threading'). Defaults to 'loky'.

    Attributes:
        detection_patterns (Dict): Compiled regex patterns by component type
        weights (Dict): Component weighting factors for scoring
        quality_thresholds (Dict): Quality tier classification thresholds
        logger (logging.Logger): Performance and error logger

    Methods:
        calculate_feasibility_score: Score single indicator
        batch_score: Score multiple indicators with optional parallel processing
        detect_components: Detect all components in text
        batch_score_with_monitoring: Score with comprehensive performance monitoring
        get_detection_rules_documentation: Get complete documentation of detection rules

    Example:
        >>> scorer = FeasibilityScorer(enable_parallel=True, n_jobs=4)
        >>> result = scorer.calculate_feasibility_score(
        ...     "Aumentar cobertura de salud del 60% actual al 85% en 2025"
        ... )
        >>> print(f"Score: {result.feasibility_score:.2f}")
        >>> print(f"Tier: {result.quality_tier}")

    Note:
        All text processing includes Unicode NFKC normalization for consistent
        handling of composed/decomposed characters and encoding variations.
        The system supports both Spanish and English pattern detection.

    """

    def __init__(self, enable_parallel=True, n_jobs=None, backend="loky"):
        """Initialize the feasibility scorer.

        Args:
            enable_parallel: Enable parallel processing for batch operations.
            n_jobs: Number of parallel jobs (default: min(CPU count, 8)).
            backend: Parallel processing backend ('loky' or 'threading').
        """
        self.detection_patterns = self._initialize_patterns()
        self.weights = {
            ComponentType.BASELINE: 0.4,
            ComponentType.TARGET: 0.4,
            ComponentType.TIME_HORIZON: 0.2,
            ComponentType.NUMERICAL: 0.1,
            ComponentType.DATE: 0.1,
        }
        self.quality_thresholds = {"high": 0.8, "medium": 0.5, "low": 0.2}

        # Override with CLI arguments if available
        cli_workers = os.environ.get("CLI_WORKERS")
        if cli_workers:
            n_jobs = int(cli_workers)

        # Parallel processing configuration
        self.enable_parallel = enable_parallel and JOBLIB_AVAILABLE
        self.n_jobs = n_jobs if n_jobs is not None else min(
            os.cpu_count() or 1, 8)
        self.backend = backend

        # Performance logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """
        Setup comprehensive performance and error logging.

        Configures logging with appropriate formatters and handlers for
        production monitoring and debugging capabilities.

        """
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @staticmethod
    def _initialize_patterns() -> Dict[ComponentType, List[Dict]]:
        """
        Initialize comprehensive regex patterns for multilingual component detection.

        Creates and compiles regex patterns for detecting baseline values, targets,
        time horizons, numerical values, and dates in both Spanish and English text.

        Returns:
            Dict[ComponentType, List[Dict]]: Dictionary mapping component types to
                                           lists of pattern dictionaries with regex,
                                           confidence, and language information

        Note:
            Patterns are optimized for policy indicator analysis and include
            confidence levels based on specificity and reliability of detection.

        """
        return {
            ComponentType.BASELINE: [
                {
                    "pattern": r"(?:línea\s+base|baseline|valor\s+inicial|situación\s+inicial|estado\s+actual)",
                    "confidence": 0.9,
                    "language": "es/en",
                },
                {
                    "pattern": r"(?:punto\s+de\s+partida|referencia\s+inicial|nivel\s+base)",
                    "confidence": 0.8,
                    "language": "es",
                },
                {
                    "pattern": r"(?:current\s+level|initial\s+value|starting\s+point)",
                    "confidence": 0.8,
                    "language": "en",
                },
            ],
            ComponentType.TARGET: [
                {
                    "pattern": r"(?:meta|objetivo|target|goal)",
                    "confidence": 0.9,
                    "language": "es/en",
                },
                {
                    "pattern": r"(?:propósito|finalidad|alcanzar|lograr|hasta)",
                    "confidence": 0.7,
                    "language": "es",
                },
                {
                    "pattern": r"(?:achieve|reach|attain|aim|to\s)",
                    "confidence": 0.7,
                    "language": "en",
                },
            ],
            ComponentType.TIME_HORIZON: [
                {
                    "pattern": r"(?:horizonte\s+temporal|plazo|período|periodo|duración)",
                    "confidence": 0.9,
                    "language": "es",
                },
                {
                    "pattern": r"(?:time\s+horizon|timeline|timeframe|duration|period)",
                    "confidence": 0.9,
                    "language": "en",
                },
                {
                    "pattern": r"(?:para\s+el\s+año|hasta\s+el|en\s+los\s+próximos|within|by\s+\d{4})",
                    "confidence": 0.8,
                    "language": "es/en",
                },
            ],
            ComponentType.NUMERICAL: [
                {
                    "pattern": r"\d+(?:[.,]\d+)?(?:\s*%|\s*por\s*ciento|\s*percent)",
                    "confidence": 0.95,
                    "language": "universal",
                },
                {
                    "pattern": r"\d+(?:[.,]\d+)?\s*(?:millones?|millions?|mil|thousand)",
                    "confidence": 0.9,
                    "language": "es/en",
                },
                {
                    "pattern": r"(?:incrementar|aumentar|reducir|disminuir|increase|reduce)\s+(?:en\s+|by\s+)?\d+",
                    "confidence": 0.85,
                    "language": "es/en",
                },
            ],
            ComponentType.DATE: [
                {
                    "pattern": r"\b(?:20\d{2}|19\d{2})\b",
                    "confidence": 0.9,
                    "language": "universal",
                },
                {
                    "pattern": r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?20\d{2}",
                    "confidence": 0.95,
                    "language": "es",
                },
                {
                    "pattern": r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+20\d{2}",
                    "confidence": 0.95,
                    "language": "en",
                },
                {
                    "pattern": r"\d{1,2}[-/]\d{1,2}[-/](?:20\d{2}|\d{2})",
                    "confidence": 0.8,
                    "language": "universal",
                },
            ],
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text using Unicode NFKC normalization for consistent character representation.

        Args:
            text (str): Input text to normalize

        Returns:
            str: Normalized text with consistent Unicode representation

        Note:
            Uses NFKC normalization to handle composed/decomposed characters,
            compatibility characters, and encoding variations consistently.

        """
        return unicodedata.normalize("NFKC", text)

    def detect_components(self, text: str) -> List[DetectionResult]:
        """
        Detect all components in the given text using comprehensive regex patterns.

        Args:
            text (str): Text to analyze for indicator components

        Returns:
            List[DetectionResult]: List of detected components with confidence scores
                                  and position information

        Note:
            Applies Unicode normalization before pattern matching to ensure
            consistent detection across different text encodings and formats.

        """
        results = []
        # Apply Unicode normalization before processing
        normalized_text = FeasibilityScorer._normalize_text(text)
        text_lower = normalized_text.lower()

        for component_type, patterns in self.detection_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]

                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    result = DetectionResult(
                        component_type=component_type,
                        matched_text=match.group(),
                        confidence=confidence,
                        position=match.start(),
                    )
                    results.append(result)

        return results

    def _has_quantitative_component(
        self, text: str, component_type: ComponentType
    ) -> bool:
        """
        Check if a component has quantitative elements in nearby context.

        Analyzes text around component mentions to determine if quantitative
        values are associated with baseline or target components.

        Args:
            text (str): Text to analyze
            component_type (ComponentType): Component type to check for quantitative association

        Returns:
            bool: True if quantitative elements are found within 30 characters of component mentions

        Note:
            Uses a 30-character context window around component mentions to identify
            associated numerical values, percentages, or quantitative indicators.

        """
        # Apply Unicode normalization before processing
        normalized_text = FeasibilityScorer._normalize_text(text)
        text_lower = normalized_text.lower()

        # Find component mentions
        component_positions = []
        for pattern_info in self.detection_patterns[component_type]:
            pattern = pattern_info["pattern"]
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
                if re.search(pattern_info["pattern"], context, re.IGNORECASE):
                    return True

        return False

    def calculate_feasibility_score(
        self, text: str, evidencia_soporte: Optional[int] = None
    ) -> IndicatorScore:
        """
        Calculate comprehensive feasibility score based on detected components and quality.

        Implements sophisticated scoring algorithm with mandatory requirements and
        bonus components for comprehensive indicator quality assessment.

        Args:
            text (str): Indicator text to evaluate
            evidencia_soporte: Optional evidence support override (0 forces failure)

        Returns:
            IndicatorScore: Comprehensive score with detailed analysis

        Scoring Requirements:
            - Must have both baseline and target components for positive score
            - Base score: 0.8 (0.4 baseline + 0.4 target)
            - Quantitative bonuses: +0.2 each for quantitative baseline and target
            - Component bonuses: +0.2 time horizon, +0.1 numerical, +0.1 dates
            - Confidence weighting: Final score multiplied by average pattern confidence

        Quality Tiers:
            - "high": ≥0.8 score
            - "medium": ≥0.5 score
            - "low": ≥0.2 score
            - "poor": <0.2 score
            - "insufficient": Missing baseline or target

        Note:
            All text processing includes Unicode normalization for consistent
            handling of different character encodings and compositions.

        """
        # Check for zero evidence support condition first
        if evidencia_soporte is not None and evidencia_soporte == 0:
            self.logger.warning(
                "Zero evidence support detected - overriding normal scoring logic"
            )
            self.logger.info(
                "Risk level set to HIGH due to lack of supporting evidence"
            )
            self.logger.info(
                "Final recommendation overridden to: REQUIERE MAYOR EVIDENCIA"
            )

            return IndicatorScore(
                feasibility_score=0.0,  # High risk = low feasibility score
                components_detected=[],
                detailed_matches=[],
                has_quantitative_baseline=False,
                has_quantitative_target=False,
                quality_tier="REQUIERE MAYOR EVIDENCIA",  # Override recommendation
            )

        # Apply Unicode normalization at entry point
        normalized_text = FeasibilityScorer._normalize_text(text)
        detected_components = self.detect_components(normalized_text)
        component_types = set(
            result.component_type for result in detected_components)

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
                quality_tier="insufficient",
            )

        # Calculate base score from mandatory components
        base_score = (
            self.weights[ComponentType.BASELINE] +
            self.weights[ComponentType.TARGET]
        )

        # Check for quantitative components (use normalized text)
        has_quantitative_baseline = self._has_quantitative_component(
            normalized_text, ComponentType.BASELINE
        )
        has_quantitative_target = self._has_quantitative_component(
            normalized_text, ComponentType.TARGET
        )

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
        avg_confidence = sum(result.confidence for result in detected_components) / len(
            detected_components
        )
        final_score = min(1.0, base_score * avg_confidence)

        # Determine quality tier - check again for zero evidence support override
        if evidencia_soporte is not None and evidencia_soporte == 0:
            quality_tier = "REQUIERE MAYOR EVIDENCIA"
        elif final_score >= self.quality_thresholds["high"]:
            quality_tier = "high"
        elif final_score >= self.quality_thresholds["medium"]:
            quality_tier = "medium"
        elif final_score >= self.quality_thresholds["low"]:
            quality_tier = "low"
        else:
            quality_tier = "poor"

        return IndicatorScore(
            feasibility_score=final_score,
            components_detected=list(component_types),
            detailed_matches=detected_components,
            has_quantitative_baseline=has_quantitative_baseline,
            has_quantitative_target=has_quantitative_target,
            quality_tier=quality_tier,
        )

    def _score_single_indicator(
        self, indicator: str, evidencia_soporte: Optional[int] = None
    ) -> IndicatorScore:
        """Score a single indicator - helper function for parallel processing.

        Args:
            indicator: Indicator text to score.

        Returns:
            IndicatorScore for the given indicator.
        """
        return self.calculate_feasibility_score(indicator, evidencia_soporte)

    def batch_score(
        self,
        indicators: List[str],
        compare_backends=False,
        use_parallel: bool = False,
        evidencia_soporte_list: Optional[List[Optional[int]]] = None,
    ) -> List[IndicatorScore]:
        """Score multiple indicators with optional parallel processing.

        Processes a batch of indicators with automatic selection of sequential
        or parallel processing based on batch size and system capabilities.

        Args:
            indicators: List of indicator strings to score.
            compare_backends: If True, compare performance between threading and loky backends.
            use_parallel: Legacy parameter for backward compatibility.
            evidencia_soporte_list: Optional list with per-indicator evidence overrides.

        Returns:
            List of IndicatorScore results in the same order as input indicators.
        """
        if not indicators:
            return []

        # Validate evidencia_soporte_list length if provided
        if evidencia_soporte_list is not None and len(evidencia_soporte_list) != len(
            indicators
        ):
            raise ValueError(
                "evidencia_soporte_list must have the same length as indicators"
            )

        # For small batches, use sequential processing to avoid overhead
        if len(indicators) < 10 or not self.enable_parallel or not JOBLIB_AVAILABLE:
            return self._batch_score_sequential(indicators, evidencia_soporte_list)

        if compare_backends:
            return self._batch_score_with_comparison(indicators, evidencia_soporte_list)
        else:
            return self._batch_score_parallel(
                indicators, self.backend, evidencia_soporte_list
            )

    def _batch_score_sequential(
        self,
        indicators: List[str],
        evidencia_soporte_list: Optional[List[Optional[int]]] = None,
    ) -> List[IndicatorScore]:
        """Sequential batch scoring."""
        start_time = time.time()
        results = []
        for i, indicator in enumerate(indicators):
            evidencia_soporte = (
                evidencia_soporte_list[i] if evidencia_soporte_list else None
            )
            results.append(
                self.calculate_feasibility_score(indicator, evidencia_soporte)
            )
        elapsed = time.time() - start_time

        self.logger.info(
            f"Sequential processing: {len(indicators)} indicators in {elapsed:.3f}s "
            f"({elapsed / len(indicators) * 1000:.1f}ms per indicator)"
        )
        return results

    def _batch_score_parallel(
        self,
        indicators: List[str],
        backend: str,
        evidencia_soporte_list: Optional[List[Optional[int]]] = None,
    ) -> List[IndicatorScore]:
        """Parallel batch scoring using specified backend."""
        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib not available for parallel processing")

        start_time = time.time()

        # Create a picklable instance for parallel processing
        scorer_copy = self._create_picklable_copy()

        with Parallel(n_jobs=self.n_jobs, backend=backend) as parallel:
            results = parallel(
                delayed(scorer_copy._score_single_indicator)(
                    indicator,
                    evidencia_soporte_list[i] if evidencia_soporte_list else None,
                )
                for i, indicator in enumerate(indicators)
            )

        elapsed = time.time() - start_time
        self.logger.info(
            f"Parallel processing ({backend}): {len(indicators)} indicators in {elapsed:.3f}s "
            f"({elapsed / len(indicators) * 1000:.1f}ms per indicator, {self.n_jobs} workers)"
        )
        return results

    def _batch_score_with_comparison(
        self,
        indicators: List[str],
        evidencia_soporte_list: Optional[List[Optional[int]]] = None,
    ) -> List[IndicatorScore]:
        """Score batch with performance comparison between backends."""
        if not JOBLIB_AVAILABLE:
            self.logger.warning(
                "joblib not available, falling back to sequential processing"
            )
            return self._batch_score_sequential(indicators, evidencia_soporte_list)

        self.logger.info("Comparing parallel processing backends...")

        # Test with threading backend
        try:
            threading_results = self._batch_score_parallel(
                indicators, "threading", evidencia_soporte_list
            )
        except Exception as e:
            self.logger.warning(f"Threading backend failed: {e}")
            threading_results = None

        # Test with loky backend
        try:
            loky_results = self._batch_score_parallel(
                indicators, "loky", evidencia_soporte_list
            )
        except Exception as e:
            self.logger.warning(f"Loky backend failed: {e}")
            loky_results = None

        # Fallback to sequential if both failed
        if loky_results is None and threading_results is None:
            self.logger.warning(
                "Both parallel backends failed, falling back to sequential"
            )
            return self._batch_score_sequential(indicators, evidencia_soporte_list)

        # Return loky results if available, otherwise threading results
        return loky_results if loky_results is not None else threading_results

    def _create_picklable_copy(self):
        """Create a copy of the scorer that can be safely pickled for multiprocessing."""
        # Create new instance with same configuration but fresh logger
        new_scorer = FeasibilityScorer(
            enable_parallel=False,  # Disable parallel in workers to avoid recursion
            n_jobs=self.n_jobs,
            backend=self.backend,
        )

        # Copy over the patterns and weights (these are picklable)
        new_scorer.detection_patterns = self.detection_patterns
        new_scorer.weights = self.weights
        new_scorer.quality_thresholds = self.quality_thresholds

        return new_scorer

    def batch_score_with_monitoring(
        self,
        indicators: List[str],
        evidencia_soporte_list: Optional[List[Optional[int]]] = None,
    ) -> BatchScoreResult:
        """Score multiple indicators with execution monitoring."""
        start_time = time.time()

        scores = []
        for i, indicator in enumerate(indicators):
            evidencia_soporte = (
                evidencia_soporte_list[i] if evidencia_soporte_list else None
            )
            scores.append(
                self.calculate_feasibility_score(indicator, evidencia_soporte)
            )

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
            planes_por_minuto=planes_por_minuto,
        )

    @staticmethod
    def get_detection_rules_documentation() -> str:
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
        normalized_text = unicodedata.normalize("NFKC", fragment.strip())
        text_lower = normalized_text.lower()

        # Initialize scoring components
        scores = {
            "monetary": 0.0,
            "dates": 0.0,
            "terminology": 0.0,
            "structure_penalty": 0.0,
        }

        # Weights for different scoring components
        weights = {
            "monetary": 0.35,
            "dates": 0.25,
            "terminology": 0.25,
            "structure_penalty": -0.15,
        }

        # 1. Monetary amount detection
        scores["monetary"] = self._detect_monetary_values(text_lower)

        # 2. Date detection
        scores["dates"] = self._detect_temporal_indicators(text_lower)

        # 3. Measurement terminology detection
        scores["terminology"] = self._detect_measurement_terminology(
            text_lower)

        # 4. Structure penalty for title-only indicators
        scores["structure_penalty"] = self._calculate_structure_penalty(
            normalized_text)

        # Calculate weighted final score
        final_score = sum(
            scores[component] * weights[component] for component in scores.keys()
        )

        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, final_score))

    @staticmethod
    def _detect_monetary_values(text: str) -> float:
        """Detect monetary amounts and return normalized score."""
        monetary_patterns = [
            # Colombian pesos with COP
            r"cop\s*[\$]?\s*[\d,.\s]+(?:millones?|mil|thousands?|millions?)?",
            # Dollar amounts with various formats
            r"[\$]\s*[\d,.\s]+(?:millones?|mil|thousands?|millions?)?",
            r"[\d,.\s]+\s*(?:dollars?|dolares?|usd)",
            # Millions/thousands indicators in Spanish/English
            r"[\d,.\s]+\s*(?:millones?|millions?)\s*(?:de\s*)?(?:pesos?|cop|[\$])?",
            r"[\d,.\s]+\s*mil(?:es)?\s*(?:pesos?|cop|[\$])?",
            # Percentage with monetary context
            r"[\d,.\s]+\s*%\s*(?:del\s*)?(?:presupuesto|budget|recursos?)",
            # Investment/cost terminology
            r"(?:inversion|investment|costo|cost|gasto|expense).*?[\d,.\s]+",
            r"[\d,.\s]+.*?(?:inversion|investment|costo|cost)",
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
            if re.search(r"[\d]+[.,][\d]{1,3}", match_text):
                precision_bonus += 0.1
            # Look for currency symbols
            if re.search(r"[\$]|cop|usd", match_text):
                precision_bonus += 0.1

        return min(base_score + precision_bonus, 1.0)

    @staticmethod
    def _detect_temporal_indicators(text: str) -> float:
        """Detect dates and temporal indicators."""
        temporal_patterns = [
            # Year patterns (YYYY)
            r"\b(?:20[0-9]{2}|19[0-9]{2})\b",
            # Quarter patterns (Q1-Q4)
            r"q[1-4](?:\s+20[0-9]{2})?",
            r"(?:trimestre|quarter)\s*[1-4]",
            r"(?:primer|segundo|tercer|cuarto)\s*trimestre",
            # Month patterns in Spanish
            r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+(?:de\s+)?20[0-9]{2})?",
            # Month patterns in English
            r"(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+20[0-9]{2})?",
            # Date formats
            r"\b\d{1,2}[-/]\d{1,2}[-/](?:20[0-9]{2}|\d{2})\b",
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
            # Relative temporal references
            r"(?:periodicidad|periodicity|frequency).*?(?:anual|annual|mensual|monthly|trimestral|quarterly)",
            r"(?:cada|every)\s+(?:\d+\s+)?(?:años?|years?|meses?|months?|trimestres?|quarters?)",
            # Time horizons
            r"(?:para|by|hasta|until|en)\s+(?:el\s+)?(?:año\s+)?20[0-9]{2}",
            r"(?:horizon|horizonte).*?(?:20[0-9]{2}|\d+\s+años?)",
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
            if re.search(r"20[0-9]{2}", match_text):
                score += 0.4
            elif re.search(r"q[1-4]|trimestre|quarter", match_text):
                score += 0.3
            elif re.search(r"enero|febrero|january|february", match_text):
                score += 0.25
            else:
                score += 0.15

        return min(score, 1.0)

    @staticmethod
    def _detect_measurement_terminology(text: str) -> float:
        """Detect measurement and evaluation terminology."""
        terminology_patterns = [
            # Baseline terminology
            r"(?:baseline|línea\s+base|valor\s+inicial|situación\s+inicial)",
            r"(?:punto\s+de\s+partida|referencia\s+inicial|estado\s+actual)",
            # Target/goal terminology
            r"(?:meta|objetivo|target|goal|propósito)",
            r"(?:alcanzar|lograr|achieve|reach)",
            # Measurement concepts
            r"(?:periodicidad|periodicity|frecuencia|frequency)",
            r"(?:indicador|indicator|métrica|metric|medición|measurement)",
            r"(?:monitoreo|monitoring|seguimiento|tracking)",
            # Performance terminology
            r"(?:desempeño|performance|resultado|result|impacto|impact)",
            r"(?:evaluación|evaluation|assessment|valoración)",
            # Quantitative terms
            r"(?:incremento|aumento|reducción|mejora|improvement)",
            r"(?:porcentaje|percentage|proporción|proportion|ratio)",
            # Comparative terms
            r"(?:comparado\s+con|compared\s+to|respecto\s+a|versus)",
            r"(?:mayor\s+que|menor\s+que|igual\s+a|greater\s+than|less\s+than)",
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
        measurement_terms = [
            "periodicidad",
            "periodicity",
            "indicador",
            "indicator",
            "monitoreo",
            "monitoring",
            "evaluación",
            "evaluation",
        ]

        for term in measurement_terms:
            if term in text:
                measurement_bonus += 0.15

        return min(richness_score + measurement_bonus, 1.0)

    @staticmethod
    def _calculate_structure_penalty(text: str) -> float:
        """Calculate penalty for indicators in titles/bullets without values."""
        # Check for title/bullet point patterns
        title_patterns = [
            r"^[-•*]\s+",  # Bullet points
            r"^#{1,6}\s+",  # Markdown headers
            r"^[A-Z\s]+:$",  # All caps titles with colon
            r"^[^\w]*(?:[A-Z][^.]*[^.]|[A-Z\s]+)$",  # Title-like structure
        ]

        is_title_like = any(
            re.match(pattern, text, re.MULTILINE) for pattern in title_patterns
        )

        if not is_title_like:
            return 0.0

        # Check if title has associated quantitative values
        value_patterns = [
            r"\d+(?:[.,]\d+)?(?:\s*%|\s*millones?|\s*mil)",
            r"[\$][\d,.\s]+",
            r"cop\s*[\d,.\s]+",
            r"\d{4}",  # Years
            r"q[1-4]",  # Quarters
        ]

        has_values = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in value_patterns
        )

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
        temp_file = (
            output_file.parent /
            f"{output_file.name}.tmp.{uuid.uuid4().hex[:8]}"
        )

        try:
            # Generate the complete report content
            report_content = self._generate_report_content(indicators)

            # Write to temporary file first
            with temp_file.open("w", encoding="utf-8") as f:
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
            content_parts.append(
                f"**Quantitative Baseline:** {'Yes' if result.has_quantitative_baseline else 'No'}"
            )
            content_parts.append(
                f"**Quantitative Target:** {'Yes' if result.has_quantitative_target else 'No'}"
            )

            if result.components_detected:
                content_parts.append(
                    f"**Components Detected:** {', '.join(c.value for c in result.components_detected)}"
                )

            if result.detailed_matches:
                content_parts.append("**Pattern Matches:**")
                for match in result.detailed_matches:
                    content_parts.append(
                        f"  - {match.component_type.value}: '{match.matched_text}' (confidence: {match.confidence:.2f})"
                    )

            content_parts.append("")

        # Recommendations
        content_parts.append("## Recommendations")

        low_quality_count = sum(
            1 for result in results if result.feasibility_score < 0.5
        )
        if low_quality_count > 0:
            content_parts.append(
                f"- {low_quality_count} indicators have scores below 0.5 and need improvement"
            )
            content_parts.append(
                "- Focus on adding quantitative baselines and targets")
            content_parts.append(
                "- Include specific time horizons where missing")

        insufficient_count = sum(
            1 for result in results if result.quality_tier == "insufficient"
        )
        if insufficient_count > 0:
            content_parts.append(
                f"- {insufficient_count} indicators are missing core components (baseline or target)"
            )
            content_parts.append(
                "- These require fundamental restructuring to be measurable"
            )

        content_parts.append("")
        content_parts.append("---")
        content_parts.append("*Report generated by Feasibility Scorer v1.0*")

        return "\n".join(content_parts)

    def generate_traceability_matrix_csv(
        self, results: Dict[str, IndicatorScore], output_dir: str = "."
    ) -> str:
        """
        Generate consolidated traceability matrix CSV with all evaluation dimensions.

        Args:
            results: Dictionary mapping plan filenames to IndicatorScore objects
            output_dir: Directory to save the CSV file

        Returns:
            Path to the generated CSV file

        Raises:
            ImportError: If pandas is not available
        """
        if not PANDAS_AVAILABLE:
            # Fallback: generate CSV manually without pandas
            return self._generate_csv_fallback(results, output_dir)

        # Spanish column headers for the traceability matrix
        columns = {
            "archivo_plan": "Archivo del Plan",
            "puntuacion_factibilidad": "Puntuación de Factibilidad",
            "nivel_calidad": "Nivel de Calidad",
            "linea_base_cuantitativa": "Línea Base Cuantitativa",
            "meta_cuantitativa": "Meta Cuantitativa",
            "componentes_detectados": "Componentes Detectados",
            "tiene_linea_base": "Tiene Línea Base",
            "tiene_meta": "Tiene Meta",
            "tiene_horizonte_temporal": "Tiene Horizonte Temporal",
            "tiene_valores_numericos": "Tiene Valores Numéricos",
            "tiene_fechas": "Tiene Fechas",
            "coincidencias_detalladas": "Coincidencias Detalladas",
            "recomendacion_general": "Recomendación General",
        }

        # Build rows for the DataFrame
        rows = []
        for plan_filename, score in results.items():
            # Generate overall recommendation based on quality tier
            recommendation = FeasibilityScorer._get_recommendation_spanish(
                score.quality_tier
            )

            # Count component types
            components = score.components_detected
            has_baseline = ComponentType.BASELINE in components
            has_target = ComponentType.TARGET in components
            has_time_horizon = ComponentType.TIME_HORIZON in components
            has_numerical = ComponentType.NUMERICAL in components
            has_dates = ComponentType.DATE in components

            # Format detected components in Spanish
            component_names_spanish = {
                ComponentType.BASELINE: "línea base",
                ComponentType.TARGET: "meta",
                ComponentType.TIME_HORIZON: "horizonte temporal",
                ComponentType.NUMERICAL: "valores numéricos",
                ComponentType.DATE: "fechas",
            }

            components_list = [
                component_names_spanish.get(comp, comp.value) for comp in components
            ]
            components_str = (
                ", ".join(components_list) if components_list else "ninguno"
            )

            # Format detailed matches
            matches_details = []
            for match in score.detailed_matches:
                match_info = f"{component_names_spanish.get(match.component_type, match.component_type.value)}: '{match.matched_text}' (confianza: {match.confidence:.2f})"
                matches_details.append(match_info)
            matches_str = (
                "; ".join(matches_details)
                if matches_details
                else "ninguna coincidencia"
            )

            row = {
                "archivo_plan": plan_filename,
                "puntuacion_factibilidad": round(score.feasibility_score, 3),
                "nivel_calidad": FeasibilityScorer._translate_quality_tier_spanish(
                    score.quality_tier
                ),
                "linea_base_cuantitativa": (
                    "Sí" if score.has_quantitative_baseline else "No"
                ),
                "meta_cuantitativa": "Sí" if score.has_quantitative_target else "No",
                "componentes_detectados": components_str,
                "tiene_linea_base": "Sí" if has_baseline else "No",
                "tiene_meta": "Sí" if has_target else "No",
                "tiene_horizonte_temporal": "Sí" if has_time_horizon else "No",
                "tiene_valores_numericos": "Sí" if has_numerical else "No",
                "tiene_fechas": "Sí" if has_dates else "No",
                "coincidencias_detalladas": matches_str,
                "recomendacion_general": recommendation,
            }
            rows.append(row)

        # Create DataFrame with Spanish column names
        df = pd.DataFrame(rows)
        df = df.rename(columns=columns)

        # Sort by feasibility score (descending)
        df = df.sort_values("Puntuación de Factibilidad", ascending=False)

        # Generate output filename
        output_path = Path(output_dir) / "matriz_trazabilidad_factibilidad.csv"

        # Check if we need gzip compression
        csv_content = df.to_csv(
            index=False, encoding="utf-8-sig"
        )  # utf-8-sig for Excel compatibility
        file_size_mb = len(csv_content.encode("utf-8")) / (1024 * 1024)

        if file_size_mb > 5.0:  # Compress if larger than 5MB
            output_path = output_path.with_suffix(".csv.gz")
            with gzip.open(output_path, "wt", encoding="utf-8-sig") as f:
                f.write(csv_content)
            print(
                f"CSV exportado con compresión gzip: {output_path} (tamaño original: {file_size_mb:.1f}MB)"
            )
        else:
            with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
                f.write(csv_content)
            print(
                f"CSV exportado: {output_path} (tamaño: {file_size_mb:.1f}MB)")

        return str(output_path)

    @staticmethod
    def translate_quality_tier_spanish(tier: str) -> str:
        """Public wrapper to translate quality tier to Spanish."""
        return FeasibilityScorer._translate_quality_tier_spanish(tier)

    @staticmethod
    def get_recommendation_spanish(quality_tier: str) -> str:
        """Public wrapper to obtain recommendation in Spanish for a quality tier."""
        return FeasibilityScorer._get_recommendation_spanish(quality_tier)

    @staticmethod
    def _translate_quality_tier_spanish(tier: str) -> str:
        """Translate quality tier to Spanish."""
        translations = {
            "high": "Alto",
            "medium": "Medio",
            "low": "Bajo",
            "poor": "Deficiente",
            "insufficient": "Insuficiente",
        }
        return translations.get(tier, tier)

    @staticmethod
    def _get_recommendation_spanish(quality_tier: str) -> str:
        """Generate recommendation in Spanish based on quality tier."""
        recommendations = {
            "high": "Indicador de alta calidad. Mantener el nivel de especificidad.",
            "medium": "Indicador aceptable. Considerar agregar elementos cuantitativos adicionales.",
            "low": "Indicador básico. Requiere mejoras en elementos cuantitativos y horizonte temporal.",
            "poor": "Indicador deficiente. Necesita revisión integral para incluir línea base y metas claras.",
            "insufficient": "Indicador insuficiente. Debe incluir tanto línea base como metas para ser viable.",
        }
        return recommendations.get(quality_tier, "Evaluación pendiente.")

    @staticmethod
    def _generate_csv_fallback(
        results: Dict[str, IndicatorScore], output_dir: str = "."
    ) -> str:
        """
        Fallback CSV generation without pandas dependency.

        Args:
            results: Dictionary mapping plan filenames to IndicatorScore objects
            output_dir: Directory to save the CSV file

        Returns:
            Path to the generated CSV file
        """
        import csv

        # CSV headers in Spanish
        headers = [
            "Archivo del Plan",
            "Puntuación de Factibilidad",
            "Nivel de Calidad",
            "Línea Base Cuantitativa",
            "Meta Cuantitativa",
            "Componentes Detectados",
            "Tiene Línea Base",
            "Tiene Meta",
            "Tiene Horizonte Temporal",
            "Tiene Valores Numéricos",
            "Tiene Fechas",
            "Coincidencias Detalladas",
            "Recomendación General",
        ]

        # Generate rows
        rows = []
        for plan_filename, score in results.items():
            # Generate overall recommendation based on quality tier
            recommendation = FeasibilityScorer._get_recommendation_spanish(
                score.quality_tier
            )

            # Count component types
            components = score.components_detected
            has_baseline = ComponentType.BASELINE in components
            has_target = ComponentType.TARGET in components
            has_time_horizon = ComponentType.TIME_HORIZON in components
            has_numerical = ComponentType.NUMERICAL in components
            has_dates = ComponentType.DATE in components

            # Format detected components in Spanish
            component_names_spanish = {
                ComponentType.BASELINE: "línea base",
                ComponentType.TARGET: "meta",
                ComponentType.TIME_HORIZON: "horizonte temporal",
                ComponentType.NUMERICAL: "valores numéricos",
                ComponentType.DATE: "fechas",
            }

            components_list = [
                component_names_spanish.get(comp, comp.value) for comp in components
            ]
            components_str = (
                ", ".join(components_list) if components_list else "ninguno"
            )

            # Format detailed matches
            matches_details = []
            for match in score.detailed_matches:
                match_info = f"{component_names_spanish.get(match.component_type, match.component_type.value)}: '{match.matched_text}' (confianza: {match.confidence:.2f})"
                matches_details.append(match_info)
            matches_str = (
                "; ".join(matches_details)
                if matches_details
                else "ninguna coincidencia"
            )

            row = [
                plan_filename,
                f"{score.feasibility_score:.3f}",
                FeasibilityScorer._translate_quality_tier_spanish(
                    score.quality_tier),
                "Sí" if score.has_quantitative_baseline else "No",
                "Sí" if score.has_quantitative_target else "No",
                components_str,
                "Sí" if has_baseline else "No",
                "Sí" if has_target else "No",
                "Sí" if has_time_horizon else "No",
                "Sí" if has_numerical else "No",
                "Sí" if has_dates else "No",
                matches_str,
                recommendation,
            ]
            # Include score for sorting
            rows.append((score.feasibility_score, row))

        # Sort by feasibility score (descending)
        rows.sort(key=lambda x: x[0], reverse=True)
        sorted_rows = [row[1] for row in rows]

        # Generate output filename
        output_path = Path(output_dir) / "matriz_trazabilidad_factibilidad.csv"

        # Write CSV and check size for compression
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(sorted_rows)

        # Check file size for compression
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        if file_size_mb > 5.0:  # Compress if larger than 5MB
            compressed_path = output_path.with_suffix(".csv.gz")
            with open(output_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    f_out.write(f_in.read())

            # Remove uncompressed file
            output_path.unlink()

            print(
                f"CSV exportado con compresión gzip: {compressed_path} (tamaño original: {file_size_mb:.1f}MB)"
            )
            return str(compressed_path)
        else:
            print(
                f"CSV exportado: {output_path} (tamaño: {file_size_mb:.1f}MB)")
            return str(output_path)


def main():
    """Command-line interface for the Feasibility Scorer."""
    parser = argparse.ArgumentParser(
        description="Feasibility Scorer - Evaluate indicator quality based on baseline, targets, and time horizons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single indicator
  python feasibility_scorer.py --text "Incrementar la línea base de 65% a una meta de 85% para 2025"
  
  # Export CSV traceability matrix 
  python feasibility_scorer.py --export-csv --output-dir ./reports/
  
  # Process batch from file
  python feasibility_scorer.py --batch-file indicators.txt --export-csv
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", type=str, help="Single indicator text to evaluate"
    )
    input_group.add_argument(
        "--batch-file",
        type=str,
        help="Path to file containing multiple indicators (one per line)",
    )
    input_group.add_argument(
        "--demo", action="store_true", help="Run demonstration with built-in examples"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for reports (default: current directory)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Generate consolidated traceability matrix CSV file",
    )
    parser.add_argument(
        "--export-json", action="store_true", help="Export detailed results as JSON"
    )
    parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Export results as Markdown report",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed component matches"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scorer = FeasibilityScorer()
    results = {}

    if args.demo:
        # Run built-in demonstration
        demo_indicators = {
            "ejemplo_alta_calidad.txt": "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025",
            "ejemplo_calidad_media.txt": "Mejorar desde la situación inicial hasta el objetivo propuesto con incremento del 20%",
            "ejemplo_calidad_baja.txt": "Partir de la línea base para alcanzar el objetivo",
            "ejemplo_insuficiente.txt": "Aumentar el acceso a servicios de salud en la región",
        }

        print("DEMOSTRACIÓN - Feasibility Scorer")
        print("=" * 50)

        for filename, text in demo_indicators.items():
            score = scorer.calculate_feasibility_score(text)
            results[filename] = score

            print(f"\nArchivo: {filename}")
            print(f"Texto: {text}")
            print(f"Puntuación: {score.feasibility_score:.3f}")
            print(
                f"Nivel: {scorer.translate_quality_tier_spanish(score.quality_tier)}")
            print(
                f"Recomendación: {scorer.get_recommendation_spanish(score.quality_tier)}"
            )

            if args.verbose:
                print(
                    f"Componentes: {[c.value for c in score.components_detected]}")
                if score.detailed_matches:
                    print("Coincidencias detalladas:")
                    for match in score.detailed_matches:
                        print(
                            f"  - {match.component_type.value}: '{match.matched_text}' (confianza: {match.confidence:.2f})"
                        )

    elif args.text:
        # Single text evaluation
        filename = "input_text.txt"
        score = scorer.calculate_feasibility_score(args.text)
        results[filename] = score

        print(f"Texto evaluado: {args.text}")
        print(f"Puntuación de factibilidad: {score.feasibility_score:.3f}")
        print(
            f"Nivel de calidad: {scorer.translate_quality_tier_spanish(score.quality_tier)}"
        )
        print(
            f"Línea base cuantitativa: {'Sí' if score.has_quantitative_baseline else 'No'}"
        )
        print(
            f"Meta cuantitativa: {'Sí' if score.has_quantitative_target else 'No'}")
        print(
            f"Recomendación: {scorer.get_recommendation_spanish(score.quality_tier)}")

        if args.verbose and score.detailed_matches:
            print("\nCoincidencias detalladas:")
            for match in score.detailed_matches:
                print(
                    f"  - {match.component_type.value}: '{match.matched_text}' (confianza: {match.confidence:.2f})"
                )

    elif args.batch_file:
        # Batch processing from file
        batch_path = Path(args.batch_file)
        if not batch_path.exists():
            print(f"Error: El archivo {args.batch_file} no existe.")
            return 1

        with open(batch_path, "r", encoding="utf-8") as f:
            indicators = [line.strip() for line in f if line.strip()]

        print(
            f"Procesando {len(indicators)} indicadores desde {args.batch_file}...")

        for i, indicator in enumerate(indicators, 1):
            filename = f"indicador_{i:03d}.txt"
            score = scorer.calculate_feasibility_score(indicator)
            results[filename] = score

            if not args.export_csv:  # Only show individual results if not exporting CSV
                print(f"\n{i}. {filename}")
                print(f"   Puntuación: {score.feasibility_score:.3f}")
                print(
                    f"   Nivel: {scorer.translate_quality_tier_spanish(score.quality_tier)}"
                )

    # Export results in requested formats
    if args.export_csv:
        try:
            csv_path = scorer.generate_traceability_matrix_csv(
                results, str(output_dir))
            print(f"✓ Matriz de trazabilidad CSV generada: {csv_path}")
        except ImportError as e:
            print(f"Error: {e}")
            print("Instalar pandas con: pip install pandas")
            return 1

    if args.export_json:
        json_path = output_dir / "resultados_factibilidad.json"
        json_data = {}
        for filename, score in results.items():
            json_data[filename] = {
                "feasibility_score": score.feasibility_score,
                "quality_tier": score.quality_tier,
                "components_detected": [c.value for c in score.components_detected],
                "has_quantitative_baseline": score.has_quantitative_baseline,
                "has_quantitative_target": score.has_quantitative_target,
                "detailed_matches": [
                    {
                        "component_type": match.component_type.value,
                        "matched_text": match.matched_text,
                        "confidence": match.confidence,
                        "position": match.position,
                    }
                    for match in score.detailed_matches
                ],
            }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Resultados JSON exportados: {json_path}")

    if args.export_markdown:
        md_path = output_dir / "reporte_factibilidad.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Reporte de Evaluación de Factibilidad de Indicadores\n\n")
            f.write(
                f"**Fecha de generación:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                f"**Número total de indicadores evaluados:** {len(results)}\n\n")

            # Summary statistics
            scores = [score.feasibility_score for score in results.values()]
            tiers = [score.quality_tier for score in results.values()]

            f.write("## Resumen Estadístico\n\n")
            if scores:
                f.write(
                    f"- **Puntuación promedio:** {sum(scores) / len(scores):.3f}\n")
                f.write(f"- **Puntuación máxima:** {max(scores):.3f}\n")
                f.write(f"- **Puntuación mínima:** {min(scores):.3f}\n")

                tier_counts = {tier: tiers.count(tier) for tier in set(tiers)}
                f.write(f"\n### Distribución por Nivel de Calidad\n\n")
                for tier, count in tier_counts.items():
                    tier_spanish = scorer.translate_quality_tier_spanish(tier)
                    percentage = (count / len(tiers)) * 100
                    f.write(
                        f"- **{tier_spanish}:** {count} indicadores ({percentage:.1f}%)\n"
                    )

            f.write("\n## Resultados Detallados\n\n")

            # Sort results by score
            sorted_results = sorted(
                results.items(), key=lambda x: x[1].feasibility_score, reverse=True
            )

            for filename, score in sorted_results:
                f.write(f"### {filename}\n\n")
                f.write(f"- **Puntuación:** {score.feasibility_score:.3f}\n")
                f.write(
                    f"- **Nivel de calidad:** {scorer.translate_quality_tier_spanish(score.quality_tier)}\n"
                )
                f.write(
                    f"- **Línea base cuantitativa:** {'Sí' if score.has_quantitative_baseline else 'No'}\n"
                )
                f.write(
                    f"- **Meta cuantitativa:** {'Sí' if score.has_quantitative_target else 'No'}\n"
                )
                f.write(
                    f"- **Componentes detectados:** {len(score.components_detected)}\n"
                )
                f.write(
                    f"- **Recomendación:** {scorer.get_recommendation_spanish(score.quality_tier)}\n\n"
                )

        print(f"✓ Reporte Markdown exportado: {md_path}")

    return 0


if __name__ == "__main__":
    exit(main())
