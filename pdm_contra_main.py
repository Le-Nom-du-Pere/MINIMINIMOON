#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDM Contradiction Detection Package
====================================
Detección de contradicciones, incompatibilidades y riesgos de gobernanza 
en Planes de Desarrollo Municipal (PDM) de Colombia.

@novelty_manifest:
- sentence-transformers v3.3.1 (2024-11): SOTA multilingual embeddings, release: https://github.com/UKPLab/sentence-transformers/releases/tag/v3.3.1
- transformers v4.46.0 (2024-10): Latest NLI models for Spanish, release: https://github.com/huggingface/transformers/releases/tag/v4.46.0  
- typer v0.12.5 (2024-08): Modern CLI framework, release: https://github.com/tiangolo/typer/releases/tag/0.12.5
- pydantic v2.10.0 (2024-11): Data validation with strict typing, release: https://github.com/pydantic/pydantic/releases/tag/v2.10.0
- polars v1.15.0 (2024-11): High-performance dataframes, release: https://github.com/pola-rs/polars/releases/tag/py-1.15.0
- uv v0.5.10 (2024-11): Fast Python package installer, release: https://github.com/astral-sh/uv/releases/tag/0.5.10
- ruff v0.8.0 (2024-11): Fast Python linter/formatter, release: https://github.com/astral-sh/ruff/releases/tag/v0.8.0
- pypdf v5.1.0 (2024-10): Modern PDF processing, release: https://github.com/py-pdf/pypdf/releases/tag/5.1.0
- python-docx v1.1.2 (2024-06): DOCX processing, release: https://github.com/python-openxml/python-docx/releases/tag/v1.1.2
- mapie v0.9.1 (2024-09): Conformal prediction for uncertainty, release: https://github.com/scikit-learn-contrib/MAPIE/releases/tag/v0.9.1
- torch v2.5.0 (2024-10): Deep learning backend, release: https://github.com/pytorch/pytorch/releases/tag/v2.5.0
"""

__version__ = "1.0.0"
__author__ = "PDM Analysis Team"

# pyproject.toml
PYPROJECT_TOML = '''
[build-system]
requires = ["setuptools>=70.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pdm_contra"
version = "1.0.0"
description = "Detección de contradicciones en Planes de Desarrollo Municipal de Colombia"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "PDM Analysis Team", email = "pdm-analysis@example.org"}
]
keywords = ["nlp", "policy-analysis", "spanish", "municipal-planning", "colombia"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Government",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Natural Language :: Spanish"
]

dependencies = [
    "sentence-transformers>=3.3.1",
    "transformers>=4.46.0",
    "torch>=2.5.0",
    "typer>=0.12.5",
    "pydantic>=2.10.0",
    "polars>=1.15.0",
    "pypdf>=5.1.0",
    "python-docx>=1.1.2",
    "mapie>=0.9.1",
    "scikit-learn>=1.5.0",
    "numpy>=1.26.0",
    "rich>=13.9.0",
    "httpx>=0.27.0",
    "orjson>=3.10.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "ipykernel>=6.29.0"
]

[project.scripts]
pdm-contradict = "pdm_contra.cli:app"

[tool.setuptools]
packages = ["pdm_contra"]

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "N", "W", "UP", "B", "C90", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=pdm_contra --cov-report=term-missing"
'''

# pdm_contra/__init__.py
from pdm_contra.core import ContradictionDetector
from pdm_contra.models import (
    ContradictionMatch,
    ContradictionAnalysis, 
    CompetenceValidation,
    PDMDocument
)
from pdm_contra.utils.guard_novelty import check_dependencies

__all__ = [
    "ContradictionDetector",
    "ContradictionMatch",
    "ContradictionAnalysis",
    "CompetenceValidation",
    "PDMDocument",
    "check_dependencies"
]

# pdm_contra/core.py
"""
Core contradiction detection engine with hybrid neuro-symbolic approach.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import re
import unicodedata

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import numpy as np
from pydantic import BaseModel, Field

from pdm_contra.models import (
    ContradictionMatch,
    ContradictionAnalysis,
    RiskLevel,
    CompetenceValidation
)
from pdm_contra.nlp.nli import SpanishNLIDetector
from pdm_contra.nlp.patterns import PatternMatcher
from pdm_contra.policy.competence import CompetenceValidator
from pdm_contra.scoring.risk import RiskScorer
from pdm_contra.explain.tracer import ExplanationTracer

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """
    Hybrid contradiction detector for PDM analysis.
    Combines pattern matching, NLI, and policy validation.
    """
    
    def __init__(
        self,
        competence_matrix_path: Optional[Path] = None,
        language: str = "es",
        mode_light: bool = False
    ):
        """
        Initialize the contradiction detector.
        
        Args:
            competence_matrix_path: Path to JSON file with competence matrix
            language: Language code (default: "es" for Spanish)
            mode_light: Use lightweight models if True
        """
        self.language = language
        self.mode_light = mode_light
        
        # Initialize components
        self.pattern_matcher = PatternMatcher(language=language)
        self.nli_detector = SpanishNLIDetector(light_mode=mode_light)
        self.competence_validator = CompetenceValidator(
            matrix_path=competence_matrix_path
        )
        self.risk_scorer = RiskScorer()
        self.explainer = ExplanationTracer()
        
        # Load sentence encoder for semantic similarity
        model_name = (
            "sentence-transformers/all-MiniLM-L6-v2" if mode_light
            else "sentence-transformers/multilingual-e5-large"
        )
        self.encoder = SentenceTransformer(model_name)
        
        logger.info(f"Initialized ContradictionDetector in {'light' if mode_light else 'full'} mode")
    
    def detect_contradictions(
        self,
        text: str,
        sectors: Optional[List[str]] = None,
        pdm_structure: Optional[Dict[str, Any]] = None
    ) -> ContradictionAnalysis:
        """
        Detect contradictions in PDM text.
        
        Args:
            text: Input PDM text
            sectors: List of sectors to analyze
            pdm_structure: Optional structured PDM data
            
        Returns:
            ContradictionAnalysis with all findings
        """
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Extract document segments
        segments = self._segment_document(normalized_text, pdm_structure)
        
        contradictions = []
        competence_issues = []
        agenda_gaps = []
        
        # Analyze each segment
        for seg_type, seg_text, seg_meta in segments:
            # Pattern-based detection
            pattern_matches = self.pattern_matcher.find_adversatives(seg_text)
            
            # NLI-based detection for semantic contradictions
            if pdm_structure and seg_type in ["objetivos", "metas", "indicadores"]:
                nli_results = self._check_nli_contradictions(
                    seg_text, pdm_structure, seg_type
                )
                pattern_matches.extend(nli_results)
            
            # Competence validation
            if sectors and seg_type in ["programas", "acciones"]:
                comp_results = self.competence_validator.validate_segment(
                    seg_text, sectors, seg_meta.get("nivel", "municipal")
                )
                competence_issues.extend(comp_results)
            
            # Agenda-setting alignment check
            if pdm_structure:
                agenda_issues = self._check_agenda_alignment(
                    seg_text, seg_type, pdm_structure
                )
                agenda_gaps.extend(agenda_issues)
            
            contradictions.extend(pattern_matches)
        
        # Calculate risk scores with conformal prediction
        risk_analysis = self.risk_scorer.calculate_risk(
            contradictions, competence_issues, agenda_gaps
        )
        
        # Generate explanations
        explanations = self.explainer.generate_explanations(
            contradictions, competence_issues, agenda_gaps
        )
        
        return ContradictionAnalysis(
            contradictions=contradictions,
            competence_mismatches=competence_issues,
            agenda_gaps=agenda_gaps,
            total_contradictions=len(contradictions),
            total_competence_issues=len(competence_issues),
            total_agenda_gaps=len(agenda_gaps),
            risk_score=risk_analysis["overall_risk"],
            risk_level=risk_analysis["risk_level"],
            confidence_intervals=risk_analysis["confidence_intervals"],
            explanations=explanations,
            calibration_info={
                "method": "conformal_prediction",
                "alpha": 0.1,
                "coverage": risk_analysis.get("empirical_coverage", 0.9)
            }
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text using NFKC and clean whitespace."""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
    
    def _segment_document(
        self,
        text: str,
        structure: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, str, Dict]]:
        """
        Segment PDM document into analyzable chunks.
        
        Returns:
            List of (segment_type, text, metadata) tuples
        """
        segments = []
        
        # Common PDM section patterns
        section_patterns = {
            "diagnostico": r"(?i)(diagnóstico|análisis\s+situacional|contexto)",
            "objetivos": r"(?i)(objetivos?\s+estratégicos?|objetivos?\s+generales?)",
            "metas": r"(?i)(metas?\s+de?\s+resultado|metas?\s+de?\s+producto)",
            "indicadores": r"(?i)(indicadores?\s+de?\s+gestión|indicadores?\s+de?\s+impacto)",
            "programas": r"(?i)(programas?\s+y?\s+proyectos?|líneas?\s+estratégicas?)",
            "presupuesto": r"(?i)(presupuesto|plan\s+plurianual|recursos)",
        }
        
        # Try to extract sections
        current_pos = 0
        for sec_type, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text))
            for match in matches:
                # Extract text until next section or 2000 chars
                start = match.end()
                end = min(start + 2000, len(text))
                
                # Find next section boundary
                next_sections = []
                for other_pattern in section_patterns.values():
                    next_match = re.search(other_pattern, text[start:end])
                    if next_match:
                        next_sections.append(start + next_match.start())
                
                if next_sections:
                    end = min(next_sections)
                
                segment_text = text[start:end].strip()
                if segment_text:
                    segments.append((
                        sec_type,
                        segment_text,
                        {"start": start, "end": end, "header": match.group()}
                    ))
        
        # If no sections found, treat as single segment
        if not segments:
            segments.append(("general", text, {"start": 0, "end": len(text)}))
        
        return segments
    
    def _check_nli_contradictions(
        self,
        text: str,
        structure: Dict[str, Any],
        segment_type: str
    ) -> List[ContradictionMatch]:
        """
        Check for semantic contradictions using NLI.
        
        Args:
            text: Segment text
            structure: PDM structure data
            segment_type: Type of segment being analyzed
            
        Returns:
            List of contradiction matches found via NLI
        """
        nli_matches = []
        
        # Extract comparison pairs based on segment type
        pairs = []
        if segment_type == "objetivos" and "metas" in structure:
            for objetivo in self._extract_items(text, "objetivo"):
                for meta in structure.get("metas", []):
                    pairs.append((objetivo, meta, "objetivo-meta"))
        
        elif segment_type == "metas" and "indicadores" in structure:
            for meta in self._extract_items(text, "meta"):
                for indicador in structure.get("indicadores", []):
                    pairs.append((meta, indicador, "meta-indicador"))
        
        # Run NLI on pairs
        for premise, hypothesis, pair_type in pairs:
            result = self.nli_detector.check_contradiction(premise, hypothesis)
            
            if result["label"] == "contradiction" and result["score"] > 0.7:
                match = ContradictionMatch(
                    type="semantic_contradiction",
                    pair_type=pair_type,
                    premise=premise[:200],
                    hypothesis=hypothesis[:200],
                    nli_score=result["score"],
                    nli_label=result["label"],
                    confidence=result["score"],
                    risk_level=self._score_to_risk(result["score"]),
                    explanation=f"Contradicción semántica detectada entre {pair_type}"
                )
                nli_matches.append(match)
        
        return nli_matches
    
    def _check_agenda_alignment(
        self,
        text: str,
        segment_type: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check alignment in the agenda-setting chain.
        
        Returns:
            List of agenda alignment issues
        """
        issues = []
        
        # Check for missing links in chain
        chain = ["diagnostico", "objetivos", "estrategias", "metas", "indicadores", "presupuesto"]
        
        if segment_type in chain:
            idx = chain.index(segment_type)
            
            # Check backward alignment
            if idx > 0:
                prev_type = chain[idx - 1]
                if prev_type in structure and not self._has_alignment(text, structure[prev_type]):
                    issues.append({
                        "type": "missing_backward_alignment",
                        "from": segment_type,
                        "to": prev_type,
                        "severity": "medium",
                        "explanation": f"Falta alineación entre {segment_type} y {prev_type}"
                    })
            
            # Check forward alignment
            if idx < len(chain) - 1:
                next_type = chain[idx + 1]
                if next_type not in structure or not structure[next_type]:
                    issues.append({
                        "type": "missing_forward_element",
                        "from": segment_type,
                        "expected": next_type,
                        "severity": "high",
                        "explanation": f"No se encontró {next_type} para {segment_type}"
                    })
        
        return issues
    
    def _extract_items(self, text: str, item_type: str) -> List[str]:
        """Extract items of a specific type from text."""
        items = []
        
        patterns = {
            "objetivo": r"objetivo\s*\d*[:.]?\s*([^.]+\.)",
            "meta": r"meta\s*\d*[:.]?\s*([^.]+\.)",
            "indicador": r"indicador\s*[:.]?\s*([^.]+\.)",
        }
        
        if item_type in patterns:
            matches = re.finditer(patterns[item_type], text, re.IGNORECASE)
            items = [match.group(1).strip() for match in matches]
        
        return items
    
    def _has_alignment(self, text1: str, text2_items: List[str]) -> bool:
        """Check if text1 has semantic alignment with items in text2."""
        if not text2_items:
            return False
        
        # Encode texts
        emb1 = self.encoder.encode(text1[:500], convert_to_tensor=True)
        emb2 = self.encoder.encode(text2_items[:5], convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.pytorch_cos_sim(emb1, emb2)
        max_sim = torch.max(similarities).item()
        
        return max_sim > 0.6  # Threshold for considering alignment
    
    def _score_to_risk(self, score: float) -> RiskLevel:
        """Convert confidence score to risk level."""
        if score >= 0.8:
            return RiskLevel.HIGH
        elif score >= 0.6:
            return RiskLevel.MEDIUM_HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW