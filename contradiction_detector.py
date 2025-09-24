"""
Contradiction Detection Module using Adversative Connectors and Goal/Activity Analysis

This module detects contradictory statements in text by analyzing adversative connectors
("sin embargo", "aunque", "pero", "no obstante") in proximity to goal indicators,
action verbs, and quantitative targets, flagging these as medium-high risk contradictions.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium-high"
    HIGH = "high"


@dataclass
class ContradictionMatch:
    """Represents a detected contradiction with detailed information."""

    adversative_connector: str
    goal_keywords: List[str]
    action_verbs: List[str]
    quantitative_targets: List[str]
    full_text: str
    start_pos: int
    end_pos: int
    risk_level: RiskLevel
    confidence: float
    context_window: str


@dataclass
class ContradictionAnalysis:
    """Complete contradiction analysis results for a text segment."""

    contradictions: List[ContradictionMatch]
    total_contradictions: int
    risk_score: float
    risk_level: RiskLevel
    highest_confidence_contradiction: Optional[ContradictionMatch]
    summary: Dict[str, int]


class ContradictionDetector:
    """Detects contradictions by identifying adversative patterns near goal/activity keywords."""

    def __init__(self, context_window: int = 150):
        """
        Initialize the contradiction detector.

        Args:
            context_window: Number of characters to analyze around adversative connectors
        """
        self.context_window = context_window
        self.logger = logging.getLogger(__name__)

        # Adversative connectors (Spanish)
        self.adversative_patterns = [
            r"\bsin\s+embargo\b",
            r"\baunque\b",
            r"\bpero\b",
            r"\bno\s+obstante\b",
            r"\ba\s+pesar\s+de\b",
            r"\bno\s+obstante\s+lo\s+anterior\b",
            r"\bsin\s+embargo,?\s+",
            r"\bno\s+obstante,?\s+",
            r"\bmás\b",  # as in "más sin embargo"
            r"\bempero\b",
            r"\bmientras\s+que\b",
        ]

        # Goal indicators
        self.goal_patterns = [
            r"\b(?:meta|metas)\b",
            r"\b(?:objetivo|objetivos)\b",
            r"\b(?:propósito|propósitos)\b",
            r"\b(?:finalidad|finalidades)\b",
            r"\b(?:resultado\s+esperado)\b",
            r"\b(?:lograr|alcanzar|conseguir|obtener)\b",
            r"\b(?:pretende|pretender|busca|buscar)\b",
            r"\b(?:se\s+espera|se\s+proyecta)\b",
            r"\b(?:aspiración|aspiraciones)\b",
            r"\b(?:target|targets)\b",
            r"\b(?:goal|goals)\b",
        ]

        # Action verbs
        self.action_patterns = [
            r"\b(?:implementar|ejecutar|desarrollar|realizar)\b",
            r"\b(?:establecer|crear|construir|formar)\b",
            r"\b(?:promover|fomentar|impulsar|fortalecer)\b",
            r"\b(?:mejorar|optimizar|incrementar|aumentar)\b",
            r"\b(?:reducir|disminuir|minimizar|controlar)\b",
            r"\b(?:garantizar|asegurar|proteger|defender)\b",
            r"\b(?:coordinar|articular|gestionar|liderar)\b",
            r"\b(?:monitorear|supervisar|evaluar|verificar)\b",
            r"\b(?:capacitar|formar|educar|sensibilizar)\b",
            r"\b(?:prevenir|atender|resolver|solucionar)\b",
        ]

        # Quantitative target patterns
        self.quantitative_patterns = [
            r"\b\d+(?:[.,]\d+)?\s*(?:%|por\s+ciento|porciento)\b",
            r"\b\d+(?:[.,]\d+)?\s*(?:millones?|mil|thousands?|millions?)\b",
            r"\b(?:incrementar|aumentar|reducir|disminuir)\s+(?:en\s+|hasta\s+)?\d+(?:[.,]\d+)?\b",
            r"\b\d+(?:[.,]\d+)?\s*(?:COP|\$|USD|pesos|dollars?)\b",
            r"\bmeta\s+de\s+\d+(?:[.,]\d+)?\b",
            r"\bhasta\s+el\s+\d+(?:[.,]\d+)?\b",
            r"\b\d+\s*(?:personas|beneficiarios|familias|hogares)\b",
            r"\b\d+\s*(?:hectáreas|km²|metros)\b",
            r"\b\d+(?:[.,]\d+)?\s*(?:puntos|bases)\b",
        ]

        # Compile patterns for performance
        self.compiled_adversative = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in self.adversative_patterns
        ]
        self.compiled_goals = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in self.goal_patterns
        ]
        self.compiled_actions = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in self.action_patterns
        ]
        self.compiled_quantitative = [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in self.quantitative_patterns
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text using Unicode NFKC normalization."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _find_pattern_matches(
        text: str, patterns: List[re.Pattern], pattern_type: str
    ) -> List[Tuple[str, int, int]]:
        """Find all matches for a given pattern type."""
        matches = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                matches.append((match.group(), match.start(), match.end()))
        return matches

    def _extract_context_window(self, text: str, center_pos: int) -> str:
        """Extract context window around a center position."""
        start = max(0, center_pos - self.context_window // 2)
        end = min(len(text), center_pos + self.context_window // 2)
        return text[start:end].strip()

    @staticmethod
    def _calculate_contradiction_confidence(
        adversative_pos: int,
        goal_matches: List[Tuple[str, int, int]],
        action_matches: List[Tuple[str, int, int]],
        quantitative_matches: List[Tuple[str, int, int]],
    ) -> float:
        """Calculate confidence score for a contradiction based on proximity and context."""
        confidence = 0.0

        # Base confidence for having an adversative connector
        confidence += 0.3

        # Proximity scoring - closer matches get higher scores
        for matches, weight in [
            (goal_matches, 0.4),
            (action_matches, 0.2),
            (quantitative_matches, 0.3),
        ]:
            if matches:
                min_distance = min(abs(match[1] - adversative_pos) for match in matches)
                # Exponential decay based on distance
                proximity_score = weight * (0.5 ** (min_distance / 50))
                confidence += proximity_score

        # Bonus for having multiple types of matches
        match_types = sum(
            [bool(goal_matches), bool(action_matches), bool(quantitative_matches)]
        )
        confidence += match_types * 0.1

        return min(1.0, confidence)

    @staticmethod
    def _determine_risk_level(confidence: float, context_complexity: int) -> RiskLevel:
        """Determine risk level based on confidence and context complexity."""
        # Adjust confidence based on context complexity
        adjusted_confidence = confidence * (1 + context_complexity * 0.1)

        if adjusted_confidence >= 0.8:
            return RiskLevel.HIGH
        elif adjusted_confidence >= 0.6:
            return RiskLevel.MEDIUM_HIGH
        elif adjusted_confidence >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def detect_contradictions(self, text: str) -> ContradictionAnalysis:
        """
        Detect contradictions in text by analyzing adversative connectors near goals/actions.

        Args:
            text: Input text to analyze

        Returns:
            ContradictionAnalysis with detected contradictions and risk assessment
        """
        # Normalize text
        normalized_text = self._normalize_text(text)

        # Find adversative connectors
        adversative_matches = self._find_pattern_matches(
            normalized_text, self.compiled_adversative, "adversative"
        )

        if not adversative_matches:
            return ContradictionAnalysis(
                contradictions=[],
                total_contradictions=0,
                risk_score=0.0,
                risk_level=RiskLevel.LOW,
                highest_confidence_contradiction=None,
                summary={"low": 0, "medium": 0, "medium-high": 0, "high": 0},
            )

        contradictions = []

        for adversative_text, adv_start, adv_end in adversative_matches:
            # Extract context window around adversative connector
            context = self._extract_context_window(normalized_text, adv_start)
            context_start = max(0, adv_start - self.context_window // 2)

            # Find goal, action, and quantitative patterns in context
            goal_matches = self._find_pattern_matches(
                context, self.compiled_goals, "goal"
            )
            action_matches = self._find_pattern_matches(
                context, self.compiled_actions, "action"
            )
            quantitative_matches = self._find_pattern_matches(
                context, self.compiled_quantitative, "quantitative"
            )

            # Only create contradiction if we have at least one goal/action/quantitative match
            if goal_matches or action_matches or quantitative_matches:
                # Calculate confidence
                confidence = self._calculate_contradiction_confidence(
                    adv_start - context_start,  # Adjust position for context window
                    goal_matches,
                    action_matches,
                    quantitative_matches,
                )

                # Determine risk level
                context_complexity = (
                    len(goal_matches) + len(action_matches) + len(quantitative_matches)
                )
                risk_level = self._determine_risk_level(confidence, context_complexity)

                # Create contradiction match
                contradiction = ContradictionMatch(
                    adversative_connector=adversative_text,
                    goal_keywords=[m[0] for m in goal_matches],
                    action_verbs=[m[0] for m in action_matches],
                    quantitative_targets=[m[0] for m in quantitative_matches],
                    full_text=context,
                    start_pos=adv_start,
                    end_pos=adv_end,
                    risk_level=risk_level,
                    confidence=confidence,
                    context_window=context,
                )

                contradictions.append(contradiction)

        # Calculate overall risk score
        if contradictions:
            risk_score = sum(c.confidence for c in contradictions) / len(contradictions)
            highest_confidence = max(contradictions, key=lambda x: x.confidence)
            overall_risk = self._determine_risk_level(risk_score, len(contradictions))
        else:
            risk_score = 0.0
            highest_confidence = None
            overall_risk = RiskLevel.LOW

        # Create summary
        summary = {"low": 0, "medium": 0, "medium-high": 0, "high": 0}
        for contradiction in contradictions:
            if contradiction.risk_level == RiskLevel.LOW:
                summary["low"] += 1
            elif contradiction.risk_level == RiskLevel.MEDIUM:
                summary["medium"] += 1
            elif contradiction.risk_level == RiskLevel.MEDIUM_HIGH:
                summary["medium-high"] += 1
            elif contradiction.risk_level == RiskLevel.HIGH:
                summary["high"] += 1

        return ContradictionAnalysis(
            contradictions=contradictions,
            total_contradictions=len(contradictions),
            risk_score=risk_score,
            risk_level=overall_risk,
            highest_confidence_contradiction=highest_confidence,
            summary=summary,
        )

    def integrate_with_risk_assessment(
        self, text: str, existing_score: float = 0.0
    ) -> Dict[str, float]:
        """
        Integrate contradiction detection with existing risk assessment scoring system.

        Args:
            text: Text to analyze
            existing_score: Existing risk score to integrate with

        Returns:
            Dictionary with updated risk scores
        """
        contradiction_analysis = self.detect_contradictions(text)

        # Calculate contradiction risk contribution
        contradiction_risk = 0.0

        if contradiction_analysis.total_contradictions > 0:
            # Base risk from having contradictions
            base_risk = min(0.3, contradiction_analysis.total_contradictions * 0.1)

            # Risk from confidence levels
            confidence_risk = contradiction_analysis.risk_score * 0.4

            # Risk from high-severity contradictions
            high_risk_count = contradiction_analysis.summary.get("high", 0)
            medium_high_count = contradiction_analysis.summary.get("medium-high", 0)
            severity_risk = (high_risk_count * 0.2) + (medium_high_count * 0.15)

            contradiction_risk = min(1.0, base_risk + confidence_risk + severity_risk)

        # Integrate with existing score
        integrated_score = existing_score + (
            contradiction_risk * 0.3
        )  # 30% weight for contradictions
        integrated_score = min(1.0, integrated_score)  # Cap at 1.0

        return {
            "base_score": existing_score,
            "contradiction_risk": contradiction_risk,
            "integrated_score": integrated_score,
            "contradiction_count": contradiction_analysis.total_contradictions,
            "highest_contradiction_confidence": (
                contradiction_analysis.highest_confidence_contradiction.confidence
                if contradiction_analysis.highest_confidence_contradiction
                else 0.0
            ),
        }


# Example usage and testing
if __name__ == "__main__":
    detector = ContradictionDetector()

    # Test samples with contradictions
    test_texts = [
        # High-risk contradiction
        "El objetivo es aumentar la cobertura educativa al 95% para 2027, sin embargo, los recursos presupuestales han sido reducidos en un 30% este año.",
        # Medium-high risk contradiction
        "La meta es crear 1000 empleos en el sector agrícola, aunque las políticas actuales no contemplan incentivos para nuevas empresas.",
        # Medium risk contradiction
        "Se busca fortalecer la seguridad ciudadana, pero no se ha definido una estrategia clara de implementación.",
        # Low-risk (no contradiction)
        "El programa pretende mejorar la calidad de vida mediante la construcción de viviendas sociales con un presupuesto de 500 millones de pesos.",
        # Complex contradiction with multiple elements
        "La Secretaría de Salud establecerá 10 nuevos centros de atención primaria con meta del 80% de cobertura para 2025, no obstante, el presupuesto actual solo cubre 3 centros y no hay planes de financiación adicional.",
        # Multiple contradictions in one text
        "El objetivo es reducir la pobreza al 15% mediante programas sociales, sin embargo, estos programas han perdido financiación. Aunque se pretende alcanzar 50,000 beneficiarios, pero la capacidad operativa actual es de solo 20,000 personas.",
    ]

    print("=== ANÁLISIS DE CONTRADICCIONES ===\n")

    for i, text in enumerate(test_texts, 1):
        print(f"TEXTO {i}:")
        print(f"'{text[:100]}...' \n")

        # Analyze contradictions
        analysis = detector.detect_contradictions(text)

        print(f"Contradicciones detectadas: {analysis.total_contradictions}")
        print(f"Nivel de riesgo general: {analysis.risk_level.value}")
        print(f"Puntuación de riesgo: {analysis.risk_score:.3f}")
        print(f"Resumen: {analysis.summary}")

        if analysis.contradictions:
            print("\nDETALLE DE CONTRADICCIONES:")
            for j, contradiction in enumerate(analysis.contradictions, 1):
                print(
                    f"  {j}. Conector adversativo: '{contradiction.adversative_connector}'"
                )
                print(f"     Palabras clave de meta: {contradiction.goal_keywords}")
                print(f"     Verbos de acción: {contradiction.action_verbs}")
                print(
                    f"     Objetivos cuantitativos: {contradiction.quantitative_targets}"
                )
                print(f"     Confianza: {contradiction.confidence:.3f}")
                print(f"     Nivel de riesgo: {contradiction.risk_level.value}")
                print()

        # Test integration with existing risk system
        risk_integration = detector.integrate_with_risk_assessment(
            text, existing_score=0.2
        )
        print(f"INTEGRACIÓN CON SISTEMA DE RIESGO:")
        print(f"  Puntuación base: {risk_integration['base_score']:.3f}")
        print(
            f"  Riesgo por contradicciones: {risk_integration['contradiction_risk']:.3f}"
        )
        print(f"  Puntuación integrada: {risk_integration['integrated_score']:.3f}")

        print("\n" + "=" * 80 + "\n")
