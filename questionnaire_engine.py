#!/usr/bin/env python3
"""
AUTHORITATIVE QUESTIONNAIRE ENGINE v1.0
======================================

CRITICAL: This module enforces the EXACT 300-question structure (30 base questions × 10 thematic points)
as defined in the Master Prompt. ANY modification to this structure is FORBIDDEN.

PURPOSE: Automate the answering of a structured questionnaire with EXACTLY:
- 10 Policy Domains (P1-P10): Each corresponding to a human rights catalog item  
- 30 Questions per Domain: Standardized across 6 dimensions (D1-D6, 5 questions each)
- Total: 300 Questions (10 × 30): Complete automated evaluation framework

IMMUTABLE STRUCTURE: The system MUST preserve the exact wording and structure of the 
original questionnaire while automating evidence detection and response generation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuestionnaireStructure:
    """IMMUTABLE: Enforces the 30×10 structure"""
    TOTAL_QUESTIONS = 300
    DOMAINS = 10  # P1-P10
    QUESTIONS_PER_DOMAIN = 30  # 6 dimensions × 5 questions each
    DIMENSIONS = 6  # D1-D6
    QUESTIONS_PER_DIMENSION = 5
    VERSION = "1.0"
    
    def validate_structure(self) -> bool:
        """Validates that 10 × 30 = 300 questions"""
        return (self.DOMAINS * self.QUESTIONS_PER_DOMAIN == self.TOTAL_QUESTIONS and
                self.DIMENSIONS * self.QUESTIONS_PER_DIMENSION == self.QUESTIONS_PER_DOMAIN)

@dataclass
class ThematicPoint:
    """Represents one of the 10 thematic points (P1-P10)"""
    id: str  # P1, P2, ..., P10
    title: str
    keywords: List[str]
    hints: List[str]
    relevant_programs: List[str] = field(default_factory=list)
    pdm_sections: List[str] = field(default_factory=list)

@dataclass
class BaseQuestion:
    """Base question that gets parametrized for each thematic point"""
    id: str  # D1-Q1, D1-Q2, etc.
    dimension: str  # D1, D2, D3, D4, D5, D6
    question_no: int  # 1-30
    template: str  # Question template with {PUNTO_TEMATICO} placeholder
    search_patterns: Dict[str, Any]
    scoring_rule: str
    max_score: float = 3.0

@dataclass
class EvaluationResult:
    """Result of evaluating one question for one thematic point"""
    question_id: str  # P1-D1-Q1, P2-D1-Q1, etc.
    point_code: str  # P1, P2, etc.
    point_title: str
    dimension: str
    question_no: int
    prompt: str  # Fully parametrized question
    score: float
    max_score: float
    elements_found: Dict[str, bool]
    evidence: List[Dict[str, Any]]
    missing_elements: List[str]
    recommendation: str

class QuestionnaireEngine:
    """
    AUTHORITATIVE ENGINE: Enforces strict adherence to 30×10 structure
    
    CRITICAL: This class is the ONLY way to execute questionnaire evaluation.
    It ensures that EXACTLY 30 questions are applied to EXACTLY 10 thematic points,
    resulting in EXACTLY 300 evaluations.
    """
    
    def __init__(self):
        """Initialize with immutable structure validation"""
        self.structure = QuestionnaireStructure()
        if not self.structure.validate_structure():
            raise ValueError("CRITICAL: Questionnaire structure validation FAILED")
        
        self.base_questions = self._load_base_questions()
        self.thematic_points = self._load_thematic_points()
        
        # Validate exact counts
        if len(self.base_questions) != 30:
            raise ValueError(f"CRITICAL: Must have exactly 30 base questions, got {len(self.base_questions)}")
        if len(self.thematic_points) != 10:
            raise ValueError(f"CRITICAL: Must have exactly 10 thematic points, got {len(self.thematic_points)}")
        
        logger.info("✅ QuestionnaireEngine initialized with EXACT 30×10 structure")
    
    def _load_base_questions(self) -> List[BaseQuestion]:
        """Load the 30 base questions as defined in the Master Prompt"""
        
        # DIMENSION D1: DIAGNÓSTICO Y RECURSOS (Q1-Q5)
        d1_questions = [
            BaseQuestion(
                id="D1-Q1",
                dimension="D1", 
                question_no=1,
                template="¿El diagnóstico presenta líneas base con fuentes, series temporales, unidades, cobertura y método de medición para {PUNTO_TEMATICO}?",
                search_patterns={
                    "valor_numerico": r"\d+[.,]?\d*\s*(%|casos|personas|tasa|porcentaje|índice)",
                    "año": r"(20\d{2}|periodo|año|vigencia)",
                    "fuente": r"(fuente:|según|DANE|Ministerio|Secretaría|Encuesta|Censo|SISBEN|SIVIGILA|\(20\d{2}\))",
                    "serie_temporal": r"(20\d{2}.{0,50}20\d{2}|serie|histórico|evolución|tendencia)"
                },
                scoring_rule="(elementos_encontrados / 4) * 3"
            ),
            BaseQuestion(
                id="D1-Q2",
                dimension="D1",
                question_no=2, 
                template="¿Las líneas base capturan la magnitud del problema y los vacíos de información, explicitando sesgos, supuestos y calidad de datos para {PUNTO_TEMATICO}?",
                search_patterns={
                    "poblacion_afectada": r"(\d+\s*(personas|habitantes|casos|familias|mujeres|niños)|población.*\d+|afectados.*\d+)",
                    "brecha_deficit": r"(brecha|déficit|diferencia|faltante|carencia|necesidad insatisfecha).{0,30}\d+",
                    "vacios_info": r"(sin datos|no.*disponible|vacío|falta.*información|se requiere.*datos|limitación.*información|no se cuenta con)"
                },
                scoring_rule="sum of elements found (max 3)"
            ),
            BaseQuestion(
                id="D1-Q3",
                dimension="D1",
                question_no=3,
                template="¿Los recursos del PPI/Plan Indicativo están asignados explícitamente a {PUNTO_TEMATICO}, con trazabilidad programática y suficiencia relativa a la brecha?",
                search_patterns={
                    "presupuesto_total": r"\$\s*\d+([.,]\d+)?\s*(millones|miles de millones|mil|COP|pesos)",
                    "desglose": r"(20\d{2}.*\$|Producto.*\$|Meta.*\$|anual|vigencia.*presupuesto|por año)",
                    "trazabilidad": r"(Programa.{0,50}(inversión|presupuesto|recursos))"
                },
                scoring_rule="sum of elements found (max 3)"
            ),
            BaseQuestion(
                id="D1-Q4", 
                dimension="D1",
                question_no=4,
                template="¿Las capacidades institucionales (talento, procesos, datos, gobernanza) necesarias para activar los mecanismos causales en {PUNTO_TEMATICO} están descritas con sus cuellos de botella?",
                search_patterns={
                    "recursos_humanos": r"(profesionales|técnicos|funcionarios|personal|equipo|contratación|psicólogo|trabajador social|profesional).{0,50}\d+|se requiere.*personal|brecha.*talento",
                    "infraestructura": r"(infraestructura|equipamiento|sede|oficina|espacios|dotación|vehículos|adecuación)",
                    "procesos_instancias": r"(Secretaría|Comisaría|Comité|Mesa|Consejo|Sistema|procedimiento|protocolo|ruta|proceso de)"
                },
                scoring_rule="sum of elements found (max 3)"
            ),
            BaseQuestion(
                id="D1-Q5",
                dimension="D1", 
                question_no=5,
                template="¿Existe coherencia entre objetivos, recursos y capacidades para {PUNTO_TEMATICO}, con restricciones legales, presupuestales y temporales modeladas?",
                search_patterns={
                    "coherencia_presupuesto_productos": "logical rule: budget > 0 AND products defined"
                },
                scoring_rule="if presupuesto > 0 and num_productos > 0: 3 elif presupuesto > 0: 2 else: 0"
            )
        ]
        
        # DIMENSION D2: DISEÑO DE INTERVENCIÓN (Q6-Q10)
        d2_questions = [
            BaseQuestion(
                id="D2-Q6",
                dimension="D2",
                question_no=6,
                template="¿Las actividades para {PUNTO_TEMATICO} están formalizadas en tablas (responsable, insumo, output, calendario, costo unitario) y no sólo en narrativa?",
                search_patterns={
                    "tabla_productos": "detect table with columns: Producto, Meta, Unidad, Responsable"
                },
                scoring_rule="(columnas_encontradas / 4) * 3"
            ),
            BaseQuestion(
                id="D2-Q7",
                dimension="D2",
                question_no=7,
                template="¿Cada actividad especifica el instrumento y su mecanismo causal pretendido y la población diana en {PUNTO_TEMATICO}?",
                search_patterns={
                    "poblacion_objetivo": r"(mujeres|niños|niñas|adolescentes|jóvenes|víctimas|familias|comunidad|población|adultos mayores|personas con discapacidad)",
                    "cuantificacion": r"\d+\s*(personas|beneficiarios|familias|atenciones|servicios|participantes)",
                    "focalizacion": r"(zona rural|urbano|cabecera|prioridad|vulnerable|focalización|criterios|selección|población objetivo)"
                },
                scoring_rule="sum of elements found (max 3)"
            ),
            BaseQuestion(
                id="D2-Q8",
                dimension="D2", 
                question_no=8,
                template="¿Cada problema priorizado en {PUNTO_TEMATICO} tiene al menos una actividad que ataca el eslabón causal relevante (causa raíz o mediador)?",
                search_patterns={
                    "semantic_matching": "use embeddings to match problems with products"
                },
                scoring_rule="ratio_cobertura >= 0.80: 3, >= 0.50: 2, >= 0.30: 1, else: 0"
            ),
            BaseQuestion(
                id="D2-Q9",
                dimension="D2",
                question_no=9, 
                template="¿Se identifican riesgos de desplazamiento de efectos, cuñas de implementación y conflictos entre actividades en {PUNTO_TEMATICO}, con mitigaciones?",
                search_patterns={
                    "riesgos_explicitos": r"(riesgo|limitación|restricción|dificultad|cuello de botella|matriz.*riesgo|desafío|obstáculo)",
                    "factores_externos": r"(depende|articulación|coordinación|transversal|nivel nacional|competencia de|requiere apoyo|sujeto a)"
                },
                scoring_rule="(elementos_encontrados / 2) * 3"
            ),
            BaseQuestion(
                id="D2-Q10",
                dimension="D2",
                question_no=10,
                template="¿Las actividades de {PUNTO_TEMATICO} forman una teoría de intervención coherente (complementariedades, secuenciación, no redundancias)?",
                search_patterns={
                    "integracion": r"(articulación|complementa|sinergia|coordinación|integra|transversal|en conjunto|simultáneamente)",
                    "referencia_cruzada": r"(programa de|articulado con|en el marco de|junto con|además de)"
                },
                scoring_rule="(elementos_encontrados / 2) * 3"
            )
        ]
        
        # Continue with D3-D6 questions following the same pattern...
        # For brevity, I'll create placeholder structures for the remaining dimensions
        
        # DIMENSION D3: PRODUCTOS Y OUTPUTS (Q11-Q15)
        d3_questions = [
            BaseQuestion(
                id=f"D3-Q{11+i}", dimension="D3", question_no=11+i,
                template=f"¿Los productos de {{PUNTO_TEMATICO}} están definidos con indicadores verificables (fórmula, fuente, línea base, meta) y validados contra el mecanismo?" if i==0 else
                        f"¿Cobertura y dosificación de productos en {{PUNTO_TEMATICO}} son proporcionales a la brecha y al tamaño de efecto esperado?" if i==1 else
                        f"¿Los productos de {{PUNTO_TEMATICO}} tienen trazabilidad presupuestal y organizacional y están vinculados a actividades específicas?" if i==2 else
                        f"¿No hay contradicciones actividad→producto en {{PUNTO_TEMATICO}} (la actividad puede producir el output con la dosificación y plazos planteados)?" if i==3 else
                        f"¿Los productos de {{PUNTO_TEMATICO}} actúan como puentes causales hacia resultados, con mediadores clave especificados?",
                search_patterns={"placeholder": "to_be_implemented"},
                scoring_rule="placeholder"
            ) for i in range(5)
        ]
        
        # DIMENSION D4: RESULTADOS Y OUTCOMES (Q16-Q20)
        d4_questions = [
            BaseQuestion(
                id=f"D4-Q{16+i}", dimension="D4", question_no=16+i,
                template=f"¿Los resultados de {{PUNTO_TEMATICO}} están definidos con métricas de outcome, líneas base, metas y ventana de maduración?" if i==0 else
                        f"¿Se explicita el encadenamiento causal productos→resultados en {{PUNTO_TEMATICO}}, incluyendo supuestos y condiciones habilitantes?" if i==1 else
                        f"¿El nivel de ambición de resultados en {{PUNTO_TEMATICO}} es consistente con recursos y capacidades, justificado con evidencia comparada?" if i==2 else
                        f"¿Los resultados en {{PUNTO_TEMATICO}} atienden los problemas diagnosticados, con criterios de éxito/fallo y umbrales claros?" if i==3 else
                        f"¿Los resultados de {{PUNTO_TEMATICO}} se alinean con PND/ODS sin romper la lógica causal local?",
                search_patterns={"placeholder": "to_be_implemented"},
                scoring_rule="placeholder"
            ) for i in range(5)
        ]
        
        # DIMENSION D5: IMPACTOS Y EFECTOS DE LARGO PLAZO (Q21-Q25)
        d5_questions = [
            BaseQuestion(
                id=f"D5-Q{21+i}", dimension="D5", question_no=21+i,
                template=f"¿Los impactos de largo plazo para {{PUNTO_TEMATICO}} están definidos y son medibles, con la ruta de transmisión desde resultados y sus rezagos?" if i==0 else
                        f"¿Existe integración resultados↔impactos en {{PUNTO_TEMATICO}} mediante indicadores compuestos o proxies y su validez para captar el mecanismo?" if i==1 else
                        f"¿Se usan proxies mensurables cuando el impacto en {{PUNTO_TEMATICO}} es difícil de observar, documentando validez externa y limitaciones?" if i==2 else
                        f"¿Los impactos en {{PUNTO_TEMATICO}} se alinean con marcos nacionales/globales y consideran riesgos sistémicos que pueden romper el mecanismo?" if i==3 else
                        f"¿La ambición del impacto en {{PUNTO_TEMATICO}} es realista dado recursos y posibles efectos no deseados, con hipótesis-límite declaradas?",
                search_patterns={"placeholder": "to_be_implemented"},
                scoring_rule="placeholder"
            ) for i in range(5)
        ]
        
        # DIMENSION D6: TEORÍA DE CAMBIO Y COHERENCIA CAUSAL (Q26-Q30)
        d6_questions = [
            BaseQuestion(
                id=f"D6-Q{26+i}", dimension="D6", question_no=26+i,
                template=f"¿La teoría de cambio de {{PUNTO_TEMATICO}} está explícita (diagrama causal) con causas, mediadores, moderadores y supuestos verificables?" if i==0 else
                        f"¿Los enlaces causales en {{PUNTO_TEMATICO}} son proporcionales y sin saltos no realistas (no hay "milagros" de implementación)?" if i==1 else
                        f"¿Se identifican inconsistencias en la cadena causal de {{PUNTO_TEMATICO}} y se proponen validaciones (pilotos, pruebas de mecanismo)?" if i==2 else
                        f"¿Se monitorean patrones de fallo (drift, layering, conversión) en {{PUNTO_TEMATICO}} con mecanismos de corrección y aprendizaje continuo?" if i==3 else
                        f"¿La lógica causal de {{PUNTO_TEMATICO}} reconoce grupos afectados y restricciones contextuales (territoriales, culturales, regulatorias)?",
                search_patterns={"placeholder": "to_be_implemented"},
                scoring_rule="placeholder"
            ) for i in range(5)
        ]
        
        # Combine all dimensions
        all_questions = d1_questions + d2_questions + d3_questions + d4_questions + d5_questions + d6_questions
        
        # Final validation
        if len(all_questions) != 30:
            raise ValueError(f"CRITICAL: Must have exactly 30 base questions, constructed {len(all_questions)}")
        
        return all_questions
    
    def _load_thematic_points(self) -> List[ThematicPoint]:
        """Load the 10 thematic points from the authoritative JSON"""
        
        json_path = Path(__file__).parent / "decalogo_industrial.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"CRITICAL: Authoritative questionnaire file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract unique thematic points
            points_dict = {}
            for question in data.get('questions', []):
                point_code = question.get('point_code')
                if point_code and point_code not in points_dict:
                    points_dict[point_code] = ThematicPoint(
                        id=point_code,
                        title=question.get('point_title', ''),
                        keywords=[],  # Will be populated from hints
                        hints=question.get('hints', [])
                    )
            
            # Convert to list and sort by ID
            points = list(points_dict.values())
            points.sort(key=lambda p: int(p.id[1:]))  # Sort P1, P2, ..., P10
            
            if len(points) != 10:
                raise ValueError(f"CRITICAL: Must have exactly 10 thematic points, found {len(points)}")
            
            return points
            
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load thematic points: {e}")
    
    def execute_full_evaluation(self, pdm_document: Union[str, Path], 
                               municipality: str = "", department: str = "") -> Dict[str, Any]:
        """
        MAIN EXECUTION: Applies exactly 30 questions to exactly 10 thematic points
        
        This is the ONLY way to execute a complete evaluation.
        Results in EXACTLY 300 question evaluations.
        
        Args:
            pdm_document: Path to the PDM document
            municipality: Municipality name
            department: Department name
            
        Returns:
            Complete evaluation results with exactly 300 question-point combinations
        """
        
        logger.info(f"🚀 Starting FULL evaluation: 30 questions × 10 points = 300 evaluations")
        
        evaluation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        results = {
            "metadata": {
                "evaluation_id": evaluation_id,
                "version": self.structure.VERSION,
                "timestamp": start_time.isoformat(),
                "municipality": municipality,
                "department": department,
                "pdm_document": str(pdm_document),
                "total_evaluations": self.structure.TOTAL_QUESTIONS,
                "structure_validation": "PASSED"
            },
            "questionnaire_structure": {
                "total_questions": self.structure.TOTAL_QUESTIONS,
                "domains": self.structure.DOMAINS,
                "questions_per_domain": self.structure.QUESTIONS_PER_DOMAIN,
                "dimensions": self.structure.DIMENSIONS,
                "questions_per_dimension": self.structure.QUESTIONS_PER_DIMENSION
            },
            "thematic_points": [],
            "evaluation_matrix": [],
            "summary": {}
        }
        
        evaluation_count = 0
        
        # Execute evaluation for each thematic point
        for point in self.thematic_points:
            logger.info(f"📋 Evaluating {point.id}: {point.title}")
            
            point_results = {
                "point_id": point.id,
                "point_title": point.title,
                "questions_evaluated": [],
                "dimension_scores": {},
                "total_score": 0.0
            }
            
            # Apply all 30 questions to this thematic point
            for question in self.base_questions:
                
                # Parametrize question for this thematic point
                parametrized_question = question.template.replace("{PUNTO_TEMATICO}", point.title)
                
                # Generate unique ID for this question-point combination
                question_point_id = f"{point.id}-{question.id}"
                
                # Execute evaluation (placeholder - will be implemented with actual search logic)
                evaluation_result = self._evaluate_single_question(
                    question=question,
                    thematic_point=point,
                    pdm_document=pdm_document,
                    parametrized_question=parametrized_question
                )
                
                evaluation_result.question_id = question_point_id
                evaluation_result.point_code = point.id
                evaluation_result.point_title = point.title
                evaluation_result.prompt = parametrized_question
                
                point_results["questions_evaluated"].append(evaluation_result)
                results["evaluation_matrix"].append(evaluation_result)
                
                evaluation_count += 1
                
                logger.debug(f"  ✓ {question_point_id}: Score {evaluation_result.score}/{evaluation_result.max_score}")
            
            # Calculate dimension scores for this point
            for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
                dim_questions = [r for r in point_results["questions_evaluated"] if r.dimension == dim]
                dim_score = sum(q.score for q in dim_questions) / len(dim_questions) if dim_questions else 0.0
                point_results["dimension_scores"][dim] = dim_score
            
            # Calculate total score for this point
            point_results["total_score"] = sum(point_results["dimension_scores"].values()) / 6
            
            results["thematic_points"].append(point_results)
            
            logger.info(f"  ✅ {point.id} completed: Score {point_results['total_score']:.2f}/3.0")
        
        # Final validation
        if evaluation_count != self.structure.TOTAL_QUESTIONS:
            raise RuntimeError(f"CRITICAL: Expected {self.structure.TOTAL_QUESTIONS} evaluations, executed {evaluation_count}")
        
        # Calculate global summary
        results["summary"] = self._calculate_global_summary(results)
        
        end_time = datetime.now()
        results["metadata"]["processing_time_seconds"] = (end_time - start_time).total_seconds()
        
        logger.info(f"🎉 EVALUATION COMPLETE: {evaluation_count} questions evaluated across {len(self.thematic_points)} points")
        logger.info(f"📊 Global Score: {results['summary']['global_score']:.2f}/3.0")
        
        return results
    
    def _evaluate_single_question(self, question: BaseQuestion, thematic_point: ThematicPoint,
                                 pdm_document: Union[str, Path], parametrized_question: str) -> EvaluationResult:
        """
        Evaluate a single question for a single thematic point
        
        NOTE: This is a placeholder implementation. The actual implementation should include:
        - PDF text extraction
        - Pattern matching using the search_patterns
        - Evidence extraction
        - Score calculation using scoring_rule
        """
        
        # PLACEHOLDER IMPLEMENTATION
        # TODO: Replace with actual PDF processing and pattern matching
        
        return EvaluationResult(
            question_id="",  # Will be set by caller
            point_code="",  # Will be set by caller  
            point_title="", # Will be set by caller
            dimension=question.dimension,
            question_no=question.question_no,
            prompt="",      # Will be set by caller
            score=1.5,      # Placeholder score
            max_score=question.max_score,
            elements_found={"placeholder": True},
            evidence=[{"texto": "Placeholder evidence", "ubicacion": "page 1", "confianza": 0.8}],
            missing_elements=["placeholder"],
            recommendation="Placeholder recommendation"
        )
    
    def _calculate_global_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate global summary statistics"""
        
        all_scores = [q.score for q in results["evaluation_matrix"]]
        dimension_averages = {}
        
        for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
            dim_scores = [q.score for q in results["evaluation_matrix"] if q.dimension == dim]
            dimension_averages[dim] = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0
        
        global_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "global_score": global_score,
            "total_evaluations": len(results["evaluation_matrix"]),
            "dimension_averages": dimension_averages,
            "score_distribution": {
                "excellent": len([s for s in all_scores if s >= 2.5]),
                "good": len([s for s in all_scores if 2.0 <= s < 2.5]),
                "satisfactory": len([s for s in all_scores if 1.5 <= s < 2.0]),  
                "insufficient": len([s for s in all_scores if 0.5 <= s < 1.5]),
                "deficient": len([s for s in all_scores if s < 0.5])
            },
            "validation_passed": len(results["evaluation_matrix"]) == 300
        }
    
    def validate_execution(self, results: Dict[str, Any]) -> bool:
        """Validate that execution followed exact 30×10 structure"""
        
        checks = {
            "total_evaluations": len(results.get("evaluation_matrix", [])) == 300,
            "thematic_points": len(results.get("thematic_points", [])) == 10,
            "questions_per_point": all(
                len(p.get("questions_evaluated", [])) == 30 
                for p in results.get("thematic_points", [])
            ),
            "unique_question_ids": len(set(
                q.question_id for q in results.get("evaluation_matrix", [])
            )) == 300
        }
        
        all_passed = all(checks.values())
        
        logger.info(f"🔍 VALIDATION RESULTS:")
        for check, passed in checks.items():
            logger.info(f"  {'✅' if passed else '❌'} {check}: {passed}")
        
        return all_passed

# Global singleton instance
_questionnaire_engine = None

def get_questionnaire_engine() -> QuestionnaireEngine:
    """Get the global questionnaire engine singleton"""
    global _questionnaire_engine
    if _questionnaire_engine is None:
        _questionnaire_engine = QuestionnaireEngine()
    return _questionnaire_engine

if __name__ == "__main__":
    # Validation test
    engine = QuestionnaireEngine()
    logger.info("🎯 QuestionnaireEngine ready for 300-question evaluation")
    logger.info(f"📊 Structure: {engine.structure.DOMAINS} points × {engine.structure.QUESTIONS_PER_DOMAIN} questions = {engine.structure.TOTAL_QUESTIONS} total")