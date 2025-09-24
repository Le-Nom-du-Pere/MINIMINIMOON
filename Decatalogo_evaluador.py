#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decatalogo_evaluator_full.py
Versión: 1.0 — Evaluador Industrial Completo del Decálogo de Derechos Humanos
Propósito: Evaluar 10 puntos del Decálogo en 4 dimensiones (DE-1 a DE-4) con scoring riguroso, evidencia textual y alineación temática estricta.
Conexión Garantizada: Se integra PERFECTAMENTE con el sistema industrial principal (Decatalogo_principal.py).
Autor: Dr. en Políticas Públicas (Extensión)
Enfoque: Calidad del dato de entrada para garantizar la robustez del análisis causal de alto nivel.
"""

import logging

import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import sys

assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

# Importar componentes del sistema industrial principal
# from sistema_industrial.componentes import ...

from Decatalogo_principal import (
    DimensionDecalogo,
    SistemaEvaluacionIndustrial,
    ResultadoDimensionIndustrial,
    EvaluacionCausalIndustrial,
    DecalogoContext,
    ClusterMetadata,
    obtener_decalogo_contexto,
)

from feasibility_scorer import (
    FeasibilityScorer,
    ComponentType,
    IndicatorScore,
    DetectionResult,
)
from responsibility_detector import ResponsibilityDetector, ResponsibilityEntity

LOGGER = logging.getLogger("DecatalogoEvaluatorFull")

DECALOGO_CONTEXT: DecalogoContext = obtener_decalogo_contexto()


@dataclass
class AnalisisEvidenciaDecalogo:
    """Resultados estandarizados del análisis automático de evidencia."""

    indicador_scores: List[IndicatorScore]
    detecciones_por_tipo: Dict[ComponentType, List[DetectionResult]]
    responsabilidades: List[ResponsibilityEntity]
    recursos: int
    plazos: int
    riesgos: int

    @property
    def max_score(self) -> float:
        return max((score.feasibility_score for score in self.indicador_scores), default=0.0)

    def detecciones(self, componente: ComponentType) -> List[DetectionResult]:
        return self.detecciones_por_tipo.get(componente, [])


# ==================== EVALUADOR INDUSTRIAL COMPLETO ====================

@dataclass
class EvaluacionPregunta:
    """Evaluación de una pregunta específica del cuestionario."""
    pregunta_id: str
    dimension: str
    punto_id: int
    respuesta: str  # Sí / Parcial / No / NI
    evidencia_textual: str
    evidencia_contraria: str
    puntaje: float

@dataclass
class EvaluacionDimensionPunto:
    """Evaluación de una dimensión para un punto específico del Decálogo."""
    punto_id: int
    dimension: str
    evaluaciones_preguntas: List[EvaluacionPregunta]
    puntaje_dimension: float
    matriz_causal: Optional[Dict[str, Any]] = None

@dataclass
class EvaluacionPuntoCompleto:
    """Evaluación completa de un punto del Decálogo en las 4 dimensiones."""
    punto_id: int
    nombre_punto: str
    evaluaciones_dimensiones: List[EvaluacionDimensionPunto]
    puntaje_agregado_punto: float

@dataclass
class EvaluacionClusterCompleto:
    """Evaluación completa de un cluster del Decálogo."""
    cluster_nombre: str
    evaluaciones_puntos: List[EvaluacionPuntoCompleto]
    puntaje_agregado_cluster: float
    clasificacion_cualitativa: str

@dataclass
class ReporteFinalDecatalogo:
    """Reporte final completo del análisis del Decálogo."""
    metadata: Dict[str, Any]
    resumen_ejecutivo: Dict[str, Any]
    reporte_macro: Dict[str, Any]
    reporte_meso_por_cluster: List[EvaluacionClusterCompleto]
    reporte_por_punto: List[EvaluacionPuntoCompleto]
    reporte_por_pregunta: List[EvaluacionPregunta]

class IndustrialDecatalogoEvaluatorFull:
    """Evaluador del decálogo apoyado en detectores especializados."""

    def __init__(self, contexto: Optional[DecalogoContext] = None):
        self.contexto = contexto or DECALOGO_CONTEXT
        self.scorer = FeasibilityScorer(enable_parallel=False)
        self.responsibility_detector = ResponsibilityDetector()
        self.preguntas_de1 = self._definir_preguntas_de1()
        self.preguntas_de2 = self._definir_preguntas_de2()
        self.preguntas_de3 = self._definir_preguntas_de3()
        self.preguntas_de4 = self._definir_preguntas_de4()

    @staticmethod
    def _definir_preguntas_de1() -> Dict[str, str]:
        return {
            "Q1": "¿El PDT define productos medibles alineados con la prioridad?",
            "Q2": "¿Las metas de producto incluyen responsable institucional?",
            "Q3": "¿Formula resultados medibles con línea base y meta al 2027?",
            "Q4": "¿Resultados y productos están lógicamente vinculados según la cadena de valor?",
            "Q5": "¿El impacto esperado está definido y alineado al Decálogo?",
            "Q6": "¿Existe una explicación explícita de la lógica de intervención completa?",
        }

    @staticmethod
    def _definir_preguntas_de2() -> Dict[str, str]:
        return {
            "D1": "Línea base 2023 con fuente citada",
            "D2": "Serie histórica ≥ 5 años",
            "O1": "Objetivo específico alineado con transformaciones del PND",
            "O2": "Indicador de resultado con línea base y meta transformadora",
            "T1": "Proyecto codificado en BPIN o código interno",
            "T2": "Monto plurianual 2024-2027",
            "S1": "Indicadores de producto y resultado en SUIFP-T",
            "S2": "Periodicidad de reporte especificada",
        }

    @staticmethod
    def _definir_preguntas_de3() -> Dict[str, str]:
        return {
            "G1": "¿Existe identificación de fuentes de financiación diversificadas?",
            "G2": "¿Se presenta distribución presupuestal anualizada?",
            "A1": "¿Los montos son coherentes con la ambición de las metas?",
            "A2": "¿Hay estrategia de gestión de recursos adicionales?",
            "R1": "¿Los recursos están trazados en el PPI con códigos?",
            "R2": "¿Se identifica necesidad de fortalecer capacidades?",
            "S1": "¿El presupuesto está alineado con el plan indicativo?",
            "S2": "¿Existe plan de contingencia presupuestal?",
        }

    @staticmethod
    def _definir_preguntas_de4() -> List[str]:
        return [
            "Diagnóstico con línea base y brechas claras",
            "Causalidad explícita entre productos, resultados e impacto",
            "Metas formuladas con claridad y ambición transformadora",
            "Programas o acciones detalladas con responsable y presupuesto",
            "Territorialización de las intervenciones (geográfica o sectorial)",
            "Vinculación institucional (articulación con sectores o niveles)",
            "Seguimiento con indicadores y calendario definido",
            "Proyección de impacto o beneficio con alineación al Decálogo",
        ]

    @staticmethod
    def _extraer_texto_entry(entry: Any) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for key in ("texto", "texto_evidencia", "segmento", "contenido"):
                valor = entry.get(key)
                if valor:
                    return str(valor)
        return ""

    def _extraer_textos(self, evidencia: Dict[str, List[Any]], *keys: str) -> List[str]:
        textos: List[str] = []
        for key in keys:
            for entry in evidencia.get(key, []):
                texto = self._extraer_texto_entry(entry)
                if texto:
                    textos.append(texto)
        return textos

    def _obtener_muestra(self, evidencia: Dict[str, List[Any]], key: str) -> str:
        textos = self._extraer_textos(evidencia, key)
        return textos[0] if textos else ""

    def _analizar_evidencia(self, evidencia: Dict[str, List[Any]]) -> AnalisisEvidenciaDecalogo:
        textos_indicadores = self._extraer_textos(evidencia, "indicadores", "metas")
        indicador_scores = [self.scorer.calculate_feasibility_score(texto) for texto in textos_indicadores]

        detecciones_por_tipo: Dict[ComponentType, List[DetectionResult]] = {component: [] for component in ComponentType}
        for score in indicador_scores:
            for deteccion in score.detailed_matches:
                detecciones_por_tipo[deteccion.component_type].append(deteccion)

        responsabilidades: List[ResponsibilityEntity] = []
        for texto in self._extraer_textos(evidencia, "responsables"):
            responsabilidades.extend(self.responsibility_detector.detect_entities(texto))

        detecciones_por_tipo = {k: v for k, v in detecciones_por_tipo.items() if v}

        return AnalisisEvidenciaDecalogo(
            indicador_scores=indicador_scores,
            detecciones_por_tipo=detecciones_por_tipo,
            responsabilidades=responsabilidades,
            recursos=len(evidencia.get("recursos", [])),
            plazos=len(evidencia.get("plazos", [])),
            riesgos=len(evidencia.get("riesgos", [])),
        )

from typing import List, Optional

class Evaluador:
    @staticmethod
    def _seleccionar_mejor_deteccion(detecciones: List["DetectionResult"]) -> Optional["DetectionResult"]:
        return max(detecciones, key=lambda d: d.confidence, default=None)

    @staticmethod
    def _seleccionar_mejor_responsable(responsables: List["ResponsibilityEntity"]) -> Optional["ResponsibilityEntity"]:
        return max(responsables, key=lambda r: r.confidence, default=None)

    @staticmethod
    def _formatear_deteccion(deteccion: Optional["DetectionResult"]) -> str:
        if not deteccion:
            return ""
        return f"{deteccion.matched_text} (conf. {deteccion.confidence:.2f})"

    @staticmethod
    def _formatear_responsable(responsable: Optional["ResponsibilityEntity"]) -> str:
        if not responsable:
            return ""
        rol = responsable.role or responsable.entity_type.value
        return f"{responsable.text} ({rol}, conf. {responsable.confidence:.2f})"

    @staticmethod
    def _valor_a_respuesta(valor: float) -> str:
        if valor >= 0.75:
            return "Sí"
        if valor >= 0.4:
            return "Parcial"
        return "No"

    @staticmethod
    def _clamp(valor: float) -> float:
        return max(0.0, min(1.0, valor))

    def _crear_evaluacion(
        self,
        pregunta_id: str,
        dimension: str,
        punto_id: int,
        valor: float,
        evidencia: str,
        descripcion: Optional[str] = None,
    ) -> "EvaluacionPregunta":
        valor = self._clamp(valor)
        respuesta = self._valor_a_respuesta(valor)

        # Construcción de evidencia textual priorizando descripción cuando aporta contexto.
        if descripcion and evidencia:
            evidencia_textual = f"{descripcion} → {evidencia}"
        elif descripcion and valor > 0:
            evidencia_textual = descripcion
        else:
            evidencia_textual = evidencia or ""

        # Si el valor es 0, consignar explícitamente ausencia de evidencia a favor.
        evidencia_contraria = (
            "" if valor > 0
            else "No se identificaron elementos que respondan a la pregunta con la evidencia disponible."
        )

        # Ensamblado final del objeto dominio
        return EvaluacionPregunta(
            pregunta_id=pregunta_id,
            dimension=dimension,
            punto_id=punto_id,
            valor=valor,
            respuesta=respuesta,
            evidencia=evidencia_textual,
            evidencia_contraria=evidencia_contraria,
        )

        return EvaluacionPregunta(
            pregunta_id=pregunta_id,
            dimension=dimension,
            punto_id=punto_id,
            respuesta=respuesta,
            evidencia_textual=evidencia_textual,
            evidencia_contraria=evidencia_contraria,
            puntaje=valor,
        )

    def evaluar_dimension_de1(
        self, analisis: AnalisisEvidenciaDecalogo, evidencia: Dict[str, List[Any]], punto_id: int
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        max_score = analisis.max_score
        baseline_dets = analisis.detecciones(ComponentType.BASELINE)
        target_dets = analisis.detecciones(ComponentType.TARGET)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)
        date_dets = analisis.detecciones(ComponentType.DATE)

        mejor_numerico = self._seleccionar_mejor_deteccion(numerical_dets)
        evaluaciones.append(
            self._crear_evaluacion(
                "Q1",
                "DE-1",
                punto_id,
                max_score if mejor_numerico else 0.0,
                self._formatear_deteccion(mejor_numerico),
                self.preguntas_de1["Q1"],
            )
        )

        mejor_responsable = self._seleccionar_mejor_responsable(analisis.responsabilidades)
        valor_q2 = self._clamp(0.6 + 0.4 * max_score) if mejor_responsable else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "Q2",
                "DE-1",
                punto_id,
                valor_q2,
                self._formatear_responsable(mejor_responsable),
                self.preguntas_de1["Q2"],
            )
        )

        indicadores_cobertura = [bool(baseline_dets), bool(target_dets), bool(timeframe_dets)]
        cobertura = sum(indicadores_cobertura) / len(indicadores_cobertura) if indicadores_cobertura else 0.0
        evidencia_q3 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(self._seleccionar_mejor_deteccion(baseline_dets)),
                    self._formatear_deteccion(self._seleccionar_mejor_deteccion(target_dets)),
                    self._formatear_deteccion(self._seleccionar_mejor_deteccion(timeframe_dets)),
                ],
            )
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "Q3",
                "DE-1",
                punto_id,
                max_score * cobertura,
                evidencia_q3,
                self.preguntas_de1["Q3"],
            )
        )

        factor_responsable = 1.0 if mejor_responsable else 0.0
        factor_recursos = 1.0 if analisis.recursos else 0.0
        valor_q4 = self._clamp(0.4 * cobertura + 0.3 * max_score + 0.2 * factor_responsable + 0.1 * factor_recursos)
        evidencia_q4 = "; ".join(filter(None, [evidencia_q3, self._formatear_responsable(mejor_responsable)]))
        evaluaciones.append(
            self._crear_evaluacion(
                "Q4",
                "DE-1",
                punto_id,
                valor_q4,
                evidencia_q4,
                self.preguntas_de1["Q4"],
            )
        )

        mejor_impacto = self._seleccionar_mejor_deteccion(target_dets + date_dets)
        valor_q5 = self._clamp(max_score if mejor_impacto else (max_score * 0.5 if target_dets else 0.0))
        evaluaciones.append(
            self._crear_evaluacion(
                "Q5",
                "DE-1",
                punto_id,
                valor_q5,
                self._formatear_deteccion(mejor_impacto),
                self.preguntas_de1["Q5"],
            )
        )

        valor_q6 = self._clamp((valor_q4 + max_score + cobertura) / 3)
        evaluaciones.append(
            self._crear_evaluacion(
                "Q6",
                "DE-1",
                punto_id,
                valor_q6,
                evidencia_q4 or evidencia_q3,
                self.preguntas_de1["Q6"],
            )
        )

        puntaje_base = sum(e.puntaje for e in evaluaciones) / len(evaluaciones) if evaluaciones else 0.0
        matriz_causal = {
            "C1": "Sí" if baseline_dets else "No",
            "C2": "Sí" if target_dets else "No",
            "C3": "Sí" if timeframe_dets else "No",
            "C4": "Sí" if mejor_responsable else "No",
            "Puntaje_Causalidad_Total": round(puntaje_base * 100, 2),
            "Factor_Causal": round(0.5 + (cobertura / 2), 2),
        }
        puntaje_final = self._clamp(puntaje_base * matriz_causal["Factor_Causal"])

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-1",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_final * 100,
            matriz_causal=matriz_causal,
        )

    def evaluar_dimension_de2(
        self, analisis: AnalisisEvidenciaDecalogo, evidencia: Dict[str, List[Any]], punto_id: int
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        max_score = analisis.max_score
        baseline_dets = analisis.detecciones(ComponentType.BASELINE)
        target_dets = analisis.detecciones(ComponentType.TARGET)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)

        evaluaciones.append(
            self._crear_evaluacion(
                "D1",
                "DE-2",
                punto_id,
                max_score if baseline_dets else 0.0,
                self._formatear_deteccion(self._seleccionar_mejor_deteccion(baseline_dets)),
                self.preguntas_de2["D1"],
            )
        )

        evaluaciones.append(
            self._crear_evaluacion(
                "D2",
                "DE-2",
                punto_id,
                max_score * 0.6 if baseline_dets else 0.0,
                self._formatear_deteccion(self._seleccionar_mejor_deteccion(baseline_dets)),
                self.preguntas_de2["D2"],
            )
        )

        evaluaciones.append(
            self._crear_evaluacion(
                "O1",
                "DE-2",
                punto_id,
                max_score if target_dets else 0.0,
                self._formatear_deteccion(self._seleccionar_mejor_deteccion(target_dets)),
                self.preguntas_de2["O1"],
            )
        )

        cobertura_resultados = 0.5 * (1 if baseline_dets else 0) + 0.5 * (1 if target_dets else 0)
        evidencia_o2 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(self._seleccionar_mejor_deteccion(baseline_dets)),
                    self._formatear_deteccion(self._seleccionar_mejor_deteccion(target_dets)),
                ],
            )
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "O2",
                "DE-2",
                punto_id,
                max_score * cobertura_resultados,
                evidencia_o2,
                self.preguntas_de2["O2"],
            )
        )

        valor_t1 = self._clamp(0.4 + 0.1 * analisis.recursos) if analisis.recursos else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "T1",
                "DE-2",
                punto_id,
                valor_t1,
                self._obtener_muestra(evidencia, "recursos"),
                self.preguntas_de2["T1"],
            )
        )

        mejor_t2 = self._seleccionar_mejor_deteccion(numerical_dets + timeframe_dets)
        evaluaciones.append(
            self._crear_evaluacion(
                "T2",
                "DE-2",
                punto_id,
                max_score if mejor_t2 else 0.0,
                self._formatear_deteccion(mejor_t2),
                self.preguntas_de2["T2"],
            )
        )

        indicadores_count = len(evidencia.get("indicadores", []))
        valor_s1 = self._clamp(min(1.0, max_score + indicadores_count * 0.1))
        evaluaciones.append(
            self._crear_evaluacion(
                "S1",
                "DE-2",
                punto_id,
                valor_s1,
                self._obtener_muestra(evidencia, "indicadores"),
                self.preguntas_de2["S1"],
            )
        )

        mejor_s2 = self._seleccionar_mejor_deteccion(timeframe_dets)
        evidencia_plazos = self._formatear_deteccion(mejor_s2) or self._obtener_muestra(evidencia, "plazos")
        valor_s2 = max_score if (timeframe_dets or analisis.plazos) else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "S2",
                "DE-2",
                punto_id,
                valor_s2,
                evidencia_plazos,
                self.preguntas_de2["S2"],
            )
        )

        puntaje_dimension = sum(e.puntaje for e in evaluaciones) / len(evaluaciones) if evaluaciones else 0.0
        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-2",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension * 100,
        )

    def evaluar_dimension_de3(
        self, analisis: AnalisisEvidenciaDecalogo, evidencia: Dict[str, List[Any]], punto_id: int
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        max_score = analisis.max_score
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        mejor_numerico = self._seleccionar_mejor_deteccion(numerical_dets)
        mejor_plazo = self._seleccionar_mejor_deteccion(timeframe_dets)
        mejor_responsable = self._seleccionar_mejor_responsable(analisis.responsabilidades)

        valor_g1 = self._clamp(0.5 + 0.1 * analisis.recursos) if analisis.recursos else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "G1",
                "DE-3",
                punto_id,
                valor_g1,
                self._obtener_muestra(evidencia, "recursos"),
                self.preguntas_de3["G1"],
            )
        )

        valor_g2 = self._clamp(0.6 if (analisis.plazos or mejor_plazo) else 0.0)
        evaluaciones.append(
            self._crear_evaluacion(
                "G2",
                "DE-3",
                punto_id,
                valor_g2,
                self._formatear_deteccion(mejor_plazo) or self._obtener_muestra(evidencia, "plazos"),
                self.preguntas_de3["G2"],
            )
        )

        valor_a1 = max_score if mejor_numerico else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "A1",
                "DE-3",
                punto_id,
                valor_a1,
                self._formatear_deteccion(mejor_numerico),
                self.preguntas_de3["A1"],
            )
        )

        valor_a2 = self._clamp(0.4 * (1 if analisis.recursos else 0) + 0.4 * (1 if mejor_responsable else 0) + 0.2 * max_score)
        evaluaciones.append(
            self._crear_evaluacion(
                "A2",
                "DE-3",
                punto_id,
                valor_a2,
                self._formatear_responsable(mejor_responsable),
                self.preguntas_de3["A2"],
            )
        )

        valor_r1 = self._clamp(0.5 + 0.1 * analisis.recursos) if analisis.recursos else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "R1",
                "DE-3",
                punto_id,
                valor_r1,
                self._obtener_muestra(evidencia, "recursos"),
                self.preguntas_de3["R1"],
            )
        )

        valor_r2 = self._clamp(0.7 if mejor_responsable else 0.0)
        evaluaciones.append(
            self._crear_evaluacion(
                "R2",
                "DE-3",
                punto_id,
                valor_r2,
                self._formatear_responsable(mejor_responsable),
                self.preguntas_de3["R2"],
            )
        )

        valor_s1 = self._clamp(0.4 * max_score + 0.3 * (1 if timeframe_dets else 0) + 0.3 * (1 if analisis.plazos else 0))
        evaluaciones.append(
            self._crear_evaluacion(
                "S1",
                "DE-3",
                punto_id,
                valor_s1,
                self._formatear_deteccion(mejor_plazo) or self._obtener_muestra(evidencia, "plazos"),
                self.preguntas_de3["S1"],
            )
        )

        valor_s2 = self._clamp(0.5 + 0.1 * analisis.riesgos) if analisis.riesgos else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "S2",
                "DE-3",
                punto_id,
                valor_s2,
                self._obtener_muestra(evidencia, "riesgos"),
                self.preguntas_de3["S2"],
            )
        )

        puntaje_dimension = sum(e.puntaje for e in evaluaciones) / len(evaluaciones) if evaluaciones else 0.0
        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-3",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension * 100,
        )

    def evaluar_dimension_de4(
        self, analisis: AnalisisEvidenciaDecalogo, evidencia: Dict[str, List[Any]], punto_id: int
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        baseline_dets = analisis.detecciones(ComponentType.BASELINE)
        target_dets = analisis.detecciones(ComponentType.TARGET)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)
        best_responsable = self._seleccionar_mejor_responsable(analisis.responsabilidades)

        cobertura_baseline = 1 if baseline_dets else 0
        cobertura_target = 1 if target_dets else 0
        cobertura_time = 1 if timeframe_dets or analisis.plazos else 0
        cobertura_responsables = 1 if best_responsable else 0
        cobertura_recursos = 1 if analisis.recursos else 0
        cobertura_numerica = 1 if numerical_dets else 0

        for idx, descripcion in enumerate(self.preguntas_de4, start=1):
            pregunta_id = f"DE4_{idx}"
            if idx == 1:
                valor = self._clamp(0.6 * analisis.max_score + 0.4 * cobertura_baseline)
                evidencia_texto = self._formatear_deteccion(self._seleccionar_mejor_deteccion(baseline_dets))
            elif idx == 2:
                valor = self._clamp((cobertura_baseline + cobertura_target + cobertura_time) / 3)
                evidencia_texto = " | ".join(
                    filter(
                        None,
                        [
                            self._formatear_deteccion(self._seleccionar_mejor_deteccion(baseline_dets)),
                            self._formatear_deteccion(self._seleccionar_mejor_deteccion(target_dets)),
                            self._formatear_deteccion(self._seleccionar_mejor_deteccion(timeframe_dets)),
                        ],
                    )
                )
            elif idx == 3:
                valor = self._clamp(0.5 * analisis.max_score + 0.5 * cobertura_target)
                evidencia_texto = self._formatear_deteccion(self._seleccionar_mejor_deteccion(target_dets))
            elif idx == 4:
                valor = self._clamp(0.4 * cobertura_responsables + 0.4 * cobertura_recursos + 0.2 * analisis.max_score)
                evidencia_texto = "; ".join(
                    filter(
                        None,
                        [
                            self._formatear_responsable(best_responsable),
                            self._obtener_muestra(evidencia, "recursos"),
                        ],
                    )
                )
            elif idx == 5:
                valor = self._clamp(0.5 * cobertura_time + 0.3 * cobertura_recursos + 0.2 * analisis.max_score)
                evidencia_texto = self._obtener_muestra(evidencia, "plazos")
            elif idx == 6:
                valor = self._clamp(0.7 if best_responsable else 0.0)
                evidencia_texto = self._formatear_responsable(best_responsable)
            elif idx == 7:
                valor = self._clamp(0.5 * cobertura_time + 0.3 * analisis.max_score + 0.2 * cobertura_numerica)
                evidencia_texto = self._formatear_deteccion(self._seleccionar_mejor_deteccion(timeframe_dets)) or self._obtener_muestra(evidencia, "indicadores")
            else:
                valor = self._clamp((cobertura_target + cobertura_time + cobertura_numerica) / 3)
                evidencia_texto = self._formatear_deteccion(self._seleccionar_mejor_deteccion(target_dets + numerical_dets))

            evaluaciones.append(
                self._crear_evaluacion(
                    pregunta_id,
                    "DE-4",
                    punto_id,
                    valor,
                    evidencia_texto,
                    descripcion,
                )
            )

        puntaje_dimension = sum(e.puntaje for e in evaluaciones) / len(evaluaciones) if evaluaciones else 0.0
        matriz_causal = {
            "componentes_detectados": {
                "baseline": bool(baseline_dets),
                "target": bool(target_dets),
                "plazos": bool(timeframe_dets or analisis.plazos),
                "responsables": bool(analisis.responsabilidades),
            },
            "indicadores_analizados": len(analisis.indicador_scores),
            "recursos_identificados": analisis.recursos,
        }
        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-4",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension * 100,
            matriz_causal=matriz_causal,
        )

    def evaluar_punto_completo(
        self, evidencia: Dict[str, List[Any]], punto_id: int
    ) -> Tuple[EvaluacionPuntoCompleto, AnalisisEvidenciaDecalogo]:
        analisis = self._analizar_evidencia(evidencia)
        evaluaciones_dimensiones = [
            self.evaluar_dimension_de1(analisis, evidencia, punto_id),
            self.evaluar_dimension_de2(analisis, evidencia, punto_id),
            self.evaluar_dimension_de3(analisis, evidencia, punto_id),
            self.evaluar_dimension_de4(analisis, evidencia, punto_id),
        ]
        puntaje_agregado = (
            sum(dim.puntaje_dimension for dim in evaluaciones_dimensiones) / len(evaluaciones_dimensiones)
            if evaluaciones_dimensiones
            else 0.0
        )
        dimension = self.contexto.dimensiones_por_id.get(punto_id)
        nombre_punto = dimension.nombre if dimension else f"Punto {punto_id}"
        return (
            EvaluacionPuntoCompleto(
                punto_id=punto_id,
                nombre_punto=nombre_punto,
                evaluaciones_dimensiones=evaluaciones_dimensiones,
                puntaje_agregado_punto=puntaje_agregado,
            ),
            analisis,
        )

    def evaluar_cluster_completo(
        self,
        evidencias_por_punto: Dict[int, Dict[str, List[Any]]],
        cluster: ClusterMetadata,
    ) -> EvaluacionClusterCompleto:
        evaluaciones_puntos: List[EvaluacionPuntoCompleto] = []
        for punto_id in cluster.puntos:
            evaluacion_punto, _ = self.evaluar_punto_completo(evidencias_por_punto.get(punto_id, {}), punto_id)
            evaluaciones_puntos.append(evaluacion_punto)

        puntaje_agregado = (
            sum(e.puntaje_agregado_punto for e in evaluaciones_puntos) / len(evaluaciones_puntos)
            if evaluaciones_puntos
            else 0.0
        )

        if puntaje_agregado >= 80:
            clasificacion = "Desempeño Óptimo"
        elif puntaje_agregado >= 60:
            clasificacion = "Desempeño Satisfactorio"
        elif puntaje_agregado >= 40:
            clasificacion = "Desempeño Básico"
        else:
            clasificacion = "Desempeño Insuficiente"

        return EvaluacionClusterCompleto(
            cluster_nombre=cluster.titulo,
            evaluaciones_puntos=evaluaciones_puntos,
            puntaje_agregado_cluster=puntaje_agregado,
            clasificacion_cualitativa=clasificacion,
        )

    def generar_reporte_final(
        self, evidencias_por_punto: Dict[int, Dict[str, List[Any]]], nombre_plan: str
    ) -> ReporteFinalDecatalogo:
        evaluaciones_clusters = [
            self.evaluar_cluster_completo(evidencias_por_punto, cluster)
            for cluster in self.contexto.clusters_por_id.values()
        ]
        evaluaciones_puntos = [
            eval_punto
            for eval_cluster in evaluaciones_clusters
            for eval_punto in eval_cluster.evaluaciones_puntos
        ]
        evaluaciones_preguntas = [
            eval_pregunta
            for eval_punto in evaluaciones_puntos
            for eval_dim in eval_punto.evaluaciones_dimensiones
            for eval_pregunta in eval_dim.evaluaciones_preguntas
        ]

        puntajes_clusters = [cluster.puntaje_agregado_cluster for cluster in evaluaciones_clusters]
        puntaje_global = sum(puntajes_clusters) / len(puntajes_clusters) if puntajes_clusters else 0.0

        cluster_mejor = (
            max(evaluaciones_clusters, key=lambda x: x.puntaje_agregado_cluster).cluster_nombre
            if evaluaciones_clusters
            else ""
        )
        cluster_peor = (
            min(evaluaciones_clusters, key=lambda x: x.puntaje_agregado_cluster).cluster_nombre
            if evaluaciones_clusters
            else ""
        )

        resumen_ejecutivo = {
            "puntaje_global": puntaje_global,
            "cluster_mejor_desempeno": cluster_mejor,
            "cluster_peor_desempeno": cluster_peor,
            "numero_puntos_evaluados": len(evaluaciones_puntos),
            "recomendacion_estrategica_global": "IMPLEMENTACIÓN RECOMENDADA" if puntaje_global >= 70 else "REDISEÑO PARCIAL REQUERIDO",
        }

        if puntaje_global >= 75:
            alineacion = "ALTA"
        elif puntaje_global >= 50:
            alineacion = "MEDIA"
        else:
            alineacion = "BAJA"

        reporte_macro = {
            "alineacion_global_decatalogo": alineacion,
            "explicacion_extensa_cualitativa": (
                "El Plan de Desarrollo Territorial presenta un desempeño "
                f"{'óptimo' if puntaje_global >= 80 else 'satisfactorio' if puntaje_global >= 60 else 'básico' if puntaje_global >= 40 else 'insuficiente'} "
                "en su alineación con el Decálogo de Derechos Humanos."
            ),
        }

        return ReporteFinalDecatalogo(
            metadata={
                "nombre_plan": nombre_plan,
                "fecha_evaluacion": datetime.now().isoformat(),
                "version_evaluador": "1.0-industrial-full",
            },
            resumen_ejecutivo=resumen_ejecutivo,
            reporte_macro=reporte_macro,
            reporte_meso_por_cluster=evaluaciones_clusters,
            reporte_por_punto=evaluaciones_puntos,
            reporte_por_pregunta=evaluaciones_preguntas,
        )
def integrar_evaluador_decatalogo(
    sistema: SistemaEvaluacionIndustrial, dimension: DimensionDecalogo
) -> Optional[ResultadoDimensionIndustrial]:
    """Integra el evaluador especializado del decálogo con la evaluación industrial."""

    if not sistema.extractor:
        raise ValueError("Extractor no inicializado - Error industrial crítico")

    try:
        evaluador = IndustrialDecatalogoEvaluatorFull()
        evidencia_dimension = sistema.extractor.extraer_variables_operativas(dimension)
        matriz_trazabilidad = sistema.extractor.generar_matriz_trazabilidad(dimension)
        cluster_metadata = DECALOGO_CONTEXT.cluster_por_dimension.get(dimension.id)

        if not cluster_metadata:
            LOGGER.warning(
                "⚠️  [DECÁLOGO] Dimensión %s no se encuentra asociada a un cluster definido", dimension.id
            )
            return None

        evaluacion_punto, analisis = evaluador.evaluar_punto_completo(evidencia_dimension, dimension.id)
        puntajes_dim = {
            dim_eval.dimension: dim_eval.puntaje_dimension for dim_eval in evaluacion_punto.evaluaciones_dimensiones
        }

        consistencia_logica = puntajes_dim.get("DE-1", 0.0) / 100
        factibilidad_operativa = puntajes_dim.get("DE-3", 0.0) / 100
        robustez_causal = puntajes_dim.get("DE-4", 0.0) / 100
        identificabilidad_causal = min(1.0, (consistencia_logica + robustez_causal) / 2)
        certeza_probabilistica = analisis.max_score if analisis.indicador_scores else consistencia_logica

        riesgos = [
            entry.get("texto")
            for entry in evidencia_dimension.get("riesgos", [])
            if isinstance(entry, dict) and entry.get("texto")
        ]
        if not riesgos and factibilidad_operativa < 0.5:
            riesgos.append("Planificación presupuestal con evidencia insuficiente según DE-3.")

        evaluacion_causal = EvaluacionCausalIndustrial(
            consistencia_logica=max(0.0, min(1.0, consistencia_logica)),
            identificabilidad_causal=max(0.0, min(1.0, identificabilidad_causal)),
            factibilidad_operativa=max(0.0, min(1.0, factibilidad_operativa)),
            certeza_probabilistica=max(0.0, min(1.0, certeza_probabilistica)),
            robustez_causal=max(0.0, min(1.0, robustez_causal)),
            riesgos_implementacion=riesgos,
            supuestos_criticos=[],
            evidencia_soporte=len(analisis.indicador_scores),
            brechas_criticas=sum(1 for score in puntajes_dim.values() if score < 60),
        )

        brechas_identificadas = [
            f"{eval_dim.dimension} presenta puntaje {eval_dim.puntaje_dimension:.1f}/100; ampliar respaldo operativo."
            for eval_dim in evaluacion_punto.evaluaciones_dimensiones
            if eval_dim.puntaje_dimension < 60
        ]

        recomendaciones = [
            f"Fortalecer evidencia trazable para {cluster_metadata.titulo} con base en los eslabones analizados."
        ]
        if factibilidad_operativa < 0.6:
            recomendaciones.append("Reforzar programación financiera y responsabilidades operativas detectadas.")

        resultado = ResultadoDimensionIndustrial(
            dimension=dimension,
            evaluacion_causal=evaluacion_causal,
            evidencia=evidencia_dimension,
            brechas_identificadas=brechas_identificadas,
            recomendaciones=recomendaciones,
            matriz_trazabilidad=matriz_trazabilidad,
            timestamp_evaluacion=datetime.now().isoformat(),
        )

        LOGGER.info(
            "✅ Evaluación decálogo integrada para dimensión %s (cluster %s): %.1f/100",
            dimension.id,
            cluster_metadata.titulo,
            resultado.puntaje_final,
        )

        return resultado

    except Exception as exc:
        LOGGER.error(f"❌ Error en integración del evaluador del decálogo: {exc}")
        return None
