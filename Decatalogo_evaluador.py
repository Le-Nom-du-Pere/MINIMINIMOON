#!/usr/bin/env python3
# coding=utf-8
# -*- coding: utf-8 -*-
"""
decatalogo_evaluator_full.py
Versión: 1.0 — Evaluador Industrial Completo del Decálogo de Derechos Humanos
Propósito: Evaluar 10 puntos del Decálogo en 4 dimensiones (DE-1 a DE-4) con scoring riguroso, evidencia textual y alineación temática estricta.
Conexión Garantizada: Se integra PERFECTAMENTE con el sistema industrial principal (Decatalogo_principal.py).
Autor: Dr. en Políticas Públicas (Extensión)
Enfoque: Calidad del dato de entrada para garantizar la robustez del análisis causal de alto nivel.
"""

import hashlib
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from Decatalogo_principal import (
    ClusterMetadata,
    DecalogoContext,
    DimensionDecalogo,
    EvaluacionCausalIndustrial,
    ResultadoDimensionIndustrial,
    SistemaEvaluacionIndustrial,
    obtener_decalogo_contexto,
)
from feasibility_scorer import (
    ComponentType,
    DetectionResult,
    FeasibilityScorer,
    IndicatorScore,
)
from responsibility_detector import ResponsibilityDetector, ResponsibilityEntity

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

LOGGER = logging.getLogger("DecatalogoEvaluatorFull")

DECALOGO_CONTEXT: DecalogoContext = obtener_decalogo_contexto()


@dataclass
class AnalisisEvidenciaDecalogo:
    """Resultados estandarizados del análisis automático de evidencia."""

    indicador_scores: List[IndicatorScore]
    indicadores_evaluados: List[Tuple[str, IndicatorScore]]
    detecciones_por_tipo: Dict[ComponentType, List[DetectionResult]]
    responsabilidades: List[ResponsibilityEntity]
    recursos: int
    plazos: int
    riesgos: int

    @property
    def max_score(self) -> float:
        return max(
            (score.feasibility_score for score in self.indicador_scores), default=0.0
        )

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
    resultados_dimension: List[ResultadoDimensionIndustrial]
    reporte_meso_por_cluster: List[EvaluacionClusterCompleto]
    reporte_por_punto: List[EvaluacionPuntoCompleto]
    reporte_por_pregunta: List[EvaluacionPregunta]
    anexos_serializables: Dict[str, Any] = field(default_factory=dict)


class _EvaluadorBase:
    """Utilidades compartidas para construir evaluaciones del decálogo."""

    @staticmethod
    def _seleccionar_mejor_deteccion(
        detecciones: List["DetectionResult"],
    ) -> Optional["DetectionResult"]:
        return max(detecciones, key=lambda d: d.confidence, default=None)

    @staticmethod
    def _seleccionar_mejor_responsable(
        responsables: List["ResponsibilityEntity"],
    ) -> Optional["ResponsibilityEntity"]:
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

        if descripcion and evidencia:
            evidencia_textual = f"{descripcion} → {evidencia}"
        elif descripcion and valor > 0:
            evidencia_textual = descripcion
        else:
            evidencia_textual = evidencia or ""

        evidencia_contraria = (
            ""
            if valor > 0
            else "No se identificaron elementos que respondan a la pregunta con la evidencia disponible."
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


class IndustrialDecatalogoEvaluatorFull(_EvaluadorBase):
    """Evaluador del decálogo apoyado en detectores especializados."""

    def __init__(self, contexto: Optional[DecalogoContext] = None):
        self.contexto = contexto or DECALOGO_CONTEXT
        self.scorer = FeasibilityScorer(enable_parallel=True)
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

    def _analizar_evidencia(
        self, evidencia: Dict[str, List[Any]]
    ) -> AnalisisEvidenciaDecalogo:
        textos_indicadores = self._extraer_textos(evidencia, "indicadores", "metas")
        if textos_indicadores:
            indicador_scores = self.scorer.batch_score(
                textos_indicadores, use_parallel=True
            )
        else:
            indicador_scores = []

        indicadores_evaluados = list(zip(textos_indicadores, indicador_scores))

        detecciones_por_tipo: Dict[ComponentType, List[DetectionResult]] = {
            component: [] for component in ComponentType
        }
        for _, score in indicadores_evaluados:
            for deteccion in score.detailed_matches:
                detecciones_por_tipo[deteccion.component_type].append(
                    deteccion)

        responsabilidades: List[ResponsibilityEntity] = []
        for texto in self._extraer_textos(evidencia, "responsables"):
            responsabilidades.extend(
                self.responsibility_detector.detect_entities(texto)
            )

        detecciones_por_tipo = {k: v for k,
                                v in detecciones_por_tipo.items() if v}

        return AnalisisEvidenciaDecalogo(
            indicador_scores=indicador_scores,
            indicadores_evaluados=indicadores_evaluados,
            detecciones_por_tipo=detecciones_por_tipo,
            responsabilidades=responsabilidades,
            recursos=len(evidencia.get("recursos", [])),
            plazos=len(evidencia.get("plazos", [])),
            riesgos=len(evidencia.get("riesgos", [])),
        )

    def _obtener_dimension(self, punto_id: int) -> Optional[DimensionDecalogo]:
        return self.contexto.dimensiones_por_id.get(punto_id)

    def _calcular_metricas_dimension(
        self, punto_id: int, analisis: AnalisisEvidenciaDecalogo
    ) -> Optional[Dict[str, Any]]:
        dimension = self._obtener_dimension(punto_id)
        if not dimension:
            return None

        riesgo_matriz = dimension.generar_matriz_riesgos()
        eslabon_metrics: Dict[str, Dict[str, Any]] = {}
        cobertura_por_tipo: Dict[str, List[float]] = defaultdict(list)
        cobertura_ponderada = 0.0
        ponderacion_total = 0.0
        total_matches = 0
        baseline_matches = 0
        target_matches = 0
        time_matches = 0
        numerical_matches = 0

        for eslabon in dimension.eslabones:
            indicadores_norm = [indicador.lower() for indicador in eslabon.indicadores]
            matches: List[IndicatorScore] = [
                score
                for texto, score in analisis.indicadores_evaluados
                if any(indicador in texto.lower() for indicador in indicadores_norm)
            ]

            match_count = len(matches)
            total_matches += match_count
            baseline_matches += sum(
                1
                for score in matches
                if ComponentType.BASELINE in score.components_detected
            )
            target_matches += sum(
                1
                for score in matches
                if ComponentType.TARGET in score.components_detected
            )
            time_matches += sum(
                1
                for score in matches
                if ComponentType.TIME_HORIZON in score.components_detected
            )
            numerical_matches += sum(
                1
                for score in matches
                if ComponentType.NUMERICAL in score.components_detected
            )

            calidad_promedio = (
                sum(score.feasibility_score for score in matches) / match_count
                if match_count
                else 0.0
            )
            cobertura_ratio = match_count / max(1, len(eslabon.indicadores))
            cobertura = self._clamp(0.5 * calidad_promedio + 0.5 * cobertura_ratio)

            ponderacion = eslabon.kpi_ponderacion
            ponderacion_total += ponderacion
            cobertura_ponderada += cobertura * ponderacion
            cobertura_por_tipo[eslabon.tipo.name].append(cobertura)

            eslabon_metrics[eslabon.id] = {
                "tipo": eslabon.tipo.name,
                "cobertura": cobertura,
                "calidad_promedio": calidad_promedio,
                "indicadores_enlazados": match_count,
                "riesgos": riesgo_matriz.get(eslabon.id, []),
                "kpi": ponderacion,
                "lead_time": eslabon.calcular_lead_time(),
            }

        cobertura_ponderada_final = (
            cobertura_ponderada / ponderacion_total if ponderacion_total else 0.0
        )
        cobertura_tipo_promedio = {
            tipo: (sum(valores) / len(valores) if valores else 0.0)
            for tipo, valores in cobertura_por_tipo.items()
        }

        riesgo_total = sum(len(riesgos) for riesgos in riesgo_matriz.values())
        riesgo_promedio = riesgo_total / max(1, len(dimension.eslabones))
        riesgo_factor = max(0.55, 1.0 - 0.15 * riesgo_promedio)

        riesgo_indicadores = sum(
            1
            for riesgos in riesgo_matriz.values()
            if any("indicadores" in r.lower() for r in riesgos)
        )
        riesgo_temporal = sum(
            1
            for riesgos in riesgo_matriz.values()
            if any("temporal" in r.lower() for r in riesgos)
        )
        riesgo_capacidades = sum(
            1
            for riesgos in riesgo_matriz.values()
            if any("capacidades" in r.lower() for r in riesgos)
        )
        divisor_riesgos = max(1, len(dimension.eslabones))
        riesgo_breakdown = {
            "indicadores": min(1.0, riesgo_indicadores / divisor_riesgos),
            "temporal": min(1.0, riesgo_temporal / divisor_riesgos),
            "capacidades": min(1.0, riesgo_capacidades / divisor_riesgos),
        }

        feasibility_avg = (
            sum(score.feasibility_score for score in analisis.indicador_scores)
            / len(analisis.indicador_scores)
            if analisis.indicador_scores
            else 0.0
        )

        baseline_ratio = baseline_matches / max(1, total_matches)
        target_ratio = target_matches / max(1, total_matches)
        time_ratio = time_matches / max(1, total_matches)
        numerical_ratio = numerical_matches / max(1, total_matches)

        responsable_ratio = min(
            1.0, len(analisis.responsabilidades) / max(1, len(dimension.eslabones))
        resource_ratio = min(1.0, analisis.recursos / max(1, len(dimension.eslabones)))
        plazo_ratio = min(1.0, analisis.plazos / max(1, len(dimension.eslabones)))
        riesgo_evidencia_ratio = min(
            1.0, analisis.riesgos / max(1, len(dimension.eslabones))
        )

        kpi_global = dimension.calcular_kpi_global()
        kpi_norm = min(1.0, kpi_global / 1.6) if dimension.eslabones else 0.0

        return {
            "dimension": dimension,
            "eslabon_metrics": eslabon_metrics,
            "cobertura_ponderada": cobertura_ponderada_final,
            "cobertura_por_tipo": cobertura_tipo_promedio,
            "riesgo_matriz": riesgo_matriz,
            "riesgo_promedio": riesgo_promedio,
            "riesgo_factor": riesgo_factor,
            "riesgo_breakdown": riesgo_breakdown,
            "feasibility_avg": feasibility_avg,
            "baseline_ratio": baseline_ratio,
            "target_ratio": target_ratio,
            "time_ratio": time_ratio,
            "numerical_ratio": numerical_ratio,
            "responsable_ratio": responsable_ratio,
            "resource_ratio": resource_ratio,
            "plazo_ratio": plazo_ratio,
            "riesgo_evidencia_ratio": riesgo_evidencia_ratio,
            "kpi_norm": kpi_norm,
            "kpi_global": kpi_global,
            "total_matches": total_matches,
        }

    def _generar_matriz_trazabilidad(
        self, metricas: Optional[Dict[str, Any]]
    ) -> Optional[pd.DataFrame]:
        if not metricas or pd is None:
            return None

        filas = []
        for eslabon_id, datos in metricas.get("eslabon_metrics", {}).items():
            filas.append(
                {
                    "eslabon_id": eslabon_id,
                    "tipo": datos.get("tipo"),
                    "cobertura": round(datos.get("cobertura", 0.0) * 100, 2),
                    "indicadores_enlazados": datos.get("indicadores_enlazados", 0),
                    "riesgos": ", ".join(datos.get("riesgos", [])),
                    "kpi": datos.get("kpi", 0.0),
                    "lead_time": datos.get("lead_time", 0.0),
                }
            )

        if not filas:
            return None

        return pd.DataFrame(filas)

    def evaluar_dimension_de1(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        max_score = analisis.max_score
        baseline_dets = analisis.detecciones(ComponentType.BASELINE)
        target_dets = analisis.detecciones(ComponentType.TARGET)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)
        date_dets = analisis.detecciones(ComponentType.DATE)

        metrics = self._calcular_metricas_dimension(punto_id, analisis)
        cobertura_factor = metrics["cobertura_ponderada"] if metrics else max_score
        riesgo_factor = metrics["riesgo_factor"] if metrics else 1.0

        mejor_numerico = self._seleccionar_mejor_deteccion(numerical_dets)
        evaluaciones.append(
            self._crear_evaluacion(
                "Q1",
                "DE-1",
                punto_id,
                self._clamp(max_score * riesgo_factor) if mejor_numerico else 0.0,
                self._formatear_deteccion(mejor_numerico),
                self.preguntas_de1["Q1"],
            )
        )

        mejor_responsable = self._seleccionar_mejor_responsable(
            analisis.responsabilidades
        )
        valor_q2 = self._clamp(0.6 + 0.4 * max_score) if mejor_responsable else 0.0
        evaluaciones.append(
            self._crear_evaluacion(
                "Q2",
                "DE-1",
                punto_id,
                valor_q2 * riesgo_factor,
                self._formatear_responsable(mejor_responsable),
                self.preguntas_de1["Q2"],
            )
        )

        indicadores_cobertura = [
            bool(baseline_dets),
            bool(target_dets),
            bool(timeframe_dets),
        ]
        cobertura = (
            sum(indicadores_cobertura) / len(indicadores_cobertura)
            if indicadores_cobertura
            else 0.0
        )
        evidencia_q3 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(baseline_dets)
                    ),
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(target_dets)
                    ),
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(timeframe_dets)
                    ),
                ],
            )
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "Q3",
                "DE-1",
                punto_id,
                self._clamp((max_score + cobertura) / 2) * riesgo_factor,
                evidencia_q3,
                self.preguntas_de1["Q3"],
            )
        )

        evidencia_q4 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(numerical_dets)
                    ),
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(timeframe_dets)
                    ),
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(date_dets)
                    ),
                ],
            )
        )
        valor_q4 = self._clamp((max_score + cobertura_factor) / 2)
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

        evaluaciones.append(
            self._crear_evaluacion(
                "Q5",
                "DE-1",
                punto_id,
                self._clamp(max_score * riesgo_factor),
                self._obtener_muestra(evidencia, "impacto"),
                self.preguntas_de1["Q5"],
            )
        )

        evidencia_q6 = evidencia_q4 or evidencia_q3
        valor_q6 = self._clamp((valor_q4 + max_score + cobertura) / 3)
        evaluaciones.append(
            self._crear_evaluacion(
                "Q6",
                "DE-1",
                punto_id,
                valor_q6,
                evidencia_q6,
                self.preguntas_de1["Q6"],
            )
        )

        puntaje_base = (
            sum(e.puntaje for e in evaluaciones) / len(evaluaciones)
            if evaluaciones
            else 0.0
        )
        matriz_causal = {
            "componentes_detectados": {
                "baseline": bool(baseline_dets),
                "target": bool(target_dets),
                "plazos": bool(timeframe_dets),
                "responsables": bool(analisis.responsabilidades),
            },
            "puntaje_causal": round(puntaje_base * 100, 2),
            "cobertura_ponderada": round(
                (metrics["cobertura_ponderada"] if metrics else 0.0) * 100, 2
            ),
            "riesgo_factor": round(riesgo_factor, 2),
        }
        puntaje_final = self._clamp(
            puntaje_base * (matriz_causal["riesgo_factor"] or 1.0)
        )

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-1",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_final * 100,
            matriz_causal=matriz_causal,
        )

    def evaluar_dimension_de2(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        baseline_dets = analisis.detecciones(ComponentType.BASELINE)
        target_dets = analisis.detecciones(ComponentType.TARGET)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)

        metrics = self._calcular_metricas_dimension(punto_id, analisis)
        coverage_por_tipo = metrics["cobertura_por_tipo"] if metrics else {}
        coverage_insumos = coverage_por_tipo.get("INSUMOS", 0.0)
        coverage_procesos = coverage_por_tipo.get("PROCESOS", 0.0)
        coverage_productos = coverage_por_tipo.get("PRODUCTOS", 0.0)
        coverage_resultados = coverage_por_tipo.get("RESULTADOS", 0.0)
        coverage_ponderada = metrics["cobertura_ponderada"] if metrics else 0.0
        riesgo_factor = metrics["riesgo_factor"] if metrics else 1.0
        riesgo_breakdown = (
            metrics["riesgo_breakdown"]
            if metrics
            else {"indicadores": 0.0, "temporal": 0.0, "capacidades": 0.0}
        )
        feasibility_avg = metrics["feasibility_avg"] if metrics else analisis.max_score
        baseline_ratio = (
            metrics["baseline_ratio"] if metrics else (1.0 if baseline_dets else 0.0)
        )
        target_ratio = (
            metrics["target_ratio"] if metrics else (1.0 if target_dets else 0.0)
        )
        time_ratio = (
            metrics["time_ratio"] if metrics else (1.0 if timeframe_dets else 0.0)
        )
        numerical_ratio = (
            metrics["numerical_ratio"] if metrics else (1.0 if numerical_dets else 0.0)
        )
        resource_ratio = (
            metrics["resource_ratio"]
            if metrics
            else (1.0 if analisis.recursos else 0.0)
        )
        plazo_ratio = (
            metrics["plazo_ratio"] if metrics else (1.0 if analisis.plazos else 0.0)
        )
        kpi_norm = metrics["kpi_norm"] if metrics else 0.0

        evidencia_d1 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(baseline_dets)
                    ),
                    f"Cobertura insumos {coverage_insumos:.0%}" if metrics else "",
                ],
            )
        )
        valor_d1 = self._clamp(
            (0.6 * baseline_ratio + 0.4 * coverage_insumos) * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "D1", "DE-2", punto_id, valor_d1, evidencia_d1, self.preguntas_de2["D1"]
            )
        )

        evidencia_d2 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(timeframe_dets)
                    ),
                    (
                        f"Cobertura resultados {coverage_resultados:.0%}"
                        if metrics
                        else ""
                    ),
                ],
            )
        )
        valor_d2 = self._clamp(
            (0.4 * time_ratio + 0.3 * coverage_resultados + 0.3 * feasibility_avg)
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "D2", "DE-2", punto_id, valor_d2, evidencia_d2, self.preguntas_de2["D2"]
            )
        )

        evidencia_o1 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(target_dets)
                    ),
                    (
                        f"Cobertura resultados {coverage_resultados:.0%}"
                        if metrics
                        else ""
                    ),
                ],
            )
        )
        valor_o1 = self._clamp(
            (0.5 * target_ratio + 0.3 * coverage_resultados + 0.2 * kpi_norm)
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "O1", "DE-2", punto_id, valor_o1, evidencia_o1, self.preguntas_de2["O1"]
            )
        )

        evidencia_o2 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(baseline_dets)
                    ),
                    self._formatear_deteccion(
                        self._seleccionar_mejor_deteccion(target_dets)
                    ),
                ],
            )
        )
        valor_o2 = self._clamp(
            ((baseline_ratio + target_ratio + feasibility_avg) / 3) * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "O2", "DE-2", punto_id, valor_o2, evidencia_o2, self.preguntas_de2["O2"]
            )
        )

        evidencia_t1 = " | ".join(
            filter(
                None,
                [
                    self._obtener_muestra(evidencia, "recursos"),
                    f"Cobertura procesos {coverage_procesos:.0%}" if metrics else "",
                ],
            )
        )
        valor_t1 = self._clamp(
            (
                0.5 * coverage_procesos
                + 0.3 * resource_ratio
                + 0.2 * (1 - riesgo_breakdown["indicadores"])
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "T1", "DE-2", punto_id, valor_t1, evidencia_t1, self.preguntas_de2["T1"]
            )
        )

        mejor_t2 = self._seleccionar_mejor_deteccion(numerical_dets + timeframe_dets)
        evidencia_t2 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(mejor_t2),
                    f"Cobertura insumos {coverage_insumos:.0%}" if metrics else "",
                ],
            )
        )
        valor_t2 = self._clamp(
            (0.4 * time_ratio + 0.3 * numerical_ratio + 0.3 * coverage_insumos)
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "T2", "DE-2", punto_id, valor_t2, evidencia_t2, self.preguntas_de2["T2"]
            )
        )

        evidencia_s1 = " | ".join(
            filter(
                None,
                [
                    self._obtener_muestra(evidencia, "indicadores"),
                    f"Cobertura productos {coverage_productos:.0%}" if metrics else "",
                ],
            )
        )
        valor_s1 = self._clamp(
            (
                0.5 * coverage_productos
                + 0.3 * coverage_ponderada
                + 0.2 * feasibility_avg
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "S1", "DE-2", punto_id, valor_s1, evidencia_s1, self.preguntas_de2["S1"]
            )
        )

        mejor_s2 = self._seleccionar_mejor_deteccion(timeframe_dets)
        evidencia_s2 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(mejor_s2),
                    self._obtener_muestra(evidencia, "plazos"),
                    (
                        f"Planificación en procesos {coverage_procesos:.0%}"
                        if metrics
                        else ""
                    ),
                ],
            )
        )
        valor_s2 = self._clamp(
            (
                0.6 * time_ratio
                + 0.2 * coverage_ponderada
                + 0.2 * (1 - riesgo_breakdown["temporal"])
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "S2", "DE-2", punto_id, valor_s2, evidencia_s2, self.preguntas_de2["S2"]
            )
        )

        puntaje_dimension = (
            sum(e.puntaje for e in evaluaciones) / len(evaluaciones)
            if evaluaciones
            else 0.0
        )
        matriz_causal = {
            "kpi_global": round((metrics["kpi_global"] if metrics else 0.0), 2),
            "cobertura_por_tipo": {
                tipo: round(valor * 100, 2)
                for tipo, valor in (
                    metrics["cobertura_por_tipo"] if metrics else {}
                ).items()
            },
            "riesgo_breakdown": riesgo_breakdown,
            "componentes": {
                "baseline_ratio": round(baseline_ratio, 2),
                "target_ratio": round(target_ratio, 2),
                "time_ratio": round(time_ratio, 2),
                "numerical_ratio": round(numerical_ratio, 2),
            },
            "feasibility_promedio": round(feasibility_avg, 2),
        }

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-2",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension * 100,
            matriz_causal=matriz_causal,
        )

    def evaluar_dimension_de3(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        mejor_numerico = self._seleccionar_mejor_deteccion(numerical_dets)
        mejor_plazo = self._seleccionar_mejor_deteccion(timeframe_dets)
        mejor_responsable = self._seleccionar_mejor_responsable(
            analisis.responsabilidades
        )

        metrics = self._calcular_metricas_dimension(punto_id, analisis)
        coverage_por_tipo = metrics["cobertura_por_tipo"] if metrics else {}
        coverage_insumos = coverage_por_tipo.get("INSUMOS", 0.0)
        coverage_procesos = coverage_por_tipo.get("PROCESOS", 0.0)
        coverage_productos = coverage_por_tipo.get("PRODUCTOS", 0.0)
        coverage_resultados = coverage_por_tipo.get("RESULTADOS", 0.0)
        coverage_ponderada = metrics["cobertura_ponderada"] if metrics else 0.0
        riesgo_factor = metrics["riesgo_factor"] if metrics else 1.0
        riesgo_breakdown = (
            metrics["riesgo_breakdown"]
            if metrics
            else {"indicadores": 0.0, "temporal": 0.0, "capacidades": 0.0}
        )
        feasibility_avg = (
            metrics["feasibility_avg"]
            if metrics
            else (mejor_numerico.feasibility_score if mejor_numerico else 0.0)
        )
        time_ratio = (
            metrics["time_ratio"] if metrics else (1.0 if timeframe_dets else 0.0)
        )
        numerical_ratio = (
            metrics["numerical_ratio"] if metrics else (1.0 if numerical_dets else 0.0)
        )
        responsable_ratio = (
            metrics["responsable_ratio"]
            if metrics
            else (1.0 if analisis.responsabilidades else 0.0)
        )
        resource_ratio = (
            metrics["resource_ratio"]
            if metrics
            else (1.0 if analisis.recursos else 0.0)
        )
        plazo_ratio = (
            metrics["plazo_ratio"] if metrics else (1.0 if analisis.plazos else 0.0)
        )
        riesgo_evidencia_ratio = (
            metrics["riesgo_evidencia_ratio"]
            if metrics
            else (1.0 if analisis.riesgos else 0.0)
        )
        kpi_norm = metrics["kpi_norm"] if metrics else 0.0

        evidencia_g1 = " | ".join(
            filter(
                None,
                [
                    self._obtener_muestra(evidencia, "recursos"),
                    f"Cobertura insumos {coverage_insumos:.0%}" if metrics else "",
                ],
            )
        )
        valor_g1 = self._clamp(
            (
                0.5 * coverage_insumos
                + 0.3 * resource_ratio
                + 0.2 * (1 - riesgo_breakdown["indicadores"])
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "G1", "DE-3", punto_id, valor_g1, evidencia_g1, self.preguntas_de3["G1"]
            )
        )

        evidencia_g2 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(mejor_plazo)
                    or self._obtener_muestra(evidencia, "plazos"),
                    f"Cobertura procesos {coverage_procesos:.0%}" if metrics else "",
                ],
            )
        )
        valor_g2 = self._clamp(
            (
                0.4 * time_ratio
                + 0.3 * plazo_ratio
                + 0.3 * (1 - riesgo_breakdown["temporal"])
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "G2", "DE-3", punto_id, valor_g2, evidencia_g2, self.preguntas_de3["G2"]
            )
        )

        evidencia_a1 = self._formatear_deteccion(mejor_numerico)
        valor_a1 = self._clamp(
            (0.5 * numerical_ratio + 0.3 * feasibility_avg + 0.2 * coverage_resultados)
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "A1", "DE-3", punto_id, valor_a1, evidencia_a1, self.preguntas_de3["A1"]
            )
        )

        evidencia_a2 = " | ".join(
            filter(
                None,
                [
                    self._formatear_responsable(mejor_responsable),
                    f"Cobertura procesos {coverage_procesos:.0%}" if metrics else "",
                ],
            )
        )
        valor_a2 = self._clamp(
            (
                0.4 * coverage_procesos
                + 0.3 * responsable_ratio
                + 0.3 * (1 - riesgo_breakdown["capacidades"])
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "A2", "DE-3", punto_id, valor_a2, evidencia_a2, self.preguntas_de3["A2"]
            )
        )

        evidencia_r1 = " | ".join(
            filter(
                None,
                [
                    self._obtener_muestra(evidencia, "recursos"),
                    f"Cobertura insumos {coverage_insumos:.0%}" if metrics else "",
                ],
            )
        )
        valor_r1 = self._clamp(
            (
                0.5 * coverage_insumos
                + 0.3 * kpi_norm
                + 0.2 * (1 - riesgo_breakdown["indicadores"])
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "R1", "DE-3", punto_id, valor_r1, evidencia_r1, self.preguntas_de3["R1"]
            )
        )

        evidencia_r2 = self._formatear_responsable(mejor_responsable)
        valor_r2 = self._clamp(
            (
                0.5 * (1 - riesgo_breakdown["capacidades"])
                + 0.3 * responsable_ratio
                + 0.2 * coverage_procesos
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "R2", "DE-3", punto_id, valor_r2, evidencia_r2, self.preguntas_de3["R2"]
            )
        )

        evidencia_s1 = " | ".join(
            filter(
                None,
                [
                    self._formatear_deteccion(mejor_plazo)
                    or self._obtener_muestra(evidencia, "plazos"),
                    f"Cobertura productos {coverage_productos:.0%}" if metrics else "",
                ],
            )
        )
        valor_s1 = self._clamp(
            (0.4 * coverage_productos + 0.4 * time_ratio + 0.2 * coverage_ponderada)
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "S1", "DE-3", punto_id, valor_s1, evidencia_s1, self.preguntas_de3["S1"]
            )
        )

        evidencia_s2 = " | ".join(
            filter(
                None,
                [
                    self._obtener_muestra(evidencia, "riesgos"),
                    f"Cobertura procesos {coverage_procesos:.0%}" if metrics else "",
                ],
            )
        )
        valor_s2 = self._clamp(
            (
                0.4 * (1 - riesgo_breakdown["temporal"])
                + 0.3 * (1 - riesgo_breakdown["capacidades"])
                + 0.3 * riesgo_evidencia_ratio
            )
            * riesgo_factor
        )
        evaluaciones.append(
            self._crear_evaluacion(
                "S2", "DE-3", punto_id, valor_s2, evidencia_s2, self.preguntas_de3["S2"]
            )
        )

        puntaje_dimension = (
            sum(e.puntaje for e in evaluaciones) / len(evaluaciones)
            if evaluaciones
            else 0.0
        )
        matriz_causal = {
            "cobertura_por_tipo": {
                tipo: round(valor * 100, 2)
                for tipo, valor in (
                    metrics["cobertura_por_tipo"] if metrics else {}
                ).items()
            },
            "riesgo_breakdown": riesgo_breakdown,
            "factores": {
                "resource_ratio": round(resource_ratio, 2),
                "responsable_ratio": round(responsable_ratio, 2),
                "time_ratio": round(time_ratio, 2),
                "numerical_ratio": round(numerical_ratio, 2),
            },
            "cobertura_ponderada": round(coverage_ponderada * 100, 2),
        }

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-3",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension * 100,
            matriz_causal=matriz_causal,
        )

    def evaluar_dimension_de4(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        evaluaciones: List[EvaluacionPregunta] = []
        baseline_dets = analisis.detecciones(ComponentType.BASELINE)
        target_dets = analisis.detecciones(ComponentType.TARGET)
        timeframe_dets = analisis.detecciones(ComponentType.TIME_HORIZON)
        numerical_dets = analisis.detecciones(ComponentType.NUMERICAL)
        best_responsable = self._seleccionar_mejor_responsable(
            analisis.responsabilidades
        )

        metrics = self._calcular_metricas_dimension(punto_id, analisis)
        coverage_por_tipo = metrics["cobertura_por_tipo"] if metrics else {}
        coverage_insumos = coverage_por_tipo.get("INSUMOS", 0.0)
        coverage_procesos = coverage_por_tipo.get("PROCESOS", 0.0)
        coverage_productos = coverage_por_tipo.get("PRODUCTOS", 0.0)
        coverage_resultados = coverage_por_tipo.get("RESULTADOS", 0.0)
        coverage_impactos = coverage_por_tipo.get("IMPACTOS", 0.0)
        coverage_ponderada = metrics["cobertura_ponderada"] if metrics else 0.0
        riesgo_factor = metrics["riesgo_factor"] if metrics else 1.0
        riesgo_breakdown = (
            metrics["riesgo_breakdown"]
            if metrics
            else {"indicadores": 0.0, "temporal": 0.0, "capacidades": 0.0}
        )
        feasibility_avg = metrics["feasibility_avg"] if metrics else analisis.max_score
        baseline_ratio = (
            metrics["baseline_ratio"] if metrics else (1.0 if baseline_dets else 0.0)
        )
        target_ratio = (
            metrics["target_ratio"] if metrics else (1.0 if target_dets else 0.0)
        )
        time_ratio = (
            metrics["time_ratio"] if metrics else (1.0 if timeframe_dets else 0.0)
        )
        numerical_ratio = (
            metrics["numerical_ratio"] if metrics else (1.0 if numerical_dets else 0.0)
        )
        responsable_ratio = (
            metrics["responsable_ratio"]
            if metrics
            else (1.0 if analisis.responsabilidades else 0.0)
        )
        resource_ratio = (
            metrics["resource_ratio"]
            if metrics
            else (1.0 if analisis.recursos else 0.0)
        )
        plazo_ratio = (
            metrics["plazo_ratio"] if metrics else (1.0 if analisis.plazos else 0.0)
        )
        kpi_norm = metrics["kpi_norm"] if metrics else 0.0

        preguntas_valores: List[Tuple[str, float, str]] = []

        preguntas_valores.append(
            (
                "DE4_1",
                self._clamp(
                    (
                        0.5 * baseline_ratio
                        + 0.3 * coverage_insumos
                        + 0.2 * feasibility_avg
                    )
                    * riesgo_factor
                ),
                " | ".join(
                    filter(
                        None,
                        [
                            self._formatear_deteccion(
                                self._seleccionar_mejor_deteccion(baseline_dets)
                            ),
                            (
                                f"Cobertura insumos {coverage_insumos:.0%}"
                                if metrics
                                else ""
                            ),
                        ],
                    )
                ),
            )
        )

        preguntas_valores.append(
            (
                "DE4_2",
                self._clamp(
                    (
                        (baseline_ratio + target_ratio + time_ratio) / 3
                        + coverage_ponderada
                    )
                    / 2
                    * riesgo_factor
                ),
                " | ".join(
                    filter(
                        None,
                        [
                            self._formatear_deteccion(
                                self._seleccionar_mejor_deteccion(baseline_dets)
                            ),
                            self._formatear_deteccion(
                                self._seleccionar_mejor_deteccion(target_dets)
                            ),
                            self._formatear_deteccion(
                                self._seleccionar_mejor_deteccion(timeframe_dets)
                            ),
                        ],
                    )
                ),
            )
        )

        preguntas_valores.append(
            (
                "DE4_3",
                self._clamp(
                    (
                        0.5 * target_ratio
                        + 0.3 * coverage_resultados
                        + 0.2 * feasibility_avg
                    )
                    * riesgo_factor
                ),
                self._formatear_deteccion(
                    self._seleccionar_mejor_deteccion(target_dets)
                ),
            )
        )

        preguntas_valores.append(
            (
                "DE4_4",
                self._clamp(
                    (
                        0.4 * coverage_procesos
                        + 0.3 * resource_ratio
                        + 0.3 * responsable_ratio
                    )
                    * riesgo_factor
                ),
                " | ".join(
                    filter(
                        None,
                        [
                            self._formatear_responsable(best_responsable),
                            self._obtener_muestra(evidencia, "recursos"),
                        ],
                    )
                ),
            )
        )

        preguntas_valores.append(
            (
                "DE4_5",
                self._clamp(
                    (
                        0.4 * plazo_ratio
                        + 0.3 * coverage_productos
                        + 0.3 * coverage_procesos
                    )
                    * riesgo_factor
                ),
                " | ".join(
                    filter(
                        None,
                        [
                            self._obtener_muestra(evidencia, "plazos"),
                            (
                                f"Cobertura productos {coverage_productos:.0%}"
                                if metrics
                                else ""
                            ),
                        ],
                    )
                ),
            )
        )

        preguntas_valores.append(
            (
                "DE4_6",
                self._clamp(
                    (
                        0.5 * responsable_ratio
                        + 0.3 * coverage_procesos
                        + 0.2 * (1 - riesgo_breakdown["capacidades"])
                    )
                    * riesgo_factor
                ),
                self._formatear_responsable(best_responsable),
            )
        )

        preguntas_valores.append(
            (
                "DE4_7",
                self._clamp(
                    (0.5 * time_ratio + 0.3 * feasibility_avg + 0.2 * numerical_ratio)
                    * riesgo_factor
                ),
                self._formatear_deteccion(
                    self._seleccionar_mejor_deteccion(timeframe_dets)
                )
                or self._obtener_muestra(evidencia, "indicadores"),
            )
        )

        preguntas_valores.append(
            (
                "DE4_8",
                self._clamp(
                    (
                        0.4 * coverage_resultados
                        + 0.3 * coverage_impactos
                        + 0.3 * kpi_norm
                    )
                    * riesgo_factor
                ),
                self._formatear_deteccion(
                    self._seleccionar_mejor_deteccion(target_dets + numerical_dets)
                ),
            )
        )

        for idx, (pregunta_id, valor, evidencia_texto) in enumerate(
            preguntas_valores, start=1
        ):
            evaluaciones.append(
                self._crear_evaluacion(
                    pregunta_id,
                    "DE-4",
                    punto_id,
                    valor,
                    evidencia_texto,
                    self.preguntas_de4[idx - 1],
                )
            )

        puntaje_dimension = (
            sum(e.puntaje for e in evaluaciones) / len(evaluaciones)
            if evaluaciones
            else 0.0
        )
        matriz_causal = {
            "componentes_detectados": {
                "baseline": bool(baseline_dets),
                "target": bool(target_dets),
                "plazos": bool(timeframe_dets or analisis.plazos),
                "responsables": bool(analisis.responsabilidades),
            },
            "cobertura_por_tipo": {
                tipo: round(valor * 100, 2)
                for tipo, valor in (coverage_por_tipo if metrics else {}).items()
            },
            "metricas": {
                "cobertura_ponderada": round(coverage_ponderada * 100, 2),
                "feasibility": round(feasibility_avg, 2),
                "riesgo_factor": round(riesgo_factor, 2),
            },
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
    ) -> Tuple[
        EvaluacionPuntoCompleto,
        AnalisisEvidenciaDecalogo,
        Optional[ResultadoDimensionIndustrial],
    ]:
        analisis = self._analizar_evidencia(evidencia)
        evaluaciones_dimensiones = [
            self.evaluar_dimension_de1(analisis, evidencia, punto_id),
            self.evaluar_dimension_de2(analisis, evidencia, punto_id),
            self.evaluar_dimension_de3(analisis, evidencia, punto_id),
            self.evaluar_dimension_de4(analisis, evidencia, punto_id),
        ]
        puntaje_agregado = (
            sum(dim.puntaje_dimension for dim in evaluaciones_dimensiones)
            / len(evaluaciones_dimensiones)
            if evaluaciones_dimensiones
            else 0.0
        )
        dimension = self.contexto.dimensiones_por_id.get(punto_id)
        nombre_punto = dimension.nombre if dimension else f"Punto {punto_id}"
        metricas = self._calcular_metricas_dimension(punto_id, analisis)
        puntajes_norm = {
            dim_eval.dimension: dim_eval.puntaje_dimension / 100
            for dim_eval in evaluaciones_dimensiones
        }

        resultado_industrial: Optional[ResultadoDimensionIndustrial] = None
        if dimension:
            matriz_trazabilidad = self._generar_matriz_trazabilidad(metricas)
            riesgos_list = []
            if metricas:
                for eslabon_id, riesgos_eslabon in metricas.get(
                    "riesgo_matriz", {}
                ).items():
                    for riesgo in riesgos_eslabon:
                        riesgos_list.append(f"{eslabon_id}: {riesgo}")

            factibilidad_operativa = self._clamp(puntajes_norm.get("DE-3", 0.0))
            robustez_causal = self._clamp(puntajes_norm.get("DE-4", 0.0))
            consistencia_logica = self._clamp(puntajes_norm.get("DE-1", 0.0))
            certeza_probabilistica = (
                self._clamp(
                    max(metricas["feasibility_avg"], puntajes_norm.get("DE-2", 0.0))
                )
                if metricas
                else self._clamp(puntajes_norm.get("DE-2", 0.0))
            )
            identificabilidad_causal = self._clamp(
                (consistencia_logica + robustez_causal) / 2
            )

            brechas_identificadas = []
            cobertura_descriptor = (
                f"Cobertura ponderada {metricas['cobertura_ponderada'] * 100:.1f}%"
                if metricas
                else "Cobertura limitada"
            )
            for dim_eval in evaluaciones_dimensiones:
                if dim_eval.puntaje_dimension < 60:
                    brechas_identificadas.append(
                        f"{dim_eval.dimension}: {dim_eval.puntaje_dimension:.1f}/100 - {cobertura_descriptor}"
                    )

            recomendaciones = []
            if metricas:
                for eslabon_id, riesgos_eslabon in metricas.get(
                    "riesgo_matriz", {}
                ).items():
                    for riesgo in riesgos_eslabon:
                        recomendaciones.append(
                            f"Mitigar riesgo en {eslabon_id}: {riesgo}"
                        )
                if metricas["feasibility_avg"] < 0.6:
                    recomendaciones.append(
                        "Fortalecer evidencia cuantitativa de indicadores para incrementar la factibilidad."
                    )
            if not recomendaciones:
                recomendaciones.append(
                    "Mantener seguimiento industrial sobre la implementación del decálogo."
                )

            evaluacion_causal = EvaluacionCausalIndustrial(
                consistencia_logica=consistencia_logica,
                identificabilidad_causal=identificabilidad_causal,
                factibilidad_operativa=factibilidad_operativa,
                certeza_probabilistica=certeza_probabilistica,
                robustez_causal=robustez_causal,
                riesgos_implementacion=riesgos_list[:10],
                supuestos_criticos=dimension.teoria_cambio.supuestos_causales,
                evidencia_soporte=len(analisis.indicador_scores),
                brechas_criticas=sum(
                    1 for valor in puntajes_norm.values() if valor < 0.6
                ),
            )

            resultado_industrial = ResultadoDimensionIndustrial(
                dimension=dimension,
                evaluacion_causal=evaluacion_causal,
                evidencia=evidencia,
                brechas_identificadas=brechas_identificadas,
                recomendaciones=recomendaciones[:10],
                matriz_trazabilidad=matriz_trazabilidad,
            )

        return (
            EvaluacionPuntoCompleto(
                punto_id=punto_id,
                nombre_punto=nombre_punto,
                evaluaciones_dimensiones=evaluaciones_dimensiones,
                puntaje_agregado_punto=puntaje_agregado,
            ),
            analisis,
            resultado_industrial,
        )

    def evaluar_cluster_completo(
        self,
        evidencias_por_punto: Dict[int, Dict[str, List[Any]]],
        cluster: ClusterMetadata,
    ) -> Tuple[EvaluacionClusterCompleto, List[ResultadoDimensionIndustrial]]:
        evaluaciones_puntos: List[EvaluacionPuntoCompleto] = []
        resultados_industriales: List[ResultadoDimensionIndustrial] = []
        for punto_id in cluster.puntos:
            evaluacion_punto, _, resultado_industrial = self.evaluar_punto_completo(
                evidencias_por_punto.get(punto_id, {}), punto_id
            )
            evaluaciones_puntos.append(evaluacion_punto)
            if resultado_industrial:
                resultados_industriales.append(resultado_industrial)

        puntaje_agregado = (
            sum(e.puntaje_agregado_punto for e in evaluaciones_puntos)
            / len(evaluaciones_puntos)
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

        return (
            EvaluacionClusterCompleto(
                cluster_nombre=cluster.titulo,
                evaluaciones_puntos=evaluaciones_puntos,
                puntaje_agregado_cluster=puntaje_agregado,
                clasificacion_cualitativa=clasificacion,
            ),
            resultados_industriales,
        )

    def generar_reporte_final(
        self, evidencias_por_punto: Dict[int, Dict[str, List[Any]]], nombre_plan: str
    ) -> ReporteFinalDecatalogo:
        evaluaciones_clusters: List[EvaluacionClusterCompleto] = []
        resultados_dimension: List[ResultadoDimensionIndustrial] = []
        for cluster in self.contexto.clusters_por_id.values():
            evaluacion_cluster, resultados = self.evaluar_cluster_completo(
                evidencias_por_punto, cluster
            )
            evaluaciones_clusters.append(evaluacion_cluster)
            resultados_dimension.extend(resultados)
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

        puntajes_clusters = [
            cluster.puntaje_agregado_cluster for cluster in evaluaciones_clusters
        ]
        if resultados_dimension:
            puntaje_global = sum(r.puntaje_final for r in resultados_dimension) / len(
                resultados_dimension
            )
        else:
            puntaje_global = (
                sum(puntajes_clusters) / len(puntajes_clusters)
                if puntajes_clusters
                else 0.0
            )

        cluster_mejor = (
            max(
                evaluaciones_clusters, key=lambda x: x.puntaje_agregado_cluster
            ).cluster_nombre
            if evaluaciones_clusters
            else ""
        )
        cluster_peor = (
            min(
                evaluaciones_clusters, key=lambda x: x.puntaje_agregado_cluster
            ).cluster_nombre
            if evaluaciones_clusters
            else ""
        )

        resumen_ejecutivo = {
            "puntaje_global": puntaje_global,
            "cluster_mejor_desempeno": cluster_mejor,
            "cluster_peor_desempeno": cluster_peor,
            "numero_puntos_evaluados": len(evaluaciones_puntos),
            "recomendacion_estrategica_global": (
                "IMPLEMENTACIÓN RECOMENDADA"
                if puntaje_global >= 70
                else "REDISEÑO PARCIAL REQUERIDO"
            ),
        }

        if puntaje_global >= 75:
            alineacion = "ALTA"
        elif puntaje_global >= 50:
            alineacion = "MEDIA"
        else:
            alineacion = "BAJA"

        brechas_globales = [
            brecha
            for resultado in resultados_dimension
            for brecha in resultado.brechas_identificadas
        ]
        recomendaciones_globales: List[str] = []
        for resultado in resultados_dimension:
            for recomendacion in resultado.recomendaciones:
                if recomendacion not in recomendaciones_globales:
                    recomendaciones_globales.append(recomendacion)
        recomendaciones_globales = recomendaciones_globales[:10]

        reporte_macro = {
            "alineacion_global_decatalogo": alineacion,
            "explicacion_extensa_cualitativa": (
                "El Plan de Desarrollo Territorial presenta un desempeño "
                f"{'óptimo' if puntaje_global >= 80 else 'satisfactorio' if puntaje_global >= 60 else 'básico' if puntaje_global >= 40 else 'insuficiente'} "
                "en su alineación con el Decálogo de Derechos Humanos."
            ),
            "brechas_globales": brechas_globales[:10],
            "recomendaciones_globales": recomendaciones_globales,
        }

        anexos_serializables = {
            "resultados_industriales": [
                resultado.generar_reporte_tecnico()
                for resultado in resultados_dimension
            ]
        }

        return ReporteFinalDecatalogo(
            metadata={
                "nombre_plan": nombre_plan,
                "fecha_evaluacion": datetime.now().isoformat(),
                "version_evaluador": "1.0-industrial-full",
            },
            resumen_ejecutivo=resumen_ejecutivo,
            reporte_macro=reporte_macro,
            resultados_dimension=resultados_dimension,
            reporte_meso_por_cluster=evaluaciones_clusters,
            reporte_por_punto=evaluaciones_puntos,
            reporte_por_pregunta=evaluaciones_preguntas,
            anexos_serializables=anexos_serializables,
        )


def integrar_evaluador_decatalogo(
    sistema: SistemaEvaluacionIndustrial, dimension: DimensionDecalogo
) -> Optional[ResultadoDimensionIndustrial]:
    """Integra el evaluador especializado del decálogo con la evaluación industrial."""

    if not sistema.extractor:
        raise ValueError(
            "Extractor no inicializado - Error industrial crítico")

    try:
        evaluador = IndustrialDecatalogoEvaluatorFull()
        evidencia_dimension = sistema.extractor.extraer_variables_operativas(
            dimension)
        matriz_trazabilidad = sistema.extractor.generar_matriz_trazabilidad(
            dimension)
        cluster_metadata = DECALOGO_CONTEXT.cluster_por_dimension.get(
            dimension.id)

        if not cluster_metadata:
            LOGGER.warning(
                "⚠️  [DECÁLOGO] Dimensión %s no se encuentra asociada a un cluster definido",
                dimension.id,
            )
            return None

        evaluacion_punto, analisis, resultado_industrial = (
            evaluador.evaluar_punto_completo(evidencia_dimension, dimension.id)
        )

        if resultado_industrial is None:
            LOGGER.warning(
                "⚠️  [DECÁLOGO] No se pudo construir resultado industrial para la dimensión %s",
                dimension.id,
            )
            return None

        resultado_industrial.evidencia = evidencia_dimension
        if matriz_trazabilidad is not None:
            resultado_industrial.matriz_trazabilidad = matriz_trazabilidad

        if cluster_metadata.titulo not in " ".join(
            resultado_industrial.recomendaciones
        ):
            resultado_industrial.recomendaciones.insert(
                0,
                f"Alinear acciones del cluster {cluster_metadata.titulo} con la teoría de cambio industrial.",
            )

        LOGGER.info(
            "✅ Evaluación decálogo integrada para dimensión %s (cluster %s): %.1f/100",
            dimension.id,
            cluster_metadata.titulo,
            resultado_industrial.puntaje_final,
        )

        return resultado_industrial

    except Exception as exc:
        LOGGER.error(
            f"❌ Error en integración del evaluador del decálogo: {exc}")
        return None