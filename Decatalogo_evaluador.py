#!/usr/bin/env python3
# coding=utf-8
# -*- coding: utf-8 -*-
"""
decatalogo_evaluator_full.py
Versión: 3.0 PRODUCTION — Evaluador Industrial Completo del Decálogo de Derechos Humanos
Propósito: Evaluar 10 puntos del Decálogo en 4 dimensiones (DE-1 a DE-4) con scoring riguroso
Conexión: INTEGRACIÓN TOTAL con todos los módulos del proyecto
Autor: Sistema Doctoral de Políticas Públicas - Versión Industrial
Fecha: 2025 - Versión de Producción
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from contradiction_detector import (
    ContradictionAnalysis,
    ContradictionDetector,
    ContradictionMatch,
)

# Importaciones del sistema principal (Decatalogo_principal.py)
from Decatalogo_principal import (  # Clases adicionales necesarias
    ClusterMetadata,
    DecalogoContext,
    DimensionDecalogo,
    ResultadoDimensionIndustrial,
    SistemaEvaluacionIndustrial,
    obtener_decalogo_contexto,
)

# Importaciones de detectores especializados (VERIFICADOS)
from feasibility_scorer import (
    ComponentType,
    DetectionResult,
    FeasibilityScorer,
    IndicatorScore,
)
from pdm_contra.bridges.decatalogo_provider import provide_decalogos
from responsibility_detector import (
    EntityType,
    ResponsibilityDetector,
    ResponsibilityEntity,
)

# ==================== IMPORTACIONES VERIFICADAS DEL SISTEMA ====================

# Importaciones opcionales pero útiles
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    from monetary_detector import MonetaryDetector, MonetaryMatch

    HAS_MONETARY = True
except ImportError:
    HAS_MONETARY = False

# Verificación de versión Python
assert sys.version_info >= (
    3, 11), "Python 3.11+ required for production deployment"

# ==================== CONFIGURACIÓN DE LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(
        "decatalogo_evaluator.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("DecatalogoEvaluatorFull.v3.0")

# ==================== CONTEXTO GLOBAL ====================

BUNDLE = provide_decalogos()
DECALOGO_CONTEXT: DecalogoContext = obtener_decalogo_contexto()

# ==================== ESTRUCTURAS DE DATOS COMPLETAS ====================


@dataclass
class AnalisisEvidenciaDecalogo:
    """Resultados completos del análisis automático de evidencia con todos los detectores."""

    indicador_scores: List[IndicatorScore]
    indicadores_evaluados: List[Tuple[str, IndicatorScore]]
    detecciones_por_tipo: Dict[ComponentType, List[DetectionResult]]
    responsabilidades: List[ResponsibilityEntity]
    contradicciones: List[ContradictionMatch]
    analisis_contradicciones: Optional[ContradictionAnalysis] = None
    valores_monetarios: List["MonetaryMatch"] = field(default_factory=list)
    recursos: int = 0
    plazos: int = 0
    riesgos: int = 0
    confidence_global: float = 0.0
    metadata_analisis: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_score(self) -> float:
        """Score máximo de feasibility encontrado."""
        return max(
            (score.feasibility_score for score in self.indicador_scores), default=0.0
        )

    @property
    def tiene_linea_base(self) -> bool:
        """Verifica si hay línea base detectada."""
        return len(self.detecciones_por_tipo.get(ComponentType.BASELINE, [])) > 0

    @property
    def tiene_metas(self) -> bool:
        """Verifica si hay metas detectadas."""
        return len(self.detecciones_por_tipo.get(ComponentType.TARGET, [])) > 0

    def detecciones(self, componente: ComponentType) -> List[DetectionResult]:
        """Obtiene detecciones por tipo de componente."""
        return self.detecciones_por_tipo.get(componente, [])


@dataclass
class EvaluacionPregunta:
    """Evaluación completa de una pregunta específica del cuestionario."""

    pregunta_id: str
    dimension: str
    punto_id: int
    respuesta: str  # Sí / Parcial / No / NI
    evidencia_textual: str
    evidencia_contraria: str
    puntaje: float
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    detecciones_asociadas: List[DetectionResult] = field(default_factory=list)


@dataclass
class EvaluacionDimensionPunto:
    """Evaluación completa de una dimensión para un punto del Decálogo."""

    punto_id: int
    dimension: str
    evaluaciones_preguntas: List[EvaluacionPregunta]
    puntaje_dimension: float
    matriz_causal: Optional[Dict[str, Any]] = None
    teoria_cambio_validada: bool = False
    eslabones_completos: int = 0
    coherencia_interna: float = 0.0
    metadata_dimension: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluacionPuntoCompleto:
    """Evaluación integral de un punto del Decálogo en todas las dimensiones."""

    punto_id: int
    nombre_punto: str
    evaluaciones_dimensiones: List[EvaluacionDimensionPunto]
    puntaje_agregado_punto: float
    clasificacion_cualitativa: str = ""
    explicacion_extensa: str = ""
    brechas_identificadas: List[str] = field(default_factory=list)
    recomendaciones_especificas: List[str] = field(default_factory=list)
    fortalezas_identificadas: List[str] = field(default_factory=list)
    riesgos_detectados: List[str] = field(default_factory=list)


@dataclass
class EvaluacionClusterCompleto:
    """Evaluación completa de un cluster del Decálogo con análisis meso."""

    cluster_nombre: str
    cluster_id: int
    evaluaciones_puntos: List[EvaluacionPuntoCompleto]
    puntaje_agregado_cluster: float
    clasificacion_cualitativa: str
    analisis_meso_completo: Dict[str, Any] = field(default_factory=dict)
    interdependencias_cluster: Dict[int, float] = field(default_factory=dict)
    coherencia_cluster: float = 0.0
    varianza_interna: float = 0.0


@dataclass
class ReporteFinalDecatalogo:
    """Reporte final exhaustivo del análisis del Decálogo."""

    metadata: Dict[str, Any]
    resumen_ejecutivo: Dict[str, Any]
    reporte_macro: Dict[str, Any]
    reporte_meso_por_cluster: List[EvaluacionClusterCompleto]
    reporte_por_punto: List[EvaluacionPuntoCompleto]
    reporte_por_pregunta: List[EvaluacionPregunta]
    resultados_dimension: List[ResultadoDimensionIndustrial]
    anexos_serializables: Dict[str, Any] = field(default_factory=dict)
    matriz_trazabilidad_global: Optional[pd.DataFrame] = None
    estadisticas_globales: Dict[str, Any] = field(default_factory=dict)


# ==================== EVALUADOR BASE MEJORADO ====================


class _EvaluadorBase:
    """Utilidades compartidas mejoradas para evaluaciones del decálogo."""

    def __init__(self):
        """Inicializa el evaluador base con detectores."""
        self.contradiction_detector = ContradictionDetector()
        if HAS_MONETARY:
            self.monetary_detector = MonetaryDetector()
        else:
            self.monetary_detector = None

    @staticmethod
    def _seleccionar_mejor_deteccion(
        detecciones: List[DetectionResult],
    ) -> Optional[DetectionResult]:
        """Selecciona la detección con mayor confianza."""
        return max(detecciones, key=lambda d: d.confidence, default=None)

    @staticmethod
    def _seleccionar_mejor_responsable(
        responsables: List[ResponsibilityEntity],
    ) -> Optional[ResponsibilityEntity]:
        """Selecciona el responsable con mayor confianza."""
        return max(responsables, key=lambda r: r.confidence, default=None)

    @staticmethod
    def _formatear_deteccion(deteccion: Optional[DetectionResult]) -> str:
        """Formatea una detección para presentación."""
        if not deteccion:
            return ""
        return f"{deteccion.matched_text} (conf. {deteccion.confidence:.2f})"

    @staticmethod
    def _formatear_responsable(responsable: Optional[ResponsibilityEntity]) -> str:
        """Formatea un responsable para presentación."""
        if not responsable:
            return ""
        rol = responsable.role or responsable.entity_type.value
        return f"{responsable.text} ({rol}, conf. {responsable.confidence:.2f})"

    @staticmethod
    def _valor_a_respuesta(valor: float) -> str:
        """Convierte valor numérico a respuesta categórica."""
        if valor >= 0.75:
            return "Sí"
        if valor >= 0.4:
            return "Parcial"
        if valor > 0:
            return "No"
        return "NI"  # No Identificado

    @staticmethod
    def _clamp(valor: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
        """Limita un valor entre mínimo y máximo."""
        return max(minimo, min(maximo, valor))

    def _crear_evaluacion(
        self,
        pregunta_id: str,
        dimension: str,
        punto_id: int,
        valor: float,
        evidencia: str,
        descripcion: Optional[str] = None,
        confidence: float = 0.0,
        detecciones: List[DetectionResult] = None,
    ) -> EvaluacionPregunta:
        """Crea una evaluación completa de pregunta."""
        valor = self._clamp(valor)
        respuesta = self._valor_a_respuesta(valor)

        # Formatear evidencia textual
        if descripcion and evidencia:
            evidencia_textual = f"{descripcion} → {evidencia}"
        elif descripcion and valor > 0:
            evidencia_textual = descripcion
        else:
            evidencia_textual = evidencia or ""

        # Generar evidencia contraria si aplica
        evidencia_contraria = (
            ""
            if valor > 0
            else "No se identificaron elementos suficientes que respondan a la pregunta con la evidencia disponible."
        )

        return EvaluacionPregunta(
            pregunta_id=pregunta_id,
            dimension=dimension,
            punto_id=punto_id,
            respuesta=respuesta,
            evidencia_textual=evidencia_textual,
            evidencia_contraria=evidencia_contraria,
            puntaje=valor,
            confidence=confidence,
            detecciones_asociadas=detecciones or [],
        )


# ==================== EVALUADOR INDUSTRIAL PRINCIPAL ====================


class IndustrialDecatalogoEvaluatorFull(_EvaluadorBase):
    """
    Evaluador industrial completo del decálogo con todos los detectores integrados.
    Versión de producción para evaluar 117 planes de desarrollo.
    """

    def __init__(self, contexto: Optional[DecalogoContext] = None):
        """Inicializa el evaluador con todos los componentes."""
        super().__init__()
        self.contexto = contexto or DECALOGO_CONTEXT

        # Inicializar detectores especializados
        self.scorer = FeasibilityScorer(enable_parallel=True)
        self.responsibility_detector = ResponsibilityDetector()

        # Definir preguntas por dimensión
        self.preguntas_de1 = self._definir_preguntas_de1()
        self.preguntas_de2 = self._definir_preguntas_de2()
        self.preguntas_de3 = self._definir_preguntas_de3()
        self.preguntas_de4 = self._definir_preguntas_de4()

        # Mapeo de puntos a clusters
        self.punto_a_cluster = self._construir_mapeo_punto_cluster()

        # Matriz de ponderación por dimensión (FIJA PARA COMPARABILIDAD)
        self.ponderacion_dimensiones = {
            "DE-1": 0.30,  # Lógica de Intervención y Coherencia Interna
            "DE-2": 0.25,  # Inclusión Temática
            "DE-3": 0.25,  # Participación y Gobernanza
            "DE-4": 0.20,  # Orientación a Resultados
        }

        # Umbrales de evaluación (FIJOS PARA ESTANDARIZACIÓN)
        self.umbrales = {
            "optimo": 80,
            "satisfactorio": 60,
            "basico": 40,
            "insuficiente": 0,
        }

        LOGGER.info("✅ Evaluador Industrial v3.0 inicializado correctamente")

    def _construir_mapeo_punto_cluster(self) -> Dict[int, int]:
        """Construye mapeo de puntos del decálogo a clusters."""
        # Basado en el agrupamiento del metodología
        return {
            # Cluster 1: PAZ, SEGURIDAD Y PROTECCIÓN DE DEFENSORES
            1: 1,  # Prevención de violencia
            5: 1,  # Derechos de víctimas
            8: 1,  # Líderes y defensores
            # Cluster 2: POBLACIONES VULNERABLES
            2: 2,  # Infancia y adolescencia
            3: 2,  # Mujeres y equidad de género
            4: 2,  # Personas con discapacidad
            # Cluster 3: DERECHOS SOCIALES Y ECONÓMICOS
            6: 3,  # Educación
            7: 3,  # Salud
            9: 3,  # Trabajo decente
            10: 3,  # Vivienda digna
        }

    @staticmethod
    def _definir_preguntas_de1() -> Dict[str, str]:
        """Define preguntas para DE-1: Lógica de Intervención."""
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
        """Define preguntas para DE-2: Inclusión Temática."""
        return {
            # Subdimensión 2.1: Diagnóstico con enfoque de derechos
            "D1": "Línea base 2023 con fuente citada",
            "D2": "Serie histórica ≥ 5 años",
            "D3": "Identificación de causas directas",
            "D4": "Identificación de causas estructurales",
            "D5": "Brechas territoriales detalladas",
            "D6": "Grupos poblacionales afectados identificados",
            # Subdimensión 2.2: Alineación Estratégica
            "O1": "Objetivo específico alineado con transformaciones del PND",
            "O2": "Indicador de resultado con línea base y meta transformadora",
            "O3": "Meta que aborde problemas estructurales del territorio",
            "O4": "Relación explícita con múltiples ODS",
            "O5": "Acción o programa con visión de largo plazo",
            "O6": "Articulación con determinantes ambientales del ordenamiento",
            # Subdimensión 2.3: Territorialización y PPI
            "T1": "Proyecto codificado en BPIN o código interno",
            "T2": "Monto plurianual 2024-2027",
            "T3": "Fuente de financiación identificada",
            "T4": "Localización geográfica específica",
            "T5": "Articulación con otros proyectos del territorio",
            "T6": "Participación comunitaria en formulación",
        }

    @staticmethod
    def _definir_preguntas_de3() -> List[str]:
        """Define preguntas para DE-3: Participación y Gobernanza."""
        return [
            "¿Mesas técnicas con actores clave documentadas?",
            "¿Diálogos ciudadanos con metodología verificable?",
            "¿Consulta previa con grupos étnicos (si aplica)?",
            "¿Mecanismos digitales de participación implementados?",
            "¿Rendición de cuentas con evidencia de respuesta ciudadana?",
            "¿Consejo Territorial de Planeación activo y documentado?",
        ]

    @staticmethod
    def _definir_preguntas_de4() -> List[str]:
        """Define preguntas para DE-4: Orientación a Resultados."""
        return [
            "¿Indicadores de producto con KPI bien definidos?",
            "¿Indicadores de resultado con línea base y meta?",
            "¿Indicadores de impacto para transformación sistémica?",
            "¿Sistema de monitoreo con instrumentos verificables?",
            "¿Ruta de evaluación con hitos temporales?",
            "¿Tablero de control para seguimiento público?",
            "¿Plan de gestión de riesgos identificados?",
            "¿Estrategia de sostenibilidad post-periodo de gobierno?",
        ]

    def _extraer_texto_entry(self, entry: Any) -> str:
        """Extrae texto de diferentes formatos de entrada."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            # Buscar texto en diferentes campos posibles
            for key in [
                "texto",
                "texto_evidencia",
                "segmento",
                "contenido",
                "descripcion",
                "value",
            ]:
                valor = entry.get(key)
                if valor:
                    return str(valor)
        if hasattr(entry, "__str__"):
            return str(entry)
        return ""

    def _extraer_textos(self, evidencia: Dict[str, List[Any]], *keys: str) -> List[str]:
        """Extrae todos los textos de las claves especificadas."""
        textos: List[str] = []
        for key in keys:
            elementos = evidencia.get(key, [])
            if isinstance(elementos, str):
                elementos = [elementos]
            for entry in elementos:
                texto = self._extraer_texto_entry(entry)
                if texto and len(texto.strip()) > 10:
                    textos.append(texto)
        return textos

    def _analizar_evidencia(
        self, evidencia: Dict[str, List[Any]]
    ) -> AnalisisEvidenciaDecalogo:
        """
        Análisis exhaustivo de evidencia con todos los detectores.
        Punto crítico para la calidad del análisis doctoral.
        """

        # 1. ANÁLISIS DE INDICADORES (FeasibilityScorer)
        textos_indicadores = self._extraer_textos(
            evidencia, "indicadores", "metas", "productos", "resultados", "impactos"
        )

        if textos_indicadores:
            # Usar batch_score para eficiencia
            batch_result = self.scorer.batch_score(
                textos_indicadores, use_parallel=True
            )
            indicador_scores = batch_result.scores
        else:
            indicador_scores = []

        indicadores_evaluados = list(zip(textos_indicadores, indicador_scores))

        # 2. ORGANIZAR DETECCIONES POR TIPO
        detecciones_por_tipo: Dict[ComponentType, List[DetectionResult]] = {
            component: [] for component in ComponentType
        }

        for _, score in indicadores_evaluados:
            if score.detailed_matches:
                for deteccion in score.detailed_matches:
                    detecciones_por_tipo[deteccion.component_type].append(
                        deteccion)

        # 3. DETECTAR RESPONSABILIDADES
        responsabilidades: List[ResponsibilityEntity] = []
        textos_responsables = self._extraer_textos(
            evidencia, "responsables", "actores", "entidades", "programas", "proyectos"
        )

        for texto in textos_responsables:
            entities = self.responsibility_detector.detect_entities(texto)
            responsabilidades.extend(entities)

        # 4. DETECTAR CONTRADICCIONES
        textos_analisis = self._extraer_textos(
            evidencia, "diagnostico", "objetivos", "estrategias", "justificacion"
        )

        todas_contradicciones = []
        analisis_contradicciones_global = None

        for texto in textos_analisis[:10]:  # Limitar para performance
            analisis_contradiccion = self.contradiction_detector.detect_contradictions(
                texto
            )
            if analisis_contradiccion.total_contradictions > 0:
                todas_contradicciones.extend(
                    analisis_contradiccion.contradictions)
                # Mantener el análisis más severo
                if (
                    not analisis_contradicciones_global
                    or analisis_contradiccion.risk_score
                    > analisis_contradicciones_global.risk_score
                ):
                    analisis_contradicciones_global = analisis_contradiccion

        # 5. DETECTAR VALORES MONETARIOS (si está disponible)
        valores_monetarios = []
        if self.monetary_detector:
            textos_presupuesto = self._extraer_textos(
                evidencia, "presupuesto", "recursos", "financiacion", "inversion"
            )
            for texto in textos_presupuesto:
                matches = self.monetary_detector.detect_monetary_expressions(
                    texto)
                valores_monetarios.extend(matches)

        # 6. CALCULAR MÉTRICAS AGREGADAS
        recursos = len(valores_monetarios) + \
            len(evidencia.get("presupuesto", []))
        plazos = len(detecciones_por_tipo.get(ComponentType.TIME_HORIZON, [])) + len(
            detecciones_por_tipo.get(ComponentType.DATE, [])
        )
        riesgos = len(todas_contradicciones)

        # 7. CALCULAR CONFIDENCE GLOBAL
        confidence_scores = []
        for score in indicador_scores:
            confidence_scores.append(score.feasibility_score)
        for resp in responsabilidades:
            confidence_scores.append(resp.confidence)

        confidence_global = np.mean(
            confidence_scores) if confidence_scores else 0.0

        # 8. METADATA DEL ANÁLISIS
        metadata_analisis = {
            "total_textos_analizados": len(textos_indicadores)
            + len(textos_responsables)
            + len(textos_analisis),
            "indicadores_de_calidad_alta": sum(
                1 for score in indicador_scores if score.feasibility_score >= 0.8
            ),
            "responsables_identificados": len(set(r.text for r in responsabilidades)),
            "contradicciones_criticas": sum(
                1
                for c in todas_contradicciones
                if c.risk_level.value in ["high", "medium-high"]
            ),
            "timestamp_analisis": datetime.now().isoformat(),
        }

        return AnalisisEvidenciaDecalogo(
            indicador_scores=indicador_scores,
            indicadores_evaluados=indicadores_evaluados,
            detecciones_por_tipo={k: v for k,
                                  v in detecciones_por_tipo.items() if v},
            responsabilidades=responsabilidades,
            contradicciones=todas_contradicciones,
            analisis_contradicciones=analisis_contradicciones_global,
            valores_monetarios=valores_monetarios,
            recursos=recursos,
            plazos=plazos,
            riesgos=riesgos,
            confidence_global=confidence_global,
            metadata_analisis=metadata_analisis,
        )

    def evaluar_dimension_de1(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        """
        Evalúa Dimensión DE-1: Lógica de Intervención y Coherencia Interna.
        Dimensión crítica para la causalidad del plan.
        """

        evaluaciones = []

        # Q1: Productos medibles alineados
        productos_detections = analisis.detecciones(ComponentType.TARGET)
        numerical_detections = analisis.detecciones(ComponentType.NUMERICAL)
        valor_q1 = 0.0
        if productos_detections:
            valor_q1 = min(1.0, len(productos_detections) * 0.2)
            if numerical_detections:
                valor_q1 = min(1.0, valor_q1 + 0.2)

        evaluaciones.append(
            self._crear_evaluacion(
                "Q1",
                "DE-1",
                punto_id,
                valor_q1,
                self._formatear_deteccion(
                    self._seleccionar_mejor_deteccion(productos_detections)
                ),
                self.preguntas_de1["Q1"],
                confidence=analisis.confidence_global,
                detecciones=productos_detections[:3],
            )
        )

        # Q2: Responsables institucionales
        valor_q2 = 0.0
        if analisis.responsabilidades:
            # Filtrar solo responsables institucionales
            institucionales = [
                r
                for r in analisis.responsabilidades
                if r.entity_type in [EntityType.ORGANIZATION, EntityType.POSITION]
            ]
            valor_q2 = min(1.0, len(institucionales) *
                           0.25) if institucionales else 0.1

        evaluaciones.append(
            self._crear_evaluacion(
                "Q2",
                "DE-1",
                punto_id,
                valor_q2,
                self._formatear_responsable(
                    self._seleccionar_mejor_responsable(
                        analisis.responsabilidades)
                ),
                self.preguntas_de1["Q2"],
                confidence=analisis.confidence_global,
            )
        )

        # Q3: Resultados con línea base y meta
        baseline_dets = analisis.detecciones(ComponentType.BASELINE)
        target_dets = analisis.detecciones(ComponentType.TARGET)
        valor_q3 = 0.0
        if baseline_dets:
            valor_q3 += 0.5
        if target_dets:
            valor_q3 += 0.5

        evidencia_q3 = f"Líneas base: {len(baseline_dets)}, Metas: {len(target_dets)}"

        evaluaciones.append(
            self._crear_evaluacion(
                "Q3",
                "DE-1",
                punto_id,
                valor_q3,
                evidencia_q3,
                self.preguntas_de1["Q3"],
                confidence=analisis.confidence_global,
            )
        )

        # Q4: Vinculación lógica cadena de valor
        valor_q4 = 0.0
        if baseline_dets and target_dets:
            valor_q4 = 0.6
            if analisis.plazos > 0:
                valor_q4 += 0.2
            # Penalizar por contradicciones
            if analisis.riesgos > 0:
                valor_q4 *= 1 - min(0.3, analisis.riesgos * 0.1)

        evaluaciones.append(
            self._crear_evaluacion(
                "Q4",
                "DE-1",
                punto_id,
                valor_q4,
                f"Coherencia causal {'identificada' if valor_q4 > 0.5 else 'parcial'} - Riesgos: {analisis.riesgos}",
                self.preguntas_de1["Q4"],
                confidence=analisis.confidence_global,
            )
        )

        # Q5: Impacto alineado al Decálogo
        valor_q5 = analisis.max_score * 0.8
        if valor_q5 > 0:
            # Bonus por alineación con punto específico
            if punto_id in self.punto_a_cluster:
                valor_q5 = min(1.0, valor_q5 + 0.1)

        evaluaciones.append(
            self._crear_evaluacion(
                "Q5",
                "DE-1",
                punto_id,
                valor_q5,
                f"Alineación con Decálogo: {valor_q5 * 100:.1f}%",
                self.preguntas_de1["Q5"],
                confidence=analisis.confidence_global,
            )
        )

        # Q6: Lógica de intervención explícita
        valor_q6 = (valor_q3 * 0.3) + (valor_q4 * 0.4) + (valor_q5 * 0.3)

        evaluaciones.append(
            self._crear_evaluacion(
                "Q6",
                "DE-1",
                punto_id,
                valor_q6,
                f"Completitud de la lógica: {valor_q6 * 100:.1f}%",
                self.preguntas_de1["Q6"],
                confidence=analisis.confidence_global,
            )
        )

        # CALCULAR MATRIZ CAUSAL AVANZADA
        puntajes = [e.puntaje for e in evaluaciones]

        # Criterios causales
        coherencia = np.mean([puntajes[0], puntajes[1]]
                             ) if puntajes[:2] else 0.0
        articulacion = np.mean([puntajes[2], puntajes[3]]) if len(
            puntajes) > 3 else 0.0
        consistencia = np.mean([puntajes[4], puntajes[5]]) if len(
            puntajes) > 5 else 0.0
        suficiencia = analisis.confidence_global

        # Ajuste por contradicciones
        if analisis.analisis_contradicciones:
            factor_contradiccion = 1 - (
                analisis.analisis_contradicciones.risk_score * 0.3
            )
            coherencia *= factor_contradiccion
            articulacion *= factor_contradiccion

        puntaje_causalidad = (
            coherencia + articulacion + consistencia + suficiencia
        ) / 4
        factor_causal = 0.5 + (puntaje_causalidad * 0.5)

        # CÁLCULO PONDERADO FINAL
        puntaje_bienestar = np.mean([puntajes[4], puntajes[5]]) * 60
        puntaje_resultado = np.mean([puntajes[2], puntajes[3]]) * 30
        puntaje_gestion = np.mean([puntajes[0], puntajes[1]]) * 10

        puntaje_dimension = (
            puntaje_bienestar * factor_causal + puntaje_resultado + puntaje_gestion
        )

        matriz_causal = {
            "coherencia": round(coherencia, 3),
            "articulacion": round(articulacion, 3),
            "consistencia": round(consistencia, 3),
            "suficiencia": round(suficiencia, 3),
            "factor_causal": round(factor_causal, 3),
            "puntaje_causalidad": round(puntaje_causalidad, 3),
            "contradicciones_detectadas": analisis.riesgos,
        }

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-1",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension,
            matriz_causal=matriz_causal,
            teoria_cambio_validada=factor_causal > 0.7,
            eslabones_completos=sum(1 for p in puntajes if p > 0.5),
            coherencia_interna=coherencia,
        )

    def evaluar_dimension_de2(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        """
        Evalúa Dimensión DE-2: Inclusión Temática.
        Verifica completitud del diagnóstico y alineación estratégica.
        """

        evaluaciones = []
        preguntas = list(self.preguntas_de2.items())

        # SUBDIMENSIÓN 2.1: DIAGNÓSTICO CON ENFOQUE DE DERECHOS (25%)
        valores_diagnostico = []

        # D1: Línea base 2023
        valor_d1 = 1.0 if analisis.tiene_linea_base else 0.0
        valores_diagnostico.append(valor_d1)

        # D2: Serie histórica
        dates_found = len(analisis.detecciones(ComponentType.DATE))
        valor_d2 = min(1.0, dates_found /
                       5) if dates_found >= 5 else dates_found / 10
        valores_diagnostico.append(valor_d2)

        # D3-D6: Análisis de causas y brechas
        for key in [
            "causas_directas",
            "causas_estructurales",
            "brechas_territoriales",
            "grupos_poblacionales",
        ]:
            elementos = evidencia.get(key, [])
            valor = min(1.0, len(elementos) * 0.33) if elementos else 0.0
            valores_diagnostico.append(valor)

        # SUBDIMENSIÓN 2.2: ALINEACIÓN ESTRATÉGICA (30%)
        valores_alineacion = []

        # O1-O6: Objetivos y alineación
        claves_alineacion = [
            "objetivos_pnd",
            "indicadores_transformadores",
            "problemas_estructurales",
            "ods",
            "vision_largo_plazo",
            "determinantes_ambientales",
        ]

        for clave in claves_alineacion:
            elementos = evidencia.get(clave, [])
            if clave == "ods":
                # Requiere múltiples ODS
                valor = (
                    min(1.0, len(elementos) / 3)
                    if len(elementos) >= 3
                    else len(elementos) / 6
                )
            else:
                valor = min(1.0, len(elementos) * 0.5) if elementos else 0.0
            valores_alineacion.append(valor)

        # SUBDIMENSIÓN 2.3: TERRITORIALIZACIÓN Y PPI (25%)
        valores_territorializacion = []

        # T1-T6: Aspectos de territorialización
        for key in [
            "codigo_bpin",
            "presupuesto_plurianual",
            "fuentes_financiacion",
            "localizacion",
            "articulacion_proyectos",
            "participacion_formulacion",
        ]:
            elementos = evidencia.get(key, [])
            valor = 1.0 if elementos else 0.0
            valores_territorializacion.append(valor)

        # SUBDIMENSIÓN 2.4: PARTICIPACIÓN (20%) - Se evalúa parcialmente aquí
        valor_participacion = len(evidencia.get("participacion", [])) > 0

        # Crear evaluaciones para cada pregunta
        todos_valores = (
            valores_diagnostico + valores_alineacion + valores_territorializacion
        )

        for idx, (pregunta_id, pregunta_texto) in enumerate(
            preguntas[: len(todos_valores)]
        ):
            valor = todos_valores[idx] if idx < len(todos_valores) else 0.0
            evaluaciones.append(
                self._crear_evaluacion(
                    pregunta_id,
                    "DE-2",
                    punto_id,
                    valor,
                    f"Criterio {'cumplido' if valor >= 0.5 else 'parcialmente cumplido' if valor > 0 else 'no cumplido'}",
                    pregunta_texto,
                    confidence=analisis.confidence_global,
                )
            )

        # CÁLCULO DE PUNTAJES PONDERADOS
        puntaje_diagnostico = (
            np.mean(valores_diagnostico) * 25 if valores_diagnostico else 0
        )
        puntaje_alineacion = (
            np.mean(valores_alineacion) * 30 if valores_alineacion else 0
        )
        puntaje_territorializacion = (
            np.mean(valores_territorializacion) * 25
            if valores_territorializacion
            else 0
        )
        puntaje_participacion_parcial = valor_participacion * 20

        puntaje_dimension = (
            puntaje_diagnostico
            + puntaje_alineacion
            + puntaje_territorializacion
            + puntaje_participacion_parcial
        )

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-2",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension,
            matriz_causal={
                "diagnostico": round(puntaje_diagnostico, 2),
                "alineacion": round(puntaje_alineacion, 2),
                "territorializacion": round(puntaje_territorializacion, 2),
                "participacion_parcial": round(puntaje_participacion_parcial, 2),
                "completitud_tematica": round(
                    sum(todos_valores) /
                    len(todos_valores) if todos_valores else 0, 3
                ),
            },
            coherencia_interna=np.mean(todos_valores) if todos_valores else 0,
        )

    def evaluar_dimension_de3(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        """
        Evalúa Dimensión DE-3: Participación y Gobernanza.
        Crítica para legitimidad democrática del plan.
        """

        evaluaciones = []

        # Mapeo de preguntas a claves de evidencia
        mapeo_evidencia = [
            ("mesas_tecnicas", 0.5),
            ("dialogos_ciudadanos", 0.4),
            ("consulta_previa", 0.8),  # Mayor peso si aplica
            ("mecanismos_digitales", 0.3),
            ("rendicion_cuentas", 0.6),
            ("consejo_territorial", 0.7),
        ]

        valores_participacion = []

        for idx, (pregunta_texto, (clave_evidencia, peso_base)) in enumerate(
            zip(self.preguntas_de3, mapeo_evidencia)
        ):
            elementos = evidencia.get(clave_evidencia, [])

            # Calcular valor base
            if elementos:
                valor = min(1.0, len(elementos) * peso_base)
                # Bonus por calidad de documentación
                if any("metodolog" in str(e).lower() for e in elementos):
                    valor = min(1.0, valor + 0.1)
                if any(
                    "acta" in str(e).lower() or "informe" in str(e).lower()
                    for e in elementos
                ):
                    valor = min(1.0, valor + 0.1)
            else:
                valor = 0.0

            # Caso especial: consulta previa
            if clave_evidencia == "consulta_previa":
                # Si no aplica, dar puntaje neutral
                if not evidencia.get("grupos_etnicos"):
                    valor = 0.5  # Neutro cuando no aplica

            valores_participacion.append(valor)

            evaluaciones.append(
                self._crear_evaluacion(
                    f"P{idx + 1}",
                    "DE-3",
                    punto_id,
                    valor,
                    f"{'Documentado' if valor > 0.5 else 'Parcialmente documentado' if valor > 0 else 'No documentado'}",
                    pregunta_texto,
                    confidence=analisis.confidence_global,
                )
            )

        # Análisis de calidad de gobernanza
        gobernanza_score = sum(v > 0.5 for v in valores_participacion) / len(
            valores_participacion
        )
        mecanismos_formales = sum(
            1 for v in valores_participacion[:3] if v > 0)
        mecanismos_digitales = sum(
            1 for v in valores_participacion[3:] if v > 0)

        # Ajuste por responsables identificados
        if len(analisis.responsabilidades) > 5:
            gobernanza_score = min(1.0, gobernanza_score + 0.1)

        puntaje_dimension = np.mean(valores_participacion) * 100

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-3",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension,
            matriz_causal={
                "participacion_activa": round(gobernanza_score, 3),
                "mecanismos_formales": mecanismos_formales,
                "mecanismos_digitales": mecanismos_digitales,
                "actores_identificados": len(
                    set(r.text for r in analisis.responsabilidades)
                ),
                "calidad_documental": round(
                    (
                        np.mean([v for v in valores_participacion if v > 0])
                        if any(valores_participacion)
                        else 0
                    ),
                    3,
                ),
            },
            coherencia_interna=gobernanza_score,
        )

    def evaluar_dimension_de4(
        self,
        analisis: AnalisisEvidenciaDecalogo,
        evidencia: Dict[str, List[Any]],
        punto_id: int,
    ) -> EvaluacionDimensionPunto:
        """
        Evalúa Dimensión DE-4: Orientación a Resultados.
        Fundamental para seguimiento y evaluación del plan.
        """

        evaluaciones = []

        # Métricas de cobertura por tipo de indicador
        coverage_productos = min(
            1.0,
            len([s for s in analisis.indicador_scores if s.has_quantitative_target])
            * 0.1,
        )
        coverage_resultados = analisis.max_score
        coverage_impactos = (
            coverage_resultados * 0.8 if coverage_resultados > 0.5 else 0
        )

        # Factor de riesgo basado en contradicciones
        if analisis.analisis_contradicciones:
            riesgo_factor = 1 - \
                (analisis.analisis_contradicciones.risk_score * 0.4)
        else:
            riesgo_factor = 1 - min(0.3, analisis.riesgos * 0.05)

        # Evaluar cada pregunta de orientación a resultados
        valores_preguntas = []

        # R1: Indicadores de producto con KPI
        valor_r1 = coverage_productos
        valores_preguntas.append(valor_r1)

        # R2: Indicadores de resultado con línea base
        valor_r2 = (
            coverage_resultados
            if analisis.tiene_linea_base
            else coverage_resultados * 0.5
        )
        valores_preguntas.append(valor_r2)

        # R3: Indicadores de impacto transformador
        valor_r3 = coverage_impactos
        valores_preguntas.append(valor_r3)

        # R4: Sistema de monitoreo
        monitoreo = evidencia.get("sistema_monitoreo", [])
        valor_r4 = min(1.0, len(monitoreo) * 0.5) if monitoreo else 0
        valores_preguntas.append(valor_r4)

        # R5: Ruta de evaluación con hitos
        hitos = evidencia.get("hitos", [])
        valor_r5 = min(1.0, len(hitos) * 0.25) if hitos else 0
        if analisis.plazos > 3:
            valor_r5 = min(1.0, valor_r5 + 0.2)
        valores_preguntas.append(valor_r5)

        # R6: Tablero de control
        tablero = evidencia.get("tablero_control", [])
        valor_r6 = 1.0 if tablero else 0.0
        valores_preguntas.append(valor_r6)

        # R7: Gestión de riesgos
        riesgos_plan = evidencia.get("plan_riesgos", [])
        if riesgos_plan:
            valor_r7 = min(1.0, len(riesgos_plan) * 0.3)
        else:
            # Penalizar si hay contradicciones pero no plan de riesgos
            valor_r7 = 0.0 if analisis.riesgos > 3 else 0.2
        valores_preguntas.append(valor_r7)

        # R8: Sostenibilidad post-gobierno
        sostenibilidad = evidencia.get("sostenibilidad", [])
        valor_r8 = min(1.0, len(sostenibilidad) * 0.5) if sostenibilidad else 0
        valores_preguntas.append(valor_r8)

        # Aplicar factor de riesgo a todas las evaluaciones
        for idx, (pregunta_texto, valor_base) in enumerate(
            zip(self.preguntas_de4, valores_preguntas)
        ):
            valor_ajustado = valor_base * riesgo_factor

            evaluaciones.append(
                self._crear_evaluacion(
                    f"R{idx + 1}",
                    "DE-4",
                    punto_id,
                    valor_ajustado,
                    f"Cobertura: {valor_base * 100:.1f}% (ajustado por riesgo: {valor_ajustado * 100:.1f}%)",
                    pregunta_texto,
                    confidence=analisis.confidence_global,
                )
            )

        # Cálculo agregado
        puntaje_dimension = np.mean([e.puntaje for e in evaluaciones]) * 100

        # Análisis por subdimensiones
        orientacion_resultados = np.mean(valores_preguntas[:3])
        sistemas_monitoreo = np.mean(valores_preguntas[3:6])
        gestion_sostenibilidad = np.mean(valores_preguntas[6:])

        return EvaluacionDimensionPunto(
            punto_id=punto_id,
            dimension="DE-4",
            evaluaciones_preguntas=evaluaciones,
            puntaje_dimension=puntaje_dimension,
            matriz_causal={
                "orientacion_resultados": round(orientacion_resultados, 3),
                "sistemas_monitoreo": round(sistemas_monitoreo, 3),
                "gestion_sostenibilidad": round(gestion_sostenibilidad, 3),
                "factor_riesgo": round(riesgo_factor, 3),
                "indicadores_cuantitativos": sum(
                    1 for s in analisis.indicador_scores if s.has_quantitative_target
                ),
                "cobertura_temporal": analisis.plazos,
            },
            coherencia_interna=orientacion_resultados * riesgo_factor,
        )

    def evaluar_punto_completo(
        self, evidencia: Dict[str, List[Any]], punto_id: int
    ) -> Tuple[
        EvaluacionPuntoCompleto,
        AnalisisEvidenciaDecalogo,
        Optional[ResultadoDimensionIndustrial],
    ]:
        """
        Evaluación completa de un punto del Decálogo en todas las dimensiones.
        Retorna evaluación detallada, análisis de evidencia y resultado industrial.
        """

        # ANÁLISIS DE EVIDENCIA (Punto crítico)
        analisis = self._analizar_evidencia(evidencia)

        # EVALUAR CADA DIMENSIÓN
        evaluaciones_dimensiones = [
            self.evaluar_dimension_de1(analisis, evidencia, punto_id),
            self.evaluar_dimension_de2(analisis, evidencia, punto_id),
            self.evaluar_dimension_de3(analisis, evidencia, punto_id),
            self.evaluar_dimension_de4(analisis, evidencia, punto_id),
        ]

        # CALCULAR PUNTAJE AGREGADO PONDERADO
        puntaje_agregado = sum(
            dim.puntaje_dimension * self.ponderacion_dimensiones[dim.dimension]
            for dim in evaluaciones_dimensiones
        )

        # CLASIFICACIÓN CUALITATIVA
        if puntaje_agregado >= self.umbrales["optimo"]:
            clasificacion = "DESEMPEÑO ÓPTIMO"
        elif puntaje_agregado >= self.umbrales["satisfactorio"]:
            clasificacion = "DESEMPEÑO SATISFACTORIO"
        elif puntaje_agregado >= self.umbrales["basico"]:
            clasificacion = "DESEMPEÑO BÁSICO"
        else:
            clasificacion = "DESEMPEÑO INSUFICIENTE"

        # GENERAR EXPLICACIÓN EXTENSA (Análisis doctoral)
        explicacion_partes = []

        # Evaluación general
        explicacion_partes.append(
            f"El punto {punto_id} del Decálogo presenta un {clasificacion.lower()} "
            f"con un puntaje agregado de {puntaje_agregado:.1f}/100. "
        )

        # Análisis dimensional
        for dim in evaluaciones_dimensiones:
            nivel = (
                "alto"
                if dim.puntaje_dimension >= 70
                else "medio"
                if dim.puntaje_dimension >= 40
                else "bajo"
            )
            explicacion_partes.append(
                f"En la dimensión {dim.dimension}, se observa un nivel {nivel} de desarrollo "
                f"({dim.puntaje_dimension:.1f}/100) con {dim.eslabones_completos if hasattr(dim, 'eslabones_completos') else 'varios'} "
                f"componentes verificados. "
            )

        # Análisis de evidencia
        if analisis.confidence_global >= 0.7:
            explicacion_partes.append(
                "La evidencia analizada muestra alta consistencia y confiabilidad. "
            )
        elif analisis.confidence_global >= 0.4:
            explicacion_partes.append(
                "La evidencia presenta consistencia moderada con oportunidades de fortalecimiento. "
            )
        else:
            explicacion_partes.append(
                "Se identifican brechas significativas en la evidencia disponible. "
            )

        # Análisis de riesgos
        if analisis.riesgos > 0:
            nivel_riesgo = (
                "crítico"
                if analisis.riesgos > 5
                else "moderado"
                if analisis.riesgos > 2
                else "bajo"
            )
            explicacion_partes.append(
                f"Se detectaron {analisis.riesgos} contradicciones potenciales, "
                f"representando un nivel de riesgo {nivel_riesgo} para la implementación. "
            )

        # Análisis de recursos
        if analisis.recursos > 0 or analisis.valores_monetarios:
            explicacion_partes.append(
                f"Se identificaron {analisis.recursos} elementos de asignación presupuestal "
                f"{'con valores monetarios específicos' if analisis.valores_monetarios else ''}. "
            )

        # IDENTIFICAR BRECHAS PRINCIPALES
        brechas = []
        for dim in evaluaciones_dimensiones:
            for evaluacion in dim.evaluaciones_preguntas:
                if evaluacion.puntaje < 0.5:
                    brecha = f"{dim.dimension}-{evaluacion.pregunta_id}: {evaluacion.evidencia_contraria[:100] if evaluacion.evidencia_contraria else 'Evidencia insuficiente'}"
                    brechas.append(brecha)

        # IDENTIFICAR FORTALEZAS
        fortalezas = []
        for dim in evaluaciones_dimensiones:
            if dim.puntaje_dimension >= 70:
                fortalezas.append(f"Alto desarrollo en {dim.dimension}")
            for evaluacion in dim.evaluaciones_preguntas:
                if evaluacion.puntaje >= 0.8:
                    fortalezas.append(
                        f"{evaluacion.pregunta_id}: {evaluacion.evidencia_textual[:50]}..."
                    )

        # GENERAR RECOMENDACIONES ESPECÍFICAS
        recomendaciones = []

        # Recomendaciones por dimensión
        for dim in evaluaciones_dimensiones:
            if dim.dimension == "DE-1" and dim.puntaje_dimension < 60:
                recomendaciones.append(
                    "PRIORIDAD ALTA: Fortalecer la lógica causal estableciendo vínculos explícitos "
                    "entre productos, resultados e impactos con indicadores medibles."
                )
            elif dim.dimension == "DE-2" and dim.puntaje_dimension < 60:
                recomendaciones.append(
                    "PRIORIDAD MEDIA: Mejorar el diagnóstico territorial incluyendo series históricas "
                    "y alineación con marcos estratégicos nacionales (PND, ODS)."
                )
            elif dim.dimension == "DE-3" and dim.puntaje_dimension < 60:
                recomendaciones.append(
                    "PRIORIDAD MEDIA: Ampliar mecanismos de participación ciudadana documentando "
                    "metodologías y resultados de espacios de diálogo."
                )
            elif dim.dimension == "DE-4" and dim.puntaje_dimension < 60:
                recomendaciones.append(
                    "PRIORIDAD ALTA: Desarrollar sistema robusto de monitoreo con tablero de control "
                    "y plan de gestión de riesgos identificados."
                )

        # Recomendaciones por contradicciones
        if analisis.riesgos > 3:
            recomendaciones.append(
                "RIESGO IDENTIFICADO: Resolver contradicciones detectadas entre objetivos y recursos "
                "para asegurar viabilidad de implementación."
            )

        # IDENTIFICAR RIESGOS
        riesgos_detectados = []
        if analisis.analisis_contradicciones:
            for contradiccion in analisis.contradicciones[:3]:
                riesgos_detectados.append(
                    f"Contradicción {contradiccion.risk_level.value}: {contradiccion.adversative_connector} "
                    f"en contexto de {', '.join(contradiccion.goal_keywords[:2])}"
                )

        if not analisis.tiene_linea_base:
            riesgos_detectados.append(
                "Ausencia de línea base dificulta medición de avances"
            )

        if not analisis.responsabilidades:
            riesgos_detectados.append(
                "Sin responsables institucionales claros para implementación"
            )

        # CONSTRUIR RESULTADO INDUSTRIAL
        nombre_punto = self.contexto.dimension_por_id.get(
            punto_id, DimensionDecalogo(
                punto_id, f"Punto {punto_id} del Decálogo", "")
        ).nombre

        resultado_industrial = ResultadoDimensionIndustrial(
            dimension_id=punto_id,
            puntaje_final=puntaje_agregado,
            componentes_evaluados={
                dim.dimension: dim.puntaje_dimension for dim in evaluaciones_dimensiones
            },
            brechas_identificadas=brechas[:10],
            recomendaciones=recomendaciones[:5],
            evidencia=evidencia,
            metadata={
                "fecha_evaluacion": datetime.now().isoformat(),
                "confidence_global": round(analisis.confidence_global, 3),
                "contradicciones_detectadas": analisis.riesgos,
                "responsables_identificados": len(
                    set(r.text for r in analisis.responsabilidades)
                ),
                "indicadores_evaluados": len(analisis.indicador_scores),
                "clasificacion": clasificacion,
                "cluster_id": self.punto_a_cluster.get(punto_id, 0),
            },
        )

        # CONSTRUIR EVALUACIÓN COMPLETA
        evaluacion_punto = EvaluacionPuntoCompleto(
            punto_id=punto_id,
            nombre_punto=nombre_punto,
            evaluaciones_dimensiones=evaluaciones_dimensiones,
            puntaje_agregado_punto=puntaje_agregado,
            clasificacion_cualitativa=clasificacion,
            explicacion_extensa=" ".join(explicacion_partes),
            brechas_identificadas=brechas[:10],
            recomendaciones_especificas=recomendaciones[:5],
            fortalezas_identificadas=fortalezas[:5],
            riesgos_detectados=riesgos_detectados[:5],
        )

        return evaluacion_punto, analisis, resultado_industrial

    def evaluar_cluster(
        self,
        cluster: ClusterMetadata,
        evidencias_por_punto: Dict[int, Dict[str, List[Any]]],
    ) -> Tuple[EvaluacionClusterCompleto, List[ResultadoDimensionIndustrial]]:
        """
        Evalúa un cluster completo del Decálogo (análisis meso).
        Agregación coherente de puntos relacionados.
        """

        evaluaciones_puntos = []
        resultados_industriales = []
        puntajes_puntos = []

        # Evaluar cada punto del cluster
        for punto_id in cluster.puntos:
            if punto_id not in evidencias_por_punto:
                LOGGER.warning(
                    f"⚠️ Sin evidencia para punto {punto_id} en cluster {cluster.titulo}"
                )
                continue

            evaluacion_punto, analisis, resultado_industrial = (
                self.evaluar_punto_completo(
                    evidencias_por_punto[punto_id], punto_id)
            )

            evaluaciones_puntos.append(evaluacion_punto)
            puntajes_puntos.append(evaluacion_punto.puntaje_agregado_punto)

            if resultado_industrial:
                resultados_industriales.append(resultado_industrial)

        if not evaluaciones_puntos:
            LOGGER.error(f"❌ Sin evaluaciones para cluster {cluster.titulo}")
            return self._crear_cluster_vacio(cluster), []

        # Calcular métricas del cluster
        puntaje_agregado = np.mean(puntajes_puntos)
        varianza_interna = np.std(puntajes_puntos)
        coherencia_cluster = (
            1 - (varianza_interna / 100) if varianza_interna < 100 else 0
        )

        # Clasificación del cluster
        if puntaje_agregado >= self.umbrales["optimo"]:
            clasificacion = "Cluster de Alto Desempeño"
        elif puntaje_agregado >= self.umbrales["satisfactorio"]:
            clasificacion = "Cluster de Desempeño Satisfactorio"
        elif puntaje_agregado >= self.umbrales["basico"]:
            clasificacion = "Cluster de Desempeño Básico"
        else:
            clasificacion = "Cluster de Desempeño Crítico"

        # Análisis meso detallado
        analisis_meso = {
            "puntaje_promedio": round(puntaje_agregado, 2),
            "desviacion_estandar": round(varianza_interna, 2),
            "coherencia_interna": round(coherencia_cluster, 3),
            "puntos_fuertes": [
                p.nombre_punto
                for p in evaluaciones_puntos
                if p.puntaje_agregado_punto >= self.umbrales["satisfactorio"]
            ],
            "puntos_criticos": [
                p.nombre_punto
                for p in evaluaciones_puntos
                if p.puntaje_agregado_punto < self.umbrales["basico"]
            ],
            "promedio_por_dimension": {},
            "distribucion_clasificaciones": {},
        }

        # Análisis por dimensión
        for dimension in ["DE-1", "DE-2", "DE-3", "DE-4"]:
            puntajes_dimension = []
            for punto in evaluaciones_puntos:
                for eval_dim in punto.evaluaciones_dimensiones:
                    if eval_dim.dimension == dimension:
                        puntajes_dimension.append(eval_dim.puntaje_dimension)
            if puntajes_dimension:
                analisis_meso["promedio_por_dimension"][dimension] = round(
                    np.mean(puntajes_dimension), 2
                )

        # Distribución de clasificaciones
        for punto in evaluaciones_puntos:
            clasif = punto.clasificacion_cualitativa
            analisis_meso["distribucion_clasificaciones"][clasif] = (
                analisis_meso["distribucion_clasificaciones"].get(
                    clasif, 0) + 1
            )

        # Identificar interdependencias (simulado por ahora)
        interdependencias = {}
        for i, punto_i in enumerate(cluster.puntos):
            for j, punto_j in enumerate(cluster.puntos):
                if i < j:
                    # Correlación simulada basada en puntajes similares
                    if len(puntajes_puntos) > max(i, j):
                        correlacion = (
                            1 - abs(puntajes_puntos[i] -
                                    puntajes_puntos[j]) / 100
                        )
                        interdependencias[f"{punto_i}-{punto_j}"] = round(
                            correlacion, 3
                        )

        return (
            EvaluacionClusterCompleto(
                cluster_nombre=cluster.titulo,
                cluster_id=cluster.cluster_id,
                evaluaciones_puntos=evaluaciones_puntos,
                puntaje_agregado_cluster=puntaje_agregado,
                clasificacion_cualitativa=clasificacion,
                analisis_meso_completo=analisis_meso,
                interdependencias_cluster=interdependencias,
                coherencia_cluster=coherencia_cluster,
                varianza_interna=varianza_interna,
            ),
            resultados_industriales,
        )

    def _crear_cluster_vacio(
        self, cluster: ClusterMetadata
    ) -> EvaluacionClusterCompleto:
        """Crea evaluación vacía para cluster sin evidencia."""
        return EvaluacionClusterCompleto(
            cluster_nombre=cluster.titulo,
            cluster_id=cluster.cluster_id,
            evaluaciones_puntos=[],
            puntaje_agregado_cluster=0.0,
            clasificacion_cualitativa="Sin Evaluación",
            analisis_meso_completo={"error": "Sin evidencia disponible"},
        )

    def generar_reporte_final(
        self, evidencias_por_punto: Dict[int, Dict[str, List[Any]]], nombre_plan: str
    ) -> ReporteFinalDecatalogo:
        """
        Genera el reporte final completo con análisis macro, meso y micro.
        Este es el punto de entrada principal para evaluación de un plan completo.
        """

        LOGGER.info(f"📊 Generando reporte final para: {nombre_plan}")

        # Contenedores para resultados
        evaluaciones_clusters = []
        resultados_dimension = []
        evaluaciones_puntos = []
        evaluaciones_preguntas = []
        todas_brechas = []
        todas_recomendaciones = []

        # EVALUAR CADA CLUSTER
        for cluster in self.contexto.clusters:
            LOGGER.info(f"  Evaluando cluster: {cluster.titulo}")

            eval_cluster, resultados_ind = self.evaluar_cluster(
                cluster, evidencias_por_punto
            )

            evaluaciones_clusters.append(eval_cluster)
            resultados_dimension.extend(resultados_ind)

            # Agregar evaluaciones detalladas
            for eval_punto in eval_cluster.evaluaciones_puntos:
                evaluaciones_puntos.append(eval_punto)
                todas_brechas.extend(eval_punto.brechas_identificadas)
                todas_recomendaciones.extend(
                    eval_punto.recomendaciones_especificas)

                for eval_dim in eval_punto.evaluaciones_dimensiones:
                    evaluaciones_preguntas.extend(
                        eval_dim.evaluaciones_preguntas)

        # CALCULAR PUNTAJE GLOBAL
        if evaluaciones_clusters:
            puntaje_global = np.mean(
                [c.puntaje_agregado_cluster for c in evaluaciones_clusters]
            )
        else:
            puntaje_global = 0.0

        # IDENTIFICAR MEJOR Y PEOR CLUSTER
        if evaluaciones_clusters:
            clusters_ordenados = sorted(
                evaluaciones_clusters,
                key=lambda c: c.puntaje_agregado_cluster,
                reverse=True,
            )
            cluster_mejor = clusters_ordenados[0].cluster_nombre
            puntaje_mejor = clusters_ordenados[0].puntaje_agregado_cluster
            cluster_peor = clusters_ordenados[-1].cluster_nombre
            puntaje_peor = clusters_ordenados[-1].puntaje_agregado_cluster
        else:
            cluster_mejor = cluster_peor = "N/A"
            puntaje_mejor = puntaje_peor = 0.0

        # ANÁLISIS DE ALINEACIÓN
        if puntaje_global >= 75:
            alineacion = "ALTA"
        elif puntaje_global >= 50:
            alineacion = "MEDIA"
        else:
            alineacion = "BAJA"

        # RESUMEN EJECUTIVO
        resumen_ejecutivo = {
            "puntaje_global": round(puntaje_global, 2),
            "nivel_alineacion": alineacion,
            "cluster_mejor_desempeno": f"{cluster_mejor} ({puntaje_mejor:.1f}/100)",
            "cluster_peor_desempeno": f"{cluster_peor} ({puntaje_peor:.1f}/100)",
            "numero_puntos_evaluados": len(evaluaciones_puntos),
            "numero_preguntas_evaluadas": len(evaluaciones_preguntas),
            "numero_clusters_evaluados": len(evaluaciones_clusters),
            "recomendacion_estrategica_global": (
                "✅ PLAN APTO PARA IMPLEMENTACIÓN - Alta alineación con el Decálogo de DDHH"
                if puntaje_global >= 70
                else (
                    "⚠️ PLAN REQUIERE AJUSTES - Brechas moderadas en alineación con DDHH"
                    if puntaje_global >= 50
                    else "❌ PLAN REQUIERE REFORMULACIÓN - Brechas críticas en DDHH"
                )
            ),
            "nivel_confianza_evaluacion": (
                round(
                    np.mean(
                        [
                            p.confidence
                            for p in evaluaciones_preguntas
                            if hasattr(p, "confidence") and p.confidence > 0
                        ]
                    ),
                    3,
                )
                if evaluaciones_preguntas
                else 0.0
            ),
            "fecha_evaluacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # ANÁLISIS ESTADÍSTICO GLOBAL
        estadisticas_globales = {
            "distribucion_puntajes": {
                "media": (
                    round(
                        np.mean(
                            [p.puntaje_agregado_punto for p in evaluaciones_puntos]
                        ),
                        2,
                    )
                    if evaluaciones_puntos
                    else 0
                ),
                "mediana": (
                    round(
                        np.median(
                            [p.puntaje_agregado_punto for p in evaluaciones_puntos]
                        ),
                        2,
                    )
                    if evaluaciones_puntos
                    else 0
                ),
                "desviacion_estandar": (
                    round(
                        np.std(
                            [p.puntaje_agregado_punto for p in evaluaciones_puntos]),
                        2,
                    )
                    if evaluaciones_puntos
                    else 0
                ),
                "minimo": (
                    round(
                        min([p.puntaje_agregado_punto for p in evaluaciones_puntos]), 2
                    )
                    if evaluaciones_puntos
                    else 0
                ),
                "maximo": (
                    round(
                        max([p.puntaje_agregado_punto for p in evaluaciones_puntos]), 2
                    )
                    if evaluaciones_puntos
                    else 0
                ),
            },
            "distribucion_respuestas": {
                "si": sum(1 for q in evaluaciones_preguntas if q.respuesta == "Sí"),
                "parcial": sum(
                    1 for q in evaluaciones_preguntas if q.respuesta == "Parcial"
                ),
                "no": sum(1 for q in evaluaciones_preguntas if q.respuesta == "No"),
                "ni": sum(1 for q in evaluaciones_preguntas if q.respuesta == "NI"),
            },
            "analisis_dimensional": {},
            "indicadores_clave": {
                "puntos_optimos": sum(
                    1
                    for p in evaluaciones_puntos
                    if p.puntaje_agregado_punto >= self.umbrales["optimo"]
                ),
                "puntos_satisfactorios": sum(
                    1
                    for p in evaluaciones_puntos
                    if self.umbrales["satisfactorio"]
                    <= p.puntaje_agregado_punto
                    < self.umbrales["optimo"]
                ),
                "puntos_basicos": sum(
                    1
                    for p in evaluaciones_puntos
                    if self.umbrales["basico"]
                    <= p.puntaje_agregado_punto
                    < self.umbrales["satisfactorio"]
                ),
                "puntos_insuficientes": sum(
                    1
                    for p in evaluaciones_puntos
                    if p.puntaje_agregado_punto < self.umbrales["basico"]
                ),
            },
        }

        # Análisis por dimensión
        for dim in ["DE-1", "DE-2", "DE-3", "DE-4"]:
            puntajes_dim = []
            for punto in evaluaciones_puntos:
                for eval_dim in punto.evaluaciones_dimensiones:
                    if eval_dim.dimension == dim:
                        puntajes_dim.append(eval_dim.puntaje_dimension)

            if puntajes_dim:
                estadisticas_globales["analisis_dimensional"][dim] = {
                    "promedio": round(np.mean(puntajes_dim), 2),
                    "desviacion": round(np.std(puntajes_dim), 2),
                    "minimo": round(min(puntajes_dim), 2),
                    "maximo": round(max(puntajes_dim), 2),
                    "coeficiente_variacion": round(
                        (
                            np.std(puntajes_dim) / np.mean(puntajes_dim)
                            if np.mean(puntajes_dim) > 0
                            else 0
                        ),
                        3,
                    ),
                }

        # GENERAR EXPLICACIÓN MACRO EXTENSA
        explicacion_macro = f"""
        EVALUACIÓN INTEGRAL DEL PLAN DE DESARROLLO TERRITORIAL '{nombre_plan}'

        El presente análisis evalúa exhaustivamente el Plan de Desarrollo Territorial contra 
        el Decálogo de Derechos Humanos de la Defensoría del Pueblo, examinando {len(evaluaciones_puntos)} 
        puntos fundamentales a través de {len(evaluaciones_preguntas)} criterios específicos 
        distribuidos en 4 dimensiones evaluativas.

        RESULTADO GLOBAL:
        El plan obtiene un puntaje global de {puntaje_global:.1f}/100, lo que representa una 
        alineación {alineacion.lower()} con los estándares del Decálogo. Este resultado se 
        fundamenta en el análisis sistemático de {len(evaluaciones_clusters)} clusters temáticos 
        que agrupan los derechos evaluados.

        ANÁLISIS POR CLUSTERS:
        El cluster con mejor desempeño es '{cluster_mejor}' con {puntaje_mejor:.1f} puntos, 
        demostrando fortalezas significativas en su formulación e implementación propuesta. 
        En contraste, el cluster '{cluster_peor}' con {puntaje_peor:.1f} puntos requiere 
        atención prioritaria para alcanzar los estándares mínimos del Decálogo.

        DISTRIBUCIÓN DE CALIDAD:
        - Puntos con desempeño óptimo: {estadisticas_globales["indicadores_clave"]["puntos_optimos"]}
        - Puntos con desempeño satisfactorio: {estadisticas_globales["indicadores_clave"]["puntos_satisfactorios"]}
        - Puntos con desempeño básico: {estadisticas_globales["indicadores_clave"]["puntos_basicos"]}
        - Puntos con desempeño insuficiente: {estadisticas_globales["indicadores_clave"]["puntos_insuficientes"]}

        ANÁLISIS DIMENSIONAL:
        La evaluación dimensional revela patrones importantes en la estructura del plan:
        {self._generar_analisis_dimensional(estadisticas_globales["analisis_dimensional"])}

        BRECHAS Y OPORTUNIDADES:
        Se identificaron {len(set(todas_brechas))} brechas únicas que requieren intervención, 
        con {len(set(todas_recomendaciones))} recomendaciones específicas para fortalecer 
        la alineación del plan con los estándares de derechos humanos.

        RECOMENDACIÓN ESTRATÉGICA:
        {resumen_ejecutivo["recomendacion_estrategica_global"]}

        NIVEL DE CONFIANZA:
        Esta evaluación se realiza con un nivel de confianza del {resumen_ejecutivo["nivel_confianza_evaluacion"] * 100:.1f}% 
        basado en la calidad y consistencia de la evidencia analizada.
        """

        # PROCESAR BRECHAS Y RECOMENDACIONES ÚNICAS
        brechas_unicas = list(dict.fromkeys(todas_brechas))[:30]
        recomendaciones_unicas = list(
            dict.fromkeys(todas_recomendaciones))[:20]

        # REPORTE MACRO
        reporte_macro = {
            "alineacion_global_decatalogo": alineacion,
            "explicacion_extensa_cualitativa": explicacion_macro.strip(),
            "brechas_globales": brechas_unicas,
            "recomendaciones_globales": recomendaciones_unicas,
            "indicadores_resumen": estadisticas_globales["indicadores_clave"],
            "analisis_dimensional": estadisticas_globales["analisis_dimensional"],
            "distribucion_calidad": {
                "optimo": (
                    estadisticas_globales["indicadores_clave"]["puntos_optimos"]
                    / len(evaluaciones_puntos)
                    * 100
                    if evaluaciones_puntos
                    else 0
                ),
                "satisfactorio": (
                    estadisticas_globales["indicadores_clave"]["puntos_satisfactorios"]
                    / len(evaluaciones_puntos)
                    * 100
                    if evaluaciones_puntos
                    else 0
                ),
                "basico": (
                    estadisticas_globales["indicadores_clave"]["puntos_basicos"]
                    / len(evaluaciones_puntos)
                    * 100
                    if evaluaciones_puntos
                    else 0
                ),
                "insuficiente": (
                    estadisticas_globales["indicadores_clave"]["puntos_insuficientes"]
                    / len(evaluaciones_puntos)
                    * 100
                    if evaluaciones_puntos
                    else 0
                ),
            },
        }

        # GENERAR MATRIZ DE TRAZABILIDAD (si pandas está disponible)
        matriz_trazabilidad = None
        if HAS_PANDAS and evaluaciones_preguntas:
            try:
                data_matriz = []
                for pregunta in evaluaciones_preguntas:
                    data_matriz.append(
                        {
                            "Punto": pregunta.punto_id,
                            "Cluster": self.punto_a_cluster.get(pregunta.punto_id, 0),
                            "Dimensión": pregunta.dimension,
                            "Pregunta": pregunta.pregunta_id,
                            "Respuesta": pregunta.respuesta,
                            "Puntaje": round(pregunta.puntaje * 100, 1),
                            "Confianza": (
                                round(pregunta.confidence * 100, 1)
                                if hasattr(pregunta, "confidence")
                                else 0.0
                            ),
                            "Evidencia": (
                                pregunta.evidencia_textual[:150]
                                if pregunta.evidencia_textual
                                else ""
                            ),
                        }
                    )
                matriz_trazabilidad = pd.DataFrame(data_matriz)

                # Guardar CSV para comparabilidad entre municipios
                archivo_csv = f"trazabilidad_{nombre_plan.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
                matriz_trazabilidad.to_csv(
                    archivo_csv, index=False, encoding="utf-8-sig"
                )
                LOGGER.info(
                    f"  ✅ Matriz de trazabilidad guardada: {archivo_csv}")

            except Exception as e:
                LOGGER.error(f"  ❌ Error generando matriz trazabilidad: {e}")

        # ANEXOS SERIALIZABLES
        anexos_serializables = {
            "resultados_industriales": [
                {
                    "dimension_id": r.dimension_id,
                    "puntaje": round(r.puntaje_final, 2),
                    "componentes": r.componentes_evaluados,
                    "metadata": r.metadata if hasattr(r, "metadata") else {},
                    "brechas_principales": (
                        r.brechas_identificadas[:3]
                        if hasattr(r, "brechas_identificadas")
                        else []
                    ),
                }
                for r in resultados_dimension
            ],
            "estadisticas_evaluacion": {
                "total_evidencias_procesadas": sum(
                    len(evidencias_por_punto.get(i, {})) for i in range(1, 11)
                ),
                "tiempo_procesamiento": datetime.now().isoformat(),
                "version_evaluador": "3.0-PRODUCTION",
                "modo_evaluacion": "INDUSTRIAL_COMPLETO",
                "municipio": nombre_plan,
                "parametros_evaluacion": {
                    "ponderacion_dimensiones": self.ponderacion_dimensiones,
                    "umbrales_clasificacion": self.umbrales,
                },
            },
            "resumen_clusters": [
                {
                    "cluster": c.cluster_nombre,
                    "puntaje": round(c.puntaje_agregado_cluster, 2),
                    "clasificacion": c.clasificacion_cualitativa,
                    "coherencia": (
                        round(c.coherencia_cluster, 3)
                        if hasattr(c, "coherencia_cluster")
                        else 0
                    ),
                    "varianza": (
                        round(c.varianza_interna, 2)
                        if hasattr(c, "varianza_interna")
                        else 0
                    ),
                }
                for c in evaluaciones_clusters
            ],
        }

        # CONSTRUIR REPORTE FINAL
        reporte_final = ReporteFinalDecatalogo(
            metadata={
                "nombre_plan": nombre_plan,
                "fecha_evaluacion": datetime.now().isoformat(),
                "version_evaluador": "3.0-PRODUCTION",
                "puntos_evaluados": len(evaluaciones_puntos),
                "preguntas_procesadas": len(evaluaciones_preguntas),
                "clusters_evaluados": len(evaluaciones_clusters),
                "tiempo_procesamiento": datetime.now().isoformat(),
            },
            resumen_ejecutivo=resumen_ejecutivo,
            reporte_macro=reporte_macro,
            reporte_meso_por_cluster=evaluaciones_clusters,
            reporte_por_punto=evaluaciones_puntos,
            reporte_por_pregunta=evaluaciones_preguntas,
            resultados_dimension=resultados_dimension,
            anexos_serializables=anexos_serializables,
            matriz_trazabilidad_global=matriz_trazabilidad,
            estadisticas_globales=estadisticas_globales,
        )

        LOGGER.info(
            f"✅ Reporte final generado exitosamente para {nombre_plan}")
        LOGGER.info(f"   - Puntaje Global: {puntaje_global:.1f}/100")
        LOGGER.info(f"   - Clasificación: {alineacion}")
        LOGGER.info(
            f"   - Confianza: {resumen_ejecutivo['nivel_confianza_evaluacion'] * 100:.1f}%"
        )

        return reporte_final

    def _generar_analisis_dimensional(self, analisis_dimensional: Dict) -> str:
        """Genera texto de análisis dimensional."""
        if not analisis_dimensional:
            return "Sin datos dimensionales disponibles."

        partes = []
        for dim, stats in analisis_dimensional.items():
            interpretacion = (
                "alto"
                if stats["promedio"] >= 70
                else "medio"
                if stats["promedio"] >= 40
                else "bajo"
            )
            consistencia = (
                "alta"
                if stats["coeficiente_variacion"] < 0.2
                else "media"
                if stats["coeficiente_variacion"] < 0.4
                else "baja"
            )
            partes.append(
                f"- {dim}: Desempeño {interpretacion} (μ={stats['promedio']:.1f}, σ={stats['desviacion']:.1f}) "
                f"con consistencia {consistencia} (CV={stats['coeficiente_variacion']:.2f})"
            )
        return "\n".join(partes)


# ==================== FUNCIÓN DE INTEGRACIÓN CON SISTEMA PRINCIPAL ====================


def integrar_evaluador_decatalogo(
    sistema: SistemaEvaluacionIndustrial, dimension: DimensionDecalogo
) -> Optional[ResultadoDimensionIndustrial]:
    """
    Integración completa del evaluador con el sistema industrial principal.
    Punto de entrada desde Decatalogo_principal.py
    """

    if not sistema.extractor:
        LOGGER.error("❌ Sistema sin extractor inicializado")
        return None

    try:
        # Inicializar evaluador
        evaluador = IndustrialDecatalogoEvaluatorFull()

        # Extraer evidencia usando el extractor del sistema
        evidencia_dimension = sistema.extractor.extraer_variables_operativas(
            dimension)

        if not evidencia_dimension:
            LOGGER.warning(f"⚠️ Sin evidencia para dimensión {dimension.id}")
            return None

        # Evaluar punto completo
        evaluacion_punto, analisis, resultado_industrial = (
            evaluador.evaluar_punto_completo(evidencia_dimension, dimension.id)
        )

        if not resultado_industrial:
            LOGGER.error(
                f"❌ No se pudo generar resultado para dimensión {dimension.id}"
            )
            return None

        # Enriquecer con matriz de trazabilidad si está disponible
        if hasattr(sistema.extractor, "generar_matriz_trazabilidad"):
            try:
                matriz = sistema.extractor.generar_matriz_trazabilidad(
                    dimension)
                if matriz is not None:
                    resultado_industrial.matriz_trazabilidad = matriz
            except Exception as e:
                LOGGER.warning(
                    f"⚠️ No se pudo generar matriz trazabilidad: {e}")

        # Obtener metadata del cluster
        cluster_metadata = DECALOGO_CONTEXT.cluster_por_dimension.get(
            dimension.id)

        if cluster_metadata:
            # Agregar información del cluster al resultado
            if hasattr(resultado_industrial, "metadata"):
                resultado_industrial.metadata["cluster"] = cluster_metadata.titulo
                resultado_industrial.metadata["cluster_id"] = (
                    cluster_metadata.cluster_id
                )

        LOGGER.info(
            f"✅ Evaluación exitosa: Dimensión {dimension.id} | "
            f"Puntaje: {resultado_industrial.puntaje_final:.1f}/100 | "
            f"Confianza: {resultado_industrial.metadata.get('confidence_global', 0) * 100:.1f}%"
        )

        return resultado_industrial

    except Exception as exc:
        LOGGER.error(
            f"❌ Error en evaluación de dimensión {dimension.id}: {exc}", exc_info=True
        )
        return None


# ==================== PUNTO DE ENTRADA PRINCIPAL ====================


def main():
    """Función principal para pruebas del evaluador v3.0."""

    print("=" * 100)
    print("EVALUADOR INDUSTRIAL DEL DECÁLOGO v3.0 PRODUCTION")
    print("Sistema Doctoral de Análisis de Planes de Desarrollo")
    print("=" * 100)

    # Crear evaluador
    evaluador = IndustrialDecatalogoEvaluatorFull()

    # Evidencia de ejemplo más completa
    evidencia_ejemplo = {
        "indicadores": [
            "Incrementar la cobertura educativa del 65% (línea base 2023) al 85% para 2027",
            "Reducir la tasa de homicidios de 25 por 100,000 habitantes a 15 por 100,000 para 2027",
            "Aumentar el acceso a servicios de salud del 70% al 90% de la población vulnerable",
        ],
        "responsables": [
            "La Secretaría de Educación Municipal liderará el programa",
            "La Oficina de Seguridad Ciudadana coordinará con la Policía Nacional",
            "El Instituto Municipal de Salud implementará la estrategia",
        ],
        "diagnostico": [
            "El municipio presenta brechas significativas en acceso a educación de calidad",
            "La violencia urbana afecta principalmente a jóvenes entre 15-25 años en zonas periféricas",
            "El 30% de la población no tiene acceso regular a servicios de salud",
        ],
        "objetivos_pnd": ["Transformación educativa", "Colombia potencia de vida"],
        "ods": [
            "ODS 4: Educación de calidad",
            "ODS 16: Paz, justicia e instituciones sólidas",
            "ODS 10: Reducción de las desigualdades",
            "ODS 3: Salud y bienestar",
        ],
        "presupuesto": [
            "$500 millones COP para infraestructura educativa",
            "$300 millones COP para programas de seguridad",
            "$750 millones COP para ampliación de servicios de salud",
        ],
        "participacion": [
            "Mesa técnica con comunidad educativa realizada en enero 2024",
            "Diálogos ciudadanos en 10 comunas del municipio",
        ],
        "sostenibilidad": [
            "Plan de continuidad establecido con horizonte 2024-2035",
            "Estrategia de gestión de recursos con cooperación internacional",
        ],
    }

    # Evaluar punto de ejemplo
    print("\n📊 EVALUANDO PUNTO DE EJEMPLO...")
    print("-" * 50)

    evaluacion, analisis, resultado = evaluador.evaluar_punto_completo(
        evidencia_ejemplo, punto_id=1
    )

    print(f"\n✅ EVALUACIÓN COMPLETADA:")
    print(f"   📍 Punto: {evaluacion.nombre_punto}")
    print(f"   📊 Puntaje: {evaluacion.puntaje_agregado_punto:.1f}/100")
    print(f"   🏆 Clasificación: {evaluacion.clasificacion_cualitativa}")
    print(f"   🔍 Confianza: {analisis.confidence_global * 100:.1f}%")

    print(f"\n📝 EXPLICACIÓN DETALLADA:")
    print(f"   {evaluacion.explicacion_extensa[:500]}...")

    if evaluacion.fortalezas_identificadas:
        print(f"\n💪 FORTALEZAS IDENTIFICADAS:")
        for i, fortaleza in enumerate(evaluacion.fortalezas_identificadas[:3], 1):
            print(f"   {i}. {fortaleza}")

    if evaluacion.brechas_identificadas:
        print(f"\n⚠️ BRECHAS PRINCIPALES:")
        for i, brecha in enumerate(evaluacion.brechas_identificadas[:3], 1):
            print(f"   {i}. {brecha}")

    if evaluacion.recomendaciones_especificas:
        print(f"\n💡 RECOMENDACIONES PRIORITARIAS:")
        for i, rec in enumerate(evaluacion.recomendaciones_especificas[:3], 1):
            print(f"   {i}. {rec}")

    if evaluacion.riesgos_detectados:
        print(f"\n🚨 RIESGOS DETECTADOS:")
        for i, riesgo in enumerate(evaluacion.riesgos_detectados[:3], 1):
            print(f"   {i}. {riesgo}")

    # Análisis dimensional
    print(f"\n📈 ANÁLISIS POR DIMENSIÓN:")
    for dim in evaluacion.evaluaciones_dimensiones:
        print(
            f"   {dim.dimension}: {dim.puntaje_dimension:.1f}/100 - "
            f"{'✅' if dim.puntaje_dimension >= 60 else '⚠️' if dim.puntaje_dimension >= 40 else '❌'}"
        )
        # Mostrar detalle de evidencia analizada
        print(f"\n🔬 ANÁLISIS DE EVIDENCIA:")
        print(f"   - Indicadores evaluados: {len(analisis.indicador_scores)}")
        print(
            f"   - Responsables identificados: {len(set(r.text for r in analisis.responsabilidades))}"
        )
        print(f"   - Contradicciones detectadas: {analisis.riesgos}")
        print(
            f"   - Línea base presente: {'✅' if analisis.tiene_linea_base else '❌'}"
        )
        print(
            f"   - Metas identificadas: {'✅' if analisis.tiene_metas else '❌'}")

        # Generar reporte completo para múltiples puntos
        print("\n" + "=" * 100)
        print("GENERANDO REPORTE COMPLETO MULTI-PUNTO")
        print("=" * 100)

        # Simular evidencias para 10 puntos del decálogo
        evidencias_completas = {
            1: evidencia_ejemplo,  # Prevención de violencia
            2: {  # Infancia y adolescencia
                "indicadores": ["Reducir trabajo infantil del 8% al 2% para 2027"],
                "responsables": ["ICBF coordinará con Secretaría de Desarrollo Social"],
                "diagnostico": ["8% de menores en situación de trabajo infantil"],
            },
            3: {  # Mujeres y equidad
                "indicadores": ["Reducir violencia de género en 40% para 2027"],
                "responsables": ["Secretaría de la Mujer liderará la estrategia"],
                "diagnostico": ["Tasa de 120 casos por 100,000 habitantes"],
            },
            4: {  # Discapacidad
                "indicadores": ["100% de edificios públicos accesibles para 2027"],
                "responsables": ["Oficina de Inclusión Social"],
                "diagnostico": ["Solo 30% de edificios son accesibles actualmente"],
            },
            5: {  # Víctimas del conflicto
                "indicadores": ["Atender 100% de víctimas registradas"],
                "responsables": ["Unidad de Víctimas Municipal"],
                "diagnostico": ["3,500 víctimas registradas sin atención integral"],
            },
            6: {  # Educación
                "indicadores": ["Deserción escolar del 12% al 5% para 2027"],
                "responsables": ["Secretaría de Educación"],
                "diagnostico": ["12% de deserción en educación media"],
            },
            7: {  # Salud
                "indicadores": ["Mortalidad infantil de 15 a 8 por 1000 nacidos vivos"],
                "responsables": ["Secretaría de Salud"],
                "diagnostico": ["Mortalidad infantil sobre el promedio nacional"],
            },
            8: {  # Líderes sociales
                "indicadores": ["Implementar sistema de protección integral"],
                "responsables": ["Secretaría de Gobierno"],
                "diagnostico": ["15 amenazas a líderes en último año"],
            },
            9: {  # Trabajo decente
                "indicadores": ["Desempleo del 12% al 8% para 2027"],
                "responsables": ["Secretaría de Desarrollo Económico"],
                "diagnostico": ["12% de desempleo, 40% de informalidad"],
            },
            10: {  # Vivienda digna
                "indicadores": ["Reducir déficit habitacional del 25% al 15%"],
                "responsables": ["Instituto de Vivienda Municipal"],
                "diagnostico": ["25% de hogares en déficit habitacional"],
            },
        }

        # Generar reporte final
        reporte_final = evaluador.generar_reporte_final(
            evidencias_completas,
            "Plan de Desarrollo Municipal 2024-2027 - Municipio Ejemplo",
        )

        # Mostrar resumen del reporte final
        print(f"\n📊 RESUMEN EJECUTIVO DEL REPORTE FINAL:")
        print("-" * 50)
        for clave, valor in reporte_final.resumen_ejecutivo.items():
            print(f"   {clave}: {valor}")

        print(f"\n📈 ESTADÍSTICAS GLOBALES:")
        print("-" * 50)
        if reporte_final.estadisticas_globales:
            dist = reporte_final.estadisticas_globales.get(
                "distribucion_respuestas", {}
            )
            print(f"   Respuestas 'Sí': {dist.get('si', 0)}")
            print(f"   Respuestas 'Parcial': {dist.get('parcial', 0)}")
            print(f"   Respuestas 'No': {dist.get('no', 0)}")
            print(f"   Sin Identificar: {dist.get('ni', 0)}")

        print(f"\n🎯 ANÁLISIS POR CLUSTER:")
        print("-" * 50)
        for cluster in reporte_final.reporte_meso_por_cluster:
            print(f"   {cluster.cluster_nombre}:")
            print(
                f"      - Puntaje: {cluster.puntaje_agregado_cluster:.1f}/100")
            print(
                f"      - Clasificación: {cluster.clasificacion_cualitativa}")
            print(f"      - Coherencia: {cluster.coherencia_cluster:.2%}")

        print(f"\n📋 RECOMENDACIONES GLOBALES TOP 5:")
        print("-" * 50)
        for i, rec in enumerate(
            reporte_final.reporte_macro["recomendaciones_globales"][:5], 1
        ):
            print(f"   {i}. {rec}")

        # Exportar resultados si es necesario
        if HAS_PANDAS and reporte_final.matriz_trazabilidad_global is not None:
            print(f"\n💾 EXPORTACIÓN DE RESULTADOS:")
            print("-" * 50)
            print(f"   ✅ Matriz de trazabilidad exportada (CSV)")
            print(
                f"   ✅ {len(reporte_final.reporte_por_pregunta)} preguntas evaluadas"
            )
            print(
                f"   ✅ {len(reporte_final.reporte_por_punto)} puntos analizados")

        print("\n" + "=" * 100)
        print("EVALUACIÓN FINALIZADA EXITOSAMENTE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)

    # ==================== FUNCIONES AUXILIARES DE EXPORTACIÓN ====================

    def exportar_resultados_json(
        reporte: ReporteFinalDecatalogo, archivo: str = "reporte_decatalogo.json"
    ) -> bool:
        """
        Exporta el reporte completo a formato JSON para interoperabilidad.

        Args:
            reporte: Reporte final del decálogo
            archivo: Nombre del archivo de salida

        Returns:
            True si la exportación fue exitosa
        """
        try:
            # Convertir a diccionario serializable
            data = {
                "metadata": reporte.metadata,
                "resumen_ejecutivo": reporte.resumen_ejecutivo,
                "reporte_macro": reporte.reporte_macro,
                "estadisticas_globales": reporte.estadisticas_globales,
                "clusters": [
                    {
                        "nombre": c.cluster_nombre,
                        "puntaje": c.puntaje_agregado_cluster,
                        "clasificacion": c.clasificacion_cualitativa,
                        "analisis": c.analisis_meso_completo,
                    }
                    for c in reporte.reporte_meso_por_cluster
                ],
                "puntos": [
                    {
                        "id": p.punto_id,
                        "nombre": p.nombre_punto,
                        "puntaje": p.puntaje_agregado_punto,
                        "clasificacion": p.clasificacion_cualitativa,
                        "brechas": p.brechas_identificadas[:5],
                        "recomendaciones": p.recomendaciones_especificas[:5],
                    }
                    for p in reporte.reporte_por_punto
                ],
                "anexos": reporte.anexos_serializables,
            }

            # Guardar JSON con formato legible
            with open(archivo, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            LOGGER.info(f"✅ Reporte exportado a JSON: {archivo}")
            return True

        except Exception as e:
            LOGGER.error(f"❌ Error exportando JSON: {e}")
            return False

    def generar_reporte_markdown(
        reporte: ReporteFinalDecatalogo, archivo: str = "reporte_decatalogo.md"
    ) -> bool:
        """
        Genera un reporte en formato Markdown para documentación.

        Args:
            reporte: Reporte final del decálogo
            archivo: Nombre del archivo de salida

        Returns:
            True si la generación fue exitosa
        """
        try:
            lineas = []

            # Encabezado
            lineas.append(
                f"# Reporte de Evaluación - {reporte.metadata['nombre_plan']}"
            )
            lineas.append(
                f"\n**Fecha:** {reporte.metadata['fecha_evaluacion']}")
            lineas.append(
                f"**Versión:** {reporte.metadata['version_evaluador']}")
            lineas.append("")

            # Resumen Ejecutivo
            lineas.append("## Resumen Ejecutivo")
            lineas.append("")
            lineas.append(
                f"- **Puntaje Global:** {reporte.resumen_ejecutivo['puntaje_global']}/100"
            )
            lineas.append(
                f"- **Nivel de Alineación:** {reporte.resumen_ejecutivo['nivel_alineacion']}"
            )
            lineas.append(
                f"- **Mejor Cluster:** {reporte.resumen_ejecutivo['cluster_mejor_desempeno']}"
            )
            lineas.append(
                f"- **Cluster Crítico:** {reporte.resumen_ejecutivo['cluster_peor_desempeno']}"
            )
            lineas.append(
                f"- **Recomendación:** {reporte.resumen_ejecutivo['recomendacion_estrategica_global']}"
            )
            lineas.append("")

            # Análisis por Cluster
            lineas.append("## Análisis por Cluster")
            lineas.append("")

            for cluster in reporte.reporte_meso_por_cluster:
                lineas.append(f"### {cluster.cluster_nombre}")
                lineas.append(
                    f"- Puntaje: {cluster.puntaje_agregado_cluster:.1f}/100")
                lineas.append(
                    f"- Clasificación: {cluster.clasificacion_cualitativa}")
                lineas.append(
                    f"- Coherencia Interna: {cluster.coherencia_cluster:.2%}")

                if cluster.analisis_meso_completo.get("puntos_fuertes"):
                    lineas.append(
                        f"- Puntos Fuertes: {', '.join(cluster.analisis_meso_completo['puntos_fuertes'])}"
                    )
                if cluster.analisis_meso_completo.get("puntos_criticos"):
                    lineas.append(
                        f"- Puntos Críticos: {', '.join(cluster.analisis_meso_completo['puntos_criticos'])}"
                    )
                lineas.append("")

            # Top Recomendaciones
            lineas.append("## Recomendaciones Prioritarias")
            lineas.append("")
            for i, rec in enumerate(
                reporte.reporte_macro["recomendaciones_globales"][:10], 1
            ):
                lineas.append(f"{i}. {rec}")
            lineas.append("")

            # Brechas Identificadas
            lineas.append("## Principales Brechas Identificadas")
            lineas.append("")
            for i, brecha in enumerate(
                reporte.reporte_macro["brechas_globales"][:10], 1
            ):
                lineas.append(f"{i}. {brecha}")
            lineas.append("")

            # Estadísticas
            if reporte.estadisticas_globales:
                lineas.append("## Estadísticas de Evaluación")
                lineas.append("")
                dist = reporte.estadisticas_globales.get(
                    "distribucion_respuestas", {})
                total = sum(dist.values())
                if total > 0:
                    lineas.append("| Tipo Respuesta | Cantidad | Porcentaje |")
                    lineas.append("|----------------|----------|------------|")
                    lineas.append(
                        f"| Sí | {dist.get('si', 0)} | {dist.get('si', 0) / total * 100:.1f}% |"
                    )
                    lineas.append(
                        f"| Parcial | {dist.get('parcial', 0)} | {dist.get('parcial', 0) / total * 100:.1f}% |"
                    )
                    lineas.append(
                        f"| No | {dist.get('no', 0)} | {dist.get('no', 0) / total * 100:.1f}% |"
                    )
                    lineas.append(
                        f"| No Identificado | {dist.get('ni', 0)} | {dist.get('ni', 0) / total * 100:.1f}% |"
                    )
                    lineas.append("")

            # Footer
            lineas.append("---")
            lineas.append(
                f"*Reporte generado automáticamente por el Sistema de Evaluación del Decálogo v3.0*"
            )
            lineas.append(f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

            # Escribir archivo
            with open(archivo, "w", encoding="utf-8") as f:
                f.write("\n".join(lineas))

            LOGGER.info(f"✅ Reporte Markdown generado: {archivo}")
            return True

        except Exception as e:
            LOGGER.error(f"❌ Error generando Markdown: {e}")
            return False

    def comparar_municipios(
        reportes: List[ReporteFinalDecatalogo],
        archivo_salida: str = "comparacion_municipios.csv",
    ) -> Optional[pd.DataFrame]:
        """
        Compara múltiples municipios generando tabla comparativa.
        Esencial para el análisis de los 117 planes.

        Args:
            reportes: Lista de reportes finales
            archivo_salida: Archivo CSV de salida

        Returns:
            DataFrame con la comparación o None si falla
        """
        if not HAS_PANDAS:
            LOGGER.error("❌ Pandas no disponible para comparación")
            return None

        try:
            data = []
            for reporte in reportes:
                fila = {
                    "Municipio": reporte.metadata["nombre_plan"],
                    "Puntaje_Global": reporte.resumen_ejecutivo["puntaje_global"],
                    "Alineacion": reporte.resumen_ejecutivo["nivel_alineacion"],
                    "Fecha_Evaluacion": reporte.metadata["fecha_evaluacion"],
                    "Puntos_Evaluados": reporte.metadata["puntos_evaluados"],
                    "Confianza": reporte.resumen_ejecutivo.get(
                        "nivel_confianza_evaluacion", 0
                    ),
                }

                # Agregar puntajes por dimensión
                if (
                    reporte.estadisticas_globales
                    and "analisis_dimensional" in reporte.estadisticas_globales
                ):
                    for dim, stats in reporte.estadisticas_globales[
                        "analisis_dimensional"
                    ].items():
                        fila[f"Puntaje_{dim}"] = stats["promedio"]
                        fila[f"Desviacion_{dim}"] = stats["desviacion"]

                # Agregar puntajes por cluster
                for cluster in reporte.reporte_meso_por_cluster:
                    fila[f"Cluster_{cluster.cluster_id}"] = (
                        cluster.puntaje_agregado_cluster
                    )

                data.append(fila)

            # Crear DataFrame
            df = pd.DataFrame(data)

            # Ordenar por puntaje global
            df = df.sort_values("Puntaje_Global", ascending=False)

            # Agregar ranking
            df["Ranking"] = range(1, len(df) + 1)

            # Calcular estadísticas agregadas
            df["Z_Score"] = (df["Puntaje_Global"] - df["Puntaje_Global"].mean()) / df[
                "Puntaje_Global"
            ].std()
            df["Percentil"] = df["Puntaje_Global"].rank(pct=True) * 100

            # Guardar CSV
            df.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

            LOGGER.info(
                f"✅ Comparación de {len(reportes)} municipios guardada: {archivo_salida}"
            )

            # Mostrar resumen
            print("\n📊 RESUMEN COMPARATIVO DE MUNICIPIOS:")
            print("-" * 50)
            print(f"   Municipios evaluados: {len(df)}")
            print(f"   Puntaje promedio: {df['Puntaje_Global'].mean():.1f}")
            print(f"   Desviación estándar: {df['Puntaje_Global'].std():.1f}")
            print(f"   Puntaje máximo: {df['Puntaje_Global'].max():.1f}")
            print(f"   Puntaje mínimo: {df['Puntaje_Global'].min():.1f}")
            print("\n   TOP 5 MUNICIPIOS:")
            for idx, row in df.head(5).iterrows():
                print(
                    f"   {row['Ranking']}. {row['Municipio']}: {row['Puntaje_Global']:.1f}"
                )

            return df

        except Exception as e:
            LOGGER.error(f"❌ Error en comparación de municipios: {e}")
            return None

    # ==================== VALIDACIÓN Y CERTIFICACIÓN ====================

    def validar_integridad_evaluador() -> bool:
        """
        Valida que el evaluador esté correctamente configurado para producción.
        Ejecutar antes de procesar los 117 planes.

        Returns:
            True si todas las validaciones pasan
        """
        print("\n🔍 VALIDANDO INTEGRIDAD DEL EVALUADOR v3.0...")
        print("-" * 50)

        validaciones = []

        # 1. Verificar importaciones críticas
        try:
            from contradiction_detector import ContradictionDetector
            from Decatalogo_principal import DecalogoContext
            from feasibility_scorer import FeasibilityScorer
            from responsibility_detector import ResponsibilityDetector

            validaciones.append(("Importaciones principales", True))
        except ImportError as e:
            validaciones.append(("Importaciones principales", False))
            print(f"   ❌ Error en importaciones: {e}")

        # 2. Verificar inicialización del evaluador
        try:
            evaluador = IndustrialDecatalogoEvaluatorFull()
            validaciones.append(("Inicialización evaluador", True))
        except Exception as e:
            validaciones.append(("Inicialización evaluador", False))
            print(f"   ❌ Error inicializando: {e}")

        # 3. Verificar contexto del decálogo
        try:
            contexto = obtener_decalogo_contexto()
            tiene_clusters = len(contexto.clusters) > 0
            tiene_dimensiones = len(contexto.dimension_por_id) > 0
            validaciones.append(
                ("Contexto decálogo", tiene_clusters and tiene_dimensiones)
            )
        except Exception:
            validaciones.append(("Contexto decálogo", False))

        # 4. Verificar detectores
        try:
            scorer = FeasibilityScorer()
            resp_detector = ResponsibilityDetector()
            cont_detector = ContradictionDetector()
            validaciones.append(("Detectores especializados", True))
        except Exception:
            validaciones.append(("Detectores especializados", False))

        # 5. Verificar funciones de evaluación
        try:
            test_evidencia = {"indicadores": [
                "Test"], "responsables": ["Test"]}
            evaluador = IndustrialDecatalogoEvaluatorFull()
            analisis = evaluador._analizar_evidencia(test_evidencia)
            validaciones.append(
                ("Análisis de evidencia", analisis is not None))
        except Exception:
            validaciones.append(("Análisis de evidencia", False))

        # 6. Verificar ponderaciones y umbrales
        try:
            evaluador = IndustrialDecatalogoEvaluatorFull()
            ponderacion_correcta = (
                sum(evaluador.ponderacion_dimensiones.values()) == 1.0
            )
            umbrales_correctos = all(
                k in evaluador.umbrales
                for k in ["optimo", "satisfactorio", "basico", "insuficiente"]
            )
            validaciones.append(
                (
                    "Configuración parámetros",
                    ponderacion_correcta and umbrales_correctos,
                )
            )
        except Exception:
            validaciones.append(("Configuración parámetros", False))

        # Mostrar resultados
        print("\n📋 RESULTADOS DE VALIDACIÓN:")
        todas_validas = True
        for componente, valido in validaciones:
            estado = "✅" if valido else "❌"
            print(f"   {estado} {componente}")
            if not valido:
                todas_validas = False

        if todas_validas:
            print("\n✅ SISTEMA VALIDADO - LISTO PARA PRODUCCIÓN")
            print(
                "   El evaluador está correctamente configurado para procesar los 117 planes"
            )
        else:
            print("\n❌ VALIDACIÓN FALLIDA - REVISAR CONFIGURACIÓN")
            print("   Corrija los errores antes de procesar planes en producción")

        return todas_validas

    # ==================== CLASE ORQUESTADOR PARA 117 PLANES ====================

    class OrquestadorEvaluacion117:
        """
        Orquestador para procesar los 117 planes de desarrollo de manera eficiente.
        Incluye manejo de errores, reintentos y generación de reportes consolidados.
        """

        def __init__(self, directorio_planes: str = "planes_desarrollo"):
            """
            Inicializa el orquestador.

            Args:
                directorio_planes: Directorio con los planes a evaluar
            """
            self.directorio = Path(directorio_planes)
            self.evaluador = IndustrialDecatalogoEvaluatorFull()
            self.resultados = []
            self.errores = []
            self.timestamp_inicio = None

            LOGGER.info(
                f"📁 Orquestador inicializado. Directorio: {self.directorio}")

        def procesar_todos_los_planes(
            self,
            max_planes: Optional[int] = None,
            exportar_individual: bool = True,
            generar_comparacion: bool = True,
        ) -> Dict[str, Any]:
            """
            Procesa todos los planes de desarrollo disponibles.

            Args:
                max_planes: Límite de planes a procesar (None = todos)
                exportar_individual: Si exportar reporte individual por plan
                generar_comparacion: Si generar tabla comparativa

            Returns:
                Diccionario con estadísticas del procesamiento
            """
            self.timestamp_inicio = datetime.now()
            planes_procesados = 0
            planes_exitosos = 0

            # Buscar archivos de planes (asumiendo formato JSON por ahora)
            archivos_planes = list(self.directorio.glob("*.json"))

            if max_planes:
                archivos_planes = archivos_planes[:max_planes]

            total_planes = len(archivos_planes)

            print(f"\n🚀 INICIANDO PROCESAMIENTO DE {total_planes} PLANES")
            print("=" * 80)

            for idx, archivo_plan in enumerate(archivos_planes, 1):
                try:
                    print(
                        f"\n[{idx}/{total_planes}] Procesando: {archivo_plan.name}")

                    # Cargar evidencias del plan
                    with open(archivo_plan, "r", encoding="utf-8") as f:
                        evidencias = json.load(f)

                    # Extraer nombre del municipio
                    nombre_municipio = archivo_plan.stem.replace(
                        "_", " ").title()

                    # Generar reporte
                    reporte = self.evaluador.generar_reporte_final(
                        evidencias, nombre_municipio
                    )

                    self.resultados.append(reporte)
                    planes_exitosos += 1

                    # Exportar reporte individual si se solicita
                    if exportar_individual:
                        self._exportar_reporte_individual(
                            reporte, nombre_municipio)

                    print(
                        f"   ✅ Completado: {reporte.resumen_ejecutivo['puntaje_global']:.1f}/100"
                    )

                except Exception as e:
                    error_msg = f"Error en {archivo_plan.name}: {str(e)}"
                    LOGGER.error(error_msg)
                    self.errores.append(error_msg)
                    print(f"   ❌ Error: {str(e)[:100]}")

                planes_procesados += 1

                # Mostrar progreso cada 10 planes
                if planes_procesados % 10 == 0:
                    self._mostrar_progreso(
                        planes_procesados, total_planes, planes_exitosos
                    )

            # Generar comparación si se solicita
            df_comparacion = None
            if generar_comparacion and self.resultados:
                df_comparacion = comparar_municipios(self.resultados)

            # Calcular estadísticas finales
            tiempo_total = (datetime.now() -
                            self.timestamp_inicio).total_seconds()

            estadisticas = {
                "planes_procesados": planes_procesados,
                "planes_exitosos": planes_exitosos,
                "planes_fallidos": len(self.errores),
                "tiempo_total_segundos": tiempo_total,
                "tiempo_promedio_segundos": (
                    tiempo_total / planes_procesados if planes_procesados > 0 else 0
                ),
                "tasa_exito": (
                    planes_exitosos / planes_procesados * 100
                    if planes_procesados > 0
                    else 0
                ),
                "puntaje_promedio_global": (
                    np.mean(
                        [r.resumen_ejecutivo["puntaje_global"]
                            for r in self.resultados]
                    )
                    if self.resultados
                    else 0
                ),
                "errores": self.errores[:10],  # Primeros 10 errores
            }

            # Mostrar resumen final
            self._mostrar_resumen_final(estadisticas)

            # Generar reporte consolidado
            self._generar_reporte_consolidado(estadisticas, df_comparacion)

            return estadisticas

        def _exportar_reporte_individual(
            self, reporte: ReporteFinalDecatalogo, nombre_municipio: str
        ):
            """Exporta reporte individual de un municipio."""
            try:
                # Crear directorio de salida
                dir_salida = Path("reportes_individuales")
                dir_salida.mkdir(exist_ok=True)

                # Nombre de archivo seguro
                nombre_archivo = nombre_municipio.replace(
                    " ", "_").replace("/", "_")

                # Exportar JSON
                archivo_json = dir_salida / f"{nombre_archivo}.json"
                exportar_resultados_json(reporte, str(archivo_json))

                # Exportar Markdown
                archivo_md = dir_salida / f"{nombre_archivo}.md"
                generar_reporte_markdown(reporte, str(archivo_md))

            except Exception as e:
                LOGGER.warning(f"No se pudo exportar reporte individual: {e}")

        def _mostrar_progreso(self, procesados: int, total: int, exitosos: int):
            """Muestra progreso del procesamiento."""
            porcentaje = procesados / total * 100
            tasa_exito = exitosos / procesados * 100
            tiempo_transcurrido = (
                datetime.now() - self.timestamp_inicio
            ).total_seconds()
            tiempo_restante = (tiempo_transcurrido /
                               procesados) * (total - procesados)

            print(f"\n📊 PROGRESO: {procesados}/{total} ({porcentaje:.1f}%)")
            print(f"   Exitosos: {exitosos} ({tasa_exito:.1f}%)")
            print(
                f"   Tiempo restante estimado: {tiempo_restante / 60:.1f} minutos")

        def _mostrar_resumen_final(self, estadisticas: Dict[str, Any]):
            """Muestra resumen final del procesamiento."""
            print("\n" + "=" * 80)
            print("📊 RESUMEN FINAL DE PROCESAMIENTO")
            print("=" * 80)
            print(f"   Planes procesados: {estadisticas['planes_procesados']}")
            print(f"   Planes exitosos: {estadisticas['planes_exitosos']}")
            print(f"   Planes fallidos: {estadisticas['planes_fallidos']}")
            print(f"   Tasa de éxito: {estadisticas['tasa_exito']:.1f}%")
            print(
                f"   Puntaje promedio: {estadisticas['puntaje_promedio_global']:.1f}/100"
            )
            print(
                f"   Tiempo total: {estadisticas['tiempo_total_segundos'] / 60:.1f} minutos"
            )
            print(
                f"   Tiempo promedio por plan: {estadisticas['tiempo_promedio_segundos']:.1f} segundos"
            )

        def _generar_reporte_consolidado(
            self, estadisticas: Dict[str, Any], df_comparacion: Optional[pd.DataFrame]
        ):
            """Genera reporte consolidado final."""
            try:
                # Crear directorio
                dir_consolidado = Path("reporte_consolidado")
                dir_consolidado.mkdir(exist_ok=True)

                # Guardar estadísticas
                with open(
                    dir_consolidado / "estadisticas_procesamiento.json", "w"
                ) as f:
                    json.dump(estadisticas, f, indent=2, default=str)

                # Guardar comparación si existe
                if df_comparacion is not None:
                    df_comparacion.to_csv(
                        dir_consolidado / "comparacion_municipios.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

                    # Generar visualización HTML simple
                    self._generar_dashboard_html(
                        df_comparacion, dir_consolidado)

                print(
                    f"\n✅ Reporte consolidado generado en: {dir_consolidado}")

            except Exception as e:
                LOGGER.error(f"Error generando reporte consolidado: {e}")

        def _generar_dashboard_html(self, df: pd.DataFrame, directorio: Path):
            """Genera un dashboard HTML básico con los resultados."""
            html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Dashboard - Evaluación Decálogo DDHH</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #3498db; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        .high {{ background-color: #27ae60; color: white; }}
                        .medium {{ background-color: #f39c12; }}
                        .low {{ background-color: #e74c3c; color: white; }}
                    </style>
                </head>
                <body>
                    <h1>Dashboard de Evaluación - Decálogo de Derechos Humanos</h1>
                    <h2>Resumen de {len(df)} Municipios Evaluados</h2>
                    <p>Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                    <h3>Estadísticas Generales</h3>
                    <ul>
                        <li>Puntaje Promedio: {df["Puntaje_Global"].mean():.1f}/100</li>
                        <li>Puntaje Máximo: {df["Puntaje_Global"].max():.1f}/100</li>
                        <li>Puntaje Mínimo: {df["Puntaje_Global"].min():.1f}/100</li>
                        <li>Desviación Estándar: {df["Puntaje_Global"].std():.1f}</li>
                    </ul>

                    <h3>Ranking de Municipios</h3>
                    <table>
                        <tr>
                            <th>Ranking</th>
                            <th>Municipio</th>
                            <th>Puntaje Global</th>
                            <th>Alineación</th>
                            <th>Percentil</th>
                        </tr>
                """

            for idx, row in df.head(20).iterrows():
                clase = (
                    "high"
                    if row["Puntaje_Global"] >= 70
                    else "medium"
                    if row["Puntaje_Global"] >= 50
                    else "low"
                )
                html += f"""
                        <tr class="{clase}">
                            <td>{row["Ranking"]}</td>
                            <td>{row["Municipio"]}</td>
                            <td>{row["Puntaje_Global"]:.1f}</td>
                            <td>{row["Alineacion"]}</td>
                            <td>{row["Percentil"]:.1f}%</td>
                        </tr>
                    """

            html += """
                    </table>
                </body>
                </html>
                """

            with open(directorio / "dashboard.html", "w", encoding="utf-8") as f:
                f.write(html)

    # ==================== PUNTO DE ENTRADA PARA PRODUCCIÓN ====================

    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(
            description="Evaluador Industrial del Decálogo de DDHH v3.0 PRODUCTION"
        )
        parser.add_argument(
            "--validar",
            action="store_true",
            help="Ejecutar validación de integridad del sistema",
        )
        parser.add_argument(
            "--demo",
            action="store_true",
            help="Ejecutar demostración con datos de ejemplo",
        )
        parser.add_argument(
            "--procesar", type=str, help="Directorio con planes a procesar"
        )
        parser.add_argument(
            "--max-planes",
            type=int,
            default=None,
            help="Número máximo de planes a procesar",
        )
        parser.add_argument(
            "--exportar", action="store_true", help="Exportar reportes individuales"
        )

        args = parser.parse_args()

        if args.validar:
            print("\n🔍 EJECUTANDO VALIDACIÓN DEL SISTEMA...")
            validar_integridad_evaluador()

        elif args.demo:
            print("\n🎮 EJECUTANDO MODO DEMOSTRACIÓN...")
            main()

        elif args.procesar:
            print(f"\n📂 PROCESANDO PLANES EN: {args.procesar}")
            orquestador = OrquestadorEvaluacion117(args.procesar)
            estadisticas = orquestador.procesar_todos_los_planes(
                max_planes=args.max_planes, exportar_individual=args.exportar
            )
            print(
                f"\n✅ Procesamiento completado. {estadisticas['planes_exitosos']} planes evaluados exitosamente."
            )

        else:
            # Ejecutar demo por defecto
            main()
