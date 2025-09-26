# -*- coding: utf-8 -*-
"""
Sistema Integral de Evaluación de Cadenas de Valor en Planes de Desarrollo Municipal
Versión: 9.0 — Marco Teórico-Institucional con Análisis Causal Multinivel, Frontier AI Capabilities,
Mathematical Innovation, Sophisticated Evidence Processing y Reporting Industrial.
Framework basado en IAD + Theory of Change, con triangulación cuali-cuantitativa,
verificación causal, certeza probabilística y capacidades de frontera.
Autor: Dr. en Políticas Públicas
Enfoque: Evaluación estructural con econometría de políticas, minería causal avanzada,
procesamiento paralelo industrial y reportes masivos granulares.
"""

import argparse
import atexit
import hashlib
import heapq
import json
import logging
import re
import signal
import statistics
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Dependencias avanzadas --------------------
try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

import spacy
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer, util

# Módulos matemáticos avanzados
try:
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import cdist
    from scipy.stats import chi2_contingency, entropy, pearsonr, spearmanr

    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False

# Capacidades de frontera en NLP
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, pipeline

    FRONTIER_NLP_AVAILABLE = True
except ImportError:
    FRONTIER_NLP_AVAILABLE = False

try:
    from decalogo_loader import get_decalogo_industrial
except ImportError:

    def get_decalogo_industrial():
        return "Fallback: Decálogo industrial para desarrollo municipal con 10 dimensiones estratégicas."


# Device configuration avanzada
try:
    from device_config import (
        add_device_args,
        configure_device_from_args,
        get_device_config,
        to_device,
    )
except ImportError:

    def add_device_args(parser):
        parser.add_argument("--device", default="cpu",
                            help="Device to use (cpu/cuda)")
        parser.add_argument(
            "--precision",
            default="float32",
            choices=["float16", "float32"],
            help="Precision",
        )
        parser.add_argument(
            "--batch_size", default=16, type=int, help="Batch size for processing"
        )
        return parser

    def configure_device_from_args(args):
        return AdvancedDeviceConfig(
            args.device if hasattr(args, "device") else "cpu",
            args.precision if hasattr(args, "precision") else "float32",
            args.batch_size if hasattr(args, "batch_size") else 16,
        )

    def get_device_config():
        return AdvancedDeviceConfig("cpu", "float32", 16)

    def to_device(model):
        return model

    class AdvancedDeviceConfig:
        def __init__(self, device="cpu", precision="float32", batch_size=16):
            self.device = device
            self.precision = precision
            self.batch_size = batch_size

        def get_device(self):
            return self.device

        def get_precision(self):
            return torch.float16 if self.precision == "float16" else torch.float32

        def get_batch_size(self):
            return self.batch_size

        def get_device_info(self):
            return {
                "device_type": self.device,
                "precision": self.precision,
                "batch_size": self.batch_size,
                "num_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
                "memory_info": self._get_memory_info(),
            }

        def _get_memory_info(self):
            if torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,
                    "reserved": torch.cuda.memory_reserved() / 1024**3,
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                }
            return {"cpu_memory": "N/A"}


# Text processing avanzado
try:
    from text_truncation_logger import (
        get_truncation_logger,
        log_debug_with_text,
        log_error_with_text,
        log_info_with_text,
        log_warning_with_text,
        truncate_text_for_log,
    )
except ImportError:

    def get_truncation_logger(name):
        return logging.getLogger(name)

    def log_debug_with_text(logger, text):
        logger.debug(truncate_text_for_log(text, 500))

    def log_error_with_text(logger, text):
        logger.error(truncate_text_for_log(text, 500))

    def log_info_with_text(logger, text):
        logger.info(truncate_text_for_log(text, 500))

    def log_warning_with_text(logger, text):
        logger.warning(truncate_text_for_log(text, 500))

    def truncate_text_for_log(text, max_len=500):
        return text[:max_len] + "..." if len(text) > max_len else text


# Requerimiento de versión
assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

# Suprimir warnings innecesarios
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Logging industrial avanzado --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"evaluacion_politicas_industrial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
LOGGER = logging.getLogger("EvaluacionPoliticasPublicasIndustrial")

# -------------------- Carga de modelos con capacidades de frontera --------------------
try:
    NLP = spacy.load("es_core_news_lg")
    log_info_with_text(
        LOGGER, "✅ Modelo SpaCy avanzado cargado (es_core_news_lg)")
except OSError:
    try:
        NLP = spacy.load("es_core_news_sm")
        log_warning_with_text(
            LOGGER, "⚠️ Usando modelo SpaCy básico (es_core_news_sm)")
    except OSError as e:
        log_error_with_text(LOGGER, f"❌ Error cargando SpaCy: {e}")
        raise SystemExit(
            "Modelo SpaCy no disponible. Ejecute: python -m spacy download es_core_news_lg"
        )

try:
    EMBEDDING_MODEL = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    EMBEDDING_MODEL = to_device(EMBEDDING_MODEL)
    log_info_with_text(LOGGER, "✅ Modelo de embeddings multilingual cargado")
    log_info_with_text(
        LOGGER, f"✅ Dispositivo: {get_device_config().get_device()}")
except Exception as e:
    log_error_with_text(LOGGER, f"❌ Error cargando embeddings: {e}")
    raise SystemExit(f"Error cargando modelo de embeddings: {e}")

# Carga de modelos de frontera para análisis avanzado
ADVANCED_NLP_PIPELINE = None
if FRONTIER_NLP_AVAILABLE:
    try:
        ADVANCED_NLP_PIPELINE = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True,
        )
        log_info_with_text(
            LOGGER, "✅ Pipeline NLP avanzado cargado para análisis de sentimientos"
        )
    except Exception as e:
        log_warning_with_text(
            LOGGER, f"⚠️ Pipeline NLP avanzado no disponible: {e}")


# -------------------- Innovaciones matemáticas --------------------
class MathematicalInnovations:
    """Clase con innovaciones matemáticas para análisis de políticas públicas."""

    @staticmethod
    def calculate_causal_strength(graph: nx.DiGraph, source: str, target: str) -> float:
        """Calcula la fuerza causal entre dos nodos usando innovaciones en teoría de grafos."""
        try:
            if not nx.has_path(graph, source, target):
                return 0.0

            # Innovación: Combinación de múltiples métricas de centralidad
            paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
            if not paths:
                return 0.0

            # Cálculo de fuerza causal ponderada
            total_strength = 0.0
            for path in paths:
                path_strength = 1.0
                for i in range(len(path) - 1):
                    edge_weight = graph.get_edge_data(path[i], path[i + 1], {}).get(
                        "weight", 0.5
                    )
                    path_strength *= edge_weight

                # Penalización por longitud de camino
                length_penalty = 0.8 ** (len(path) - 2)
                total_strength += path_strength * length_penalty

            # Normalización basada en la centralidad de los nodos
            source_centrality = nx.betweenness_centrality(
                graph).get(source, 0.1)
            target_centrality = nx.betweenness_centrality(
                graph).get(target, 0.1)
            centrality_factor = (source_centrality + target_centrality) / 2

            return min(1.0, total_strength * (1 + centrality_factor))

        except Exception:
            return 0.3

    @staticmethod
    def bayesian_evidence_integration(
        evidences: List[float], priors: List[float]
    ) -> float:
        """Integración bayesiana de evidencias para cálculo de certeza probabilística."""
        if not evidences or not priors:
            return 0.5

        try:
            # Innovación: Actualización bayesiana iterativa
            posterior = priors[0] if priors else 0.5

            for i, evidence in enumerate(evidences):
                likelihood = evidence
                prior = posterior

                # Aplicación del teorema de Bayes
                numerator = likelihood * prior
                denominator = likelihood * prior + \
                    (1 - likelihood) * (1 - prior)
                posterior = numerator / denominator if denominator > 0 else prior

                # Regularización para evitar valores extremos
                posterior = max(0.01, min(0.99, posterior))

            return posterior

        except Exception:
            return np.mean(evidences) if evidences else 0.5

    @staticmethod
    def entropy_based_complexity(elements: List[str]) -> float:
        """Calcula complejidad basada en entropía de elementos."""
        if not elements:
            return 0.0

        try:
            # Distribución de frecuencias
            from collections import Counter

            freq_dist = Counter(elements)
            total = sum(freq_dist.values())
            probabilities = [count / total for count in freq_dist.values()]

            # Cálculo de entropía de Shannon
            entropy_val = -sum(p * np.log2(p) for p in probabilities if p > 0)

            # Normalización por máxima entropía posible
            max_entropy = np.log2(len(freq_dist))
            normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0

            return normalized_entropy

        except Exception:
            return 0.5

    @staticmethod
    def fuzzy_logic_aggregation(
        values: List[float], weights: List[float] = None
    ) -> Dict[str, float]:
        """Agregación difusa avanzada de valores con múltiples operadores."""
        if not values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "fuzzy_and": 0.0,
                "fuzzy_or": 0.0,
            }

        values = np.array(values)
        weights = np.array(weights) if weights else np.ones(len(values))
        weights = weights / np.sum(weights)  # Normalización

        try:
            # Operadores difusos clásicos
            fuzzy_and = np.min(values)  # T-norma mínima
            fuzzy_or = np.max(values)  # T-conorma máxima

            # Operadores avanzados
            weighted_mean = np.sum(values * weights)
            geometric_mean = np.exp(
                np.sum(weights * np.log(np.maximum(values, 1e-10))))
            harmonic_mean = 1.0 / np.sum(weights / np.maximum(values, 1e-10))

            # Agregación OWA (Ordered Weighted Averaging)
            sorted_values = np.sort(values)[::-1]  # Orden descendente
            owa_weights = np.array([0.4, 0.3, 0.2, 0.1])[: len(sorted_values)]
            owa_weights = owa_weights / np.sum(owa_weights)
            owa_result = np.sum(
                sorted_values[: len(owa_weights)] * owa_weights)

            return {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(weighted_mean),
                "geometric_mean": float(geometric_mean),
                "harmonic_mean": float(harmonic_mean),
                "fuzzy_and": float(fuzzy_and),
                "fuzzy_or": float(fuzzy_or),
                "owa": float(owa_result),
                "std": float(np.std(values)),
                "entropy": MathematicalInnovations.entropy_based_complexity(
                    [str(v) for v in values]
                ),
            }

        except Exception:
            return {
                "min": float(np.min(values)) if len(values) > 0 else 0.0,
                "max": float(np.max(values)) if len(values) > 0 else 0.0,
                "mean": float(np.mean(values)) if len(values) > 0 else 0.0,
                "fuzzy_and": 0.0,
                "fuzzy_or": 0.0,
                "owa": 0.0,
                "std": 0.0,
                "entropy": 0.0,
            }


# -------------------- Marco teórico avanzado --------------------
class NivelAnalisis(Enum):
    MACRO = "Institucional-Sistémico"
    MESO = "Organizacional-Sectorial"
    MICRO = "Operacional-Territorial"
    META = "Meta-Evaluativo"


class TipoCadenaValor(Enum):
    INSUMOS = "Recursos financieros, humanos y físicos"
    PROCESOS = "Transformación institucional y gestión"
    PRODUCTOS = "Bienes/servicios entregables medibles"
    RESULTADOS = "Cambios conductuales/institucionales"
    IMPACTOS = "Bienestar y desarrollo humano sostenible"
    OUTCOMES = "Efectos de largo plazo y sostenibilidad"


class TipoEvidencia(Enum):
    CUANTITATIVA = "Datos numéricos y estadísticas"
    CUALITATIVA = "Narrativas y descripciones"
    MIXTA = "Combinación cuanti-cualitativa"
    DOCUMENTAL = "Evidencia documental y normativa"
    TESTIMONIAL = "Testimonios y entrevistas"


@dataclass(frozen=True)
class TeoriaCambioAvanzada:
    """Teoría de cambio avanzada con capacidades matemáticas de frontera."""

    supuestos_causales: List[str]
    mediadores: Dict[str, List[str]]
    resultados_intermedios: List[str]
    precondiciones: List[str]
    moderadores: List[str] = field(default_factory=list)
    variables_contextuales: List[str] = field(default_factory=list)
    mecanismos_causales: List[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.supuestos_causales) == 0:
            raise ValueError("Supuestos causales no pueden estar vacíos")
        if len(self.mediadores) == 0:
            raise ValueError("Mediadores no pueden estar vacíos")

    def verificar_identificabilidad_avanzada(self) -> Dict[str, float]:
        """Verificación avanzada de identificabilidad causal."""
        criterios = {
            "supuestos_suficientes": len(self.supuestos_causales) >= 2,
            "mediadores_diversificados": len(self.mediadores) >= 2,
            "resultados_especificos": len(self.resultados_intermedios) >= 1,
            "precondiciones_definidas": len(self.precondiciones) >= 1,
            "moderadores_identificados": len(self.moderadores) >= 1,
            "mecanismos_explicitos": len(self.mecanismos_causales) >= 1,
        }

        puntajes = {k: 1.0 if v else 0.0 for k, v in criterios.items()}
        puntaje_global = np.mean(list(puntajes.values()))

        return {
            "puntaje_global_identificabilidad": puntaje_global,
            "criterios_individuales": puntajes,
            "nivel_identificabilidad": self._clasificar_identificabilidad(
                puntaje_global
            ),
        }

    def _clasificar_identificabilidad(self, puntaje: float) -> str:
        if puntaje >= 0.9:
            return "EXCELENTE"
        if puntaje >= 0.75:
            return "ALTA"
        if puntaje >= 0.6:
            return "MEDIA"
        if puntaje >= 0.4:
            return "BAJA"
        return "INSUFICIENTE"

    def construir_grafo_causal_avanzado(self) -> nx.DiGraph:
        """Construcción de grafo causal con propiedades avanzadas."""
        G = nx.DiGraph()

        # Nodos básicos
        G.add_node("insumos", tipo="nodo_base", nivel="input", centralidad=1.0)
        G.add_node("impactos", tipo="nodo_base",
                   nivel="outcome", centralidad=1.0)

        # Adición de nodos con atributos enriquecidos
        for categoria, lista in self.mediadores.items():
            for i, mediador in enumerate(lista):
                G.add_node(
                    mediador,
                    tipo="mediador",
                    categoria=categoria,
                    orden=i,
                    peso_teorico=0.8 + (i * 0.1),
                )
                G.add_edge("insumos", mediador, weight=0.9,
                           tipo="causal_directa")

        # Resultados intermedios con conexiones complejas
        for i, resultado in enumerate(self.resultados_intermedios):
            G.add_node(
                resultado,
                tipo="resultado_intermedio",
                orden=i,
                criticidad=0.7 + (i * 0.1),
            )

            # Conexiones desde mediadores
            mediadores_disponibles = [
                n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"
            ]
            for mediador in mediadores_disponibles:
                G.add_edge(
                    mediador, resultado, weight=0.8 - (i * 0.1), tipo="causal_mediada"
                )

            # Conexión al impacto final
            G.add_edge(
                resultado, "impactos", weight=0.9 - (i * 0.05), tipo="causal_final"
            )

        # Moderadores como nodos especiales
        for moderador in self.moderadores:
            G.add_node(moderador, tipo="moderador", influencia="contextual")
            # Los moderadores influencian las relaciones, no son parte del flujo directo

        # Precondiciones como requisitos
        for precond in self.precondiciones:
            G.add_node(precond, tipo="precondicion", necesidad="critica")
            G.add_edge(precond, "insumos", weight=1.0, tipo="prerequisito")

        return G

    def calcular_coeficiente_causal_avanzado(self) -> Dict[str, float]:
        """Cálculo avanzado de coeficientes causales."""
        G = self.construir_grafo_causal_avanzado()

        if len(G.nodes) < 3:
            return {
                "coeficiente_global": 0.3,
                "robustez_estructural": 0.2,
                "complejidad_causal": 0.1,
            }

        try:
            # Métricas estructurales
            density = nx.density(G)
            avg_clustering = nx.average_clustering(G.to_undirected())

            # Análisis de caminos causales
            mediadores = [n for n in G.nodes if G.nodes[n].get(
                "tipo") == "mediador"]
            resultados = [
                n for n in G.nodes if G.nodes[n].get("tipo") == "resultado_intermedio"
            ]

            # Innovación: Cálculo de fuerza causal usando la clase MathematicalInnovations
            fuerza_causal = MathematicalInnovations.calculate_causal_strength(
                G, "insumos", "impactos"
            )

            # Robustez estructural
            robustez = self._calcular_robustez_estructural(
                G, mediadores, resultados)

            # Complejidad causal
            elementos_causales = (
                self.supuestos_causales
                + list(self.mediadores.keys())
                + self.resultados_intermedios
                + self.moderadores
            )
            complejidad = MathematicalInnovations.entropy_based_complexity(
                elementos_causales
            )

            return {
                "coeficiente_global": fuerza_causal,
                "robustez_estructural": robustez,
                "complejidad_causal": complejidad,
                "densidad_grafo": density,
                "clustering_promedio": avg_clustering,
                "nodos_totales": len(G.nodes),
                "aristas_totales": len(G.edges),
            }

        except Exception as e:
            LOGGER.warning(f"Error en cálculo causal avanzado: {e}")
            return {
                "coeficiente_global": 0.5,
                "robustez_estructural": 0.4,
                "complejidad_causal": 0.3,
            }

    def _calcular_robustez_estructural(
        self, G: nx.DiGraph, mediadores: List[str], resultados: List[str]
    ) -> float:
        """Cálculo de robustez estructural del grafo causal."""
        try:
            # Simulación de perturbaciones
            robustez_scores = []

            for _ in range(100):  # 100 simulaciones
                G_perturbed = G.copy()

                # Remover aleatoriamente algunos nodos mediadores
                nodes_to_remove = (
                    np.random.choice(
                        mediadores, size=min(len(mediadores) // 3, 2), replace=False
                    )
                    if len(mediadores) > 2
                    else []
                )

                for node in nodes_to_remove:
                    if G_perturbed.has_node(node):
                        G_perturbed.remove_node(node)

                # Verificar si aún existe camino causal principal
                if nx.has_path(G_perturbed, "insumos", "impactos"):
                    robustez_scores.append(1.0)
                else:
                    robustez_scores.append(0.0)

            return np.mean(robustez_scores)

        except Exception:
            return 0.5


@dataclass(frozen=True)
class EslabonCadenaAvanzado:
    """Eslabón de cadena de valor con capacidades avanzadas."""

    id: str
    tipo: TipoCadenaValor
    indicadores: List[str]
    capacidades_requeridas: List[str]
    puntos_criticos: List[str]
    ventana_temporal: Tuple[int, int]
    kpi_ponderacion: float = 1.0
    riesgos_especificos: List[str] = field(default_factory=list)
    dependencias: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    recursos_estimados: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not (0 <= self.kpi_ponderacion <= 3.0):
            raise ValueError("KPI ponderación debe estar entre 0 y 3.0")
        if self.ventana_temporal[0] > self.ventana_temporal[1]:
            raise ValueError("Ventana temporal inválida")
        if len(self.indicadores) == 0:
            raise ValueError("Debe tener al menos un indicador")

    def calcular_metricas_avanzadas(self) -> Dict[str, float]:
        """Cálculo de métricas avanzadas del eslabón."""
        try:
            # Complejidad operativa
            complejidad_operativa = (
                len(self.capacidades_requeridas) * 0.3
                + len(self.puntos_criticos) * 0.4
                + len(self.dependencias) * 0.3
            ) / 10.0  # Normalización

            # Riesgo agregado
            riesgo_agregado = min(1.0, len(self.riesgos_especificos) * 0.2)

            # Intensidad de recursos
            intensidad_recursos = sum(self.recursos_estimados.values()) / max(
                1, len(self.recursos_estimados)
            )
            intensidad_recursos = min(
                1.0, intensidad_recursos / 1000000
            )  # Normalización por millones

            # Lead time normalizado
            lead_time = self.calcular_lead_time()
            lead_time_normalizado = min(
                1.0, lead_time / 24
            )  # Normalización por 24 meses

            # Factor de stakeholders
            factor_stakeholders = min(1.0, len(self.stakeholders) * 0.15)

            return {
                "complejidad_operativa": complejidad_operativa,
                "riesgo_agregado": riesgo_agregado,
                "intensidad_recursos": intensidad_recursos,
                "lead_time_normalizado": lead_time_normalizado,
                "factor_stakeholders": factor_stakeholders,
                "kpi_ponderado": self.kpi_ponderacion / 3.0,  # Normalización
                "criticidad_global": (
                    complejidad_operativa + riesgo_agregado + lead_time_normalizado
                )
                / 3,
            }

        except Exception:
            return {
                "complejidad_operativa": 0.5,
                "riesgo_agregado": 0.5,
                "intensidad_recursos": 0.5,
                "lead_time_normalizado": 0.5,
                "factor_stakeholders": 0.3,
                "kpi_ponderado": self.kpi_ponderacion / 3.0,
                "criticidad_global": 0.5,
            }

    def calcular_lead_time(self) -> float:
        """Cálculo optimizado del lead time."""
        return (self.ventana_temporal[0] + self.ventana_temporal[1]) / 2.0

    def generar_hash_avanzado(self) -> str:
        """Generación de hash avanzado del eslabón."""
        data = (
            f"{self.id}|{self.tipo.value}|{sorted(self.indicadores)}|"
            f"{sorted(self.capacidades_requeridas)}|{sorted(self.riesgos_especificos)}|"
            f"{self.ventana_temporal}|{self.kpi_ponderacion}"
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


# -------------------- Ontología avanzada --------------------
@dataclass
class OntologiaPoliticasAvanzada:
    """Ontología avanzada para políticas públicas con capacidades de frontera."""

    dimensiones: Dict[str, List[str]]
    relaciones_causales: Dict[str, List[str]]
    indicadores_ods: Dict[str, List[str]]
    taxonomia_evidencia: Dict[str, List[str]]
    patrones_linguisticos: Dict[str, List[str]]
    vocabulario_especializado: Dict[str, List[str]]
    fecha_creacion: str = field(
        default_factory=lambda: datetime.now().isoformat())
    version: str = "3.0-industrial-frontier"

    @classmethod
    def cargar_ontologia_avanzada(cls) -> "OntologiaPoliticasAvanzada":
        """Carga ontología avanzada con capacidades de frontera."""
        try:
            # Dimensiones expandidas con granularidad superior
            dimensiones_frontier = {
                "social_avanzado": [
                    "salud_preventiva",
                    "educacion_calidad",
                    "vivienda_digna",
                    "proteccion_social_integral",
                    "equidad_genero",
                    "inclusion_diversidad",
                    "cohesion_social",
                    "capital_social",
                    "bienestar_subjetivo",
                    "calidad_vida_urbana",
                    "seguridad_ciudadana",
                    "participacion_comunitaria",
                ],
                "economico_transformacional": [
                    "empleo_decente",
                    "productividad_sectorial",
                    "innovacion_tecnologica",
                    "infraestructura_inteligente",
                    "competitividad_territorial",
                    "emprendimiento_social",
                    "economia_circular",
                    "finanzas_sostenibles",
                    "comercio_justo",
                    "turismo_sostenible",
                    "agroindustria_sustentable",
                    "servicios_avanzados",
                ],
                "ambiental_regenerativo": [
                    "sostenibilidad_integral",
                    "biodiversidad_conservacion",
                    "mitigacion_climatica",
                    "adaptacion_climatica",
                    "gestion_integral_residuos",
                    "gestion_hidrica",
                    "energia_renovable",
                    "movilidad_sostenible",
                    "construccion_verde",
                    "agricultura_regenerativa",
                    "bosques_urbanos",
                    "economia_verde",
                ],
                "institucional_transformativo": [
                    "gobernanza_multinivel",
                    "transparencia_activa",
                    "participacion_ciudadana",
                    "rendicion_cuentas",
                    "eficiencia_administrativa",
                    "innovacion_publica",
                    "gobierno_abierto",
                    "justicia_social",
                    "estado_derecho",
                    "capacidades_institucionales",
                    "coordinacion_intersectorial",
                    "planificacion_estrategica",
                ],
                "territorial_inteligente": [
                    "ordenamiento_territorial",
                    "planificacion_urbana",
                    "conectividad_digital",
                    "logistica_territorial",
                    "patrimonio_cultural",
                    "identidad_territorial",
                    "resiliencia_territorial",
                    "sistemas_urbanos",
                ],
            }

            # Relaciones causales avanzadas con múltiples niveles
            relaciones_causales_avanzadas = {
                "inversion_publica_inteligente": [
                    "crecimiento_economico_sostenible",
                    "empleo_formal_calidad",
                    "infraestructura_resiliente",
                    "capacidades_institucionales",
                    "innovacion_territorial",
                    "equidad_espacial",
                ],
                "educacion_transformacional": [
                    "productividad_laboral_avanzada",
                    "innovacion_social",
                    "reduccion_desigualdades",
                    "cohesion_social",
                    "capital_humano_especializado",
                    "emprendimiento_innovador",
                ],
                "salud_integral": [
                    "productividad_economica",
                    "calidad_vida_poblacional",
                    "equidad_social_territorial",
                    "resilienza_comunitaria",
                    "capital_social_saludable",
                ],
                "gobernanza_inteligente": [
                    "transparencia_institucional",
                    "eficiencia_publica",
                    "confianza_ciudadana",
                    "participacion_democratica",
                    "legitimidad_estatal",
                    "capacidad_adaptativa",
                ],
                "sostenibilidad_regenerativa": [
                    "resiliencia_climatica",
                    "economia_circular_territorial",
                    "bienestar_ecosistemico",
                    "salud_ambiental",
                    "prosperidad_sostenible",
                    "justicia_intergeneracional",
                ],
            }

            # Taxonomía de evidencia sofisticada
            taxonomia_evidencia_avanzada = {
                "cuantitativa_robusta": [
                    "estadisticas_oficiales",
                    "encuestas_representativas",
                    "censos_poblacionales",
                    "registros_administrativos",
                    "indicadores_desempeño",
                    "metricas_impacto",
                    "series_temporales",
                    "analisis_econometricos",
                    "evaluaciones_impacto",
                ],
                "cualitativa_profunda": [
                    "entrevistas_profundidad",
                    "grupos_focales",
                    "observacion_participante",
                    "etnografia_institucional",
                    "narrativas_territoriales",
                    "historias_vida",
                    "analisis_discurso",
                    "mapeo_actores",
                    "analisis_redes_sociales",
                ],
                "mixta_integrativa": [
                    "triangulacion_metodologica",
                    "evaluacion_realista",
                    "analisis_configuracional",
                    "metodos_participativos",
                    "investigacion_accion",
                    "evaluacion_desarrollo",
                ],
                "documental_normativa": [
                    "planes_desarrollo",
                    "politicas_publicas",
                    "normatividad_vigente",
                    "reglamentaciones_tecnicas",
                    "lineamientos_sectoriales",
                    "directrices_internacionales",
                ],
            }

            # Patrones lingüísticos avanzados para detección de evidencia
            patrones_linguisticos_especializados = {
                "indicadores_desempeño": [
                    r"\b(?:indicador|metric|medidor|parametro|kpi)\b.*\b(?:de|para|del)\b.*\b(?:desempeño|resultado|impacto|logro)\b",
                    r"\b(?:medir|evaluar|monitorear|seguir|rastrear)\b.*\b(?:progreso|avance|cumplimiento|efectividad)\b",
                    r"\b(?:linea\s+base|baseline|situacion\s+inicial|punto\s+partida)\b.*\d+",
                    r"\b(?:meta|objetivo|target|proposito)\b.*\d+.*\b(?:2024|2025|2026|2027|2028)\b",
                ],
                "recursos_financieros": [
                    r"\$\s*[\d,.]+(?: millones?| mil(?:es)?| billones?)?\b",
                    r"\bpresupuesto\b.*\$?[\d,.]+(?: millones?| mil(?:es)?| billones?)?",
                    r"\b(?:inversion|asignacion|destinacion|cofinanciacion)\b.*\$?[\d,.]+(?: millones?| mil(?:es)?)?",
                    r"\b(?:recursos|fondos|capital|financiacion)\b.*\$?[\d,.]+(?: millones?| mil(?:es)?)?",
                    r"\bCOP\s*[\d,.]+(?: millones?| mil(?:es)?| billones?)?\b",
                ],
                "responsabilidades_institucionales": [
                    r"\b(?:responsable|encargado|lidera|coordina|gestiona|ejecuta)\b:\s*\w+",
                    r"\b(?:secretaria|ministerio|departamento|entidad|institucion)\b.*\b(?:responsable|cargo|funcion)\b",
                    r"\b(?:quien|que)\b.*\b(?:lidera|coordina|ejecuta|implementa)\b",
                    r"\brol\b.*\b(?:de|del|para)\b.*\b(?:secretaria|ministerio|entidad)\b",
                ],
                "temporalidad_plazos": [
                    r"\b(?:plazo|cronograma|calendario|programacion|tiempo)\b.*\b(?:de|para|del)\b.*\b(?:implementacion|ejecucion|desarrollo)\b",
                    r"\b(?:inicio|comienzo|arranque)\b.*\b(?:en|el|durante)\b.*\b(?:20\d{2}|primer|segundo|tercer|cuarto)\b.*\b(?:trimestre|semestre|año)\b",
                    r"\b(?:duracion|periodo|etapa|fase)\b.*\b(?:de|del)\b.*\b(?:\d+)\b.*\b(?:meses|años|trimestres)\b",
                    r"\b(?:hasta|para|antes|durante)\b.*\b(?:20\d{2}|diciembre|final|culminacion)\b",
                ],
                "impactos_resultados": [
                    r"\b(?:impacto|efecto|resultado|consecuencia|cambio)\b.*\b(?:en|sobre|para)\b.*\b(?:poblacion|comunidad|territorio)\b",
                    r"\b(?:beneficio|mejora|incremento|reduccion|disminucion)\b.*\b(?:del|de la|en el|en la)\b.*\b(?:\d+%|\d+ puntos)\b",
                    r"\b(?:transformacion|cambio|modificacion)\b.*\b(?:social|economica|ambiental|institucional|territorial)\b",
                ],
            }

            # Vocabulario especializado expandido
            vocabulario_especializado_ampliado = {
                "planificacion_territorial": [
                    "ordenamiento_territorial",
                    "zonificacion",
                    "uso_suelo",
                    "plan_ordenamiento",
                    "esquema_ordenamiento",
                    "plan_basico_ordenamiento",
                    "pot",
                    "eot",
                    "pbot",
                    "suelo_urbano",
                    "suelo_rural",
                    "suelo_expansion",
                    "suelo_proteccion",
                ],
                "desarrollo_sostenible": [
                    "objetivos_desarrollo_sostenible",
                    "ods",
                    "agenda_2030",
                    "sostenibilidad",
                    "desarrollo_humano",
                    "crecimiento_verde",
                    "economia_circular",
                    "resilencia_climatica",
                ],
                "gobernanza_publica": [
                    "participacion_ciudadana",
                    "transparencia",
                    "rendicion_cuentas",
                    "gobierno_abierto",
                    "cocreacion",
                    "corresponsabilidad",
                    "veeduria_ciudadana",
                    "control_social",
                ],
                "gestion_publica": [
                    "meci",
                    "modelo_integrado_planeacion_gestion",
                    "sistema_gestion_calidad",
                    "plan_desarrollo_territorial",
                    "pdt",
                    "plan_accion",
                    "seguimiento_evaluacion",
                ],
            }

            # Carga de indicadores ODS especializados
            indicadores_ods_especializados = cls._cargar_indicadores_ods_avanzados()

            return cls(
                dimensiones=dimensiones_frontier,
                relaciones_causales=relaciones_causales_avanzadas,
                indicadores_ods=indicadores_ods_especializados,
                taxonomia_evidencia=taxonomia_evidencia_avanzada,
                patrones_linguisticos=patrones_linguisticos_especializados,
                vocabulario_especializado=vocabulario_especializado_ampliado,
            )

        except Exception as e:
            log_error_with_text(
                LOGGER, f"❌ Error cargando ontología avanzada: {e}")
            raise SystemExit("Fallo en carga de ontología avanzada")

    @staticmethod
    def _cargar_indicadores_ods_avanzados() -> Dict[str, List[str]]:
        """Carga indicadores ODS con granularidad avanzada."""
        indicadores_path = Path("indicadores_ods_avanzados.json")

        # Indicadores base expandidos y especializados
        indicadores_especializados = {
            "ods1_pobreza": [
                "tasa_pobreza_monetaria",
                "tasa_pobreza_extrema",
                "indice_pobreza_multidimensional",
                "coeficiente_gini",
                "proteccion_social_cobertura",
                "acceso_servicios_basicos",
                "vulnerabilidad_economica",
                "resiliencia_economica_hogares",
                "activos_productivos_acceso",
            ],
            "ods3_salud": [
                "mortalidad_infantil",
                "mortalidad_materna",
                "esperanza_vida_nacimiento",
                "acceso_servicios_salud",
                "cobertura_vacunacion",
                "prevalencia_enfermedades_cronicas",
                "salud_mental_indicadores",
                "seguridad_alimentaria",
                "agua_potable_saneamiento_acceso",
            ],
            "ods4_educacion": [
                "tasa_alfabetizacion",
                "matriucla_educacion_basica",
                "permanencia_educativa",
                "calidad_educativa_pruebas",
                "acceso_educacion_superior",
                "formacion_tecnica_profesional",
                "educacion_digital_competencias",
                "infraestructura_educativa_calidad",
            ],
            "ods5_genero": [
                "participacion_politica_mujeres",
                "brecha_salarial_genero",
                "violencia_genero_prevalencia",
                "acceso_credito_mujeres",
                "liderazgo_empresarial_femenino",
                "uso_tiempo_trabajo_cuidado",
                "educacion_ciencia_tecnologia_mujeres",
                "derechos_reproductivos_acceso",
            ],
            "ods8_trabajo": [
                "tasa_empleo",
                "tasa_desempleo",
                "empleo_informal",
                "trabajo_decente_indicadores",
                "productividad_laboral",
                "crecimiento_economico_pib",
                "diversificacion_economica",
                "emprendimiento_formal",
                "inclusion_financiera",
                "innovacion_empresarial",
            ],
            "ods11_ciudades": [
                "vivienda_adecuada_acceso",
                "transporte_publico_acceso",
                "espacios_publicos_calidad",
                "gestion_residuos_solidos",
                "calidad_aire",
                "planificacion_urbana_participativa",
                "patrimonio_cultural_proteccion",
                "resiliencia_desastres",
                "conectividad_urbana",
            ],
            "ods13_clima": [
                "emisiones_gei_per_capita",
                "vulnerabilidad_climatica",
                "adaptacion_climatica_medidas",
                "educacion_ambiental",
                "energia_renovable_uso",
                "eficiencia_energetica",
                "conservacion_ecosistemas",
                "reforestacion_restauracion",
                "economia_baja_carbono",
            ],
            "ods16_paz": [
                "indice_transparencia",
                "percepcion_corrupcion",
                "acceso_justicia",
                "participacion_decisiones_publicas",
                "libertad_expresion",
                "seguridad_ciudadana",
                "confianza_instituciones",
                "estado_derecho_fortalecimiento",
                "inclusion_social_politica",
            ],
            "ods17_alianzas": [
                "cooperacion_internacional",
                "transferencia_tecnologia",
                "capacitacion_institucional",
                "movilizacion_recursos_domesticos",
                "comercio_internacional",
                "acceso_mercados",
                "sostenibilidad_deuda",
                "sistemas_monitoreo_datos",
                "alianzas_publico_privadas",
            ],
        }

        # Intentar cargar desde archivo si existe
        if indicadores_path.exists():
            try:
                with open(indicadores_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and len(data) >= 8:
                    LOGGER.info(
                        "✅ Indicadores ODS avanzados cargados desde archivo")
                    return data
                else:
                    LOGGER.warning(
                        "⚠️ Indicadores ODS avanzados inválidos, usando base especializada"
                    )
            except Exception as e:
                LOGGER.warning(
                    f"⚠️ Error leyendo indicadores avanzados {indicadores_path}: {e}"
                )

        # Guardar template avanzado
        try:
            with open(indicadores_path, "w", encoding="utf-8") as f:
                json.dump(indicadores_especializados, f,
                          indent=2, ensure_ascii=False)
            LOGGER.info(
                f"✅ Template ODS avanzado generado: {indicadores_path}")
        except Exception as e:
            LOGGER.error(f"❌ Error generando template ODS avanzado: {e}")

        return indicadores_especializados

    def buscar_patrones_avanzados(
        self, texto: str, categoria: str
    ) -> List[Dict[str, Any]]:
        """Búsqueda avanzada de patrones lingüísticos en texto."""
        if categoria not in self.patrones_linguisticos:
            return []

        patrones = self.patrones_linguisticos[categoria]
        resultados = []

        for i, patron in enumerate(patrones):
            try:
                matches = re.finditer(
                    patron, texto, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    resultado = {
                        "texto_encontrado": match.group(),
                        "posicion_inicio": match.start(),
                        "posicion_fin": match.end(),
                        "patron_id": i,
                        "categoria": categoria,
                        "confianza": self._calcular_confianza_patron(
                            match.group(), patron
                        ),
                        "contexto": texto[
                            max(0, match.start() - 50): match.end() + 50
                        ],
                    }
                    resultados.append(resultado)
            except re.error:
                continue

        return sorted(resultados, key=lambda x: x["confianza"], reverse=True)

    def _calcular_confianza_patron(self, texto_match: str, patron: str) -> float:
        """Calcula confianza del patrón encontrado."""
        try:
            # Factores de confianza
            longitud_factor = min(
                1.0, len(texto_match) / 50
            )  # Textos más largos = mayor confianza
            complejidad_patron = min(
                1.0, len(patron) / 100
            )  # Patrones más complejos = mayor precisión

            # Verificar presencia de números (para indicadores cuantitativos)
            tiene_numeros = bool(re.search(r"\d+", texto_match))
            factor_numerico = 0.2 if tiene_numeros else 0.0

            # Verificar presencia de fechas
            tiene_fechas = bool(re.search(r"20\d{2}", texto_match))
            factor_temporal = 0.15 if tiene_fechas else 0.0

            # Confianza base
            confianza_base = 0.6

            return min(
                1.0,
                confianza_base
                + longitud_factor * 0.2
                + complejidad_patron * 0.1
                + factor_numerico
                + factor_temporal,
            )

        except Exception:
            return 0.5  # -------------------- Contexto y evaluación industrial avanzada --------------------


@dataclass
class ClusterMetadataAvanzada:
    """Metadatos de clusters semánticos utilizados en el decálogo."""

    cluster_id: int
    nombre: str
    palabras_clave: List[str]
    vector_representativo: np.ndarray
    miembros: List[str] = field(default_factory=list)
    interdependencias: Dict[int, float] = field(default_factory=dict)

    def calcular_metricas_cluster(self) -> Dict[str, Any]:
        """Calcula métricas agregadas del cluster para análisis exploratorio."""
        tamano = len(self.miembros) or 1
        norma_vector = (
            float(np.linalg.norm(self.vector_representativo))
            if self.vector_representativo.size
            else 0.0
        )
        densidad_semantica = norma_vector / max(1, len(self.palabras_clave))
        diversidad_palabras = len(set(p.lower() for p in self.palabras_clave)) / max(
            1, len(self.palabras_clave)
        )
        centralidad = (
            float(np.mean(list(self.interdependencias.values())))
            if self.interdependencias
            else 0.0
        )

        return {
            "cluster_id": self.cluster_id,
            "nombre": self.nombre,
            "tamano": len(self.miembros),
            "densidad_semantica": round(densidad_semantica, 4),
            "diversidad_palabras": round(diversidad_palabras, 4),
            "centralidad_promedio": round(centralidad, 4),
            "vector_norma": round(norma_vector, 4),
        }

    def calcular_interdependencias_avanzadas(self, grafo: nx.Graph) -> Dict[int, float]:
        """Calcula interdependencias con otros clusters utilizando el grafo provisto."""
        if grafo.number_of_nodes() == 0:
            return {}

        centralidad = nx.degree_centrality(grafo)
        resultados: Dict[int, float] = {}
        for nodo, valor in centralidad.items():
            if isinstance(nodo, tuple) and nodo[0] == self.cluster_id:
                resultados[nodo[1]] = valor
            elif nodo in self.miembros:
                resultados.setdefault(self.cluster_id, 0.0)
                resultados[self.cluster_id] = max(
                    resultados[self.cluster_id], valor)

        if resultados:
            self.interdependencias.update(resultados)
        return resultados

    @staticmethod
    def _calcular_modularidad(
        grafo: nx.Graph, comunidades: Optional[List[set]] = None
    ) -> float:
        """Calcula la modularidad del grafo para evaluar cohesión de clusters."""
        if grafo.number_of_nodes() == 0:
            return 0.0
        try:
            from networkx.algorithms.community import (
                greedy_modularity_communities,
                modularity,
            )
        except ImportError:  # pragma: no cover
            return 0.0

        if comunidades is None:
            comunidades = list(greedy_modularity_communities(grafo))
        if not comunidades:
            return 0.0
        return float(modularity(grafo, comunidades))


@dataclass
class DecalogoContextoAvanzado:
    """Contenedor de dimensiones, clusters y metadatos del decálogo industrial."""

    dimensiones: List["DimensionDecalogoAvanzada"]
    clusters: List[ClusterMetadataAvanzada]
    metadata_general: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def cargar_decalogo_industrial_avanzado(
        cls,
        path: Optional[Union[str, Path]] = None,
    ) -> "DecalogoContextoAvanzado":
        """Carga el decálogo desde un JSON o construye uno por defecto."""
        data: Dict[str, Any] = {}
        if path:
            ruta = Path(path)
            if ruta.exists():
                try:
                    data = json.loads(ruta.read_text(encoding="utf-8"))
                except Exception as exc:  # pragma: no cover - lectura segura
                    LOGGER.warning(
                        "Error cargando decálogo personalizado: %s", exc)
                    data = {}
        if not data:
            try:
                raw = get_decalogo_industrial()
                if isinstance(raw, str):
                    data = json.loads(raw)
                else:
                    data = raw
            except Exception:
                data = {}

        if not data.get("dimensiones"):
            data["dimensiones"] = [
                {
                    "id": idx + 1,
                    "nombre": nombre,
                    "cluster": cluster,
                    "prioridad_estrategica": 1.0,
                    "complejidad_implementacion": 0.5,
                }
                for idx, (nombre, cluster) in enumerate(
                    [
                        ("Gobernanza y articulación", "institucional"),
                        ("Innovación productiva", "productivo"),
                        ("Sostenibilidad territorial", "territorial"),
                    ]
                )
            ]

        dimensiones: List[DimensionDecalogoAvanzada] = []
        clusters: List[ClusterMetadataAvanzada] = []
        for dim in data.get("dimensiones", []):
            teoria = TeoriaCambioAvanzada(
                supuestos_causales=["Supuesto de cambio principal"],
                mediadores={
                    "institucional": ["Capacidad de coordinación"],
                    "operativo": ["Infraestructura básica"],
                },
                resultados_intermedios=["Servicios fortalecidos"],
                precondiciones=["Voluntad política"],
                moderadores=["Contexto económico"],
                variables_contextuales=["Cobertura territorial"],
                mecanismos_causales=["Incentivos alineados"],
            )
            eslabones = [
                EslabonCadenaAvanzado(
                    nombre="Diagnóstico avanzado",
                    descripcion="Generación y análisis de datos para soporte causal",
                    capacidades_clave=["analítica avanzada", "gobernanza"],
                    complejidad_operativa=0.6,
                    madurez_digital=0.5,
                    evidencia_disponible=True,
                ),
                EslabonCadenaAvanzado(
                    nombre="Implementación",
                    descripcion="Ejecución y monitoreo adaptativo",
                    capacidades_clave=[
                        "gestión adaptativa", "participación social"],
                    complejidad_operativa=0.5,
                    madurez_digital=0.4,
                    evidencia_disponible=True,
                ),
                EslabonCadenaAvanzado(
                    nombre="Escalamiento",
                    descripcion="Ampliación territorial y sostenibilidad",
                    capacidades_clave=["financiamiento", "gestión del cambio"],
                    complejidad_operativa=0.7,
                    madurez_digital=0.6,
                    evidencia_disponible=False,
                ),
                EslabonCadenaAvanzado(
                    nombre="Monitoreo",
                    descripcion="Seguimiento con analítica avanzada",
                    capacidades_clave=["datos", "evaluación"],
                    complejidad_operativa=0.55,
                    madurez_digital=0.65,
                    evidencia_disponible=True,
                ),
            ]
            dimensiones.append(
                DimensionDecalogoAvanzada(
                    id=dim.get("id", len(dimensiones) + 1),
                    nombre=dim.get("nombre", "Dimensión sin nombre"),
                    cluster=dim.get("cluster", "general"),
                    teoria_cambio=teoria,
                    eslabones=eslabones,
                    prioridad_estrategica=float(
                        dim.get("prioridad_estrategica", 1.0)),
                    complejidad_implementacion=float(
                        dim.get("complejidad_implementacion", 0.5)
                    ),
                    interdependencias=dim.get("interdependencias", []),
                    contexto_territorial=dim.get("contexto_territorial", {}),
                )
            )

        for idx, cluster_data in enumerate(data.get("clusters", [])):
            vector = np.array(cluster_data.get(
                "vector", [0.0, 0.0, 0.0]), dtype=float)
            clusters.append(
                ClusterMetadataAvanzada(
                    cluster_id=cluster_data.get("id", idx + 1),
                    nombre=cluster_data.get("nombre", f"Cluster {idx + 1}"),
                    palabras_clave=cluster_data.get("palabras_clave", []),
                    vector_representativo=vector,
                    miembros=cluster_data.get("miembros", []),
                    interdependencias=cluster_data.get(
                        "interdependencias", {}),
                )
            )

        if not clusters:
            clusters = [
                ClusterMetadataAvanzada(
                    cluster_id=1,
                    nombre="Gobernanza",
                    palabras_clave=[
                        "institucionalidad",
                        "coordinación",
                        "transparencia",
                    ],
                    vector_representativo=np.array(
                        [0.7, 0.5, 0.6], dtype=float),
                    miembros=["Gobernanza y articulación"],
                ),
                ClusterMetadataAvanzada(
                    cluster_id=2,
                    nombre="Productividad",
                    palabras_clave=["innovación",
                                    "competitividad", "tecnología"],
                    vector_representativo=np.array(
                        [0.6, 0.8, 0.65], dtype=float),
                    miembros=["Innovación productiva"],
                ),
            ]

        metadata_general = {
            "fuente": data.get("fuente", "plantilla_interna"),
            "actualizado_en": datetime.now().isoformat(),
            "numero_dimensiones": len(dimensiones),
            "numero_clusters": len(clusters),
        }
        metadata_general.update(data.get("metadata_general", {}))

        return cls(
            dimensiones=dimensiones,
            clusters=clusters,
            metadata_general=metadata_general,
        )

    def obtener_decalogo_contexto_avanzado(self) -> Dict[str, Any]:
        """Retorna el contexto completo del decálogo con métricas resumidas."""
        return {
            "metadata": self.metadata_general,
            "dimensiones": [dim.to_dict() for dim in self.dimensiones],
            "clusters": [
                cluster.calcular_metricas_cluster() for cluster in self.clusters
            ],
        }


@dataclass
class DimensionDecalogoAvanzada:
    """Dimensión del decálogo con capacidades matemáticas de frontera."""

    id: int
    nombre: str
    cluster: str
    teoria_cambio: TeoriaCambioAvanzada
    eslabones: List[EslabonCadenaAvanzado]
    prioridad_estrategica: float = 1.0
    complejidad_implementacion: float = 0.5
    interdependencias: List[int] = field(default_factory=list)
    contexto_territorial: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (1 <= self.id <= 10):
            raise ValueError("ID de dimensión debe estar entre 1 y 10")
        if len(self.nombre.strip()) < 5:
            raise ValueError("Nombre de dimensión debe ser más descriptivo")
        if len(self.eslabones) < 2:
            raise ValueError(
                "Una dimensión debe contar con al menos dos eslabones")
        self.prioridad_estrategica = float(
            np.clip(self.prioridad_estrategica, 0.1, 2.0)
        )
        self.complejidad_implementacion = float(
            np.clip(self.complejidad_implementacion, 0.1, 2.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "nombre": self.nombre,
            "cluster": self.cluster,
            "prioridad_estrategica": self.prioridad_estrategica,
            "complejidad_implementacion": self.complejidad_implementacion,
            "interdependencias": self.interdependencias,
            "contexto_territorial": self.contexto_territorial,
            "eslabones": [eslabon.to_dict() for eslabon in self.eslabones],
        }


@dataclass
class EvaluacionCausalIndustrialAvanzada:
    """Evaluación causal avanzada con capacidades matemáticas de frontera."""

    consistencia_logica: float
    identificabilidad_causal: float
    factibilidad_operativa: float
    certeza_probabilistica: float
    robustez_causal: float
    innovacion_metodologica: float
    sostenibilidad_resultados: float
    escalabilidad_territorial: float
    riesgos_implementacion: List[str]
    supuestos_criticos: List[str]
    evidencia_soporte: int
    brechas_criticas: int
    factores_contextuales: Dict[str, float] = field(default_factory=dict)
    interdependencias_detectadas: List[str] = field(default_factory=list)
    evidencia_detallada: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict)
    extractor_evidencia: Optional["ExtractorEvidenciaIndustrialAvanzado"] = None

    def __post_init__(self) -> None:
        campos = [
            "consistencia_logica",
            "identificabilidad_causal",
            "factibilidad_operativa",
            "certeza_probabilistica",
            "robustez_causal",
            "innovacion_metodologica",
            "sostenibilidad_resultados",
            "escalabilidad_territorial",
        ]
        for campo in campos:
            valor = float(getattr(self, campo))
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"Campo {campo} debe estar entre 0.0 y 1.0")
            setattr(self, campo, valor)
        self.evidencia_soporte = int(max(0, self.evidencia_soporte))
        self.brechas_criticas = int(max(0, self.brechas_criticas))

    @property
    def puntaje_global_avanzado(self) -> float:
        pesos = {
            "consistencia_logica": 0.18,
            "identificabilidad_causal": 0.16,
            "factibilidad_operativa": 0.15,
            "certeza_probabilistica": 0.14,
            "robustez_causal": 0.13,
            "innovacion_metodologica": 0.10,
            "sostenibilidad_resultados": 0.08,
            "escalabilidad_territorial": 0.06,
        }
        puntaje = 0.0
        for campo, peso in pesos.items():
            valor = getattr(self, campo)
            ajuste = self.factores_contextuales.get(campo, 1.0)
            puntaje += valor * peso * ajuste
        puntaje += min(0.1, self.evidencia_soporte * 0.01)
        puntaje -= min(0.1, self.brechas_criticas * 0.015)
        return float(np.clip(puntaje, 0.0, 1.2))

    def nivel_certidumbre_avanzado(self) -> str:
        puntaje = (self.certeza_probabilistica + self.robustez_causal) / 2
        if puntaje >= 0.85:
            return "ALTA"
        if puntaje >= 0.65:
            return "MEDIA"
        if puntaje >= 0.45:
            return "BAJA"
        return "MUY BAJA"

    def recomendacion_estrategica_avanzada(self) -> str:
        puntaje = self.puntaje_global_avanzado
        nivel = self.nivel_certidumbre_avanzado()
        if puntaje >= 0.85 and nivel == "ALTA":
            return "Escalar intervención con enfoque territorial y financiar seguimiento avanzado."
        if puntaje >= 0.7:
            return (
                "Implementar con pilotos controlados y fortalecer evidencia operativa."
            )
        if puntaje >= 0.5:
            return "Priorizar fortalecimiento de capacidades institucionales antes de escalar."
        return "Replantear teoría de cambio y reforzar supuestos críticos."

    def calcular_indice_madurez_tecnologica(self) -> Dict[str, Any]:
        base = (
            self.factibilidad_operativa * 0.35
            + self.robustez_causal * 0.25
            + self.sostenibilidad_resultados * 0.2
            + self.innovacion_metodologica * 0.2
        )
        trl = int(np.clip(round(base * 9), 1, 9))
        return {
            "trl_nivel": trl,
            "recomendacion": self._generar_recomendacion_trl(trl),
            "factores_limitantes": self._identificar_factores_limitantes_trl(trl),
        }

    def _generar_recomendacion_trl(self, nivel: int) -> str:
        if nivel >= 8:
            return "Consolidar escalamiento territorial y monitoreo en tiempo real."
        if nivel >= 6:
            return "Aumentar pruebas piloto multirregionales y ajustar supuestos."
        if nivel >= 4:
            return "Fortalecer diseño y validación de prototipos institucionales."
        return "Concentrarse en generación de evidencia y pruebas conceptuales."

    def _identificar_factores_limitantes_trl(self, nivel: int) -> List[str]:
        factores: List[str] = []
        if self.innovacion_metodologica < 0.5:
            factores.append("Innovación metodológica limitada")
        if self.sostenibilidad_resultados < 0.5:
            factores.append("Sostenibilidad no comprobada")
        if self.factibilidad_operativa < 0.5:
            factores.append("Capacidades operativas insuficientes")
        if nivel < 4:
            factores.append(
                "Falta evidencia de implementación en contexto real")
        return factores

    def generar_matriz_decision_multicriterio(self) -> pd.DataFrame:
        criterios = {
            "Consistencia lógica": self.consistencia_logica,
            "Identificabilidad causal": self.identificabilidad_causal,
            "Factibilidad operativa": self.factibilidad_operativa,
            "Certeza probabilística": self.certeza_probabilistica,
            "Robustez causal": self.robustez_causal,
            "Innovación metodológica": self.innovacion_metodologica,
            "Sostenibilidad resultados": self.sostenibilidad_resultados,
            "Escalabilidad territorial": self.escalabilidad_territorial,
        }
        normalizado = pd.Series(criterios).clip(0.0, 1.0)
        pesos = normalizado / normalizado.sum()
        return pd.DataFrame(
            {
                "valor": normalizado,
                "peso_normalizado": pesos,
                "clasificacion": [
                    self._generar_recomendacion_priorizacion(v) for v in normalizado
                ],
            }
        )

    @staticmethod
    def _generar_recomendacion_priorizacion(valor: float) -> str:
        if valor >= 0.8:
            return "FORTALECER Y ESCALAR"
        if valor >= 0.6:
            return "MANTENER CON MEJORA CONTINUA"
        if valor >= 0.4:
            return "PRIORIZAR ACCIONES DE CIERRE DE BRECHAS"
        return "ATENCIÓN INMEDIATA"

    @property
    def puntaje_final_avanzado(self) -> float:
        puntaje_base = self.puntaje_global_avanzado * 100
        ajuste_evidencia = min(8.0, self.evidencia_soporte * 0.6)
        ajuste_riesgos = -min(12.0, len(self.riesgos_implementacion) * 1.2)
        ajuste_brechas = -min(10.0, self.brechas_criticas * 0.9)
        puntaje = puntaje_base + ajuste_evidencia + ajuste_riesgos + ajuste_brechas
        return float(np.clip(puntaje, 0.0, 100.0))

    def nivel_madurez_avanzado(self) -> str:
        resultado = self.calcular_indice_madurez_tecnologica()
        nivel = resultado["trl_nivel"]
        if nivel >= 8:
            return "Optimizado y listo para escalamiento"
        if nivel >= 6:
            return "Gestionado cuantitativamente"
        if nivel >= 4:
            return "Diseño avanzado con pilotos requeridos"
        if nivel >= 3:
            return "Definido con componentes clave pendientes"
        return "Inicial en formulación"

    def generar_reporte_tecnico_avanzado(self) -> Dict[str, Any]:
        matriz = self.generar_matriz_decision_multicriterio()
        riesgos = self._analizar_riesgos_probabilistico()
        return {
            "puntaje_global": self.puntaje_global_avanzado,
            "puntaje_final": self.puntaje_final_avanzado,
            "nivel_madurez": self.nivel_madurez_avanzado(),
            "recomendacion_estrategica": self.recomendacion_estrategica_avanzada(),
            "matriz_decision": matriz,
            "analisis_riesgos": riesgos,
            "supuestos_criticos": self.supuestos_criticos,
            "interdependencias_detectadas": self.interdependencias_detectadas,
        }

    def _analizar_riesgos_probabilistico(self) -> pd.DataFrame:
        if not self.riesgos_implementacion:
            return pd.DataFrame(
                columns=["riesgo", "probabilidad",
                         "impacto", "riesgo_agregado"]
            )
        probabilidades = np.linspace(
            0.6, 0.3, num=len(self.riesgos_implementacion))
        impactos = np.linspace(0.8, 0.4, num=len(self.riesgos_implementacion))
        data = []
        for riesgo, prob, imp in zip(
            self.riesgos_implementacion, probabilidades, impactos
        ):
            data.append(
                {
                    "riesgo": riesgo,
                    "probabilidad": round(float(prob), 3),
                    "impacto": round(float(imp), 3),
                    "riesgo_agregado": round(float(prob * imp), 3),
                }
            )
        return pd.DataFrame(data)

    def _generar_estrategia_mitigacion(self, matriz_riesgo: pd.DataFrame) -> List[str]:
        estrategias: List[str] = []
        if matriz_riesgo.empty:
            return ["Mantener monitoreo continuo de riesgos con revisión trimestral."]
        for _, fila in matriz_riesgo.iterrows():
            if fila["riesgo_agregado"] > 0.45:
                estrategias.append(
                    f"Implementar plan de mitigación inmediato para '{fila['riesgo']}' con responsables definidos."
                )
            elif fila["riesgo_agregado"] > 0.25:
                estrategias.append(
                    f"Desarrollar medidas preventivas y pilotos de control para '{fila['riesgo']}'."
                )
            else:
                estrategias.append(
                    f"Monitorear '{fila['riesgo']}' con indicadores de alerta temprana."
                )
        return estrategias

    def _evaluar_calidad_contenido(self, documento: str) -> float:
        if not documento:
            return 0.0
        longitud = len(documento.split())
        menciones = sum(
            1
            for palabra in ["impacto", "resultado", "evidencia", "indicador"]
            if palabra in documento.lower()
        )
        riqueza = min(1.0, longitud / 500)
        return round((0.6 * riqueza) + (0.4 * menciones / 4), 3)

    def _diversificar_resultados(
        self, evidencias: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        vistos = set()
        resultados: List[Dict[str, Any]] = []
        for evidencia in evidencias:
            clave = evidencia.get("id") or evidencia.get("titulo")
            if clave and clave not in vistos:
                resultados.append(evidencia)
                vistos.add(clave)
            elif not clave:
                resultados.append(evidencia)
        return resultados

    def _buscar_evidencia_fallback(self, query: str) -> List[Dict[str, Any]]:
        return [
            {
                "id": "fallback-1",
                "descripcion": f"Síntesis cualitativa sobre {query}",
                "confianza": 0.45,
                "tipo": "cualitativa",
            }
        ]

    def buscar_patrones_ontologicos_avanzados(
        self, conceptos: List[str]
    ) -> List[Dict[str, Any]]:
        ontologia = OntologiaPoliticasAvanzada()
        patrones = ontologia.buscar_patrones_avanzados(conceptos)
        return [patron for patron in patrones if patron.get("confianza", 0.0) >= 0.4]

    def extraer_variables_operativas_avanzadas(
        self, evidencia: Dict[str, Any]
    ) -> Dict[str, Any]:
        texto = evidencia.get("texto", "")
        return {
            "menciones_indicadores": texto.lower().count("indicador"),
            "menciones_resultados": texto.lower().count("resultado"),
            "menciones_financieras": sum(
                texto.lower().count(p)
                for p in ["millon", "presupuesto", "financiación"]
            ),
            "extension_palabras": len(texto.split()),
        }

    def _clasificar_nivel_institucional(self, evidencia: Dict[str, Any]) -> str:
        fuente = evidencia.get("fuente", "").lower()
        if any(p in fuente for p in ["ministerio", "gobierno", "secretaría"]):
            return "institucional"
        if any(p in fuente for p in ["ong", "cooperativa", "asociación"]):
            return "social"
        if "empresa" in fuente or "privado" in fuente:
            return "privado"
        return "mixto"

    def generar_matriz_trazabilidad_avanzada(self) -> pd.DataFrame:
        registros: List[Dict[str, Any]] = []
        for categoria, evidencias in self.evidencia_detallada.items():
            for evidencia in evidencias:
                registros.append(
                    {
                        "categoria": categoria,
                        "fuente": evidencia.get("fuente", "desconocida"),
                        "anio": evidencia.get("anio", datetime.now().year),
                        "confianza": evidencia.get("confianza", 0.5),
                    }
                )
        return pd.DataFrame(registros)

    def _calcular_confiabilidad_evidencia(self, evidencia: Dict[str, Any]) -> float:
        base = evidencia.get("confianza", 0.5)
        if evidencia.get("tiene_cuantificacion"):
            base += 0.15
        if evidencia.get("tipo") == "cuantitativa":
            base += 0.1
        if evidencia.get("fuente", "").lower().startswith("oficial"):
            base += 0.1
        return float(np.clip(base, 0.0, 1.0))

    def _evaluar_completitud_evidencia(self) -> float:
        categorias = {"cuantitativa", "cualitativa",
                      "documental", "testimonial"}
        presentes = {
            clave
            for clave, evidencias in self.evidencia_detallada.items()
            if evidencias
        }
        return round(len(presentes.intersection(categorias)) / len(categorias), 3)

    def _evaluar_coherencia_temporal_evidencia(self) -> float:
        anios: List[int] = []
        for evidencias in self.evidencia_detallada.values():
            anios.extend(
                [e.get("anio")
                 for e in evidencias if isinstance(e.get("anio"), int)]
            )
        if not anios:
            return 0.5
        rango = max(anios) - min(anios) + 1
        return float(np.clip(1.0 / rango, 0.2, 1.0))

    def _determinar_sentimiento_predominante(self, texto: str) -> str:
        positivo = sum(
            texto.lower().count(p) for p in ["exitoso", "positivo", "fortalecido"]
        )
        negativo = sum(texto.lower().count(p)
                       for p in ["crítico", "débil", "riesgo"])
        if positivo > negativo:
            return "positivo"
        if negativo > positivo:
            return "negativo"
        return "neutral"

    def buscar_evidencia_causal_avanzadas(
        self,
        query: str,
        conceptos_clave: List[str],
        top_k: int = 5,
        umbral_certeza: float = 0.6,
    ) -> List[Dict[str, Any]]:
        if self.extractor_evidencia is None:
            return self._buscar_evidencia_fallback(query)
        resultados = self.extractor_evidencia.buscar_evidencia_causal_avanzada(
            query=query,
            conceptos_clave=conceptos_clave,
            top_k=top_k,
        )
        resultados = [
            r for r in resultados if r.get("confianza", 0.0) >= umbral_certeza
        ]
        if not resultados:
            resultados = self._buscar_evidencia_fallback(query)
        return self._diversificar_resultados(resultados)


@dataclass
class ExtractorEvidenciaIndustrialAvanzado:
    """Extractor avanzado de evidencia causal basada en NLP híbrido."""

    corpus: List[str]
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    modelo_embeddings: Optional[SentenceTransformer] = None
    vectorizer: Optional[TfidfVectorizer] = None
    embedding_cache: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self._inicializar_capacidades_avanzadas()

    def _inicializar_capacidades_avanzadas(self) -> None:
        if self.modelo_embeddings is None:
            try:
                self.modelo_embeddings = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception as exc:  # pragma: no cover - entorno sin modelo
                LOGGER.warning(
                    "No fue posible cargar modelo SentenceTransformer: %s", exc
                )
                self.modelo_embeddings = None
        self._precomputar_tfidf()
        self._precomputar_embeddings_avanzados()

    def _precomputar_embeddings_avanzados(self) -> None:
        if self.modelo_embeddings is None or not self.corpus:
            self.embedding_cache = None
            return
        try:
            batch = self.modelo_embeddings.encode(
                self.corpus, convert_to_numpy=True, show_progress_bar=False
            )
            self.embedding_cache = batch.astype(np.float32)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Fallo en cómputo de embeddings: %s", exc)
            self.embedding_cache = None

    def _extraer_caracteristicas_texto(self, texto: str) -> Dict[str, Any]:
        tokens = texto.split()
        return {
            "longitud": len(tokens),
            "longitud_oraciones": (
                np.mean(
                    [
                        len(oracion.split())
                        for oracion in texto.split(".")
                        if oracion.strip()
                    ]
                )
                if texto
                else 0.0
            ),
            "palabras_unicas": len(set(tokens)),
        }

    def _clasificar_tipo_contenido(self, texto: str) -> str:
        texto_lower = texto.lower()
        if any(p in texto_lower for p in ["decreto", "ley", "reglamento"]):
            return "normativo"
        if any(p in texto_lower for p in ["evaluación", "indicador", "análisis"]):
            return "analítico"
        if any(p in texto_lower for p in ["caso", "experiencia", "testimonio"]):
            return "cualitativo"
        return "general"

    def _precomputar_tfidf(self) -> None:
        if not self.corpus:
            self.vectorizer = None
            return
        self.vectorizer = TfidfVectorizer(
            max_features=2048, ngram_range=(1, 2))
        try:
            self.vectorizer.fit(self.corpus)
        except ValueError:
            self.vectorizer = None

    def _analizar_estructura_documental(self, texto: str) -> Dict[str, Any]:
        parrafos = [p for p in texto.split("\n") if p.strip()]
        return {
            "numero_parrafos": len(parrafos),
            "longitud_promedio_parrafo": (
                np.mean([len(p.split())
                        for p in parrafos]) if parrafos else 0.0
            ),
        }

    def _calcular_densidad_causal_avanzada(self, texto: str) -> float:
        texto_lower = texto.lower()
        terminos = ["causa", "efecto", "impacto", "resultado", "indicador"]
        total = sum(texto_lower.count(t) for t in terminos)
        return float(np.clip(total / 20.0, 0.0, 1.0))

    def _analizar_sentimientos_texto(self, texto: str) -> Dict[str, float]:
        positivo = sum(texto.lower().count(p)
                       for p in ["fortaleza", "mejora", "éxito"])
        negativo = sum(
            texto.lower().count(p) for p in ["riesgo", "problema", "debilidad"]
        )
        total = positivo + negativo or 1
        return {
            "positivo": positivo / total,
            "negativo": negativo / total,
            "sentimiento_predominante": (
                "positivo"
                if positivo > negativo
                else "negativo"
                if negativo > positivo
                else "neutral"
            ),
        }

    def buscar_evidencia_causal_avanzada(
        self,
        query: str,
        conceptos_clave: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        resultados: List[Dict[str, Any]] = []
        if self.embedding_cache is not None and self.modelo_embeddings is not None:
            query_embedding = self.modelo_embeddings.encode(
                [query], convert_to_numpy=True, show_progress_bar=False
            )[0]
            query_embedding = query_embedding.astype(np.float32)
            similitudes = cosine_similarity(
                [query_embedding], self.embedding_cache)[0]
            indices = np.argsort(similitudes)[::-1][:top_k]
            for idx in indices:
                confianza = float(np.clip(similitudes[idx], 0.0, 1.0))
                texto = self.corpus[idx]
                resultados.append(
                    {
                        "id": (
                            self.metadata[idx].get("id")
                            if idx < len(self.metadata)
                            else f"doc-{idx}"
                        ),
                        "texto": texto,
                        "confianza": confianza,
                        "tipo": self._clasificar_tipo_contenido(texto),
                        "conceptos_relacionados": conceptos_clave,
                        "densidad_causal": self._calcular_densidad_causal_avanzada(
                            texto
                        ),
                    }
                )
        elif self.vectorizer is not None:
            matriz = self.vectorizer.transform(self.corpus)
            query_vector = self.vectorizer.transform([query])
            similitudes = cosine_similarity(query_vector, matriz)[0]
            indices = np.argsort(similitudes)[::-1][:top_k]
            for idx in indices:
                resultados.append(
                    {
                        "id": (
                            self.metadata[idx].get("id")
                            if idx < len(self.metadata)
                            else f"doc-{idx}"
                        ),
                        "texto": self.corpus[idx],
                        "confianza": float(np.clip(similitudes[idx], 0.0, 1.0)),
                        "tipo": self._clasificar_tipo_contenido(self.corpus[idx]),
                        "conceptos_relacionados": conceptos_clave,
                        "densidad_causal": self._calcular_densidad_causal_avanzada(
                            self.corpus[idx]
                        ),
                    }
                )
        else:
            resultados = [
                {
                    "id": f"doc-{idx}",
                    "texto": texto,
                    "confianza": 0.4,
                    "tipo": self._clasificar_tipo_contenido(texto),
                    "conceptos_relacionados": conceptos_clave,
                    "densidad_causal": self._calcular_densidad_causal_avanzada(texto),
                }
                for idx, texto in enumerate(self.corpus[:top_k])
            ]
        return resultados

    def _calcular_relevancia_conceptual_avanzada(
        self, query: str, documento: str
    ) -> float:
        if self.vectorizer is None:
            return 0.0
        vectores = self.vectorizer.transform([query, documento])
        return float(cosine_similarity(vectores[0], vectores[1])[0][0])

    def _calcular_puntaje_innovacion(self, evidencia: Dict[str, Any]) -> float:
        densidad = evidencia.get("densidad_causal", 0.0)
        confianza = evidencia.get("confianza", 0.0)
        tipo = evidencia.get("tipo", "general")
        puntaje = densidad * 0.5 + confianza * 0.4
        if tipo == "analítico":
            puntaje += 0.1
        return float(np.clip(puntaje, 0.0, 1.0))


@dataclass
class ResultadoDimensionIndustrialAvanzado:
    """Resultado avanzado de evaluación dimensional con capacidades de frontera."""

    dimension: DimensionDecalogoAvanzada
    evaluacion_causal: EvaluacionCausalIndustrialAvanzada
    evidencia: Dict[str, List[Dict[str, Any]]]
    brechas_identificadas: List[str]
    recomendaciones: List[str]
    matriz_trazabilidad: Optional[pd.DataFrame] = None
    analisis_interdependencias: Dict[str, Any] = field(default_factory=dict)
    proyeccion_impacto: Dict[str, Any] = field(default_factory=dict)
    analisis_costo_beneficio: Dict[str, Any] = field(default_factory=dict)
    timestamp_evaluacion: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def puntaje_final_avanzado(self) -> float:
        return self.evaluacion_causal.puntaje_final_avanzado

    @property
    def nivel_madurez_avanzado(self) -> str:
        return self.evaluacion_causal.nivel_madurez_avanzado()

    def generar_reporte_tecnico_avanzado(self) -> Dict[str, Any]:
        matriz = self.evaluacion_causal.generar_matriz_decision_multicriterio()
        riesgos = self.generar_matriz_riesgos_avanzada()
        plan = self._generar_plan_implementacion()
        impacto = self._proyectar_impacto_temporal()
        costo = self._analizar_costo_efectividad()
        trazabilidad = self._generar_resumen_trazabilidad()
        return {
            "dimension": self.dimension.nombre,
            "puntaje_final": self.puntaje_final_avanzado,
            "nivel_madurez": self.nivel_madurez_avanzado,
            "matriz_decision": matriz,
            "matriz_riesgos": riesgos,
            "plan_implementacion": plan,
            "proyeccion_impacto": impacto,
            "analisis_costo_beneficio": costo,
            "trazabilidad": trazabilidad,
            "recomendaciones": self._priorizar_recomendaciones(),
        }

    def _proyectar_impacto_temporal(self) -> Dict[str, Any]:
        factibilidad = self.evaluacion_causal.factibilidad_operativa
        sostenibilidad = self.evaluacion_causal.sostenibilidad_resultados
        escalabilidad = self.evaluacion_causal.escalabilidad_territorial
        anos = list(range(1, 6))
        impactos = []
        for ano in anos:
            if ano <= 2:
                impacto_ano = factibilidad * (0.3 + 0.2 * (ano - 1))
            elif ano <= 4:
                impacto_ano = factibilidad * (0.5 + 0.25 * (ano - 2))
            else:
                impacto_ano = factibilidad * sostenibilidad * escalabilidad
            factor_riesgo = max(
                0.7, 1.0 -
                len(self.evaluacion_causal.riesgos_implementacion) * 0.05
            )
            impacto_ajustado = impacto_ano * factor_riesgo
            impactos.append(
                {
                    "anio": ano,
                    "impacto_base": impacto_ano,
                    "impacto_ajustado": impacto_ajustado,
                    "factor_riesgo": factor_riesgo,
                }
            )
        tasa_descuento = 0.08
        vpn = sum(
            imp["impacto_ajustado"] / ((1 + tasa_descuento) ** imp["anio"])
            for imp in impactos
        )
        self.proyeccion_impacto = {
            "impactos_por_anio": impactos,
            "valor_presente_neto": vpn,
            "impacto_acumulado_5_anos": sum(
                imp["impacto_ajustado"] for imp in impactos
            ),
            "punto_equilibrio_estimado": self._estimar_punto_equilibrio(impactos),
            "tendencia_impacto": (
                "creciente" if vpn > 2.0 else "estable" if vpn > 1.0 else "decreciente"
            ),
        }
        return self.proyeccion_impacto

    def _estimar_punto_equilibrio(self, impactos: List[Dict[str, Any]]) -> int:
        for impacto in impactos:
            if impacto["impacto_ajustado"] >= 0.6:
                return impacto["anio"]
        return impactos[-1]["anio"] if impactos else 5

    def _analizar_costo_efectividad(self) -> Dict[str, Any]:
        num_eslabones = len(self.dimension.eslabones)
        complejidad_promedio = np.mean(
            [
                es.calcular_metricas_avanzadas()["complejidad_operativa"]
                for es in self.dimension.eslabones
            ]
        )
        costo_base_eslabon = 200.0
        factor_complejidad = 1.0 + complejidad_promedio
        factor_prioridad = self.dimension.prioridad_estrategica
        costo_estimado = (
            num_eslabones * costo_base_eslabon * factor_complejidad * factor_prioridad
        )
        beneficio_estimado = (
            self.evaluacion_causal.puntaje_global_avanzado
            * self.evaluacion_causal.escalabilidad_territorial
            * 1000
        )
        if costo_estimado > 0:
            ratio_ce = beneficio_estimado / costo_estimado
        else:
            ratio_ce = 0.0
        if ratio_ce >= 3.0:
            clasificacion = "MUY ALTA"
        elif ratio_ce >= 2.0:
            clasificacion = "ALTA"
        elif ratio_ce >= 1.0:
            clasificacion = "MODERADA"
        else:
            clasificacion = "BAJA"
        self.analisis_costo_beneficio = {
            "costo_estimado_millones": costo_estimado,
            "beneficio_estimado": beneficio_estimado,
            "ratio_costo_efectividad": ratio_ce,
            "clasificacion": clasificacion,
        }
        return self.analisis_costo_beneficio

    def _generar_recomendacion_inversion(self) -> str:
        ratio = self.analisis_costo_beneficio.get(
            "ratio_costo_efectividad", 0.0)
        if ratio >= 2.5:
            return "Priorizar inversión estratégica y acelerar escalamiento."
        if ratio >= 1.5:
            return "Inversión moderada condicionada a cierre de brechas críticas."
        if ratio >= 0.8:
            return "Aplicar inversión incremental con monitoreo estricto."
        return "Reconsiderar inversión hasta fortalecer fundamentos."

    def _calcular_roi_estimado(self) -> Dict[str, float]:
        inversion = self.analisis_costo_beneficio.get(
            "costo_estimado_millones", 0.0)
        beneficio = self.analisis_costo_beneficio.get(
            "beneficio_estimado", 0.0)
        if inversion <= 0:
            return {"roi_innovacion_estimado": 0.0, "roi_promedio_estimado": 0.0}
        roi_promedio = beneficio / inversion
        roi_innovacion = roi_promedio * (
            1 + self.evaluacion_causal.innovacion_metodologica * 0.3
        )
        return {
            "roi_innovacion_estimado": roi_innovacion,
            "roi_promedio_estimado": roi_promedio,
            "periodo_recuperacion_anios": max(1, int(5 / max(0.1, roi_promedio / 50))),
        }

    def _evaluar_eficiencia_recursos(self) -> float:
        recursos = self.evidencia.get("recursos", [])
        cuantificados = sum(
            1 for r in recursos if r.get("tiene_cuantificacion"))
        precision = cuantificados / len(recursos) if recursos else 0.3
        eficiencia = (
            self.evaluacion_causal.factibilidad_operativa * 0.4
            + precision * 0.3
            + min(1.0, self.evaluacion_causal.evidencia_soporte / 10) * 0.3
        )
        return float(np.clip(eficiencia, 0.0, 1.0))

    def _priorizar_recomendaciones(self) -> List[Dict[str, Any]]:
        recomendaciones = []
        for idx, recomendacion in enumerate(self.recomendaciones[:10]):
            texto = recomendacion.lower()
            if any(t in texto for t in ["crítico", "urgente", "fundamental"]):
                impacto = 0.9
            elif any(t in texto for t in ["importante", "prioritario", "necesario"]):
                impacto = 0.7
            else:
                impacto = 0.5
            if any(t in texto for t in ["simple", "inmediato", "rápido"]):
                factibilidad = 0.8
            elif any(t in texto for t in ["mediano", "gradual"]):
                factibilidad = 0.6
            else:
                factibilidad = 0.4
            score = impacto * 0.6 + factibilidad * 0.4
            if score >= 0.8:
                prioridad = "CRÍTICA"
            elif score >= 0.65:
                prioridad = "ALTA"
            elif score >= 0.5:
                prioridad = "MEDIA"
            else:
                prioridad = "BAJA"
            recomendaciones.append(
                {
                    "recomendacion": recomendacion,
                    "impacto_estimado": impacto,
                    "factibilidad_estimada": factibilidad,
                    "score_priorizacion": score,
                    "prioridad": prioridad,
                    "orden_original": idx + 1,
                }
            )
        return sorted(
            recomendaciones, key=lambda x: x["score_priorizacion"], reverse=True
        )

    def _generar_plan_implementacion(self) -> Dict[str, Any]:
        trl_info = self.evaluacion_causal.calcular_indice_madurez_tecnologica()
        trl = trl_info.get("trl_nivel", 3)
        fases: List[Dict[str, Any]] = []
        if trl <= 4:
            fases.append(
                {
                    "nombre": "Fase 1: Fortalecimiento Conceptual",
                    "duracion_meses": 6,
                    "objetivos": [
                        "Consolidar marco teórico y evidencia base",
                        "Desarrollar capacidades técnicas básicas",
                        "Establecer líneas base de indicadores",
                    ],
                    "entregables": [
                        "Marco conceptual validado",
                        "Capacidades básicas instaladas",
                        "Línea base establecida",
                    ],
                    "recursos_estimados": "20% del presupuesto total",
                }
            )
        if trl <= 6:
            fases.append(
                {
                    "nombre": "Fase 2: Implementación Piloto",
                    "duracion_meses": 12,
                    "objetivos": [
                        "Implementar intervenciones piloto",
                        "Validar modelo en entorno controlado",
                        "Ajustar diseño basado en aprendizajes",
                    ],
                    "entregables": [
                        "Piloto implementado",
                        "Modelo validado",
                        "Ajustes incorporados",
                    ],
                    "recursos_estimados": "40% del presupuesto total",
                }
            )
        if trl <= 8:
            fases.append(
                {
                    "nombre": "Fase 3: Escalamiento Gradual",
                    "duracion_meses": 18,
                    "objetivos": [
                        "Escalar intervención a nivel territorial",
                        "Consolidar sistema de monitoreo",
                        "Desarrollar capacidades de sostenibilidad",
                    ],
                    "entregables": [
                        "Sistema escalado",
                        "Monitoreo operativo",
                        "Sostenibilidad asegurada",
                    ],
                    "recursos_estimados": "40% del presupuesto total",
                }
            )
        hitos = [
            {"hito": "Aprobación marco conceptual", "mes": 3},
            {"hito": "Inicio piloto", "mes": 9},
            {"hito": "Evaluación intermedia", "mes": 15},
            {"hito": "Decisión escalamiento", "mes": 21},
            {"hito": "Sistema completamente operativo", "mes": 30},
        ]
        riesgos_por_fase = {
            "Fase 1": ["Demoras en aprobaciones", "Limitaciones de capacidad técnica"],
            "Fase 2": ["Resistencia al cambio", "Problemas de coordinación"],
            "Fase 3": ["Limitaciones presupuestales", "Cambios en contexto político"],
        }
        return {
            "fases_implementacion": fases,
            "duracion_total_meses": sum(f.get("duracion_meses", 0) for f in fases),
            "hitos_criticos": hitos,
            "riesgos_por_fase": riesgos_por_fase,
            "factores_exito_clave": [
                "Liderazgo institucional sólido",
                "Participación activa de stakeholders",
                "Sistema de monitoreo robusto",
                "Flexibilidad adaptativa",
                "Comunicación efectiva",
            ],
            "recomendacion_inicio": self._recomendar_momento_inicio(),
        }

    def _recomendar_momento_inicio(self) -> str:
        puntaje = self.puntaje_final_avanzado
        if puntaje >= 85:
            return "Iniciar inmediatamente con fase de escalamiento controlado."
        if puntaje >= 70:
            return "Iniciar en próximos 6 meses tras fortalecer capacidades clave."
        if puntaje >= 55:
            return "Planificar inicio en 12 meses priorizando cierre de brechas."
        return "Postergar inicio hasta reforzar evidencia y capacidades críticas."

    def _generar_indicadores_monitoreo_avanzados(self) -> List[Dict[str, Any]]:
        indicadores: List[Dict[str, Any]] = []
        for eslabon in self.dimension.eslabones:
            metricas = eslabon.calcular_metricas_avanzadas()
            indicadores.append(
                {
                    "eslabon": eslabon.nombre,
                    "indicador": f"Progreso {eslabon.nombre.lower()}",
                    "linea_base": metricas.get("complejidad_operativa", 0.5),
                    "meta": min(1.0, metricas.get("complejidad_operativa", 0.5) + 0.2),
                    "frecuencia": "trimestral",
                }
            )
        return indicadores

    def _generar_resumen_trazabilidad(self) -> Dict[str, Any]:
        if self.matriz_trazabilidad is None or self.matriz_trazabilidad.empty:
            self.matriz_trazabilidad = (
                self.evaluacion_causal.generar_matriz_trazabilidad_avanzada()
            )
        conteo_por_categoria = (
            self.matriz_trazabilidad.groupby("categoria").size().to_dict()
        )
        confianza_promedio = (
            self.matriz_trazabilidad["confianza"].mean()
            if not self.matriz_trazabilidad.empty
            else 0.5
        )
        recomendaciones = self._generar_recomendaciones_trazabilidad(
            conteo_por_categoria, confianza_promedio
        )
        return {
            "conteo_por_categoria": conteo_por_categoria,
            "confianza_promedio": confianza_promedio,
            "recomendaciones": recomendaciones,
        }

    def _generar_recomendaciones_trazabilidad(
        self, conteo: Dict[str, int], confianza: float
    ) -> List[str]:
        recomendaciones = []
        categorias_requeridas = {
            "cuantitativa",
            "cualitativa",
            "documental",
            "testimonial",
        }
        for categoria in categorias_requeridas:
            if conteo.get(categoria, 0) == 0:
                recomendaciones.append(
                    f"Incorporar evidencia {categoria} para balancear trazabilidad."
                )
        if confianza < 0.6:
            recomendaciones.append(
                "Fortalecer verificación de fuentes para aumentar confiabilidad."
            )
        if self._evaluar_coherencia_temporal() < 0.5:
            recomendaciones.append(
                "Actualizar evidencia reciente para mejorar coherencia temporal."
            )
        return recomendaciones or ["Sistema de trazabilidad equilibrado y robusto."]

    def _calcular_puntaje_innovacion(self) -> float:
        extractor = self.evaluacion_causal.extractor_evidencia
        if extractor is None:
            return 0.0
        evidencias = [
            evidencia for lista in self.evidencia.values() for evidencia in lista
        ]
        if not evidencias:
            return 0.0
        puntajes = [extractor._calcular_puntaje_innovacion(
            ev) for ev in evidencias]
        return float(np.mean(puntajes))

    def buscar_evidencia_causal_avanzada(
        self,
        query: str,
        conceptos_clave: List[str],
        top_k: int = 5,
        umbral_certeza: float = 0.7,
    ) -> List[Dict[str, Any]]:
        return self.evaluacion_causal.buscar_evidencia_causal_avanzadas(
            query, conceptos_clave, top_k, umbral_certeza
        )

    def cargar_decalogo_industrial_avanzado(
        self, path: Optional[Union[str, Path]] = None
    ) -> DecalogoContextoAvanzado:
        return DecalogoContextoAvanzado.cargar_decalogo_industrial_avanzado(path)

    def calcular_metricas_cluster(self, grafo: nx.Graph) -> Dict[str, Any]:
        cluster = ClusterMetadataAvanzada(
            cluster_id=self.dimension.id,
            nombre=self.dimension.nombre,
            palabras_clave=[
                cap for es in self.dimension.eslabones for cap in es.capacidades_clave
            ],
            vector_representativo=np.random.default_rng(
                self.dimension.id).random(3),
            miembros=[es.nombre for es in self.dimension.eslabones],
        )
        metricas = cluster.calcular_metricas_cluster()
        metricas["modularidad"] = ClusterMetadataAvanzada._calcular_modularidad(
            grafo)
        return metricas

    def calcular_interdependencias_avanzadas(self, grafo: nx.Graph) -> Dict[int, float]:
        cluster = ClusterMetadataAvanzada(
            cluster_id=self.dimension.id,
            nombre=self.dimension.nombre,
            palabras_clave=[
                cap for es in self.dimension.eslabones for cap in es.capacidades_clave
            ],
            vector_representativo=np.ones(3),
            miembros=[es.nombre for es in self.dimension.eslabones],
        )
        self.analisis_interdependencias = cluster.calcular_interdependencias_avanzadas(
            grafo
        )
        return self.analisis_interdependencias

    def _calcular_modularidad(self, grafo: nx.Graph) -> float:
        return ClusterMetadataAvanzada._calcular_modularidad(grafo)

    def evaluar_coherencia_causal_avanzada(self) -> Dict[str, Any]:
        temporal = self._evaluar_coherencia_temporal()
        circular = self._evaluar_dependencias_circulares()
        clasificacion = self._clasificar_coherencia(temporal, circular)
        return {
            "coherencia_temporal": temporal,
            "dependencias_circulares": circular,
            "clasificacion": clasificacion,
        }

    def _evaluar_coherencia_temporal(self) -> float:
        return self.evaluacion_causal._evaluar_coherencia_temporal_evidencia()

    def _evaluar_dependencias_circulares(self) -> float:
        total = len(self.evaluacion_causal.interdependencias_detectadas)
        if total == 0:
            return 0.1
        ciclos = sum(
            1
            for inter in self.evaluacion_causal.interdependencias_detectadas
            if "ciclo" in inter.lower()
        )
        return float(np.clip(ciclos / total, 0.0, 1.0))

    def _clasificar_coherencia(self, temporal: float, circular: float) -> str:
        if temporal >= 0.7 and circular <= 0.2:
            return "ALTA"
        if temporal >= 0.5 and circular <= 0.4:
            return "MEDIA"
        if temporal >= 0.3 and circular <= 0.6:
            return "BAJA"
        return "CRÍTICA"

    def calcular_kpi_global_avanzado(self) -> Dict[str, Any]:
        puntaje_innovacion = self._calcular_puntaje_innovacion()
        eficiencia = self._evaluar_eficiencia_recursos()
        coherencia = self.evaluar_coherencia_causal_avanzada()
        return {
            "puntaje_dimension": self.puntaje_final_avanzado,
            "innovacion": puntaje_innovacion,
            "eficiencia_recursos": eficiencia,
            "coherencia_causal": coherencia,
        }

    def generar_matriz_riesgos_avanzada(self) -> pd.DataFrame:
        matriz = self.evaluacion_causal._analizar_riesgos_probabilistico()
        if not matriz.empty:
            matriz["medidas_mitigacion"] = self._generar_medidas_mitigacion(
                matriz)
        return matriz

    def _generar_medidas_mitigacion(self, matriz_riesgo: pd.DataFrame) -> List[str]:
        estrategias = self.evaluacion_causal._generar_estrategia_mitigacion(
            matriz_riesgo
        )
        return estrategias

    def _generar_indicadores_monitoreo(self) -> List[Dict[str, Any]]:
        return self._generar_indicadores_monitoreo_avanzados()
