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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn.functional as F

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
    from scipy.spatial.distance import cdist
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.stats import entropy, chi2_contingency, pearsonr, spearmanr
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False

# Capacidades de frontera en NLP
try:
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModel
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
    from device_config import add_device_args, configure_device_from_args, get_device_config, to_device
except ImportError:
    def add_device_args(parser):
        parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
        parser.add_argument('--precision', default='float32', choices=['float16', 'float32'], help='Precision')
        parser.add_argument('--batch_size', default=16, type=int, help='Batch size for processing')
        return parser

    def configure_device_from_args(args):
        return AdvancedDeviceConfig(
            args.device if hasattr(args, 'device') else 'cpu',
            args.precision if hasattr(args, 'precision') else 'float32',
            args.batch_size if hasattr(args, 'batch_size') else 16
        )

    def get_device_config():
        return AdvancedDeviceConfig('cpu', 'float32', 16)

    def to_device(model):
        return model

    class AdvancedDeviceConfig:
        def __init__(self, device='cpu', precision='float32', batch_size=16):
            self.device = device
            self.precision = precision
            self.batch_size = batch_size
        
        def get_device(self):
            return self.device
        
        def get_precision(self):
            return torch.float16 if self.precision == 'float16' else torch.float32
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_device_info(self):
            return {
                'device_type': self.device,
                'precision': self.precision,
                'batch_size': self.batch_size,
                'num_threads': torch.get_num_threads(),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'memory_info': self._get_memory_info()
            }
        
        def _get_memory_info(self):
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated() / 1024**3,
                    'reserved': torch.cuda.memory_reserved() / 1024**3,
                    'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
                }
            return {'cpu_memory': 'N/A'}

# Text processing avanzado
try:
    from text_truncation_logger import (
        get_truncation_logger, log_debug_with_text, log_error_with_text,
        log_info_with_text, log_warning_with_text, truncate_text_for_log,
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
    log_info_with_text(LOGGER, "✅ Modelo SpaCy avanzado cargado (es_core_news_lg)")
except OSError:
    try:
        NLP = spacy.load("es_core_news_sm")
        log_warning_with_text(LOGGER, "⚠️ Usando modelo SpaCy básico (es_core_news_sm)")
    except OSError as e:
        log_error_with_text(LOGGER, f"❌ Error cargando SpaCy: {e}")
        raise SystemExit("Modelo SpaCy no disponible. Ejecute: python -m spacy download es_core_news_lg")

try:
    EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    EMBEDDING_MODEL = to_device(EMBEDDING_MODEL)
    log_info_with_text(LOGGER, "✅ Modelo de embeddings multilingual cargado")
    log_info_with_text(LOGGER, f"✅ Dispositivo: {get_device_config().get_device()}")
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
            return_all_scores=True
        )
        log_info_with_text(LOGGER, "✅ Pipeline NLP avanzado cargado para análisis de sentimientos")
    except Exception as e:
        log_warning_with_text(LOGGER, f"⚠️ Pipeline NLP avanzado no disponible: {e}")

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
                    edge_weight = graph.get_edge_data(path[i], path[i+1], {}).get('weight', 0.5)
                    path_strength *= edge_weight
                
                # Penalización por longitud de camino
                length_penalty = 0.8 ** (len(path) - 2)
                total_strength += path_strength * length_penalty
            
            # Normalización basada en la centralidad de los nodos
            source_centrality = nx.betweenness_centrality(graph).get(source, 0.1)
            target_centrality = nx.betweenness_centrality(graph).get(target, 0.1)
            centrality_factor = (source_centrality + target_centrality) / 2
            
            return min(1.0, total_strength * (1 + centrality_factor))
        
        except Exception:
            return 0.3
    
    @staticmethod
    def bayesian_evidence_integration(evidences: List[float], priors: List[float]) -> float:
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
                denominator = likelihood * prior + (1 - likelihood) * (1 - prior)
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
    def fuzzy_logic_aggregation(values: List[float], weights: List[float] = None) -> Dict[str, float]:
        """Agregación difusa avanzada de valores con múltiples operadores."""
        if not values:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'fuzzy_and': 0.0, 'fuzzy_or': 0.0}
        
        values = np.array(values)
        weights = np.array(weights) if weights else np.ones(len(values))
        weights = weights / np.sum(weights)  # Normalización
        
        try:
            # Operadores difusos clásicos
            fuzzy_and = np.min(values)  # T-norma mínima
            fuzzy_or = np.max(values)   # T-conorma máxima
            
            # Operadores avanzados
            weighted_mean = np.sum(values * weights)
            geometric_mean = np.exp(np.sum(weights * np.log(np.maximum(values, 1e-10))))
            harmonic_mean = 1.0 / np.sum(weights / np.maximum(values, 1e-10))
            
            # Agregación OWA (Ordered Weighted Averaging)
            sorted_values = np.sort(values)[::-1]  # Orden descendente
            owa_weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(sorted_values)]
            owa_weights = owa_weights / np.sum(owa_weights)
            owa_result = np.sum(sorted_values[:len(owa_weights)] * owa_weights)
            
            return {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(weighted_mean),
                'geometric_mean': float(geometric_mean),
                'harmonic_mean': float(harmonic_mean),
                'fuzzy_and': float(fuzzy_and),
                'fuzzy_or': float(fuzzy_or),
                'owa': float(owa_result),
                'std': float(np.std(values)),
                'entropy': MathematicalInnovations.entropy_based_complexity([str(v) for v in values])
            }
        
        except Exception:
            return {
                'min': float(np.min(values)) if len(values) > 0 else 0.0,
                'max': float(np.max(values)) if len(values) > 0 else 0.0,
                'mean': float(np.mean(values)) if len(values) > 0 else 0.0,
                'fuzzy_and': 0.0,
                'fuzzy_or': 0.0,
                'owa': 0.0,
                'std': 0.0,
                'entropy': 0.0
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
            'supuestos_suficientes': len(self.supuestos_causales) >= 2,
            'mediadores_diversificados': len(self.mediadores) >= 2,
            'resultados_especificos': len(self.resultados_intermedios) >= 1,
            'precondiciones_definidas': len(self.precondiciones) >= 1,
            'moderadores_identificados': len(self.moderadores) >= 1,
            'mecanismos_explicitos': len(self.mecanismos_causales) >= 1
        }
        
        puntajes = {k: 1.0 if v else 0.0 for k, v in criterios.items()}
        puntaje_global = np.mean(list(puntajes.values()))
        
        return {
            'puntaje_global_identificabilidad': puntaje_global,
            'criterios_individuales': puntajes,
            'nivel_identificabilidad': self._clasificar_identificabilidad(puntaje_global)
        }
    
    def _clasificar_identificabilidad(self, puntaje: float) -> str:
        if puntaje >= 0.9: return "EXCELENTE"
        if puntaje >= 0.75: return "ALTA"
        if puntaje >= 0.6: return "MEDIA"
        if puntaje >= 0.4: return "BAJA"
        return "INSUFICIENTE"
    
    def construir_grafo_causal_avanzado(self) -> nx.DiGraph:
        """Construcción de grafo causal con propiedades avanzadas."""
        G = nx.DiGraph()
        
        # Nodos básicos
        G.add_node("insumos", tipo="nodo_base", nivel="input", centralidad=1.0)
        G.add_node("impactos", tipo="nodo_base", nivel="outcome", centralidad=1.0)
        
        # Adición de nodos con atributos enriquecidos
        for categoria, lista in self.mediadores.items():
            for i, mediador in enumerate(lista):
                G.add_node(
                    mediador, 
                    tipo="mediador", 
                    categoria=categoria,
                    orden=i,
                    peso_teorico=0.8 + (i * 0.1)
                )
                G.add_edge("insumos", mediador, weight=0.9, tipo="causal_directa")
        
        # Resultados intermedios con conexiones complejas
        for i, resultado in enumerate(self.resultados_intermedios):
            G.add_node(
                resultado, 
                tipo="resultado_intermedio",
                orden=i,
                criticidad=0.7 + (i * 0.1)
            )
            
            # Conexiones desde mediadores
            mediadores_disponibles = [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]
            for mediador in mediadores_disponibles:
                G.add_edge(
                    mediador, 
                    resultado, 
                    weight=0.8 - (i * 0.1), 
                    tipo="causal_mediada"
                )
            
            # Conexión al impacto final
            G.add_edge(
                resultado, 
                "impactos", 
                weight=0.9 - (i * 0.05), 
                tipo="causal_final"
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
            return {'coeficiente_global': 0.3, 'robustez_estructural': 0.2, 'complejidad_causal': 0.1}
        
        try:
            # Métricas estructurales
            density = nx.density(G)
            avg_clustering = nx.average_clustering(G.to_undirected())
            
            # Análisis de caminos causales
            mediadores = [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]
            resultados = [n for n in G.nodes if G.nodes[n].get("tipo") == "resultado_intermedio"]
            
            # Innovación: Cálculo de fuerza causal usando la clase MathematicalInnovations
            fuerza_causal = MathematicalInnovations.calculate_causal_strength(G, "insumos", "impactos")
            
            # Robustez estructural
            robustez = self._calcular_robustez_estructural(G, mediadores, resultados)
            
            # Complejidad causal
            elementos_causales = (self.supuestos_causales + 
                                list(self.mediadores.keys()) + 
                                self.resultados_intermedios + 
                                self.moderadores)
            complejidad = MathematicalInnovations.entropy_based_complexity(elementos_causales)
            
            return {
                'coeficiente_global': fuerza_causal,
                'robustez_estructural': robustez,
                'complejidad_causal': complejidad,
                'densidad_grafo': density,
                'clustering_promedio': avg_clustering,
                'nodos_totales': len(G.nodes),
                'aristas_totales': len(G.edges)
            }
        
        except Exception as e:
            LOGGER.warning(f"Error en cálculo causal avanzado: {e}")
            return {'coeficiente_global': 0.5, 'robustez_estructural': 0.4, 'complejidad_causal': 0.3}
    
    def _calcular_robustez_estructural(self, G: nx.DiGraph, mediadores: List[str], resultados: List[str]) -> float:
        """Cálculo de robustez estructural del grafo causal."""
        try:
            # Simulación de perturbaciones
            robustez_scores = []
            
            for _ in range(100):  # 100 simulaciones
                G_perturbed = G.copy()
                
                # Remover aleatoriamente algunos nodos mediadores
                nodes_to_remove = np.random.choice(
                    mediadores, 
                    size=min(len(mediadores) // 3, 2), 
                    replace=False
                ) if len(mediadores) > 2 else []
                
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
                len(self.capacidades_requeridas) * 0.3 +
                len(self.puntos_criticos) * 0.4 +
                len(self.dependencias) * 0.3
            ) / 10.0  # Normalización
            
            # Riesgo agregado
            riesgo_agregado = min(1.0, len(self.riesgos_especificos) * 0.2)
            
            # Intensidad de recursos
            intensidad_recursos = sum(self.recursos_estimados.values()) / max(1, len(self.recursos_estimados))
            intensidad_recursos = min(1.0, intensidad_recursos / 1000000)  # Normalización por millones
            
            # Lead time normalizado
            lead_time = self.calcular_lead_time()
            lead_time_normalizado = min(1.0, lead_time / 24)  # Normalización por 24 meses
            
            # Factor de stakeholders
            factor_stakeholders = min(1.0, len(self.stakeholders) * 0.15)
            
            return {
                'complejidad_operativa': complejidad_operativa,
                'riesgo_agregado': riesgo_agregado,
                'intensidad_recursos': intensidad_recursos,
                'lead_time_normalizado': lead_time_normalizado,
                'factor_stakeholders': factor_stakeholders,
                'kpi_ponderado': self.kpi_ponderacion / 3.0,  # Normalización
                'criticidad_global': (complejidad_operativa + riesgo_agregado + lead_time_normalizado) / 3
            }
        
        except Exception:
            return {
                'complejidad_operativa': 0.5,
                'riesgo_agregado': 0.5,
                'intensidad_recursos': 0.5,
                'lead_time_normalizado': 0.5,
                'factor_stakeholders': 0.3,
                'kpi_ponderado': self.kpi_ponderacion / 3.0,
                'criticidad_global': 0.5
            }
    
    def calcular_lead_time(self) -> float:
        """Cálculo optimizado del lead time."""
        return (self.ventana_temporal[0] + self.ventana_temporal[1]) / 2.0
    
    def generar_hash_avanzado(self) -> str:
        """Generación de hash avanzado del eslabón."""
        data = (f"{self.id}|{self.tipo.value}|{sorted(self.indicadores)}|"
                f"{sorted(self.capacidades_requeridas)}|{sorted(self.riesgos_especificos)}|"
                f"{self.ventana_temporal}|{self.kpi_ponderacion}")
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
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "3.0-industrial-frontier"
    
    @classmethod
    def cargar_ontologia_avanzada(cls) -> 'OntologiaPoliticasAvanzada':
        """Carga ontología avanzada con capacidades de frontera."""
        try:
            # Dimensiones expandidas con granularidad superior
            dimensiones_frontier = {
                "social_avanzado": [
                    "salud_preventiva", "educacion_calidad", "vivienda_digna", "proteccion_social_integral",
                    "equidad_genero", "inclusion_diversidad", "cohesion_social", "capital_social",
                    "bienestar_subjetivo", "calidad_vida_urbana", "seguridad_ciudadana", "participacion_comunitaria"
                ],
                "economico_transformacional": [
                    "empleo_decente", "productividad_sectorial", "innovacion_tecnologica", "infraestructura_inteligente",
                    "competitividad_territorial", "emprendimiento_social", "economia_circular", "finanzas_sostenibles",
                    "comercio_justo", "turismo_sostenible", "agroindustria_sustentable", "servicios_avanzados"
                ],
                "ambiental_regenerativo": [
                    "sostenibilidad_integral", "biodiversidad_conservacion", "mitigacion_climatica", "adaptacion_climatica",
                    "gestion_integral_residuos", "gestion_hidrica", "energia_renovable", "movilidad_sostenible",
                    "construccion_verde", "agricultura_regenerativa", "bosques_urbanos", "economia_verde"
                ],
                "institucional_transformativo": [
                    "gobernanza_multinivel", "transparencia_activa", "participacion_ciudadana", "rendicion_cuentas",
                    "eficiencia_administrativa", "innovacion_publica", "gobierno_abierto", "justicia_social",
                    "estado_derecho", "capacidades_institucionales", "coordinacion_intersectorial", "planificacion_estrategica"
                ],
                "territorial_inteligente": [
                    "ordenamiento_territorial", "planificacion_urbana", "conectividad_digital", "logistica_territorial",
                    "patrimonio_cultural", "identidad_territorial", "resiliencia_territorial", "sistemas_urbanos"
                ]
            }
            
            # Relaciones causales avanzadas con múltiples niveles
            relaciones_causales_avanzadas = {
                "inversion_publica_inteligente": [
                    "crecimiento_economico_sostenible", "empleo_formal_calidad", "infraestructura_resiliente",
                    "capacidades_institucionales", "innovacion_territorial", "equidad_espacial"
                ],
                "educacion_transformacional": [
                    "productividad_laboral_avanzada", "innovacion_social", "reduccion_desigualdades",
                    "cohesion_social", "capital_humano_especializado", "emprendimiento_innovador"
                ],
                "salud_integral": [
                    "productividad_economica", "calidad_vida_poblacional", "equidad_social_territorial",
                    "resilienza_comunitaria", "capital_social_saludable"
                ],
                "gobernanza_inteligente": [
                    "transparencia_institucional", "eficiencia_publica", "confianza_ciudadana",
                    "participacion_democratica", "legitimidad_estatal", "capacidad_adaptativa"
                ],
                "sostenibilidad_regenerativa": [
                    "resiliencia_climatica", "economia_circular_territorial", "bienestar_ecosistemico",
                    "salud_ambiental", "prosperidad_sostenible", "justicia_intergeneracional"
                ]
            }
            
            # Taxonomía de evidencia sofisticada
            taxonomia_evidencia_avanzada = {
                "cuantitativa_robusta": [
                    "estadisticas_oficiales", "encuestas_representativas", "censos_poblacionales",
                    "registros_administrativos", "indicadores_desempeño", "metricas_impacto",
                    "series_temporales", "analisis_econometricos", "evaluaciones_impacto"
                ],
                "cualitativa_profunda": [
                    "entrevistas_profundidad", "grupos_focales", "observacion_participante",
                    "etnografia_institucional", "narrativas_territoriales", "historias_vida",
                    "analisis_discurso", "mapeo_actores", "analisis_redes_sociales"
                ],
                "mixta_integrativa": [
                    "triangulacion_metodologica", "evaluacion_realista", "analisis_configuracional",
                    "metodos_participativos", "investigacion_accion", "evaluacion_desarrollo"
                ],
                "documental_normativa": [
                    "planes_desarrollo", "politicas_publicas", "normatividad_vigente",
                    "reglamentaciones_tecnicas", "lineamientos_sectoriales", "directrices_internacionales"
                ]
            }
            
            # Patrones lingüísticos avanzados para detección de evidencia
            patrones_linguisticos_especializados = {
                "indicadores_desempeño": [
                    r"\b(?:indicador|metric|medidor|parametro|kpi)\b.*\b(?:de|para|del)\b.*\b(?:desempeño|resultado|impacto|logro)\b",
                    r"\b(?:medir|evaluar|monitorear|seguir|rastrear)\b.*\b(?:progreso|avance|cumplimiento|efectividad)\b",
                    r"\b(?:linea\s+base|baseline|situacion\s+inicial|punto\s+partida)\b.*\d+",
                    r"\b(?:meta|objetivo|target|proposito)\b.*\d+.*\b(?:2024|2025|2026|2027|2028)\b"
                ],
                "recursos_financieros": [
                    r"\$\s*[\d,.]+(?: millones?| mil(?:es)?| billones?)?\b",
                    r"\bpresupuesto\b.*\$?[\d,.]+(?: millones?| mil(?:es)?| billones?)?",
                    r"\b(?:inversion|asignacion|destinacion|cofinanciacion)\b.*\$?[\d,.]+(?: millones?| mil(?:es)?)?",
                    r"\b(?:recursos|fondos|capital|financiacion)\b.*\$?[\d,.]+(?: millones?| mil(?:es)?)?",
                    r"\bCOP\s*[\d,.]+(?: millones?| mil(?:es)?| billones?)?\b"
                ],
                "responsabilidades_institucionales": [
                    r"\b(?:responsable|encargado|lidera|coordina|gestiona|ejecuta)\b:\s*\w+",
                    r"\b(?:secretaria|ministerio|departamento|entidad|institucion)\b.*\b(?:responsable|cargo|funcion)\b",
                    r"\b(?:quien|que)\b.*\b(?:lidera|coordina|ejecuta|implementa)\b",
                    r"\brol\b.*\b(?:de|del|para)\b.*\b(?:secretaria|ministerio|entidad)\b"
                ],
                "temporalidad_plazos": [
                    r"\b(?:plazo|cronograma|calendario|programacion|tiempo)\b.*\b(?:de|para|del)\b.*\b(?:implementacion|ejecucion|desarrollo)\b",
                    r"\b(?:inicio|comienzo|arranque)\b.*\b(?:en|el|durante)\b.*\b(?:20\d{2}|primer|segundo|tercer|cuarto)\b.*\b(?:trimestre|semestre|año)\b",
                    r"\b(?:duracion|periodo|etapa|fase)\b.*\b(?:de|del)\b.*\b(?:\d+)\b.*\b(?:meses|años|trimestres)\b",
                    r"\b(?:hasta|para|antes|durante)\b.*\b(?:20\d{2}|diciembre|final|culminacion)\b"
                ],
                "impactos_resultados": [
                    r"\b(?:impacto|efecto|resultado|consecuencia|cambio)\b.*\b(?:en|sobre|para)\b.*\b(?:poblacion|comunidad|territorio)\b",
                    r"\b(?:beneficio|mejora|incremento|reduccion|disminucion)\b.*\b(?:del|de la|en el|en la)\b.*\b(?:\d+%|\d+ puntos)\b",
                    r"\b(?:transformacion|cambio|modificacion)\b.*\b(?:social|economica|ambiental|institucional|territorial)\b"
                ]
            }
            
            # Vocabulario especializado expandido
            vocabulario_especializado_ampliado = {
                "planificacion_territorial": [
                    "ordenamiento_territorial", "zonificacion", "uso_suelo", "plan_ordenamiento",
                    "esquema_ordenamiento", "plan_basico_ordenamiento", "pot", "eot", "pbot",
                    "suelo_urbano", "suelo_rural", "suelo_expansion", "suelo_proteccion"
                ],
                "desarrollo_sostenible": [
                    "objetivos_desarrollo_sostenible", "ods", "agenda_2030", "sostenibilidad",
                    "desarrollo_humano", "crecimiento_verde", "economia_circular", "resilencia_climatica"
                ],
                "gobernanza_publica": [
                    "participacion_ciudadana", "transparencia", "rendicion_cuentas", "gobierno_abierto",
                    "cocreacion", "corresponsabilidad", "veeduria_ciudadana", "control_social"
                ],
                "gestion_publica": [
                    "meci", "modelo_integrado_planeacion_gestion", "sistema_gestion_calidad",
                    "plan_desarrollo_territorial", "pdt", "plan_accion", "seguimiento_evaluacion"
                ]
            }
            
            # Carga de indicadores ODS especializados
            indicadores_ods_especializados = cls._cargar_indicadores_ods_avanzados()
            
            return cls(
                dimensiones=dimensiones_frontier,
                relaciones_causales=relaciones_causales_avanzadas,
                indicadores_ods=indicadores_ods_especializados,
                taxonomia_evidencia=taxonomia_evidencia_avanzada,
                patrones_linguisticos=patrones_linguisticos_especializados,
                vocabulario_especializado=vocabulario_especializado_ampliado
            )
            
        except Exception as e:
            log_error_with_text(LOGGER, f"❌ Error cargando ontología avanzada: {e}")
            raise SystemExit("Fallo en carga de ontología avanzada")
    
    @staticmethod
    def _cargar_indicadores_ods_avanzados() -> Dict[str, List[str]]:
        """Carga indicadores ODS con granularidad avanzada."""
        indicadores_path = Path("indicadores_ods_avanzados.json")
        
        # Indicadores base expandidos y especializados
        indicadores_especializados = {
            "ods1_pobreza": [
                "tasa_pobreza_monetaria", "tasa_pobreza_extrema", "indice_pobreza_multidimensional",
                "coeficiente_gini", "proteccion_social_cobertura", "acceso_servicios_basicos",
                "vulnerabilidad_economica", "resiliencia_economica_hogares", "activos_productivos_acceso"
            ],
            "ods3_salud": [
                "mortalidad_infantil", "mortalidad_materna", "esperanza_vida_nacimiento",
                "acceso_servicios_salud", "cobertura_vacunacion", "prevalencia_enfermedades_cronicas",
                "salud_mental_indicadores", "seguridad_alimentaria", "agua_potable_saneamiento_acceso"
            ],
            "ods4_educacion": [
                "tasa_alfabetizacion", "matriucla_educacion_basica", "permanencia_educativa",
                "calidad_educativa_pruebas", "acceso_educacion_superior", "formacion_tecnica_profesional",
                "educacion_digital_competencias", "infraestructura_educativa_calidad"
            ],
            "ods5_genero": [
                "participacion_politica_mujeres", "brecha_salarial_genero", "violencia_genero_prevalencia",
                "acceso_credito_mujeres", "liderazgo_empresarial_femenino", "uso_tiempo_trabajo_cuidado",
                "educacion_ciencia_tecnologia_mujeres", "derechos_reproductivos_acceso"
            ],
            "ods8_trabajo": [
                "tasa_empleo", "tasa_desempleo", "empleo_informal", "trabajo_decente_indicadores",
                "productividad_laboral", "crecimiento_economico_pib", "diversificacion_economica",
                "emprendimiento_formal", "inclusion_financiera", "innovacion_empresarial"
            ],
            "ods11_ciudades": [
                "vivienda_adecuada_acceso", "transporte_publico_acceso", "espacios_publicos_calidad",
                "gestion_residuos_solidos", "calidad_aire", "planificacion_urbana_participativa",
                "patrimonio_cultural_proteccion", "resiliencia_desastres", "conectividad_urbana"
            ],
            "ods13_clima": [
                "emisiones_gei_per_capita", "vulnerabilidad_climatica", "adaptacion_climatica_medidas",
                "educacion_ambiental", "energia_renovable_uso", "eficiencia_energetica",
                "conservacion_ecosistemas", "reforestacion_restauracion", "economia_baja_carbono"
            ],
            "ods16_paz": [
                "indice_transparencia", "percepcion_corrupcion", "acceso_justicia",
                "participacion_decisiones_publicas", "libertad_expresion", "seguridad_ciudadana",
                "confianza_instituciones", "estado_derecho_fortalecimiento", "inclusion_social_politica"
            ],
            "ods17_alianzas": [
                "cooperacion_internacional", "transferencia_tecnologia", "capacitacion_institucional",
                "movilizacion_recursos_domesticos", "comercio_internacional", "acceso_mercados",
                "sostenibilidad_deuda", "sistemas_monitoreo_datos", "alianzas_publico_privadas"
            ]
        }
        
        # Intentar cargar desde archivo si existe
        if indicadores_path.exists():
            try:
                with open(indicadores_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and len(data) >= 8:
                    LOGGER.info("✅ Indicadores ODS avanzados cargados desde archivo")
                    return data
                else:
                    LOGGER.warning("⚠️ Indicadores ODS avanzados inválidos, usando base especializada")
            except Exception as e:
                LOGGER.warning(f"⚠️ Error leyendo indicadores avanzados {indicadores_path}: {e}")
        
        # Guardar template avanzado
        try:
            with open(indicadores_path, "w", encoding="utf-8") as f:
                json.dump(indicadores_especializados, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"✅ Template ODS avanzado generado: {indicadores_path}")
        except Exception as e:
            LOGGER.error(f"❌ Error generando template ODS avanzado: {e}")
        
        return indicadores_especializados
    
    def buscar_patrones_avanzados(self, texto: str, categoria: str) -> List[Dict[str, Any]]:
        """Búsqueda avanzada de patrones lingüísticos en texto."""
        if categoria not in self.patrones_linguisticos:
            return []
        
        patrones = self.patrones_linguisticos[categoria]
        resultados = []
        
        for i, patron in enumerate(patrones):
            try:
                matches = re.finditer(patron, texto, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    resultado = {
                        'texto_encontrado': match.group(),
                        'posicion_inicio': match.start(),
                        'posicion_fin': match.end(),
                        'patron_id': i,
                        'categoria': categoria,
                        'confianza': self._calcular_confianza_patron(match.group(), patron),
                        'contexto': texto[max(0, match.start()-50):match.end()+50]
                    }
                    resultados.append(resultado)
            except re.error:
                continue
        
        return sorted(resultados, key=lambda x: x['confianza'], reverse=True)
    
    def _calcular_confianza_patron(self, texto_match: str, patron: str) -> float:
        """Calcula confianza del patrón encontrado."""
        try:
            # Factores de confianza
            longitud_factor = min(1.0, len(texto_match) / 50)  # Textos más largos = mayor confianza
            complejidad_patron = min(1.0, len(patron) / 100)  # Patrones más complejos = mayor precisión
            
            # Verificar presencia de números (para indicadores cuantitativos)
            tiene_numeros = bool(re.search(r'\d+', texto_match))
            factor_numerico = 0.2 if tiene_numeros else 0.0
            
            # Verificar presencia de fechas
            tiene_fechas = bool(re.search(r'20\d{2}', texto_match))
            factor_temporal = 0.15 if tiene_fechas else 0.0
            
            # Confianza base
            confianza_base = 0.6
            
            return min(1.0, confianza_base + longitud_factor * 0.2 + complejidad_patron * 0.1 + 
                      factor_numerico + factor_temporal)
        
        except Exception:
            return 0.5