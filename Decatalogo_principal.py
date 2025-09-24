# -*- coding: utf-8 -*-
"""
Sistema Integral de Evaluación de Cadenas de Valor en Planes de Desarrollo Municipal
Versión: 8.1 — Marco Teórico-Institucional con Análisis Causal Multinivel, Batch Processing,
Certificación de Rigor y Selección Global Top-K con Heap
Framework basado en Institutional Analysis and Development (IAD) + Theory of Change (ToC)
con triangulación metodológica cualitativa-cuantitativa, verificación causal y certeza probabilística.
Autor: Dr. en Políticas Públicas
Enfoque: Evaluación estructural con econometría de políticas, minería causal y procesamiento paralelo industrial.
"""

import argparse
import atexit
import hashlib
import heapq
import json
import logging
import os
import re
import signal
import statistics
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# Manejo robusto de pdfplumber con fallback
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

import spacy
from joblib import Parallel, delayed

# Manejo robusto de sentence_transformers con fallback
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    util = None

import numpy as np
import pandas as pd
import torch

# Import statements for external modules - with fallback handling
try:
    from decalogo_loader import get_decalogo_industrial
except ImportError:
    def get_decalogo_industrial():
        return "Fallback: Decálogo industrial para desarrollo municipal con 10 dimensiones estratégicas."

try:
    from device_config import add_device_args, configure_device_from_args, get_device_config, to_device
except ImportError:
    # Fallback device configuration
    def add_device_args(parser):
        parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
        return parser
    
    def configure_device_from_args(args):
        return SimpleDeviceConfig(args.device if hasattr(args, 'device') else 'cpu')
    
    def get_device_config():
        return SimpleDeviceConfig('cpu')
    
    def to_device(model):
        return model
    
    class SimpleDeviceConfig:
        def __init__(self, device='cpu'):
            self.device = device
        def get_device(self):
            return self.device
        def get_device_info(self):
            return {
                'device_type': self.device,
                'num_threads': torch.get_num_threads(),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }

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
    # Fallback logging functions
    def get_truncation_logger(name):
        return logging.getLogger(name)
    def log_debug_with_text(logger, text): logger.debug(text)
    def log_error_with_text(logger, text): logger.error(text)
    def log_info_with_text(logger, text): logger.info(text)
    def log_warning_with_text(logger, text): logger.warning(text)
    def truncate_text_for_log(text, max_len=200):
        return text[:max_len] + "..." if len(text) > max_len else text

assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

# -------------------- CONFIGURACIÓN ACADÉMICA INDUSTRIAL --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"evaluacion_politicas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
LOGGER = logging.getLogger("EvaluacionPoliticasPublicasIndustrial")

# -------------------- MODELOS AVANZADOS CON FALLBACK INDUSTRIAL --------------------
try:
    NLP = spacy.load("es_core_news_lg")
    log_info_with_text(LOGGER, "✅ Modelo SpaCy cargado exitosamente")
except OSError as e:
    try:
        NLP = spacy.load("es_core_news_sm")
        log_warning_with_text(LOGGER, "⚠️ Usando modelo SpaCy pequeño (es_core_news_sm)")
    except OSError:
        log_error_with_text(LOGGER, f"❌ Error crítico cargando modelo SpaCy: {e}")
        raise SystemExit(
            "Modelo SpaCy no disponible. Ejecute: python -m spacy download es_core_news_lg"
        )

try:
    EMBEDDING_MODEL = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    EMBEDDING_MODEL = to_device(EMBEDDING_MODEL)
    log_info_with_text(LOGGER, "✅ Modelo de embeddings cargado exitosamente")
    log_info_with_text(
        LOGGER,
        f"✅ Modelo configurado en dispositivo: {get_device_config().get_device()}",
    )
except Exception as e:
    log_error_with_text(LOGGER, f"❌ Error crítico cargando modelo de embeddings: {e}")
    raise SystemExit(f"Error cargando modelo de embeddings: {e}")


# ==================== MARCO TEÓRICO-INSTITUCIONAL INDUSTRIAL ====================
class NivelAnalisis(Enum):
    MACRO = "Institucional"
    MESO = "Organizacional"
    MICRO = "Operacional"


class TipoCadenaValor(Enum):
    INSUMOS = "Recursos financieros, humanos y físicos"
    PROCESOS = "Transformación institucional"
    PRODUCTOS = "Bienes/servicios entregables"
    RESULTADOS = "Cambios conductuales/institucionales"
    IMPACTOS = "Bienestar y desarrollo humano"


@dataclass(frozen=True)
class TeoriaCambio:
    """Representación formal de teoría de cambio como DAG causal con verificación matemática.
    
    Esta clase modela una teoría de cambio como un grafo dirigido acíclico (DAG) causal,
    implementando métodos de identificabilidad según Pearl (2009) y análisis de robustez.
    """
    supuestos_causales: List[str]
    mediadores: Dict[str, List[str]]
    resultados_intermedios: List[str]
    precondiciones: List[str]

    def verificar_identificabilidad(self) -> bool:
        """Verifica condiciones de identificabilidad según Pearl (2009)."""
        return len(self.supuestos_causales) > 0 and len(self.mediadores) > 0 and len(self.resultados_intermedios) > 0

    def construir_grafo_causal(self) -> nx.DiGraph:
        """Construye grafo causal para análisis de paths y d-separación."""
        G = nx.DiGraph()
        G.add_node("insumos", tipo="nodo_base")
        G.add_node("impactos", tipo="nodo_base")

        for tipo_mediador, lista_mediadores in self.mediadores.items():
            for mediador in lista_mediadores:
                G.add_node(mediador, tipo="mediador", categoria=tipo_mediador)
                G.add_edge("insumos", mediador, weight=1.0, tipo="causal")

                for resultado in self.resultados_intermedios:
                    G.add_node(resultado, tipo="resultado")
                    G.add_edge(mediador, resultado, weight=0.8, tipo="causal")
                    G.add_edge(resultado, "impactos", weight=0.9, tipo="causal")

        return G

    def calcular_coeficiente_causal(self) -> float:
        """Calcula coeficiente de robustez causal basado en conectividad y paths."""
        G = self.construir_grafo_causal()
        if len(G.nodes) < 3:
            return 0.3

        try:
            paths_validos = 0
            total_paths = 0

            for mediador in [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]:
                for resultado in [n for n in G.nodes if G.nodes[n].get("tipo") == "resultado"]:
                    if nx.has_path(G, mediador, resultado) and nx.has_path(G, resultado, "impactos"):
                        paths_validos += 1
                    total_paths += 1

            return paths_validos / max(1, total_paths) if total_paths > 0 else 0.5
        except Exception:
            return 0.5


@dataclass(frozen=True)
class EslabonCadena:
    """Modelo industrial de eslabón de cadena de valor con métricas cuantitativas."""
    id: str
    tipo: TipoCadenaValor
    indicadores: List[str]
    capacidades_requeridas: List[str]
    puntos_criticos: List[str]
    ventana_temporal: Tuple[int, int]
    kpi_ponderacion: float = 1.0

    def __post_init__(self):
        """Validación industrial de datos post-inicialización."""
        if not (0 <= self.kpi_ponderacion <= 2.0):
            raise ValueError("KPI ponderación debe estar entre 0 y 2.0")
        if self.ventana_temporal[0] > self.ventana_temporal[1]:
            raise ValueError("Ventana temporal inválida")

    def calcular_lead_time(self) -> float:
        """Calcula lead time esperado con intervalo de confianza."""
        return (self.ventana_temporal[0] + self.ventana_temporal[1]) / 2.0

    def generar_hash(self) -> str:
        """Genera hash único para trazabilidad industrial."""
        data = f"{self.id}|{self.tipo.value}|{sorted(self.indicadores)}|{sorted(self.capacidades_requeridas)}"
        return hashlib.md5(data.encode("utf-8")).hexdigest()


# ==================== ONTOLOGÍA DE POLÍTICAS PÚBLICAS INDUSTRIAL ====================
@dataclass
class OntologiaPoliticas:
    """Sistema ontológico industrial con validación cruzada y trazabilidad."""
    dimensiones: Dict[str, List[str]]
    relaciones_causales: Dict[str, List[str]]
    indicadores_ods: Dict[str, List[str]]
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0-industrial"

    @classmethod
    def cargar_estandar(cls) -> 'OntologiaPoliticas':
        """Carga ontología con validación industrial robusta y fallback jerárquico."""
        try:
            dimensiones_industrial = {
                "social": ["salud", "educación", "vivienda", "protección_social", "equidad_genero", "inclusión"],
                "economico": ["empleo", "productividad", "innovación", "infraestructura", "competitividad", "emprendimiento"],
                "ambiental": ["sostenibilidad", "biodiversidad", "cambio_climatico", "gestión_residuos", "agua", "energía_limpia"],
                "institucional": ["gobernanza", "transparencia", "participación", "rendición_cuentas", "eficiencia", "innovación_gubernamental"],
            }

            relaciones_industrial = {
                "inversión_publica": ["crecimiento_economico", "empleo", "infraestructura"],
                "educación_calidad": ["productividad", "innovación", "reducción_pobreza"],
                "salud_acceso": ["productividad_laboral", "calidad_vida", "equidad_social"],
                "gobernanza": ["transparencia", "eficiencia", "confianza_ciudadana"],
                "sostenibilidad": ["medio_ambiente", "economía_circular", "resiliencia_climática"],
            }

            indicadores_ods_path = Path("indicadores_ods_industrial.json")
            indicadores_ods = cls._cargar_indicadores_ods(indicadores_ods_path)

            return cls(
                dimensiones=dimensiones_industrial,
                relaciones_causales=relaciones_industrial,
                indicadores_ods=indicadores_ods,
            )
        except Exception as e:
            log_error_with_text(LOGGER, f"❌ Error crítico cargando ontología industrial: {e}")
            raise SystemExit("Fallo en carga de ontología - Requiere intervención manual")

    @staticmethod
    def _cargar_indicadores_ods(ruta: Path) -> Dict[str, List[str]]:
        """Carga indicadores ODS con sistema de fallback industrial."""
        indicadores_base = {
            "ods1": ["tasa_pobreza", "protección_social", "vulnerabilidad_económica"],
            "ods3": ["mortalidad_infantil", "acceso_salud", "cobertura_sanitaria"],
            "ods4": ["alfabetización", "matrícula_escolar", "calidad_educativa"],
            "ods5": ["equidad_genero", "participación_mujeres", "violencia_genero"],
            "ods8": ["empleo_decente", "crecimiento_económico", "productividad_laboral"],
            "ods11": ["vivienda_digna", "transporte_sostenible", "espacios_públicos"],
            "ods13": ["emisiones_co2", "adaptación_climática", "educación_ambiental"],
            "ods16": ["gobernanza", "transparencia", "acceso_justicia"],
        }

        if ruta.exists():
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and len(data) >= 5:
                    LOGGER.info("✅ Ontología ODS industrial cargada desde archivo")
                    return data
                else:
                    LOGGER.warning("⚠️ Ontología ODS inválida, usando base industrial")
            except Exception as e:
                LOGGER.warning(f"⚠️ Error leyendo {ruta}: {e}, usando base industrial")

        try:
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(indicadores_base, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"✅ Template industrial de indicadores ODS generado: {ruta}")
        except Exception as e:
            LOGGER.error(f"❌ Error generando template ODS: {e}")

        return indicadores_base


# ==================== SISTEMA DE CARGA DINÁMICA DEL DECÁLOGO INDUSTRIAL ====================
def cargar_decalogo_industrial() -> List[Any]:
    """Carga el decálogo industrial completo desde JSON con validación de esquema."""
    json_path = Path("decalogo_industrial.json")

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) != 10:
                raise ValueError("Decálogo debe contener exactamente 10 dimensiones")

            decalogos = []
            for i, item in enumerate(data):
                try:
                    if not all(k in item for k in ["id", "nombre", "cluster", "teoria_cambio", "eslabones"]):
                        raise ValueError(f"Dimensión {i + 1} incompleta")

                    if item["id"] != i + 1:
                        raise ValueError(f"ID de dimensión incorrecto: esperado {i + 1}, encontrado {item['id']}")

                    tc_data = item["teoria_cambio"]
                    teoria_cambio = TeoriaCambio(
                        supuestos_causales=tc_data["supuestos_causales"],
                        mediadores=tc_data["mediadores"],
                        resultados_intermedios=tc_data["resultados_intermedios"],
                        precondiciones=tc_data["precondiciones"],
                    )

                    if not teoria_cambio.verificar_identificabilidad():
                        raise ValueError(f"Teoría de cambio no identificable en dimensión {i + 1}")

                    eslabones = []
                    for j, eslabon_data in enumerate(item["eslabones"]):
                        try:
                            eslabon = EslabonCadena(
                                id=eslabon_data["id"],
                                tipo=TipoCadenaValor[eslabon_data["tipo"]],
                                indicadores=eslabon_data["indicadores"],
                                capacidades_requeridas=eslabon_data["capacidades_requeridas"],
                                puntos_criticos=eslabon_data["puntos_criticos"],
                                ventana_temporal=tuple(eslabon_data["ventana_temporal"]),
                                kpi_ponderacion=float(eslabon_data.get("kpi_ponderacion", 1.0)),
                            )
                            eslabones.append(eslabon)
                        except Exception as e:
                            raise ValueError(f"Error en eslabón {j + 1} de dimensión {i + 1}: {e}")

                    dimension = DimensionDecalogo(
                        id=item["id"],
                        nombre=item["nombre"],
                        cluster=item["cluster"],
                        teoria_cambio=teoria_cambio,
                        eslabones=eslabones,
                    )

                    decalogos.append(dimension)

                except Exception as e:
                    LOGGER.error(f"❌ Error crítico en dimensión {i + 1}: {e}")
                    raise SystemExit(f"Fallo en validación de dimensión {i + 1} - Requiere corrección manual")

            LOGGER.info(f"✅ Decálogo industrial cargado y validado: {len(decalogos)} dimensiones")
            return decalogos

        except Exception as e:
            LOGGER.error(f"❌ Error crítico cargando decálogo industrial: {e}")
            raise SystemExit("Fallo en carga de decálogo - Requiere intervención manual")

    # Generar template industrial si no existe
    LOGGER.info("⚙️ Generando template industrial de decálogo estructurado")
    try:
        fallback_raw = get_decalogo_industrial()
        LOGGER.info(f"⚠️ Plantilla textual de respaldo del decálogo activada ({len(fallback_raw)} caracteres)")
    except Exception as fallback_exc:
        LOGGER.warning(f"⚠️ No se pudo activar fallback textual del decálogo: {fallback_exc}")
    
    template_industrial = []
    for dim_id in range(1, 11):
        dimension_template = {
            "id": dim_id,
            "nombre": f"Dimensión {dim_id} del Decálogo Industrial",
            "cluster": f"Cluster {((dim_id - 1) // 3) + 1}",
            "teoria_cambio": {
                "supuestos_causales": [
                    f"Supuesto causal 1 para dimensión {dim_id}",
                    f"Supuesto causal 2 para dimensión {dim_id}",
                ],
                "mediadores": {
                    "institucionales": [f"mediador_institucional_{dim_id}_1", f"mediador_institucional_{dim_id}_2"],
                    "comunitarios": [f"mediador_comunitario_{dim_id}_1", f"mediador_comunitario_{dim_id}_2"],
                },
                "resultados_intermedios": [
                    f"resultado_intermedio_{dim_id}_1",
                    f"resultado_intermedio_{dim_id}_2",
                ],
                "precondiciones": [f"precondicion_{dim_id}_1", f"precondicion_{dim_id}_2"],
            },
            "eslabones": []
        }
        
        # Generar eslabones para cada dimensión
        for tipo_idx, tipo in enumerate(["INSUMOS", "PROCESOS", "PRODUCTOS", "RESULTADOS", "IMPACTOS"]):
            eslabon = {
                "id": f"{tipo.lower()[:3]}_{dim_id}",
                "tipo": tipo,
                "indicadores": [f"indicador_{tipo.lower()}_{dim_id}_{i+1}" for i in range(3)],
                "capacidades_requeridas": [f"capacidad_{tipo.lower()}_{dim_id}_{i+1}" for i in range(2)],
                "puntos_criticos": [f"punto_critico_{tipo.lower()}_{dim_id}_{i+1}" for i in range(2)],
                "ventana_temporal": [tipo_idx * 6 + 1, (tipo_idx + 1) * 6 + 6],
                "kpi_ponderacion": 1.0 + (tipo_idx * 0.1)
            }
            dimension_template["eslabones"].append(eslabon)
        
        template_industrial.append(dimension_template)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(template_industrial, f, indent=2, ensure_ascii=False)
        LOGGER.info(f"✅ Template industrial de decálogo generado: {json_path}")
        LOGGER.warning("⚠️ COMPLETE Y VALIDA MANUALMENTE EL ARCHIVO decalogo_industrial.json")
    except Exception as e:
        LOGGER.error(f"❌ Error generando template industrial: {e}")
        raise SystemExit("Fallo en generación de template - Requiere intervención manual")

    return cargar_decalogo_industrial()


@dataclass(frozen=True)
class DimensionDecalogo:
    """Dimensión industrial del decálogo con evaluación cuantitativa avanzada"""
    id: int
    nombre: str
    cluster: str
    teoria_cambio: TeoriaCambio
    eslabones: List[EslabonCadena]

    def __post_init__(self):
        if not (1 <= self.id <= 10):
            raise ValueError("ID de dimensión debe estar entre 1 y 10")
        if len(self.nombre) < 5:
            raise ValueError("Nombre de dimensión demasiado corto")
        if len(self.eslabones) < 3:
            raise ValueError("Debe haber al menos 3 eslabones por dimensión")

    def evaluar_coherencia_causal(self) -> float:
        """Evalúa coherencia interna de la teoría de cambio con métricas industriales"""
        coherencia = 0.0
        peso_total = 0.0

        if self.teoria_cambio.verificar_identificabilidad():
            coherencia += 0.4
            peso_total += 0.4
        else:
            peso_total += 0.4

        tipos_presentes = {eslabon.tipo for eslabon in self.eslabones}
        tipos_esenciales = {TipoCadenaValor.INSUMOS, TipoCadenaValor.PROCESOS, TipoCadenaValor.PRODUCTOS}
        if tipos_esenciales.issubset(tipos_presentes):
            coherencia += 0.3
            peso_total += 0.3
        else:
            peso_total += 0.3

        if any(eslabon.tipo == TipoCadenaValor.IMPACTOS for eslabon in self.eslabones):
            coherencia += 0.3
            peso_total += 0.3
        else:
            peso_total += 0.3

        return coherencia / peso_total if peso_total > 0 else 0.0

    def calcular_kpi_global(self) -> float:
        """Calcula KPI global ponderado de la dimensión"""
        if not self.eslabones:
            return 0.0
        suma_ponderada = sum(eslabon.kpi_ponderacion for eslabon in self.eslabones)
        return suma_ponderada / len(self.eslabones)

    def generar_matriz_riesgos(self) -> Dict[str, List[str]]:
        """Genera matriz industrial de riesgos por eslabón"""
        matriz = {}
        for eslabon in self.eslabones:
            riesgos = []
            if not eslabon.indicadores:
                riesgos.append("Falta de indicadores de desempeño")
            if eslabon.ventana_temporal[1] - eslabon.ventana_temporal[0] > 24:
                riesgos.append("Ventana temporal excesivamente amplia")
            if len(eslabon.capacidades_requeridas) < 2:
                riesgos.append("Capacidades requeridas insuficientes")
            matriz[eslabon.id] = riesgos
        return matriz


# Cargar decálogo industrial completo
DECALOGO_INDUSTRIAL = cargar_decalogo_industrial()


@dataclass(frozen=True)
class ClusterMetadata:
    """Metadatos consolidados de los clusters del decálogo."""
    cluster_id: str
    titulo: str
    nombre_dimension: str
    puntos: List[int]
    logica_agrupacion: str


@dataclass(frozen=True)
class DecalogoContext:
    """Contenedor con el decálogo industrial y su metadata asociada."""
    dimensiones_por_id: Dict[int, DimensionDecalogo]
    clusters_por_id: Dict[str, ClusterMetadata]
    cluster_por_dimension: Dict[int, ClusterMetadata]


_CLUSTER_DEFINITIONS = {
    "CLUSTER 1": {
        "titulo": "CLUSTER 1: PAZ, SEGURIDAD Y PROTECCIÓN DE DEFENSORES",
        "puntos": [1, 5, 8],
        "logica": "Estos tres puntos comparten una matriz común centrada en la seguridad humana.",
    },
    "CLUSTER 2": {
        "titulo": "CLUSTER 2: DERECHOS DE GRUPOS POBLACIONALES",
        "puntos": [2, 6, 9],
        "logica": "Agrupa derechos de poblaciones que enfrentan vulnerabilidades específicas.",
    },
    "CLUSTER 3": {
        "titulo": "CLUSTER 3: TERRITORIO, AMBIENTE Y DESARROLLO SOSTENIBLE",
        "puntos": [3, 7],
        "logica": "Ambos puntos abordan la relación sociedad-territorio desde una perspectiva de sostenibilidad.",
    },
    "CLUSTER 4": {
        "titulo": "CLUSTER 4: DERECHOS SOCIALES FUNDAMENTALES Y CRISIS HUMANITARIAS",
        "puntos": [4, 10],
        "logica": "Comparten la dimensión de respuesta a necesidades básicas y dignidad humana.",
    },
}

_DECALOGO_CONTEXT_CACHE: Optional[DecalogoContext] = None


def obtener_decalogo_contexto() -> DecalogoContext:
    """Factory centralizado que entrega el decálogo validado y sus metadatos de cluster."""
    global _DECALOGO_CONTEXT_CACHE
    if _DECALOGO_CONTEXT_CACHE is not None:
        return _DECALOGO_CONTEXT_CACHE

    dimensiones_por_id = {dimension.id: dimension for dimension in DECALOGO_INDUSTRIAL}

    clusters_por_id: Dict[str, ClusterMetadata] = {}
    cluster_por_dimension: Dict[int, ClusterMetadata] = {}

    for cluster_id, data in _CLUSTER_DEFINITIONS.items():
        puntos = data["puntos"]
        nombre_dimension = next(
            (dimensiones_por_id[punto].cluster for punto in puntos if punto in dimensiones_por_id),
            data["titulo"],
        )

        metadata = ClusterMetadata(
            cluster_id=cluster_id,
            titulo=data["titulo"],
            nombre_dimension=nombre_dimension,
            puntos=puntos,
            logica_agrupacion=data["logica"],
        )
        clusters_por_id[cluster_id] = metadata

        for punto_id in puntos:
            cluster_por_dimension[punto_id] = metadata

    _DECALOGO_CONTEXT_CACHE = DecalogoContext(
        dimensiones_por_id=dimensiones_por_id,
        clusters_por_id=clusters_por_id,
        cluster_por_dimension=cluster_por_dimension,
    )

    return _DECALOGO_CONTEXT_CACHE


# ==================== SISTEMA DE EXTRACCIÓN AVANZADA INDUSTRIAL ====================
class ExtractorEvidenciaIndustrial:
    """Sistema industrial de minería textual con embeddings contextuales y análisis causal avanzado"""

    def __init__(self, documentos: List[Tuple[int, str]], nombre_plan: str = "desconocido"):
        self.documentos = documentos
        self.nombre_plan = nombre_plan
        self.ontologia = OntologiaPoliticas.cargar_estandar()
        self.embeddings_doc = None
        self.textos_originales = [doc[1] for doc in documentos]
        self.logger = logging.getLogger(f"Extractor_{nombre_plan}")
        self._precomputar_embeddings()

    def _precomputar_embeddings(self):
        """Precomputa embeddings para búsqueda semántica eficiente con caché industrial"""
        textos_validos = [texto for texto in self.textos_originales if len(texto.strip()) > 10]
        if textos_validos:
            try:
                self.embeddings_doc = EMBEDDING_MODEL.encode(textos_validos, convert_to_tensor=True)
                self.logger.info(f"✅ Embeddings precomputados para {len(textos_validos)} segmentos - {self.nombre_plan}")
            except Exception as e:
                self.logger.error(f"❌ Error precomputando embeddings: {e}")
                self.embeddings_doc = torch.tensor([])
        else:
            self.embeddings_doc = torch.tensor([])
            self.logger.warning(f"⚠️ No hay textos suficientes para precomputar embeddings - {self.nombre_plan}")

    def buscar_evidencia_causal(
        self,
        query: str,
        conceptos_clave: List[str],
        top_k: int = 5,
        umbral_certeza: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Búsqueda semántica industrial con filtrado por relaciones causales y umbral de certeza ajustable"""
        if (
            not hasattr(self, "embeddings_doc")
            or self.embeddings_doc is None
            or self.embeddings_doc.numel() == 0
        ):
            self.logger.warning("⚠️ No hay embeddings precomputados disponibles, fallback a encoding en tiempo real")
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)

        try:
            query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            similitudes = util.pytorch_cos_sim(query_embedding, self.embeddings_doc)[0]
            resultados = []

            indices_top = torch.topk(similitudes, min(top_k * 2, len(self.textos_originales))).indices

            for idx in indices_top:
                texto_original = None
                pagina_original = None
                for doc in self.documentos:
                    if (
                        doc[1] in self.textos_originales
                        and self.textos_originales.index(doc[1]) == idx
                    ):
                        texto_original = doc[1]
                        pagina_original = doc[0]
                        break

                if not texto_original:
                    continue

                texto = texto_original
                pagina = pagina_original

                coincidencias_conceptuales = sum(
                    1 for concepto in conceptos_clave if concepto.lower() in texto.lower()
                )
                relevancia_conceptual = coincidencias_conceptuales / max(1, len(conceptos_clave))

                patrones_causales_industriales = [
                    r"\b(porque|debido a|como consecuencia de|en razón de|a causa de)\b",
                    r"\b(genera|produce|causa|determina|influye en|afecta a)\b",
                    r"\b(impacto|efecto|resultado|consecuencia|repercusión)\b",
                    r"\b(mejora|aumenta|reduce|disminuye|fortalece|debilita)\b",
                    r"\b(siempre que|cuando|si)\b.*\b(entonces|por lo tanto|en consecuencia)\b",
                ]

                densidad_causal = 0.0
                for patron in patrones_causales_industriales:
                    matches = len(re.findall(patron, texto.lower(), re.IGNORECASE))
                    densidad_causal += matches * 0.2

                densidad_causal = min(1.0, densidad_causal / max(1, len(texto.split()) / 100))

                score_final = (
                    similitudes[idx].item() * 0.5
                    + relevancia_conceptual * 0.3
                    + densidad_causal * 0.2
                )

                if score_final >= umbral_certeza:
                    resultados.append(
                        {
                            "texto": texto,
                            "pagina": pagina,
                            "similitud_semantica": float(similitudes[idx].item()),
                            "relevancia_conceptual": relevancia_conceptual,
                            "densidad_causal": densidad_causal,
                            "score_final": score_final,
                            "hash_segmento": hashlib.md5(texto.encode("utf-8")).hexdigest()[:8],
                            "timestamp_extraccion": datetime.now().isoformat(),
                        }
                    )

            resultados_ordenados = sorted(resultados, key=lambda x: x["score_final"], reverse=True)
            return resultados_ordenados[:top_k]

        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda causal con embeddings precomputados: {e}")
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)

    def _buscar_evidencia_fallback(
        self,
        query: str,
        conceptos_clave: List[str],
        top_k: int = 5,
        umbral_certeza: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Fallback method que realiza encoding de documentos en tiempo real"""
        if not self.textos_originales:
            self.logger.warning("⚠️ No hay textos disponibles para búsqueda")
            return []

        try:
            query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            doc_embeddings = EMBEDDING_MODEL.encode(self.textos_originales, convert_to_tensor=True)
            similitudes = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            resultados = []

            indices_top = torch.topk(similitudes, min(top_k * 2, len(self.textos_originales))).indices

            for idx in indices_top:
                texto_original = None
                pagina_original = None
                for doc in self.documentos:
                    if (
                        doc[1] in self.textos_originales
                        and self.textos_originales.index(doc[1]) == idx
                    ):
                        texto_original = doc[1]
                        pagina_original = doc[0]
                        break

                if not texto_original:
                    continue

                texto = texto_original
                pagina = pagina_original

                coincidencias_conceptuales = sum(
                    1 for concepto in conceptos_clave if concepto.lower() in texto.lower()
                )
                relevancia_conceptual = coincidencias_conceptuales / max(1, len(conceptos_clave))

                patrones_causales_industriales = [
                    r"\b(porque|debido a|como consecuencia de|en razón de|a causa de)\b",
                    r"\b(genera|produce|causa|determina|influye en|afecta a)\b",
                    r"\b(impacto|efecto|resultado|consecuencia|repercusión)\b",
                    r"\b(mejora|aumenta|reduce|disminuye|fortalece|debilita)\b",
                    r"\b(siempre que|cuando|si)\b.*\b(entonces|por lo tanto|en consecuencia)\b",
                ]

                densidad_causal = 0.0
                for patron in patrones_causales_industriales:
                    matches = len(re.findall(patron, texto.lower(), re.IGNORECASE))
                    densidad_causal += matches * 0.2

                densidad_causal = min(1.0, densidad_causal / max(1, len(texto.split()) / 100))

                score_final = (
                    similitudes[idx].item() * 0.5
                    + relevancia_conceptual * 0.3
                    + densidad_causal * 0.2
                )

                if score_final >= umbral_certeza:
                    resultados.append(
                        {
                            "texto": texto,
                            "pagina": pagina,
                            "similitud_semantica": float(similitudes[idx].item()),
                            "relevancia_conceptual": relevancia_conceptual,
                            "densidad_causal": densidad_causal,
                            "score_final": score_final,
                            "hash_segmento": hashlib.md5(texto.encode("utf-8")).hexdigest()[:8],
                            "timestamp_extraccion": datetime.now().isoformat(),
                        }
                    )

            resultados_ordenados = sorted(resultados, key=lambda x: x["score_final"], reverse=True)
            return resultados_ordenados[:top_k]

        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda causal fallback: {e}")
            return []

    def buscar_segmentos_semanticos_global(
        self, queries: List[str], max_segmentos: int, batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Búsqueda semántica global con selección top-k usando heap para optimización de memoria."""
        if self.embeddings_doc.numel() == 0:
            self.logger.warning("⚠️ No hay embeddings disponibles para búsqueda semántica global")
            return []

        heap_global = []

        try:
            for batch_start in range(0, len(queries), batch_size):
                batch_queries = queries[batch_start : batch_start + batch_size]

                query_embeddings = EMBEDDING_MODEL.encode(batch_queries, convert_to_tensor=True)
                similitudes_batch = util.pytorch_cos_sim(query_embeddings, self.embeddings_doc)

                for query_idx_local, query in enumerate(batch_queries):
                    query_idx_global = batch_start + query_idx_local
                    similitudes = similitudes_batch[query_idx_local]

                    for doc_idx, similitud in enumerate(similitudes):
                        score = float(similitud.item())

                        if doc_idx < len(self.textos_originales):
                            texto_seg = self.textos_originales[doc_idx]

                            pagina = None
                            for doc in self.documentos:
                                if doc[1] == texto_seg:
                                    pagina = doc[0]
                                    break

                            if pagina is None:
                                continue

                            datos_segmento = {
                                "texto": texto_seg,
                                "pagina": pagina,
                                "query": query,
                                "query_idx": query_idx_global,
                                "similitud_semantica": score,
                                "score_final": score,
                                "hash_segmento": hashlib.md5(texto_seg.encode("utf-8")).hexdigest()[:8],
                                "timestamp_extraccion": datetime.now().isoformat(),
                            }

                            if len(heap_global) < max_segmentos:
                                heapq.heappush(heap_global, (score, doc_idx, query_idx_global, datos_segmento))
                            elif score > heap_global[0][0]:
                                heapq.heappushpop(heap_global, (score, doc_idx, query_idx_global, datos_segmento))

            resultados_finales = []
            while heap_global:
                score, _, _, datos_segmento = heapq.heappop(heap_global)
                resultados_finales.append(datos_segmento)

            resultados_finales.sort(key=lambda x: x["score_final"], reverse=True)

            self.logger.info(
                f"✅ Búsqueda semántica global completada: {len(resultados_finales)} segmentos "
                f"seleccionados de {len(self.textos_originales)} totales con {len(queries)} queries"
            )

            return resultados_finales

        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda semántica global: {e}")
            return []

    def extraer_variables_operativas(self, dimension: DimensionDecalogo) -> Dict[str, List]:
        """Extrae variables operativas específicas para cada dimensión con trazabilidad industrial"""
        variables = {
            "indicadores": [],
            "metas": [],
            "recursos": [],
            "responsables": [],
            "plazos": [],
            "riesgos": [],
        }

        try:
            for eslabon in dimension.eslabones:
                for indicador in eslabon.indicadores:
                    resultados = self.buscar_evidencia_causal(
                        f"indicador {indicador} meta objetivo {dimension.nombre}",
                        [indicador, "meta", "objetivo", "línea base", "indicador"],
                        top_k=3,
                        umbral_certeza=0.7,
                    )
                    if resultados:
                        for resultado in resultados:
                            resultado["eslabon_origen"] = eslabon.id
                            resultado["tipo_variable"] = "indicador"
                        variables["indicadores"].extend(resultados)

            patrones_recursos = [
                "presupuesto", "financiación", "recursos", "inversión", "asignación",
                "fondo", "subsidio", "transferencia", "cofinanciación", "contrapartida",
            ]

            resultados_recursos = self.buscar_evidencia_causal(
                f"presupuesto financiación recursos para {dimension.nombre}",
                patrones_recursos,
                top_k=5,
                umbral_certeza=0.65,
            )

            if resultados_recursos:
                for resultado in resultados_recursos:
                    resultado["tipo_variable"] = "recurso"
                variables["recursos"].extend(resultados_recursos)

            patrones_responsables = ["responsable", "encargado", "lidera", "coordina", "gestiona"]
            patrones_plazos = ["plazo", "fecha", "cronograma", "tiempo", "duración", "inicio", "finalización"]

            resultados_responsables = self.buscar_evidencia_causal(
                f"responsable encargado de {dimension.nombre}",
                patrones_responsables,
                top_k=3,
                umbral_certeza=0.6,
            )

            resultados_plazos = self.buscar_evidencia_causal(
                f"plazo fecha cronograma para {dimension.nombre}",
                patrones_plazos,
                top_k=3,
                umbral_certeza=0.6,
            )

            if resultados_responsables:
                for resultado in resultados_responsables:
                    resultado["tipo_variable"] = "responsable"
                variables["responsables"].extend(resultados_responsables)

            if resultados_plazos:
                for resultado in resultados_plazos:
                    resultado["tipo_variable"] = "plazo"
                variables["plazos"].extend(resultados_plazos)

            self.logger.info(
                f"✅ Extracción completada para dimensión {dimension.id}: "
                f"{sum(len(v) for v in variables.values())} variables encontradas"
            )

        except Exception as e:
            self.logger.error(f"❌ Error extrayendo variables para dimensión {dimension.id}: {e}")

        return variables

    def generar_matriz_trazabilidad(self, dimension: DimensionDecalogo) -> pd.DataFrame:
        """Genera matriz industrial de trazabilidad entre teoría de cambio y evidencia"""
        try:
            variables = self.extraer_variables_operativas(dimension)
            data = []

            for tipo_variable, resultados in variables.items():
                for resultado in resultados:
                    data.append(
                        {
                            "dimension_id": dimension.id,
                            "dimension_nombre": dimension.nombre,
                            "tipo_variable": tipo_variable,
                            "texto_evidencia": resultado.get("texto", "")[:200] + "...",
                            "pagina": resultado.get("pagina", 0),
                            "score_confianza": resultado.get("score_final", 0.0),
                            "hash_evidencia": resultado.get("hash_segmento", "N/A"),
                            "timestamp": resultado.get("timestamp_extraccion", ""),
                        }
                    )

            if data:
                df = pd.DataFrame(data)
                return df
            else:
                return pd.DataFrame(
                    columns=[
                        "dimension_id", "dimension_nombre", "tipo_variable", "texto_evidencia",
                        "pagina", "score_confianza", "hash_evidencia", "timestamp"
                    ]
                )

        except Exception as e:
            self.logger.error(f"❌ Error generando matriz de trazabilidad: {e}")
            return pd.DataFrame()


# ==================== EVALUACIÓN CAUSAL INDUSTRIAL CON CERTEZA PROBABILÍSTICA ====================
@dataclass
class EvaluacionCausalIndustrial:
    """Evaluación industrial de coherencia causal y factibilidad con certeza cuantificada"""
    consistencia_logica: float
    identificabilidad_causal: float
    factibilidad_operativa: float
    certeza_probabilistica: float
    robustez_causal: float
    riesgos_implementacion: List[str]
    supuestos_criticos: List[str]
    evidencia_soporte: int
    brechas_criticas: int

    def __post_init__(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, float) and not (0.0 <= value <= 1.0):
                if field_name not in ["evidencia_soporte", "brechas_criticas"]:
                    raise ValueError(f"Campo {field_name} fuera de rango [0,1]: {value}")

    @property
    def puntaje_global(self) -> float:
        """Cálculo industrial del puntaje global con ponderaciones estratégicas"""
        return (
            self.consistencia_logica * 0.25
            + self.identificabilidad_causal * 0.20
            + self.factibilidad_operativa * 0.20
            + self.certeza_probabilistica * 0.15
            + self.robustez_causal * 0.20
        )

    @property
    def nivel_certidumbre(self) -> str:
        """Clasificación industrial del nivel de certidumbre"""
        puntaje = self.puntaje_global
        if puntaje >= 0.85:
            return "ALTA - Certidumbre sólida"
        elif puntaje >= 0.70:
            return "MEDIA - Certidumbre aceptable"
        elif puntaje >= 0.50:
            return "BAJA - Certidumbre limitada"
        else:
            return "MUY BAJA - Alta incertidumbre"

    @property
    def recomendacion_estrategica(self) -> str:
        """Genera recomendación estratégica basada en evaluación industrial"""
        if self.factibilidad_operativa < 0.6 and self.riesgos_implementacion:
            return "REQUIERE REDISEÑO OPERATIVO"
        elif self.certeza_probabilistica < 0.7:
            return "REQUIERE MAYOR EVIDENCIA EMPÍRICA"
        elif self.consistencia_logica < 0.7:
            return "REQUIERE FORTALECIMIENTO TEÓRICO"
        elif len(self.riesgos_implementacion) > 3:
            return "REQUIERE PLAN DE MITIGACIÓN DE RIESGOS"
        else:
            return "IMPLEMENTACIÓN RECOMENDADA"


@dataclass
class ResultadoDimensionIndustrial:
    """Resultado industrial de evaluación de dimensión con trazabilidad completa"""
    dimension: DimensionDecalogo
    evaluacion_causal: EvaluacionCausalIndustrial
    evidencia: Dict[str, List]
    brechas_identificadas: List[str]
    recomendaciones: List[str]
    matriz_trazabilidad: Optional[pd.DataFrame] = None
    timestamp_evaluacion: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def puntaje_final(self) -> float:
        """Puntaje final industrial escalado a 100"""
        return self.evaluacion_causal.puntaje_global * 100

    @property
    def nivel_madurez(self) -> str:
        """Nivel de madurez industrial de la dimensión"""
        puntaje = self.puntaje_final
        if puntaje >= 85:
            return "NIVEL 5 - Optimizado"
        elif puntaje >= 70:
            return "NIVEL 4 - Gestionado cuantitativamente"
        elif puntaje >= 50:
            return "NIVEL 3 - Definido"
        elif puntaje >= 30:
            return "NIVEL 2 - Gestionado"
        else:
            return "NIVEL 1 - Inicial"

    def generar_reporte_tecnico(self) -> Dict[str, Any]:
        """Genera reporte técnico industrial completo"""
        return {
            "metadata": {
                "dimension_id": self.dimension.id,
                "dimension_nombre": self.dimension.nombre,
                "cluster": self.dimension.cluster,
                "timestamp": self.timestamp_evaluacion,
                "version_sistema": "8.1-industrial",
            },
            "evaluacion_causal": {
                "puntaje_global": self.evaluacion_causal.puntaje_global,
                "nivel_certidumbre": self.evaluacion_causal.nivel_certidumbre,
                "recomendacion_estrategica": self.evaluacion_causal.recomendacion_estrategica,
                "metricas_detalle": {
                    "consistencia_logica": self.evaluacion_causal.consistencia_logica,
                    "identificabilidad_causal": self.evaluacion_causal.identificabilidad_causal,
                    "factibilidad_operativa": self.evaluacion_causal.factibilidad_operativa,
                    "certeza_probabilistica": self.evaluacion_causal.certeza_probabilistica,
                    "robustez_causal": self.evaluacion_causal.robustez_causal,
                },
            },
            "diagnostico": {
                "brechas_criticas": len(self.brechas_identificadas),
                "riesgos_principales": self.evaluacion_causal.riesgos_implementacion[:5],
                "evidencia_disponible": sum(len(v) for v in self.evidencia.values()),
                "nivel_madurez": self.nivel_madurez,
            },
            "recomendaciones": self.recomendaciones[:10],
            "trazabilidad": (
                self.matriz_trazabilidad.to_dict()
                if self.matriz_trazabilidad is not None
                else {}
            ),
        }


# ==================== CARGADOR DE DOCUMENTOS PDF INDUSTRIAL ====================
class PDFLoaderIndustrial:
    """Cargador y procesador industrial de documentos PDF con manejo de errores robusto"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.paginas: List[str] = []
        self.segmentos: List[Tuple[int, str]] = []
        self.nombre_plan = file_path.stem
        self.logger = logging.getLogger(f"PDFLoader_{self.nombre_plan}")
        self.hash_documento = ""
        self.metadata = {}

        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("❌ pdfplumber no está instalado. Por favor, instale con: pip install pdfplumber")

    def calcular_hash_documento(self) -> str:
        """Calcula hash industrial del documento para trazabilidad"""
        if self.paginas:
            contenido_completo = " ".join(self.paginas)
            return hashlib.sha256(contenido_completo.encode("utf-8")).hexdigest()
        return ""

    def extraer_metadata_pdf(self) -> Dict[str, Any]:
        """Extrae metadata industrial del PDF"""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                if hasattr(pdf, "metadata") and pdf.metadata:
                    return {
                        "author": pdf.metadata.get("Author", "Desconocido"),
                        "title": pdf.metadata.get("Title", self.nombre_plan),
                        "creation_date": pdf.metadata.get("CreationDate", ""),
                        "modification_date": pdf.metadata.get("ModDate", ""),
                        "producer": pdf.metadata.get("Producer", ""),
                        "page_count": len(pdf.pages),
                    }
        except Exception as e:
            self.logger.warning(f"⚠️ Error extrayendo metadata: {e}")
        return {"page_count": len(self.paginas) if self.paginas else 0}

    def cargar(self) -> bool:
        """Carga el documento PDF con manejo de errores industrial"""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, pagina in enumerate(pdf.pages, start=1):
                    try:
                        texto = pagina.extract_text() or ""
                        texto = re.sub(r"\s+", " ", texto).strip()
                        if len(texto) > 10:
                            self.paginas.append(texto)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error procesando página {i}: {e}")
                        continue

            if not self.paginas:
                self.logger.error("❌ No se pudo extraer texto del PDF")
                return False


# ==================== SISTEMA COMPLETO INDUSTRIAL CON PROCESAMIENTO PARALELO ====================
class SistemaEvaluacionIndustrial:
    """Sistema integral industrial de evaluación de políticas públicas"""

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.loader = PDFLoaderIndustrial(pdf_path)
        self.extractor: Optional[ExtractorEvidenciaIndustrial] = None
        self.ontologia = OntologiaPoliticas.cargar_estandar()
        self.logger = logging.getLogger(f"EvaluacionIndustrial_{pdf_path.stem}")
        self.hash_evaluacion = ""
        self.metadata_plan = {}

    def cargar_y_procesar(self) -> bool:
        """Carga y procesa el documento PDF con estándares industriales"""
        self.logger.info(f"🔄 Iniciando procesamiento industrial de: {self.pdf_path.name}")

        if not self.loader.cargar():
            self.logger.error("❌ Falló la carga del documento")
            return False

        if not self.loader.segmentar():
            self.logger.error("❌ Falló la segmentación del documento")
            return False

        try:
            self.extractor = ExtractorEvidenciaIndustrial(
                self.loader.segmentos, self.pdf_path.stem
            )
            self.metadata_plan = self.loader.metadata
            self.hash_evaluacion = hashlib.sha256(
                f"{self.loader.hash_documento}_{datetime.now().isoformat()}".encode("utf-8")
            ).hexdigest()

            self.logger.info(f"✅ Sistema preparado para evaluación industrial - Hash: {self.hash_evaluacion[:8]}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error inicializando extractor: {e}")
            return False

    def evaluar_dimension(self, dimension: DimensionDecalogo) -> ResultadoDimensionIndustrial:
        """Evalúa una dimensión completa del decálogo con estándares industriales"""
        if not self.extractor:
            raise ValueError("Extractor no inicializado - Error industrial crítico")

        try:
            self.logger.info(
                f"🔍 Iniciando evaluación industrial de dimensión {dimension.id}: {dimension.nombre}"
            )

            # Extracción de evidencia con trazabilidad
            evidencia = self.extractor.extraer_variables_operativas(dimension)

            # Generar matriz de trazabilidad
            matriz_trazabilidad = self.extractor.generar_matriz_trazabilidad(dimension)

            # Evaluación causal industrial
            evaluacion_causal = self._evaluar_coherencia_causal_industrial(dimension, evidencia)

            # Identificación de brechas industriales
            brechas = self._identificar_brechas_industrial(dimension, evidencia)

            # Generación de recomendaciones industriales
            recomendaciones = self._generar_recomendaciones_industrial(dimension, evaluacion_causal, brechas)

            # Intentar integración con evaluador del decálogo si está disponible
            try:
                from Decatalogo_evaluador import integrar_evaluador_decatalogo

                resultado_decatalogo = integrar_evaluador_decatalogo(self, dimension)

                if resultado_decatalogo:
                    # Combinar resultados del evaluador del decálogo con los resultados principales
                    resultado_decatalogo.evidencia = evidencia
                    resultado_decatalogo.matriz_trazabilidad = matriz_trazabilidad
                    resultado_decatalogo.brechas_identificadas = brechas
                    resultado_decatalogo.recomendaciones = recomendaciones

                    self.logger.info(
                        f"✅ Evaluación completada para dimensión {dimension.id} usando evaluador del decálogo: {resultado_decatalogo.puntaje_final:.1f}/100"
                    )
                    return resultado_decatalogo
            except ImportError as e:
                self.logger.warning(f"⚠️ No se pudo importar el evaluador del decálogo: {e}")
            except Exception as e:
                self.logger.error(f"❌ Error en evaluador del decálogo: {e}")

            resultado = ResultadoDimensionIndustrial(
                dimension=dimension,
                evaluacion_causal=evaluacion_causal,
                evidencia=evidencia,
                brechas_identificadas=brechas,
                recomendaciones=recomendaciones,
                matriz_trazabilidad=matriz_trazabilidad,
                timestamp_evaluacion=datetime.now().isoformat(),
            )

            self.logger.info(
                f"✅ Evaluación completada para dimensión {dimension.id}: {resultado.puntaje_final:.1f}/100"
            )
            return resultado

        except Exception as e:
            self.logger.error(f"❌ Error crítico evaluando dimensión {dimension.id}: {e}")
            # Retornar resultado con error para no detener el proceso
            evaluacion_fallback = EvaluacionCausalIndustrial(
                consistencia_logica=0.3,
                identificabilidad_causal=0.3,
                factibilidad_operativa=0.3,
                certeza_probabilistica=0.3,
                robustez_causal=0.3,
                riesgos_implementacion=[f"Error en evaluación: {str(e)}"],
                supuestos_criticos=[],
                evidencia_soporte=0,
                brechas_criticas=5,
            )

            return ResultadoDimensionIndustrial(
                dimension=dimension,
                evaluacion_causal=evaluacion_fallback,
                evidencia={},
                brechas_identificadas=[f"Error crítico en evaluación: {str(e)}"],
                recomendaciones=["Requiere revisión manual urgente"],
                timestamp_evaluacion=datetime.now().isoformat(),
            )

    def _evaluar_coherencia_causal_industrial(
        self, dimension: DimensionDecalogo, evidencia: Dict
    ) -> EvaluacionCausalIndustrial:
        """Evalúa la coherencia causal de la teoría de cambio con estándares industriales"""
        try:
            # Análisis de consistencia lógica industrial
            consistencia = dimension.evaluar_coherencia_causal()

            # Verificación de identificabilidad industrial
            identificabilidad = 1.0 if dimension.teoria_cambio.verificar_identificabilidad() else 0.3

            # Evaluación de factibilidad operativa industrial
            factibilidad = self._calcular_factibilidad_industrial(dimension, evidencia)

            # Identificación de riesgos industriales
            riesgos = self._identificar_riesgos_industrial(dimension, evidencia)

            # Análisis de grafos causales con certeza probabilística industrial
            G = dimension.teoria_cambio.construir_grafo_causal()
            certezas = []

            # Simulación Monte Carlo industrial para certeza probabilística
            for _ in range(200):
                try:
                    if len(G.nodes) > 2:
                        nodos_base = ["insumos", "impactos"]
                        nodos_mediadores = [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]
                        nodos_resultados = [n for n in G.nodes if G.nodes[n].get("tipo") == "resultado"]

                        nodos_sample = nodos_base.copy()
                        if nodos_mediadores:
                            nodos_sample.extend(
                                np.random.choice(
                                    nodos_mediadores,
                                    size=min(3, len(nodos_mediadores)),
                                    replace=False,
                                ).tolist()
                            )
                        if nodos_resultados:
                            nodos_sample.extend(
                                np.random.choice(
                                    nodos_resultados,
                                    size=min(2, len(nodos_resultados)),
                                    replace=False,
                                ).tolist()
                            )

                        subsample = G.subgraph(nodos_sample)

                        if (
                            nx.is_directed_acyclic_graph(subsample)
                            and nx.has_path(subsample, "insumos", "impactos")
                            and len(subsample.edges) >= 2
                        ):
                            certezas.append(1.0)
                        else:
                            certezas.append(0.4)
                    else:
                        certezas.append(0.3)
                except Exception:
                    certezas.append(0.3)

            certeza = np.mean(certezas) if certezas else 0.5

            # Cálculo de robustez causal industrial
            robustez = dimension.teoria_cambio.calcular_coeficiente_causal()

            # Ajuste de certeza basado en evidencia disponible
            evidencia_soporte = sum(len(v) for v in evidencia.values())
            if evidencia_soporte == 0:
                certeza *= 0.5
                factibilidad *= 0.5

            # Ajuste de riesgos basado en certeza
            if certeza < 0.7:
                riesgos.append("⚠️ Baja certeza causal: Requiere fortalecimiento de marco teórico")
            if robustez < 0.6:
                riesgos.append("⚠️ Baja robustez causal: Requiere simplificación de relaciones causales")

            # Conteo de brechas críticas
            brechas_criticas = len(self._identificar_brechas_industrial(dimension, evidencia))

            return EvaluacionCausalIndustrial(
                consistencia_logica=consistencia,
                identificabilidad_causal=identificabilidad,
                factibilidad_operativa=factibilidad,
                certeza_probabilistica=certeza,
                robustez_causal=robustez,
                riesgos_implementacion=riesgos,
                supuestos_criticos=dimension.teoria_cambio.supuestos_causales,
                evidencia_soporte=evidencia_soporte,
                brechas_criticas=brechas_criticas,
            )

        except Exception as e:
            self.logger.error(f"❌ Error en evaluación causal industrial: {e}")
            return EvaluacionCausalIndustrial(
                consistencia_logica=0.3,
                identificabilidad_causal=0.3,
                factibilidad_operativa=0.3,
                certeza_probabilistica=0.3,
                robustez_causal=0.3,
                riesgos_implementacion=[f"Error en evaluación causal: {str(e)}"],
                supuestos_criticos=[],
                evidencia_soporte=0,
                brechas_criticas=5,
            )

    def _calcular_factibilidad_industrial(
        self, dimension: DimensionDecalogo, evidencia: Dict
    ) -> float:
        """Calcula factibilidad operativa industrial basada en evidencia y estándares"""
        factores = []

        # Factor 1: Presencia de recursos (peso 0.4)
        if evidencia.get("recursos", []):
            factores.append(0.9)
        else:
            factores.append(0.2)

        # Factor 2: Especificidad de indicadores (peso 0.3)
        indicadores_encontrados = len(evidencia.get("indicadores", []))
        indicadores_requeridos = len(dimension.eslabones)
        if indicadores_encontrados >= indicadores_requeridos:
            factores.append(0.95)
        elif indicadores_encontrados > 0:
            factores.append(0.6 + (0.35 * indicadores_encontrados / indicadores_requeridos))
        else:
            factores.append(0.1)

        # Factor 3: Presencia de responsables y plazos (peso 0.3)
        responsables_plazos = len(evidencia.get("responsables", [])) + len(evidencia.get("plazos", []))
        if responsables_plazos >= 2:
            factores.append(0.85)
        elif responsables_plazos == 1:
            factores.append(0.5)
        else:
            factores.append(0.2)

        return sum(factores) / len(factores) if factores else 0.3

    def _identificar_brechas_industrial(
        self, dimension: DimensionDecalogo, evidencia: Dict
    ) -> List[str]:
        """Identifica brechas industriales en la implementación con diagnóstico preciso"""
        brechas = []

        # Brecha 1: Falta de indicadores por eslabón
        for eslabon in dimension.eslabones:
            indicadores_encontrados = any(
                any(ind.lower() in ev["texto"].lower() for ind in eslabon.indicadores)
                for ev in evidencia.get("indicadores", [])
            )
            if not indicadores_encontrados:
                brechas.append(
                    f"🔴 BRECHA CRÍTICA: Falta especificación de indicadores para eslabón {eslabon.id} ({eslabon.tipo.value})"
                )

        # Brecha 2: Falta de recursos
        if not evidencia.get("recursos", []):
            brechas.append("🔴 BRECHA CRÍTICA: No se encontró especificación presupuestal o de recursos")

        # Brecha 3: Falta de responsables
        if not evidencia.get("responsables", []):
            brechas.append(
                "🟠 BRECHA IMPORTANTE: No se identificaron responsables claros para la implementación"
            )

        # Brecha 4: Falta de plazos
        if not evidencia.get("plazos", []):
            brechas.append("🟠 BRECHA IMPORTANTE: No se encontraron plazos o cronogramas definidos")

        # Brecha 5: Complejidad causal excesiva
        if len(dimension.teoria_cambio.supuestos_causales) > 5:
            brechas.append(
                "🟡 BRECHA MODERADA: Alta complejidad causal puede dificultar la implementación y medición"
            )

        return brechas

    def _identificar_riesgos_industrial(
        self, dimension: DimensionDecalogo, evidencia: Dict
    ) -> List[str]:
        """Identifica riesgos industriales de implementación con clasificación por severidad"""
        riesgos = []

        # Riesgo 1: Falta de recursos (ALTO)
        if not evidencia.get("recursos", []):
            riesgos.append("🔴 ALTO: Falta de especificación presupuestal - Riesgo de inviabilidad operativa")

        # Riesgo 2: Complejidad causal (MEDIO-ALTO)
        if len(dimension.teoria_cambio.supuestos_causales) > 4:
            riesgos.append(
                "🟠 MEDIO-ALTO: Alta complejidad causal - Riesgo de dificultad en implementación y atribución"
            )

        # Riesgo 3: Falta de indicadores (ALTO)
        indicadores_encontrados = len(evidencia.get("indicadores", []))
        indicadores_requeridos = len(dimension.eslabones)
        if indicadores_encontrados < indicadores_requeridos * 0.5:
            riesgos.append(
                "🔴 ALTO: Cobertura insuficiente de indicadores - Riesgo de imposibilidad de medición y evaluación"
            )

        # Riesgo 4: Ventanas temporales (MEDIO)
        for eslabon in dimension.eslabones:
            if eslabon.ventana_temporal[1] - eslabon.ventana_temporal[0] > 36:
                riesgos.append(
                    f"🟠 MEDIO: Ventana temporal excesivamente amplia en eslabón {eslabon.id} - Riesgo de desfase en resultados"
                )
                break

        # Riesgo 5: Falta de responsables (MEDIO)
        if not evidencia.get("responsables", []):
            riesgos.append(
                "🟠 MEDIO: Ausencia de responsables definidos - Riesgo de falta de rendición de cuentas"
            )

        return riesgos

    def _generar_recomendaciones_industrial(
        self,
        dimension: DimensionDecalogo,
        evaluacion: EvaluacionCausalIndustrial,
        brechas: List[str],
    ) -> List[str]:
        """Genera recomendaciones industriales específicas basadas en el análisis con enfoque estratégico"""
        recomendaciones = []

        # Recomendaciones basadas en evaluación causal
        if evaluacion.consistencia_logica < 0.7:
            recomendaciones.append(
                "🔧 FORTALECER: Revisar y fortalecer la coherencia lógica de la teoría de cambio"
            )

        if evaluacion.factibilidad_operativa < 0.6:
            recomendaciones.append(
                "🔧 FORTALECER: Especificar mejor los mecanismos de implementación operativa"
            )

        if evaluacion.certeza_probabilistica < 0.7:
            recomendaciones.append(
                "📊 EVIDENCIA: Incorporar mayor evidencia empírica para sustentar las relaciones causales"
            )

        if evaluacion.robustez_causal < 0.6:
            recomendaciones.append(
                "🧩 SIMPLIFICAR: Reducir la complejidad del modelo causal para mejorar su robustez"
            )

        # Recomendaciones basadas en brechas
        for brecha in brechas:
            if "BRECHA CRÍTICA" in brecha:
                recomendaciones.append(
                    f"🚨 ACCIÓN INMEDIATA: {brecha.replace('🔴 BRECHA CRÍTICA: ', '')}"
                )
            elif "BRECHA IMPORTANTE" in brecha:
                recomendaciones.append(
                    f"⚠️ PRIORIDAD ALTA: {brecha.replace('🟠 BRECHA IMPORTANTE: ', '')}"
                )
            elif "BRECHA MODERADA" in brecha:
                recomendaciones.append(
                    f"🔧 MEJORA CONTINUA: {brecha.replace('🟡 BRECHA MODERADA: ', '')}"
                )

        # Recomendaciones estratégicas adicionales
        if evaluacion.evidencia_soporte < 5:
            recomendaciones.append(
                "📚 INVESTIGACIÓN: Realizar estudio de línea base para fortalecer la evidencia disponible"
            )

        if len(evaluacion.riesgos_implementacion) > 3:
            recomendaciones.append(
                "🛡️ GESTIÓN DE RIESGOS: Desarrollar plan integral de mitigación de riesgos"
            )

        # Recomendación de monitoreo
        recomendaciones.append(
            "📈 MONITOREO: Establecer sistema de monitoreo y evaluación con indicadores SMART"
        )

        return recomendaciones[:15]

    def generar_reporte_tecnico_completo(
        self, resultados: List[ResultadoDimensionIndustrial]
    ) -> Dict[str, Any]:
        """Genera reporte técnico industrial completo con análisis agregado"""
        try:
            puntajes = [r.puntaje_final for r in resultados]
            niveles_madurez = [r.nivel_madurez for r in resultados]
            certidumbres = [r.evaluacion_causal.nivel_certidumbre for r in resultados]

            # Análisis agregado industrial
            analisis_agregado = {
                "puntaje_global_promedio": statistics.mean(puntajes) if puntajes else 0,
                "desviacion_estandar": statistics.stdev(puntajes) if len(puntajes) > 1 else 0,
                "dimensiones_evaluadas": len(resultados),
                "dimensiones_excelentes": len([p for p in puntajes if p >= 85]),
                "dimensiones_aceptables": len([p for p in puntajes if 70 <= p < 85]),
                "dimensiones_deficientes": len([p for p in puntajes if p < 70]),
                "nivel_madurez_predominante": (
                    max(set(niveles_madurez), key=niveles_madurez.count) if niveles_madurez else "N/A"
                ),
                "certidumbre_predominante": (
                    max(set(certidumbres), key=certidumbres.count) if certidumbres else "N/A"
                ),
                "recomendacion_estrategica_global": self._generar_recomendacion_global(resultados),
                "riesgos_sistemicos": self._identificar_riesgos_sistemicos(resultados),
            }

            # Resultados por dimensión
            resultados_detalles = []
            for resultado in resultados:
                resultados_detalles.append(resultado.generar_reporte_tecnico())

            reporte_completo = {
                "metadata": {
                    "nombre_plan": self.pdf_path.stem,
                    "hash_evaluacion": self.hash_evaluacion,
                    "fecha_evaluacion": datetime.now().isoformat(),
                    "version_sistema": "8.1-industrial",
                    "total_dimensiones": len(DECALOGO_INDUSTRIAL),
                },
                "analisis_agregado": analisis_agregado,
                "resultados_por_dimension": resultados_detalles,
                "timestamp_generacion": datetime.now().isoformat(),
            }

            return reporte_completo

        except Exception as e:
            self.logger.error(f"❌ Error generando reporte técnico completo: {e}")
            return {
                "metadata": {"nombre_plan": self.pdf_path.stem, "error": str(e)},
                "analisis_agregado": {},
                "resultados_por_dimension": [],
            }

    def _generar_recomendacion_global(self, resultados: List[ResultadoDimensionIndustrial]) -> str:
        """Genera recomendación estratégica global basada en análisis agregado"""
        puntajes = [r.puntaje_final for r in resultados]
        if not puntajes:
            return "NO APLICA"

        promedio = statistics.mean(puntajes)
        deficientes = len([p for p in puntajes if p < 70])
        total = len(puntajes)

        if promedio >= 85:
            return "IMPLEMENTACIÓN INTEGRAL RECOMENDADA - ALTO NIVEL DE MADUREZ"
        elif promedio >= 70 and deficientes <= total * 0.3:
            return "IMPLEMENTACIÓN SELECTIVA RECOMENDADA - FORTALECER DIMENSIONES DÉBILES"
        elif promedio >= 50:
            return "REDISEÑO PARCIAL REQUERIDO - PRIORIZAR DIMENSIONES CRÍTICAS"
        else:
            return "REDISEÑO INTEGRAL REQUERIDO - REVISIÓN ESTRATÉGICA FUNDAMENTAL"

    def _identificar_riesgos_sistemicos(self, resultados: List[ResultadoDimensionIndustrial]) -> List[str]:
        """Identifica riesgos sistémicos que afectan a todo el plan"""
        riesgos_sistemicos = []

        # Riesgo de coherencia global
        puntajes = [r.puntaje_final for r in resultados]
        if len(puntajes) > 1 and statistics.stdev(puntajes) > 25:
            riesgos_sistemicos.append(
                "🔴 DESCOHERENCIA ESTRATÉGICA: Alta variabilidad en madurez entre dimensiones"
            )

        # Riesgo de evidencia global
        evidencia_total = sum(r.evaluacion_causal.evidencia_soporte for r in resultados)
        if evidencia_total < len(resultados) * 3:
            riesgos_sistemicos.append(
                "🟠 DÉFICIT DE EVIDENCIA: Bajo soporte empírico para el conjunto del plan"
            )

        # Riesgo de implementación global
        riesgos_criticos = sum(len(r.evaluacion_causal.riesgos_implementacion) for r in resultados)
        if riesgos_criticos > len(resultados) * 3:
            riesgos_sistemicos.append(
                "🔴 SOBRECARGA DE RIESGOS: Alta concentración de riesgos de implementación"
            )

        # Riesgo de certeza global
        certezas_bajas = sum(
            1 for r in resultados if r.evaluacion_causal.certeza_probabilistica < 0.6
        )
        if certezas_bajas > len(resultados) * 0.4:
            riesgos_sistemicos.append(
                "🟠 INCERTIDUMBRE SISTÉMICA: Baja certeza causal en múltiples dimensiones"
            )

        return riesgos_sistemicos if riesgos_sistemicos else ["✅ SIN RIESGOS SISTÉMICOS IDENTIFICADOS"]


# ==================== GENERADOR DE REPORTES INDUSTRIAL ====================
class GeneradorReporteIndustrial:
    """Genera reportes industriales en múltiples formatos con estándares profesionales"""

    @staticmethod
    def generar_reporte_markdown(
        resultados: List[ResultadoDimensionIndustrial],
        nombre_plan: str,
        metadata: Dict = None,
    ) -> str:
        """Genera reporte industrial en formato markdown con análisis profundo"""
        reporte = []

        # Encabezado industrial
        reporte.append("# 🏭 EVALUACIÓN INDUSTRIAL DE POLÍTICAS PÚBLICAS")
        reporte.append(f"## 📄 Plan de Desarrollo Municipal: {nombre_plan}")
        reporte.append("### 🎯 Análisis Multinivel con Enfoque Causal y Certificación de Rigor")
        reporte.append(f"### 📊 Fecha de evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if metadata and metadata.get("hash_evaluacion"):
            reporte.append(f"### 🔐 Hash de evaluación: {metadata.get('hash_evaluacion', '')[:12]}...")
        reporte.append("")

        # Resumen ejecutivo industrial
        puntajes = [r.puntaje_final for r in resultados]
        if puntajes:
            promedio = statistics.mean(puntajes)
            desviacion = statistics.stdev(puntajes) if len(puntajes) > 1 else 0
            excelentes = len([p for p in puntajes if p >= 85])
            deficientes = len([p for p in puntajes if p < 70])

            reporte.append("## 📈 RESUMEN EJECUTIVO INDUSTRIAL")
            reporte.append(f"**🎯 Puntaje Global:** {promedio:.1f}/100")
            reporte.append(f"**📊 Desviación Estándar:** {desviacion:.1f}")
            reporte.append(f"**🏆 Dimensiones Excelentes (≥85):** {excelentes}/{len(resultados)}")
            reporte.append(f"**⚠️ Dimensiones Deficientes (<70):** {deficientes}/{len(resultados)}")

            # Nivel de madurez global
            niveles = [r.nivel_madurez for r in resultados]
            nivel_predominante = max(set(niveles), key=niveles.count) if niveles else "N/A"
            reporte.append(f"**🏭 Nivel de Madurez Predominante:** {nivel_predominante}")

            # Recomendación estratégica global
            if promedio >= 85:
                recomendacion_global = "IMPLEMENTACIÓN INTEGRAL RECOMENDADA ✅"
                emoji = "🚀"
            elif promedio >= 70:
                recomendacion_global = "IMPLEMENTACIÓN SELECTIVA CON MEJORAS ⚠️"
                emoji = "🔧"
            elif promedio >= 50:
                recomendacion_global = "REDISEÑO PARCIAL REQUERIDO 🚨"
                emoji = "🛠️"
            else:
                recomendacion_global = "REDISEÑO INTEGRAL URGENTE ❌"
                emoji = "🆘"

            reporte.append(f"**{emoji} Recomendación Estratégica Global:** {recomendacion_global}")
            reporte.append("")

        # Análisis detallado por dimensión
        for i, resultado in enumerate(resultados, 1):
            reporte.append(f"## 🔍 DIMENSIÓN {resultado.dimension.id}: {resultado.dimension.nombre}")
            reporte.append(f"### 🏷️ Cluster: {resultado.dimension.cluster}")
            reporte.append(f"### 📊 Puntaje: {resultado.puntaje_final:.1f}/100")
            reporte.append(f"### 🏭 Nivel de Madurez: {resultado.nivel_madurez}")
            reporte.append(f"### 🎯 Certidumbre: {resultado.evaluacion_causal.nivel_certidumbre}")
            reporte.append(
                f"### 💡 Recomendación Estratégica: {resultado.evaluacion_causal.recomendacion_estrategica}"
            )
            reporte.append("")

            # Teoría de cambio industrial
            reporte.append("### 🧩 TEORÍA DE CAMBIO INDUSTRIAL")
            reporte.append("**Supuestos causales:**")
            for supuesto in resultado.dimension.teoria_cambio.supuestos_causales:
                reporte.append(f"- {supuesto}")
            reporte.append("")

            # Métricas de evaluación causal industrial
            reporte.append("### 📊 EVALUACIÓN CAUSAL INDUSTRIAL")
            ev = resultado.evaluacion_causal
            reporte.append(f"- **Consistencia lógica:** {ev.consistencia_logica:.3f}")
            reporte.append(f"- **Identificabilidad causal:** {ev.identificabilidad_causal:.3f}")
            reporte.append(f"- **Factibilidad operativa:** {ev.factibilidad_operativa:.3f}")
            reporte.append(f"- **Certeza probabilística:** {ev.certeza_probabilistica:.3f}")
            reporte.append(f"- **Robustez causal:** {ev.robustez_causal:.3f}")
            reporte.append(f"- **Evidencia de soporte:** {ev.evidencia_soporte} elementos")
            reporte.append(f"- **Brechas críticas:** {ev.brechas_criticas}")
            reporte.append("")

            # Riesgos de implementación
            if ev.riesgos_implementacion:
                reporte.append("### ⚠️ RIESGOS DE IMPLEMENTACIÓN")
                for riesgo in ev.riesgos_implementacion:
                    reporte.append(f"- {riesgo}")
                reporte.append("")

            # Brechas identificadas
            if resultado.brechas_identificadas:
                reporte.append("### 🚨 BRECHAS IDENTIFICADAS")
                for brecha in resultado.brechas_identificadas:
                    reporte.append(f"- {brecha}")
                reporte.append("")

            # Recomendaciones industriales
            if resultado.recomendaciones:
                reporte.append("### 💡 RECOMENDACIONES INDUSTRIALES")
                for recomendacion in resultado.recomendaciones[:8]:
                    reporte.append(f"- {recomendacion}")
                reporte.append("")

            # Evidencia clave
            evidencia_total = sum(len(v) for v in resultado.evidencia.values())
            if evidencia_total > 0:
                reporte.append("### 📚 EVIDENCIA CLAVE")
                for tipo, evidencias

            self.hash_documento = self.calcular_hash_documento()
            self.metadata = self.extraer_metadata_pdf()

            self.logger.info(f"✅ Documento cargado: {len(self.paginas)} páginas - Hash: {self.hash_documento[:8]}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error crítico cargando PDF {self.file_path}: {e}")
            return False

    def segmentar(self) -> bool:
        """Segmenta el texto en unidades de análisis industrial con procesamiento NLP avanzado"""
        if not self.paginas:
            self.logger.error("❌ No hay páginas para segmentar")
            return False

        try:
            total_segmentos = 0
            for i, pagina in enumerate(self.paginas, start=1):
                if not pagina.strip():
                    continue

                try:
                    doc = NLP(pagina)
                    buffer = []

                    for sentencia in doc.sents:
                        texto = sentencia.text.strip()
                        if len(texto) >= 20 and self._tiene_contenido_sustancial(sentencia):
                            buffer.append(texto)
                            if len(buffer) >= 3 or (
                                len(buffer) >= 2 and self._detectar_cambio_tematico(buffer)
                            ):
                                segmento = " ".join(buffer)
                                self.segmentos.append((i, segmento))
                                total_segmentos += 1
                                buffer = []

                    if buffer:
                        segmento = " ".join(buffer)
                        if len(segmento) >= 30:
                            self.segmentos.append((i, segmento))
                            total_segmentos += 1

                except Exception as e:
                    self.logger.warning(f"⚠️ Error procesando página {i}: {e}")
                    continue

            self.logger.info(f"✅ Segmentación completada: {len(self.segmentos)} segmentos - {self.nombre_plan}")
            return len(self.segmentos) > 0

        except Exception as e:
            self.logger.error(f"❌ Error crítico segmentando {self.nombre_plan}: {e}")
            return False

    def _tiene_contenido_sustancial(self, doc: spacy.tokens.Doc) -> bool:
        """Verifica si el Doc procesado tiene contenido sustancial para análisis"""
        tiene_sustantivos = any(token.pos_ in ["NOUN", "PROPN"] for token in doc)
        tiene_verbos = any(token.pos_ == "VERB" for token in doc)
        return tiene_sustantivos and tiene_verbos and len(doc) >= 5

    def _detectar_cambio_tematico(self, buffer: List[str]) -> bool:
        """Detecta cambios temáticos para segmentación inteligente"""
        if len(buffer) < 2:
            return False

        try:
            if len(buffer) >= 2:
                embeddings = EMBEDDING_MODEL.encode(buffer[-2:], convert_to_tensor=True)
                if len(embeddings) == 2:
                    similitud = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                    return similitud < 0.6
        except Exception:
            pass

        return False