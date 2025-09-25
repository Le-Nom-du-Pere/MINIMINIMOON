# -*- coding: utf-8 -*-
"""
Sistema Integral de Evaluación de Cadenas de Valor en Planes de Desarrollo Municipal
Versión: 8.1 — Marco Teórico-Institucional con Análisis Causal Multinivel, Batch Processing,
Certificación de Rigor y Selección Global Top-K con Heap.
Framework basado en IAD + Theory of Change, con triangulación cuali-cuantitativa,
verificación causal y certeza probabilística.
Autor: Dr. en Políticas Públicas
Enfoque: Evaluación estructural con econometría de políticas, minería causal y
procesamiento paralelo industrial.
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

# -------------------- Dependencias con fallback explícito --------------------
# pdfplumber (lectura PDF)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

# NLP (spaCy)
import spacy

# Paralelismo
from joblib import Parallel, delayed  # noqa: F401  (se mantiene para compatibilidad futura)

# sentence-transformers (embeddings semánticos)
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

# Módulos externos con fallback
try:
    from decalogo_loader import get_decalogo_industrial
except ImportError:
    def get_decalogo_industrial():
        return "Fallback: Decálogo industrial para desarrollo municipal con 10 dimensiones estratégicas."

try:
    from device_config import add_device_args, configure_device_from_args, get_device_config, to_device
except ImportError:
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
    def get_truncation_logger(name): return logging.getLogger(name)
    def log_debug_with_text(logger, text): logger.debug(text)
    def log_error_with_text(logger, text): logger.error(text)
    def log_info_with_text(logger, text): logger.info(text)
    def log_warning_with_text(logger, text): logger.warning(text)
    def truncate_text_for_log(text, max_len=200): return text[:max_len] + "..." if len(text) > max_len else text

# Requerimiento de versión
assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

# -------------------- Logging industrial --------------------
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

# -------------------- Modelos (NLP + Embeddings) --------------------
try:
    NLP = spacy.load("es_core_news_lg")
    log_info_with_text(LOGGER, "✅ Modelo SpaCy cargado (es_core_news_lg)")
except OSError as e:
    try:
        NLP = spacy.load("es_core_news_sm")
        log_warning_with_text(LOGGER, "⚠️ Usando modelo SpaCy pequeño (es_core_news_sm)")
    except OSError:
        log_error_with_text(LOGGER, f"❌ Error cargando SpaCy: {e}")
        raise SystemExit("Modelo SpaCy no disponible. Ejecute: python -m spacy download es_core_news_lg")

if not SENTENCE_TRANSFORMERS_AVAILABLE:
    raise SystemExit("sentence-transformers no disponible. Instale: pip install sentence-transformers")

try:
    EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    EMBEDDING_MODEL = to_device(EMBEDDING_MODEL)
    log_info_with_text(LOGGER, "✅ Modelo de embeddings cargado")
    log_info_with_text(LOGGER, f"✅ Dispositivo: {get_device_config().get_device()}")
except Exception as e:
    log_error_with_text(LOGGER, f"❌ Error cargando embeddings: {e}")
    raise SystemExit(f"Error cargando modelo de embeddings: {e}")

# -------------------- Marco teórico e instrumentos --------------------
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
    supuestos_causales: List[str]
    mediadores: Dict[str, List[str]]
    resultados_intermedios: List[str]
    precondiciones: List[str]

    def verificar_identificabilidad(self) -> bool:
        return bool(self.supuestos_causales and self.mediadores and self.resultados_intermedios)

    def construir_grafo_causal(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_node("insumos", tipo="nodo_base")
        G.add_node("impactos", tipo="nodo_base")
        for categoria, lista in self.mediadores.items():
            for mediador in lista:
                G.add_node(mediador, tipo="mediador", categoria=categoria)
                G.add_edge("insumos", mediador, weight=1.0, tipo="causal")
                for resultado in self.resultados_intermedios:
                    G.add_node(resultado, tipo="resultado")
                    G.add_edge(mediador, resultado, weight=0.8, tipo="causal")
                    G.add_edge(resultado, "impactos", weight=0.9, tipo="causal")
        return G

    def calcular_coeficiente_causal(self) -> float:
        G = self.construir_grafo_causal()
        if len(G.nodes) < 3:
            return 0.3
        try:
            paths_validos = 0
            total_paths = 0
            mediadores = [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]
            resultados = [n for n in G.nodes if G.nodes[n].get("tipo") == "resultado"]
            for m in mediadores:
                for r in resultados:
                    if nx.has_path(G, m, r) and nx.has_path(G, r, "impactos"):
                        paths_validos += 1
                    total_paths += 1
            return paths_validos / max(1, total_paths) if total_paths > 0 else 0.5
        except Exception:
            return 0.5

@dataclass(frozen=True)
class EslabonCadena:
    id: str
    tipo: TipoCadenaValor
    indicadores: List[str]
    capacidades_requeridas: List[str]
    puntos_criticos: List[str]
    ventana_temporal: Tuple[int, int]
    kpi_ponderacion: float = 1.0
    def __post_init__(self):
        if not (0 <= self.kpi_ponderacion <= 2.0):
            raise ValueError("KPI ponderación debe estar entre 0 y 2.0")
        if self.ventana_temporal[0] > self.ventana_temporal[1]:
            raise ValueError("Ventana temporal inválida")
    def calcular_lead_time(self) -> float:
        return (self.ventana_temporal[0] + self.ventana_temporal[1]) / 2.0
    def generar_hash(self) -> str:
        data = f"{self.id}|{self.tipo.value}|{sorted(self.indicadores)}|{sorted(self.capacidades_requeridas)}"
        return hashlib.md5(data.encode("utf-8")).hexdigest()

# -------------------- Ontología --------------------
@dataclass
class OntologiaPoliticas:
    dimensiones: Dict[str, List[str]]
    relaciones_causales: Dict[str, List[str]]
    indicadores_ods: Dict[str, List[str]]
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0-industrial"

    @classmethod
    def cargar_estandar(cls) -> 'OntologiaPoliticas':
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
            return cls(dimensiones=dimensiones_industrial,
                       relaciones_causales=relaciones_industrial,
                       indicadores_ods=indicadores_ods)
        except Exception as e:
            log_error_with_text(LOGGER, f"❌ Error cargando ontología: {e}")
            raise SystemExit("Fallo en carga de ontología")

    @staticmethod
    def _cargar_indicadores_ods(ruta: Path) -> Dict[str, List[str]]:
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
                    LOGGER.info("✅ Ontología ODS cargada desde archivo")
                    return data
                else:
                    LOGGER.warning("⚠️ Ontología ODS inválida, usando base")
            except Exception as e:
                LOGGER.warning(f"⚠️ Error leyendo {ruta}: {e}, usando base")
        try:
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(indicadores_base, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"✅ Template ODS generado: {ruta}")
        except Exception as e:
            LOGGER.error(f"❌ Error generando template ODS: {e}")
        return indicadores_base

# -------------------- Decálogo --------------------
@dataclass(frozen=True)
class DimensionDecalogo:
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
        coherencia = 0.0; peso_total = 0.0
        if self.teoria_cambio.verificar_identificabilidad(): coherencia += 0.4
        peso_total += 0.4
        tipos_presentes = {e.tipo for e in self.eslabones}
        tipos_esenciales = {TipoCadenaValor.INSUMOS, TipoCadenaValor.PROCESOS, TipoCadenaValor.PRODUCTOS}
        coherencia += 0.3 if tipos_esenciales.issubset(tipos_presentes) else 0.0
        peso_total += 0.3
        coherencia += 0.3 if any(e.tipo == TipoCadenaValor.IMPACTOS for e in self.eslabones) else 0.0
        peso_total += 0.3
        return coherencia / peso_total if peso_total > 0 else 0.0
    def calcular_kpi_global(self) -> float:
        return (sum(e.kpi_ponderacion for e in self.eslabones) / len(self.eslabones)) if self.eslabones else 0.0
    def generar_matriz_riesgos(self) -> Dict[str, List[str]]:
        matriz: Dict[str, List[str]] = {}
        for e in self.eslabones:
            riesgos = []
            if not e.indicadores: riesgos.append("Falta de indicadores de desempeño")
            if e.ventana_temporal[1] - e.ventana_temporal[0] > 24: riesgos.append("Ventana temporal excesivamente amplia")
            if len(e.capacidades_requeridas) < 2: riesgos.append("Capacidades requeridas insuficientes")
            matriz[e.id] = riesgos
        return matriz

def cargar_decalogo_industrial() -> List[Any]:
    json_path = Path("decalogo_industrial.json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) != 10:
                raise ValueError("Decálogo debe contener exactamente 10 dimensiones")
            decalogos = []
            for i, item in enumerate(data):
                if not all(k in item for k in ["id", "nombre", "cluster", "teoria_cambio", "eslabones"]):
                    raise ValueError(f"Dimensión {i+1} incompleta")
                if item["id"] != i + 1:
                    raise ValueError(f"ID incorrecto en dimensión {i+1}")
                tc_data = item["teoria_cambio"]
                teoria_cambio = TeoriaCambio(
                    supuestos_causales=tc_data["supuestos_causales"],
                    mediadores=tc_data["mediadores"],
                    resultados_intermedios=tc_data["resultados_intermedios"],
                    precondiciones=tc_data["precondiciones"],
                )
                if not teoria_cambio.verificar_identificabilidad():
                    raise ValueError(f"Teoría de cambio no identificable en dimensión {i+1}")
                eslabones = []
                for j, ed in enumerate(item["eslabones"]):
                    eslabones.append(EslabonCadena(
                        id=ed["id"],
                        tipo=TipoCadenaValor[ed["tipo"]],
                        indicadores=ed["indicadores"],
                        capacidades_requeridas=ed["capacidades_requeridas"],
                        puntos_criticos=ed["puntos_criticos"],
                        ventana_temporal=tuple(ed["ventana_temporal"]),
                        kpi_ponderacion=float(ed.get("kpi_ponderacion", 1.0)),
                    ))
                decalogos.append(DimensionDecalogo(
                    id=item["id"], nombre=item["nombre"], cluster=item["cluster"],
                    teoria_cambio=teoria_cambio, eslabones=eslabones))
            LOGGER.info(f"✅ Decálogo cargado y validado: {len(decalogos)} dimensiones")
            return decalogos
        except Exception as e:
            LOGGER.error(f"❌ Error cargando decálogo: {e}")
            raise SystemExit("Fallo en carga de decálogo")

    # Genera template si no existe.
    LOGGER.info("⚙️ Generando template industrial de decálogo estructurado")
    try:
        _ = get_decalogo_industrial()
    except Exception as fallback_exc:
        LOGGER.warning(f"⚠️ Fallback textual no disponible: {fallback_exc}")

    template = []
    for dim_id in range(1, 11):
        dim = {
            "id": dim_id,
            "nombre": f"Dimensión {dim_id} del Decálogo Industrial",
            "cluster": f"Cluster {((dim_id - 1) // 3) + 1}",
            "teoria_cambio": {
                "supuestos_causales": [f"Supuesto causal 1 ({dim_id})", f"Supuesto causal 2 ({dim_id})"],
                "mediadores": {
                    "institucionales": [f"mediador_institucional_{dim_id}_1", f"mediador_institucional_{dim_id}_2"],
                    "comunitarios": [f"mediador_comunitario_{dim_id}_1", f"mediador_comunitario_{dim_id}_2"],
                },
                "resultados_intermedios": [f"resultado_intermedio_{dim_id}_1", f"resultado_intermedio_{dim_id}_2"],
                "precondiciones": [f"precondicion_{dim_id}_1", f"precondicion_{dim_id}_2"],
            },
            "eslabones": []
        }
        for tipo_idx, tipo in enumerate(["INSUMOS", "PROCESOS", "PRODUCTOS", "RESULTADOS", "IMPACTOS"]):
            dim["eslabones"].append({
                "id": f"{tipo.lower()[:3]}_{dim_id}",
                "tipo": tipo,
                "indicadores": [f"indicador_{tipo.lower()}_{dim_id}_{i+1}" for i in range(3)],
                "capacidades_requeridas": [f"capacidad_{tipo.lower()}_{dim_id}_{i+1}" for i in range(2)],
                "puntos_criticos": [f"punto_critico_{tipo.lower()}_{dim_id}_{i+1}" for i in range(2)],
                "ventana_temporal": [tipo_idx * 6 + 1, (tipo_idx + 1) * 6 + 6],
                "kpi_ponderacion": 1.0 + (tipo_idx * 0.1),
            })
        template.append(dim)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"✅ Template de decálogo generado: {json_path}")
    LOGGER.warning("⚠️ COMPLETE Y VALIDE MANUALMENTE 'decalogo_industrial.json'")
    return cargar_decalogo_industrial()

DECALOGO_INDUSTRIAL = cargar_decalogo_industrial()

# Clusters (metadatos)
@dataclass(frozen=True)
class ClusterMetadata:
    cluster_id: str
    titulo: str
    nombre_dimension: str
    puntos: List[int]
    logica_agrupacion: str

@dataclass(frozen=True)
class DecalogoContext:
    dimensiones_por_id: Dict[int, DimensionDecalogo]
    clusters_por_id: Dict[str, ClusterMetadata]
    cluster_por_dimension: Dict[int, ClusterMetadata]

_CLUSTER_DEFINITIONS = {
    "CLUSTER 1": {"titulo": "CLUSTER 1: PAZ, SEGURIDAD Y PROTECCIÓN DE DEFENSORES", "puntos": [1, 5, 8],
                  "logica": "Estos tres puntos comparten una matriz común centrada en la seguridad humana."},
    "CLUSTER 2": {"titulo": "CLUSTER 2: DERECHOS DE GRUPOS POBLACIONALES", "puntos": [2, 6, 9],
                  "logica": "Agrupa derechos de poblaciones que enfrentan vulnerabilidades específicas."},
    "CLUSTER 3": {"titulo": "CLUSTER 3: TERRITORIO, AMBIENTE Y DESARROLLO SOSTENIBLE", "puntos": [3, 7],
                  "logica": "Relación sociedad-territorio desde sostenibilidad."},
    "CLUSTER 4": {"titulo": "CLUSTER 4: DERECHOS SOCIALES FUNDAMENTALES Y CRISIS HUMANITARIAS", "puntos": [4, 10],
                  "logica": "Respuesta a necesidades básicas y dignidad humana."},
}

_DECALOGO_CONTEXT_CACHE: Optional[DecalogoContext] = None

def obtener_decalogo_contexto() -> DecalogoContext:
    global _DECALOGO_CONTEXT_CACHE
    if _DECALOGO_CONTEXT_CACHE is not None:
        return _DECALOGO_CONTEXT_CACHE
    dimensiones_por_id = {d.id: d for d in DECALOGO_INDUSTRIAL}
    clusters_por_id: Dict[str, ClusterMetadata] = {}
    cluster_por_dimension: Dict[int, ClusterMetadata] = {}
    for cluster_id, data in _CLUSTER_DEFINITIONS.items():
        puntos = data["puntos"]
        nombre_dimension = next((dimensiones_por_id[p].cluster for p in puntos if p in dimensiones_por_id),
                                data["titulo"])
        metadata = ClusterMetadata(cluster_id=cluster_id, titulo=data["titulo"],
                                   nombre_dimension=nombre_dimension, puntos=puntos,
                                   logica_agrupacion=data["logica"])
        clusters_por_id[cluster_id] = metadata
        for pid in puntos:
            cluster_por_dimension[pid] = metadata
    _DECALOGO_CONTEXT_CACHE = DecalogoContext(dimensiones_por_id, clusters_por_id, cluster_por_dimension)
    return _DECALOGO_CONTEXT_CACHE

# -------------------- Extractor de evidencia --------------------
class ExtractorEvidenciaIndustrial:
    def __init__(self, documentos: List[Tuple[int, str]], nombre_plan: str = "desconocido"):
        self.documentos = documentos
        self.nombre_plan = nombre_plan
        self.ontologia = OntologiaPoliticas.cargar_estandar()
        self.embeddings_doc: torch.Tensor | None = None
        self.textos_originales = [doc[1] for doc in documentos]
        self.logger = logging.getLogger(f"Extractor_{nombre_plan}")
        self._precomputar_embeddings()

    def _precomputar_embeddings(self):
        textos_validos = [t for t in self.textos_originales if len(t.strip()) > 10]
        if textos_validos:
            try:
                self.embeddings_doc = EMBEDDING_MODEL.encode(textos_validos, convert_to_tensor=True)
                self.logger.info(f"✅ Embeddings precomputados: {len(textos_validos)} segmentos - {self.nombre_plan}")
            except Exception as e:
                self.logger.error(f"❌ Error precomputando embeddings: {e}")
                self.embeddings_doc = torch.tensor([])
        else:
            self.embeddings_doc = torch.tensor([])
            self.logger.warning(f"⚠️ Textos insuficientes para embeddings - {self.nombre_plan}")

    def _densidad_causal(self, texto: str) -> float:
        patrones = [
            r"\b(porque|debido a|como consecuencia de|en razón de|a causa de)\b",
            r"\b(genera|produce|causa|determina|influye en|afecta a)\b",
            r"\b(impacto|efecto|resultado|consecuencia|repercusión)\b",
            r"\b(mejora|aumenta|reduce|disminuye|fortalece|debilita)\b",
            r"\b(siempre que|cuando|si)\b.*\b(entonces|por lo tanto|en consecuencia)\b",
        ]
        densidad = 0.0
        for patron in patrones:
            matches = len(re.findall(patron, texto.lower(), re.IGNORECASE))
            densidad += matches * 0.2
        return min(1.0, densidad / max(1, len(texto.split()) / 100))

    def _build_result(self, texto: str, pagina: int, sim: float, rel: float, dens: float) -> Dict[str, Any]:
        return {
            "texto": texto,
            "pagina": pagina,
            "similitud_semantica": float(sim),
            "relevancia_conceptual": rel,
            "densidad_causal": dens,
            "score_final": sim * 0.5 + rel * 0.3 + dens * 0.2,
            "hash_segmento": hashlib.md5(texto.encode("utf-8")).hexdigest()[:8],
            "timestamp_extraccion": datetime.now().isoformat(),
        }

    def buscar_evidencia_causal(self, query: str, conceptos_clave: List[str], top_k: int = 5, umbral_certeza: float = 0.75) -> List[Dict[str, Any]]:
        if (self.embeddings_doc is None) or (self.embeddings_doc.numel() == 0):
            self.logger.warning("⚠️ Embeddings no disponibles. Se usa fallback en tiempo real.")
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)
        try:
            q_emb = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(q_emb, self.embeddings_doc)[0]
            resultados = []
            indices_top = torch.topk(sims, min(top_k * 2, len(self.textos_originales))).indices
            for idx in indices_top:
                idx_i = int(idx.item())
                if idx_i >= len(self.textos_originales):
                    continue
                texto = self.textos_originales[idx_i]
                # Página original
                pagina = None
                for p, t in self.documentos:
                    if t == texto:
                        pagina = p
                        break
                if pagina is None:
                    continue
                coincidencias = sum(1 for c in conceptos_clave if c.lower() in texto.lower())
                rel = coincidencias / max(1, len(conceptos_clave))
                dens = self._densidad_causal(texto)
                res = self._build_result(texto, pagina, sims[idx_i].item(), rel, dens)
                if res["score_final"] >= umbral_certeza:
                    resultados.append(res)
            return sorted(resultados, key=lambda x: x["score_final"], reverse=True)[:top_k]
        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda con embeddings precomputados: {e}")
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)

    def _buscar_evidencia_fallback(self, query: str, conceptos_clave: List[str], top_k: int = 5, umbral_certeza: float = 0.75) -> List[Dict[str, Any]]:
        if not self.textos_originales:
            self.logger.warning("⚠️ No hay textos para búsqueda")
            return []
        try:
            q_emb = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            d_emb = EMBEDDING_MODEL.encode(self.textos_originales, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(q_emb, d_emb)[0]
            resultados = []
            indices_top = torch.topk(sims, min(top_k * 2, len(self.textos_originales))).indices
            for idx in indices_top:
                idx_i = int(idx.item())
                texto = self.textos_originales[idx_i]
                pagina = None
                for p, t in self.documentos:
                    if t == texto:
                        pagina = p
                        break
                if pagina is None:
                    continue
                coincidencias = sum(1 for c in conceptos_clave if c.lower() in texto.lower())
                rel = coincidencias / max(1, len(conceptos_clave))
                dens = self._densidad_causal(texto)
                res = self._build_result(texto, pagina, sims[idx_i].item(), rel, dens)
                if res["score_final"] >= umbral_certeza:
                    resultados.append(res)
            return sorted(resultados, key=lambda x: x["score_final"], reverse=True)[:top_k]
        except Exception as e:
            self.logger.error(f"❌ Error en fallback de búsqueda: {e}")
            return []

    def buscar_segmentos_semanticos_global(self, queries: List[str], max_segmentos: int, batch_size: int = 32) -> List[Dict[str, Any]]:
        if self.embeddings_doc is None or self.embeddings_doc.numel() == 0:
            self.logger.warning("⚠️ Sin embeddings para búsqueda global")
            return []
        heap_global: List[Tuple[float, int, int, Dict[str, Any]]] = []
        try:
            for b in range(0, len(queries), batch_size):
                q_batch = queries[b:b + batch_size]
                q_embs = EMBEDDING_MODEL.encode(q_batch, convert_to_tensor=True)
                sims_batch = util.pytorch_cos_sim(q_embs, self.embeddings_doc)
                for qi_local, q in enumerate(q_batch):
                    qi_global = b + qi_local
                    sims = sims_batch[qi_local]
                    for doc_idx, s in enumerate(sims):
                        score = float(s.item())
                        if doc_idx < len(self.textos_originales):
                            texto_seg = self.textos_originales[doc_idx]
                            pagina = None
                            for p, t in self.documentos:
                                if t == texto_seg:
                                    pagina = p
                                    break
                            if pagina is None:
                                continue
                            datos = {
                                "texto": texto_seg,
                                "pagina": pagina,
                                "query": q,
                                "query_idx": qi_global,
                                "similitud_semantica": score,
                                "score_final": score,
                                "hash_segmento": hashlib.md5(texto_seg.encode("utf-8")).hexdigest()[:8],
                                "timestamp_extraccion": datetime.now().isoformat(),
                            }
                            if len(heap_global) < max_segmentos:
                                heapq.heappush(heap_global, (score, doc_idx, qi_global, datos))
                            elif score > heap_global[0][0]:
                                heapq.heappushpop(heap_global, (score, doc_idx, qi_global, datos))
            resultados: List[Dict[str, Any]] = []
            while heap_global:
                _, _, _, datos = heapq.heappop(heap_global)
                resultados.append(datos)
            resultados.sort(key=lambda x: x["score_final"], reverse=True)
            self.logger.info(f"✅ Búsqueda global: {len(resultados)} segmentos seleccionados")
            return resultados
        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda global: {e}")
            return []

    def extraer_variables_operativas(self, dimension: DimensionDecalogo) -> Dict[str, List]:
        variables = {"indicadores": [], "metas": [], "recursos": [], "responsables": [], "plazos": [], "riesgos": []}
        try:
            for e in dimension.eslabones:
                for indicador in e.indicadores:
                    res = self.buscar_evidencia_causal(
                        f"indicador {indicador} meta objetivo {dimension.nombre}",
                        [indicador, "meta", "objetivo", "línea base", "indicador"], top_k=3, umbral_certeza=0.7,
                    )
                    for r in res:
                        r["eslabon_origen"] = e.id
                        r["tipo_variable"] = "indicador"
                    variables["indicadores"].extend(res)
            res_recursos = self.buscar_evidencia_causal(
                f"presupuesto financiación recursos para {dimension.nombre}",
                ["presupuesto", "financiación", "recursos", "inversión", "asignación", "fondo", "subsidio", "transferencia", "cofinanciación", "contrapartida"],
                top_k=5, umbral_certeza=0.65,
            )
            for r in res_recursos:
                r["tipo_variable"] = "recurso"
            variables["recursos"].extend(res_recursos)
            res_resp = self.buscar_evidencia_causal(
                f"responsable encargado de {dimension.nombre}",
                ["responsable", "encargado", "lidera", "coordina", "gestiona"], top_k=3, umbral_certeza=0.6,
            )
            for r in res_resp:
                r["tipo_variable"] = "responsable"
            variables["responsables"].extend(res_resp)
            res_plazos = self.buscar_evidencia_causal(
                f"plazo fecha cronograma para {dimension.nombre}",
                ["plazo", "fecha", "cronograma", "tiempo", "duración", "inicio", "finalización"], top_k=3, umbral_certeza=0.6,
            )
            for r in res_plazos:
                r["tipo_variable"] = "plazo"
            variables["plazos"].extend(res_plazos)
            self.logger.info(f"✅ Extracción dimensión {dimension.id}: {sum(len(v) for v in variables.values())} variables")
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo variables (dim {dimension.id}): {e}")
        return variables

    def generar_matriz_trazabilidad(self, dimension: DimensionDecalogo) -> pd.DataFrame:
        try:
            variables = self.extraer_variables_operativas(dimension)
            data = []
            for tipo_variable, resultados in variables.items():
                for r in resultados:
                    data.append({
                        "dimension_id": dimension.id,
                        "dimension_nombre": dimension.nombre,
                        "tipo_variable": tipo_variable,
                        "texto_evidencia": (r.get("texto", "")[:200] + "...") if r.get("texto") else "",
                        "pagina": r.get("pagina", 0),
                        "score_confianza": r.get("score_final", 0.0),
                        "hash_evidencia": r.get("hash_segmento", "N/A"),
                        "timestamp": r.get("timestamp_extraccion", ""),
                    })
            return pd.DataFrame(data) if data else pd.DataFrame(
                columns=["dimension_id","dimension_nombre","tipo_variable","texto_evidencia","pagina","score_confianza","hash_evidencia","timestamp"])
        except Exception as e:
            self.logger.error(f"❌ Error generando matriz de trazabilidad: {e}")
            return pd.DataFrame()

# -------------------- Evaluación --------------------
@dataclass
class EvaluacionCausalIndustrial:
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
        for name, value in self.__dict__.items():
            if isinstance(value, float) and name not in ["evidencia_soporte", "brechas_criticas"]:
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"Campo {name} fuera de rango [0,1]: {value}")
    @property
    def puntaje_global(self) -> float:
        return (self.consistencia_logica * 0.25
                + self.identificabilidad_causal * 0.20
                + self.factibilidad_operativa * 0.20
                + self.certeza_probabilistica * 0.15
                + self.robustez_causal * 0.20)
    @property
    def nivel_certidumbre(self) -> str:
        p = self.puntaje_global
        if p >= 0.85: return "ALTA - Certidumbre sólida"
        if p >= 0.70: return "MEDIA - Certidumbre aceptable"
        if p >= 0.50: return "BAJA - Certidumbre limitada"
        return "MUY BAJA - Alta incertidumbre"
    @property
    def recomendacion_estrategica(self) -> str:
        if self.factibilidad_operativa < 0.6 and self.riesgos_implementacion:
            return "REQUIERE REDISEÑO OPERATIVO"
        if self.certeza_probabilistica < 0.7:
            return "REQUIERE MAYOR EVIDENCIA EMPÍRICA"
        if self.consistencia_logica < 0.7:
            return "REQUIERE FORTALECIMIENTO TEÓRICO"
        if len(self.riesgos_implementacion) > 3:
            return "REQUIERE PLAN DE MITIGACIÓN DE RIESGOS"
        return "IMPLEMENTACIÓN RECOMENDADA"

@dataclass
class ResultadoDimensionIndustrial:
    dimension: DimensionDecalogo
    evaluacion_causal: EvaluacionCausalIndustrial
    evidencia: Dict[str, List]
    brechas_identificadas: List[str]
    recomendaciones: List[str]
    matriz_trazabilidad: Optional[pd.DataFrame] = None
    timestamp_evaluacion: str = field(default_factory=lambda: datetime.now().isoformat())
    @property
    def puntaje_final(self) -> float:
        return self.evaluacion_causal.puntaje_global * 100
    @property
    def nivel_madurez(self) -> str:
        p = self.puntaje_final
        if p >= 85: return "NIVEL 5 - Optimizado"
        if p >= 70: return "NIVEL 4 - Gestionado cuantitativamente"
        if p >= 50: return "NIVEL 3 - Definido"
        if p >= 30: return "NIVEL 2 - Gestionado"
        return "NIVEL 1 - Inicial"
    def generar_reporte_tecnico(self) -> Dict[str, Any]:
        ev = self.evaluacion_causal
        return {
            "metadata": {
                "dimension_id": self.dimension.id,
                "dimension_nombre": self.dimension.nombre,
                "cluster": self.dimension.cluster,
                "timestamp": self.timestamp_evaluacion,
                "version_sistema": "8.1-industrial",
            },
            "evaluacion_causal": {
                "puntaje_global": ev.puntaje_global,
                "nivel_certidumbre": ev.nivel_certidumbre,
                "recomendacion_estrategica": ev.recomendacion_estrategica,
                "metricas_detalle": {
                    "consistencia_logica": ev.consistencia_logica,
                    "identificabilidad_causal": ev.identificabilidad_causal,
                    "factibilidad_operativa": ev.factibilidad_operativa,
                    "certeza_probabilistica": ev.certeza_probabilistica,
                    "robustez_causal": ev.robustez_causal,
                },
            },
            "diagnostico": {
                "brechas_criticas": len(self.brechas_identificadas),
                "riesgos_principales": ev.riesgos_implementacion[:5],
                "evidencia_disponible": sum(len(v) for v in self.evidencia.values()),
                "nivel_madurez": self.nivel_madurez,
            },
            "recomendaciones": self.recomendaciones[:10],
            "trazabilidad": (self.matriz_trazabilidad.to_dict() if self.matriz_trazabilidad is not None else {}),
        }

# -------------------- Loader PDF con segmentación avanzada --------------------
class PDFLoaderIndustrial:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.paginas: List[str] = []
        self.segmentos: List[Tuple[int, str]] = []
        self.nombre_plan = file_path.stem
        self.logger = logging.getLogger(f"PDFLoader_{self.nombre_plan}")
        self.hash_documento = ""
        self.metadata: Dict[str, Any] = {}
        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("❌ pdfplumber no está instalado. Instale: pip install pdfplumber")

    def calcular_hash_documento(self) -> str:
        if self.paginas:
            contenido = " ".join(self.paginas)
            return hashlib.sha256(contenido.encode("utf-8")).hexdigest()
        return ""

    def extraer_metadata_pdf(self) -> Dict[str, Any]:
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
            self.hash_documento = self.calcular_hash_documento()
            self.metadata = self.extraer_metadata_pdf()
            self.logger.info(f"✅ Documento cargado: {len(self.paginas)} páginas - Hash: {self.hash_documento[:8]}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error crítico cargando PDF {self.file_path}: {e}")
            return False

    def _tiene_contenido_sustancial(self, doc: spacy.tokens.Doc) -> bool:
        tiene_sustantivos = any(t.pos_ in ["NOUN", "PROPN"] for t in doc)
        tiene_verbos = any(t.pos_ == "VERB" for t in doc)
        return tiene_sustantivos and tiene_verbos and len(doc) >= 5

    def _detectar_cambio_tematico(self, buffer: List[str]) -> bool:
        if len(buffer) < 2:
            return False
        try:
            embeddings = EMBEDDING_MODEL.encode(buffer[-2:], convert_to_tensor=True)
            if len(embeddings) == 2:
                similitud = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                return similitud < 0.6
        except Exception:
            pass
        return False

    def segmentar(self) -> bool:
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
                    buffer: List[str] = []
                    for sentencia in doc.sents:
                        texto = sentencia.text.strip()
                        if len(texto) >= 20 and self._tiene_contenido_sustancial(sentencia):
                            buffer.append(texto)
                            if len(buffer) >= 3 or (len(buffer) >= 2 and self._detectar_cambio_tematico(buffer)):
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

# -------------------- Sistema maestro --------------------
class SistemaEvaluacionIndustrial:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.loader = PDFLoaderIndustrial(pdf_path)
        self.extractor: Optional[ExtractorEvidenciaIndustrial] = None
        self.ontologia = OntologiaPoliticas.cargar_estandar()
        self.logger = logging.getLogger(f"EvaluacionIndustrial_{pdf_path.stem}")
        self.hash_evaluacion = ""
        self.metadata_plan: Dict[str, Any] = {}

    def cargar_y_procesar(self) -> bool:
        self.logger.info(f"🔄 Procesando: {self.pdf_path.name}")
        if not self.loader.cargar():
            self.logger.error("❌ Falló la carga del documento")
            return False
        if not self.loader.segmentar():
            self.logger.error("❌ Falló la segmentación del documento")
            return False
        try:
            self.extractor = ExtractorEvidenciaIndustrial(self.loader.segmentos, self.pdf_path.stem)
            self.metadata_plan = self.loader.metadata
            self.hash_evaluacion = hashlib.sha256(
                f"{self.loader.hash_documento}_{datetime.now().isoformat()}".encode("utf-8")
            ).hexdigest()
            self.logger.info(f"✅ Sistema listo — Hash: {self.hash_evaluacion[:8]}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error inicializando extractor: {e}")
            return False

    def _calcular_factibilidad_industrial(self, dimension: DimensionDecalogo, evidencia: Dict) -> float:
        factores = []
        factores.append(0.9 if evidencia.get("recursos", []) else 0.2)
        indicadores_encontrados = len(evidencia.get("indicadores", []))
        indicadores_requeridos = len(dimension.eslabones)
        if indicadores_encontrados >= indicadores_requeridos:
            factores.append(0.95)
        elif indicadores_encontrados > 0:
            factores.append(0.6 + (0.35 * indicadores_encontrados / indicadores_requeridos))
        else:
            factores.append(0.1)
        rp = len(evidencia.get("responsables", [])) + len(evidencia.get("plazos", []))
        factores.append(0.85 if rp >= 2 else (0.5 if rp == 1 else 0.2))
        return sum(factores) / len(factores) if factores else 0.3

    def _identificar_brechas_industrial(self, dimension: DimensionDecalogo, evidencia: Dict) -> List[str]:
        brechas: List[str] = []
        for e in dimension.eslabones:
            tiene = any(ind.lower() in ev.get("texto", "").lower()
                        for ind in e.indicadores for ev in evidencia.get("indicadores", []))
            if not tiene:
                brechas.append(f"🔴 BRECHA CRÍTICA: Falta especificación de indicadores en {e.id} ({e.tipo.value})")
        if not evidencia.get("recursos", []):
            brechas.append("🔴 BRECHA CRÍTICA: No se encontró especificación presupuestal/recursos")
        if not evidencia.get("responsables", []):
            brechas.append("🟠 BRECHA IMPORTANTE: Sin responsables claros")
        if not evidencia.get("plazos", []):
            brechas.append("🟠 BRECHA IMPORTANTE: Sin plazos/cronogramas definidos")
        if len(dimension.teoria_cambio.supuestos_causales) > 5:
            brechas.append("🟡 BRECHA MODERADA: Complejidad causal alta")
        return brechas

    def _identificar_riesgos_industrial(self, dimension: DimensionDecalogo, evidencia: Dict) -> List[str]:
        riesgos: List[str] = []
        if not evidencia.get("recursos", []):
            riesgos.append("🔴 ALTO: Falta de especificación presupuestal")
        if len(dimension.teoria_cambio.supuestos_causales) > 4:
            riesgos.append("🟠 MEDIO-ALTO: Complejidad causal elevada")
        if len(evidencia.get("indicadores", [])) < len(dimension.eslabones) * 0.5:
            riesgos.append("🔴 ALTO: Cobertura insuficiente de indicadores")
        for e in dimension.eslabones:
            if e.ventana_temporal[1] - e.ventana_temporal[0] > 36:
                riesgos.append(f"🟠 MEDIO: Ventana temporal amplia en {e.id}")
                break
        if not evidencia.get("responsables", []):
            riesgos.append("🟠 MEDIO: Ausencia de responsables definidos")
        return riesgos

    def _generar_recomendaciones_industrial(self, dimension: DimensionDecalogo, ev: EvaluacionCausalIndustrial, brechas: List[str]) -> List[str]:
        recs: List[str] = []
        if ev.consistencia_logica < 0.7: recs.append("🔧 FORTALECER coherencia lógica de la teoría de cambio")
        if ev.factibilidad_operativa < 0.6: recs.append("🔧 FORTALECER mecanismos de implementación")
        if ev.certeza_probabilistica < 0.7: recs.append("📊 EVIDENCIA: Incrementar soporte empírico")
        if ev.robustez_causal < 0.6: recs.append("🧩 SIMPLIFICAR el modelo causal")
        for b in brechas:
            if "BRECHA CRÍTICA" in b: recs.append(f"🚨 ACCIÓN INMEDIATA: {b.split(': ',1)[1]}")
            elif "BRECHA IMPORTANTE" in b: recs.append(f"⚠️ PRIORIDAD ALTA: {b.split(': ',1)[1]}")
            elif "BRECHA MODERADA" in b: recs.append(f"🔧 MEJORA CONTINUA: {b.split(': ',1)[1]}")
        if ev.evidencia_soporte < 5: recs.append("📚 INVESTIGACIÓN: Levantar línea base robusta")
        if len(ev.riesgos_implementacion) > 3: recs.append("🛡️ GESTIÓN: Plan integral de mitigación de riesgos")
        recs.append("📈 MONITOREO: Indicadores SMART y tablero M&E")
        return recs[:15]

    def _evaluar_coherencia_causal_industrial(self, dimension: DimensionDecalogo, evidencia: Dict) -> EvaluacionCausalIndustrial:
        try:
            consistencia = dimension.evaluar_coherencia_causal()
            identificabilidad = 1.0 if dimension.teoria_cambio.verificar_identificabilidad() else 0.3
            factibilidad = self._calcular_factibilidad_industrial(dimension, evidencia)
            riesgos = self._identificar_riesgos_industrial(dimension, evidencia)
            G = dimension.teoria_cambio.construir_grafo_causal()
            certezas: List[float] = []
            for _ in range(200):
                try:
                    if len(G.nodes) > 2:
                        nb = ["insumos", "impactos"]
                        nm = [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]
                        nr = [n for n in G.nodes if G.nodes[n].get("tipo") == "resultado"]
                        nodos = nb.copy()
                        if nm: nodos.extend(np.random.choice(nm, size=min(3, len(nm)), replace=False).tolist())
                        if nr: nodos.extend(np.random.choice(nr, size=min(2, len(nr)), replace=False).tolist())
                        sub = G.subgraph(nodos)
                        if nx.is_directed_acyclic_graph(sub) and nx.has_path(sub, "insumos", "impactos") and len(sub.edges) >= 2:
                            certezas.append(1.0)
                        else:
                            certezas.append(0.4)
                    else:
                        certezas.append(0.3)
                except Exception:
                    certezas.append(0.3)
            certeza = float(np.mean(certezas)) if certezas else 0.5
            robustez = dimension.teoria_cambio.calcular_coeficiente_causal()
            evidencia_soporte = sum(len(v) for v in evidencia.values())
            if evidencia_soporte == 0:
                certeza *= 0.5
                factibilidad *= 0.5
            if certeza < 0.7: riesgos.append("⚠️ Baja certeza causal: fortalecer marco teórico")
            if robustez < 0.6: riesgos.append("⚠️ Baja robustez causal: simplificar relaciones")
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
            self.logger.error(f"❌ Error evaluación causal industrial: {e}")
            return EvaluacionCausalIndustrial(
                consistencia_logica=0.3, identificabilidad_causal=0.3, factibilidad_operativa=0.3,
                certeza_probabilistica=0.3, robustez_causal=0.3,
                riesgos_implementacion=[f"Error en evaluación causal: {str(e)}"],
                supuestos_criticos=[], evidencia_soporte=0, brechas_criticas=5)

    def evaluar_dimension(self, dimension: DimensionDecalogo) -> ResultadoDimensionIndustrial:
        if not self.extractor:
            raise ValueError("Extractor no inicializado")
        try:
            self.logger.info(f"🔍 Evaluando dimensión {dimension.id}: {dimension.nombre}")
            evidencia = self.extractor.extraer_variables_operativas(dimension)
            matriz_trazabilidad = self.extractor.generar_matriz_trazabilidad(dimension)
            evaluacion_causal = self._evaluar_coherencia_causal_industrial(dimension, evidencia)
            brechas = self._identificar_brechas_industrial(dimension, evidencia)
            recomendaciones = self._generar_recomendaciones_industrial(dimension, evaluacion_causal, brechas)
            # Integración opcional con evaluador externo
            try:
                from Decatalogo_evaluador import integrar_evaluador_decatalogo  # noqa: F401
                # Se respeta invocación; si existe, el usuario podrá usarla. Aquí no se fuerza.
            except Exception as e:
                self.logger.warning(f"⚠️ Evaluador externo no disponible: {e}")
            resultado = ResultadoDimensionIndustrial(
                dimension=dimension,
                evaluacion_causal=evaluacion_causal,
                evidencia=evidencia,
                brechas_identificadas=brechas,
                recomendaciones=recomendaciones,
                matriz_trazabilidad=matriz_trazabilidad,
                timestamp_evaluacion=datetime.now().isoformat(),
            )
            self.logger.info(f"✅ Resultado dimensión {dimension.id}: {resultado.puntaje_final:.1f}/100")
            return resultado
        except Exception as e:
            self.logger.error(f"❌ Error crítico evaluando dim {dimension.id}: {e}")
            evaluacion_fallback = EvaluacionCausalIndustrial(
                consistencia_logica=0.3, identificabilidad_causal=0.3, factibilidad_operativa=0.3,
                certeza_probabilistica=0.3, robustez_causal=0.3,
                riesgos_implementacion=[f"Error en evaluación: {str(e)}"],
                supuestos_criticos=[], evidencia_soporte=0, brechas_criticas=5)
            return ResultadoDimensionIndustrial(
                dimension=dimension, evaluacion_causal=evaluacion_fallback,
                evidencia={}, brechas_identificadas=[f"Error crítico: {str(e)}"],
                recomendaciones=["Revisión manual urgente"], timestamp_evaluacion=datetime.now().isoformat())

    def _generar_recomendacion_global(self, resultados: List[ResultadoDimensionIndustrial]) -> str:
        puntajes = [r.puntaje_final for r in resultados]
        if not puntajes: return "NO APLICA"
        promedio = statistics.mean(puntajes)
        deficientes = len([p for p in puntajes if p < 70])
        total = len(puntajes)
        if promedio >= 85: return "IMPLEMENTACIÓN INTEGRAL RECOMENDADA"
        if promedio >= 70 and deficientes <= total * 0.3: return "IMPLEMENTACIÓN SELECTIVA RECOMENDADA"
        if promedio >= 50: return "REDISEÑO PARCIAL REQUERIDO"
        return "REDISEÑO INTEGRAL REQUERIDO"

    def _identificar_riesgos_sistemicos(self, resultados: List[ResultadoDimensionIndustrial]) -> List[str]:
        riesgos: List[str] = []
        puntajes = [r.puntaje_final for r in resultados]
        if len(puntajes) > 1 and statistics.stdev(puntajes) > 25:
            riesgos.append("🔴 DESCOHERENCIA: Alta variabilidad entre dimensiones")
        evidencia_total = sum(r.evaluacion_causal.evidencia_soporte for r in resultados)
        if evidencia_total < len(resultados) * 3:
            riesgos.append("🟠 DÉFICIT DE EVIDENCIA: Soporte empírico bajo")
        riesgos_criticos = sum(len(r.evaluacion_causal.riesgos_implementacion) for r in resultados)
        if riesgos_criticos > len(resultados) * 3:
            riesgos.append("🔴 SOBRECARGA DE RIESGOS")
        certezas_bajas = sum(1 for r in resultados if r.evaluacion_causal.certeza_probabilistica < 0.6)
        if certezas_bajas > len(resultados) * 0.4:
            riesgos.append("🟠 INCERTIDUMBRE SISTÉMICA")
        return riesgos if riesgos else ["✅ SIN RIESGOS SISTÉMICOS"]

    def generar_reporte_tecnico_completo(self, resultados: List[ResultadoDimensionIndustrial]) -> Dict[str, Any]:
        try:
            puntajes = [r.puntaje_final for r in resultados]
            niveles = [r.nivel_madurez for r in resultados]
            certs = [r.evaluacion_causal.nivel_certidumbre for r in resultados]
            analisis_agregado = {
                "puntaje_global_promedio": statistics.mean(puntajes) if puntajes else 0,
                "desviacion_estandar": statistics.stdev(puntajes) if len(puntajes) > 1 else 0,
                "dimensiones_evaluadas": len(resultados),
                "dimensiones_excelentes": len([p for p in puntajes if p >= 85]),
                "dimensiones_aceptables": len([p for p in puntajes if 70 <= p < 85]),
                "dimensiones_deficientes": len([p for p in puntajes if p < 70]),
                "nivel_madurez_predominante": max(set(niveles), key=niveles.count) if niveles else "N/A",
                "certidumbre_predominante": max(set(certs), key=certs.count) if certs else "N/A",
                "recomendacion_estrategica_global": self._generar_recomendacion_global(resultados),
                "riesgos_sistemicos": self._identificar_riesgos_sistemicos(resultados),
            }
            resultados_detalle = [r.generar_reporte_tecnico() for r in resultados]
            return {
                "metadata": {
                    "nombre_plan": self.pdf_path.stem,
                    "hash_evaluacion": self.hash_evaluacion,
                    "fecha_evaluacion": datetime.now().isoformat(),
                    "version_sistema": "8.1-industrial",
                    "total_dimensiones": len(DECALOGO_INDUSTRIAL),
                },
                "analisis_agregado": analisis_agregado,
                "resultados_por_dimension": resultados_detalle,
                "timestamp_generacion": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"❌ Error generando reporte completo: {e}")
            return {"metadata": {"nombre_plan": self.pdf_path.stem, "error": str(e)},
                    "analisis_agregado": {}, "resultados_por_dimension": []}

# -------------------- Reporte Markdown --------------------
class GeneradorReporteIndustrial:
    @staticmethod
    def generar_reporte_markdown(resultados: List[ResultadoDimensionIndustrial], nombre_plan: str, metadata: Dict | None = None) -> str:
        rep: List[str] = []
        rep.append("# 🏭 EVALUACIÓN INDUSTRIAL DE POLÍTICAS PÚBLICAS")
        rep.append(f"## 📄 Plan de Desarrollo Municipal: {nombre_plan}")
        rep.append("### 🎯 Análisis Multinivel con Enfoque Causal y Certificación de Rigor")
        rep.append(f"### 📊 Fecha de evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if metadata and metadata.get("hash_evaluacion"):
            rep.append(f"### 🔐 Hash de evaluación: {metadata.get('hash_evaluacion','')[:12]}...")
        rep.append("")
        puntajes = [r.puntaje_final for r in resultados]
        if puntajes:
            promedio = statistics.mean(puntajes)
            desviacion = statistics.stdev(puntajes) if len(puntajes) > 1 else 0
            excelentes = len([p for p in puntajes if p >= 85])
            deficientes = len([p for p in puntajes if p < 70])
            rep.append("## 📈 RESUMEN EJECUTIVO INDUSTRIAL")
            rep.append(f"**🎯 Puntaje Global:** {promedio:.1f}/100")
            rep.append(f"**📊 Desviación Estándar:** {desviacion:.1f}")
            rep.append(f"**🏆 Dimensiones Excelentes (≥85):** {excelentes}/{len(resultados)}")
            rep.append(f"**⚠️ Dimensiones Deficientes (<70):** {deficientes}/{len(resultados)}")
            niveles = [r.nivel_madurez for r in resultados]
            nivel_pred = max(set(niveles), key=niveles.count) if niveles else "N/A"
            rep.append(f"**🏭 Nivel de Madurez Predominante:** {nivel_pred}")
            if promedio >= 85:
                rec_global, emoji = "IMPLEMENTACIÓN INTEGRAL RECOMENDADA ✅", "🚀"
            elif promedio >= 70:
                rec_global, emoji = "IMPLEMENTACIÓN SELECTIVA CON MEJORAS ⚠️", "🔧"
            elif promedio >= 50:
                rec_global, emoji = "REDISEÑO PARCIAL REQUERIDO 🚨", "🛠️"
            else:
                rec_global, emoji = "REDISEÑO INTEGRAL URGENTE ❌", "🆘"
            rep.append(f"**{emoji} Recomendación Estratégica Global:** {rec_global}")
            rep.append("")
        for r in resultados:
            rep.append(f"## 🔍 DIMENSIÓN {r.dimension.id}: {r.dimension.nombre}")
            rep.append(f"### 🏷️ Cluster: {r.dimension.cluster}")
            rep.append(f"### 📊 Puntaje: {r.puntaje_final:.1f}/100")
            rep.append(f"### 🏭 Nivel de Madurez: {r.nivel_madurez}")
            rep.append(f"### 🎯 Certidumbre: {r.evaluacion_causal.nivel_certidumbre}")
            rep.append(f"### 💡 Recomendación Estratégica: {r.evaluacion_causal.recomendacion_estrategica}")
            rep.append("")
            rep.append("### 🧩 TEORÍA DE CAMBIO")
            rep.append("**Supuestos causales:**")
            for s in r.dimension.teoria_cambio.supuestos_causales:
                rep.append(f"- {s}")
            rep.append("")
            ev = r.evaluacion_causal
            rep.append("### 📊 EVALUACIÓN CAUSAL")
            rep.append(f"- **Consistencia lógica:** {ev.consistencia_logica:.3f}")
            rep.append(f"- **Identificabilidad causal:** {ev.identificabilidad_causal:.3f}")
            rep.append(f"- **Factibilidad operativa:** {ev.factibilidad_operativa:.3f}")
            rep.append(f"- **Certeza probabilística:** {ev.certeza_probabilistica:.3f}")
            rep.append(f"- **Robustez causal:** {ev.robustez_causal:.3f}")
            rep.append(f"- **Evidencia de soporte:** {ev.evidencia_soporte} elementos")
            rep.append(f"- **Brechas críticas:** {ev.brechas_criticas}")
            rep.append("")
            if ev.riesgos_implementacion:
                rep.append("### ⚠️ RIESGOS DE IMPLEMENTACIÓN")
                for riesgo in ev.riesgos_implementacion:
                    rep.append(f"- {risco if (risco:=riesgo) else riesgo}")
                rep.append("")
            if r.brechas_identificadas:
                rep.append("### 🚨 BRECHAS IDENTIFICADAS")
                for b in r.brechas_identificadas:
                    rep.append(f"- {b}")
                rep.append("")
            if r.recomendaciones:
                rep.append("### 💡 RECOMENDACIONES")
                for rec in r.recomendaciones[:8]:
                    rep.append(f"- {rec}")
                rep.append("")
            evidencia_total = sum(len(v) for v in r.evidencia.values())
            if evidencia_total > 0:
                rep.append("### 📚 EVIDENCIA CLAVE")
                for tipo, evids in r.evidencia.items():
                    if not evids: continue
                    rep.append(f"- **{tipo.upper()}**")
                    for e in evids[:3]:
                        frag = (e.get("texto","")[:140] + "...") if e.get("texto") else ""
                        rep.append(f"  - p.{e.get('pagina','?')} — {frag} (score={e.get('score_final',0):.2f})")
                rep.append("")
        return "\n".join(rep)

# -------------------- CLI --------------------
def _install_signal_handlers():
    def handle_sigterm(signum, frame):
        LOGGER.warning("⚠️ Señal de terminación recibida. Saliendo con gracia.")
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(130))

def main():
    parser = argparse.ArgumentParser(description="Evaluación Industrial de Planes de Desarrollo Municipal")
    parser.add_argument("--pdf", required=True, type=str, help="Ruta al PDF del Plan")
    parser.add_argument("--salida_json", type=str, default="reporte_industrial.json", help="Archivo JSON de salida")
    parser.add_argument("--salida_md", type=str, default="reporte_industrial.md", help="Archivo Markdown de salida")
    parser.add_argument("--top_dim", type=int, default=10, help="Número de dimensiones a procesar (1-10)")
    add_device_args(parser)
    args = parser.parse_args()

    _install_signal_handlers()
    atexit.register(lambda: LOGGER.info("🧹 Finalizando ejecución"))

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        LOGGER.error(f"❌ PDF no encontrado: {pdf_path}")
        sys.exit(2)

    _ = configure_device_from_args(args)  # Se respeta wiring de device

    sistema = SistemaEvaluacionIndustrial(pdf_path)
    if not sistema.cargar_y_procesar():
        LOGGER.error("❌ Proceso abortado por fallo de carga/segmentación")
        sys.exit(3)

    contexto = obtener_decalogo_contexto()
    dimensiones = [contexto.dimensiones_por_id[i] for i in range(1, min(10, args.top_dim) + 1)]
    resultados: List[ResultadoDimensionIndustrial] = []
    for d in dimensiones:
        resultados.append(sistema.evaluar_dimension(d))

    reporte_json = sistema.generar_reporte_tecnico_completo(resultados)
    try:
        with open(args.salida_json, "w", encoding="utf-8") as f:
            json.dump(reporte_json, f, indent=2, ensure_ascii=False)
        LOGGER.info(f"💾 Reporte JSON guardado: {args.salida_json}")
    except Exception as e:
        LOGGER.error(f"❌ Error guardando JSON: {e}")

    try:
        md = GeneradorReporteIndustrial.generar_reporte_markdown(resultados, sistema.pdf_path.stem,
                                                                 {"hash_evaluacion": sistema.hash_evaluacion})
        with open(args.salida_md, "w", encoding="utf-8") as f:
            f.write(md)
        LOGGER.info(f"📝 Reporte Markdown guardado: {args.salida_md}")
    except Exception as e:
        LOGGER.error(f"❌ Error guardando Markdown: {e}")

if __name__ == "__main__":
    main()
