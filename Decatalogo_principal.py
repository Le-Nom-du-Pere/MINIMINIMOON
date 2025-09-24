#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Integral de Evaluación de Cadenas de Valor en Planes de Desarrollo Municipal
==================================================================================

Comprehensive evaluation framework for value chains in municipal development plans
based on Institutional Analysis and Development (IAD) and Theory of Change (ToC)
methodologies with advanced causal analysis and parallel processing capabilities.

This module provides a complete framework for evaluating public policy effectiveness
through sophisticated causal modeling, institutional analysis, and quantitative
assessment methods. It integrates multiple theoretical frameworks to provide
robust policy evaluation capabilities.

Version: 8.1 — Marco Teórico-Institucional con Análisis Causal Multinivel, Batch Processing, 
Certificación de Rigor y Selección Global Top-K con Heap

Key Features:
    - Institutional Analysis and Development (IAD) framework integration
    - Theory of Change (ToC) modeling with causal graph construction
    - Multilevel causal analysis with econometric validation
    - Industrial-grade batch processing and parallel execution
    - Comprehensive policy ontology with ODS indicator mapping
    - Advanced embedding models for semantic analysis
    - Statistical validation and reproducibility certification
    - PDF document processing and analysis capabilities

Classes:
    NivelAnalisis: Enumeration of analysis levels (MACRO, MESO, MICRO)
    TipoCadenaValor: Value chain types (INSUMOS, PROCESOS, PRODUCTOS, RESULTADOS, IMPACTOS)
    TeoriaCambio: Formal theory of change representation with causal DAG
    EslabonCadena: Industrial value chain link model
    OntologiaPoliticas: Industrial policy ontology system
    DimensionDecalogo: Complete dimension specification with validation

Functions:
    cargar_decalogo_industrial: Load and validate industrial decalog from JSON
    generar_template_dimension: Generate structured dimension template
    _cargar_indicadores_ods: Load ODS indicators with fallback system

Constants:
    LOGGER: Industrial-grade logger instance
    NLP: SpaCy language model (Spanish)
    EMBEDDING_MODEL: Sentence transformer model for semantic analysis

Raises:
    SystemExit: When critical models fail to load or validation errors occur
    ValueError: When configuration parameters are invalid
    OSError: When model files are not accessible

Example:
    >>> # Load decalog and perform evaluation
    >>> decalogo = cargar_decalogo_industrial()
    >>> ontologia = OntologiaPoliticas.cargar_estandar()
    >>> # Process evaluation with comprehensive analysis
    
Note:
    Requires Python 3.11+ and extensive dependencies including spaCy models,
    sentence transformers, and scientific computing libraries. All operations
    are designed for production environments with comprehensive error handling.

Author: Dr. en Políticas Públicas
Focus: Structural evaluation with policy econometrics, causal mining and industrial parallel processing
"""
import argparse
import atexit
import argparse
import hashlib
import heapq
import json
import logging
import os
import random
import re
import signal
import statistics
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

from device_config import add_device_args, configure_device_from_args, get_device_config

import numpy as np
import pandas as pd
import torch
from text_truncation_logger import (
    log_info_with_text, log_warning_with_text, log_error_with_text, 
    log_debug_with_text, truncate_text_for_log, get_truncation_logger
)

assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

# -------------------- CONFIGURACIÓN ACADÉMICA INDUSTRIAL --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"evaluacion_politicas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    ]
)
LOGGER = logging.getLogger("EvaluacionPoliticasPublicasIndustrial")

# -------------------- MODELOS AVANZADOS CON FALLBACK INDUSTRIAL --------------------
try:
    # Intentar cargar el modelo grande primero, con fallback al pequeño
    try:
        NLP = spacy.load("es_core_news_lg")
        LOGGER.info("✅ Modelo SpaCy large cargado exitosamente")
    except OSError:
        LOGGER.warning("⚠️  Modelo SpaCy large no disponible, usando small")
        NLP = spacy.load("es_core_news_sm")
        LOGGER.info("✅ Modelo SpaCy small cargado exitosamente")
except OSError as e:
    LOGGER.error(f"❌ Error crítico cargando modelo SpaCy: {e}")
    raise SystemExit("Modelo SpaCy no disponible. Ejecute: python -m spacy download es_core_news_sm")

try:
    from device_config import to_device
    EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    EMBEDDING_MODEL = to_device(EMBEDDING_MODEL)
    log_info_with_text(LOGGER, "✅ Modelo de embeddings cargado exitosamente")
    log_info_with_text(LOGGER, f"✅ Modelo configurado en dispositivo: {get_device_config().get_device()}")
except Exception as e:
    log_error_with_text(LOGGER, f"❌ Error crítico cargando modelo de embeddings: {e}")
    raise SystemExit(f"Error cargando modelo de embeddings: {e}")


# ==================== MARCO TEÓRICO-INSTITUCIONAL INDUSTRIAL ====================
class NivelAnalisis(Enum):
    """
    Analysis levels for institutional framework evaluation.
    
    Defines the three primary levels of institutional analysis following
    the IAD (Institutional Analysis and Development) framework.
    
    Attributes:
        MACRO: Institutional level - Constitutional and policy framework
        MESO: Organizational level - Collective choice and coordination
        MICRO: Operational level - Day-to-day operational activities
    """
    MACRO = "Institucional"
    MESO = "Organizacional"
    MICRO = "Operacional"


class TipoCadenaValor(Enum):
    """
    Value chain types for comprehensive policy evaluation.
    
    Represents the complete value chain from inputs to final impacts,
    following international development evaluation standards.
    
    Attributes:
        INSUMOS: Financial, human, and physical resources
        PROCESOS: Institutional transformation processes
        PRODUCTOS: Deliverable goods and services
        RESULTADOS: Behavioral and institutional changes
        IMPACTOS: Welfare and human development outcomes
    """
    INSUMOS = "Recursos financieros, humanos y físicos"
    PROCESOS = "Transformación institucional"
    PRODUCTOS = "Bienes/servicios entregables"
    RESULTADOS = "Cambios conductuales/institucionales"
    IMPACTOS = "Bienestar y desarrollo humano"


@dataclass(frozen=True)
class TeoriaCambio:
    """
    Formal theory of change representation as causal DAG with mathematical verification.
    
    Implements a rigorous theory of change model using directed acyclic graphs (DAG)
    for causal inference and identification following Pearl (2009) methodology.
    
    Args:
        supuestos_causales (List[str]): List of causal assumptions
        mediadores (Dict[str, List[str]]): Mediator variables by category
        resultados_intermedios (List[str]): Intermediate outcome variables
        precondiciones (List[str]): Required preconditions for validity
        
    Methods:
        verificar_identificabilidad: Verify identification conditions per Pearl (2009)
        construir_grafo_causal: Build causal graph for path analysis and d-separation
        calcular_coeficiente_causal: Calculate causal robustness coefficient
        
    Example:
        >>> teoria = TeoriaCambio(
        ...     supuestos_causales=["X → Y", "Y → Z"],
        ...     mediadores={"institucionales": ["gobierno", "organizaciones"]},
        ...     resultados_intermedios=["mejora_servicios"],
        ...     precondiciones=["recursos_disponibles"]
        ... )
        >>> if teoria.verificar_identificabilidad():
        ...     grafo = teoria.construir_grafo_causal()
        
    Note:
        Immutable dataclass ensuring theoretical consistency and reproducibility.
        All causal relationships must satisfy identification conditions.

    """
    supuestos_causales: List[str]
    mediadores: Dict[str, List[str]]
    resultados_intermedios: List[str]
    precondiciones: List[str]

    def verificar_identificabilidad(self) -> bool:
        """
        Verify identification conditions according to Pearl (2009).
        
        Checks that the theory of change satisfies basic identification
        requirements for causal inference validity.
        
        Returns:
            bool: True if identification conditions are met

        """
        return len(self.supuestos_causales) > 0 and len(self.mediadores) > 0 and len(self.resultados_intermedios) > 0

    def construir_grafo_causal(self) -> nx.DiGraph:
        """
        Build causal graph for path analysis and d-separation.
        
        Creates a NetworkX directed graph representing the causal structure
        for advanced path analysis and causal identification testing.
        
        Returns:
            nx.DiGraph: Directed graph with causal relationships
            
        Raises:
            ValueError: If graph construction fails due to invalid structure

        """
        G = nx.DiGraph()
        # Nodos base obligatorios
        G.add_node("insumos", tipo="nodo_base")
        G.add_node("impactos", tipo="nodo_base")

        # Construcción jerárquica del grafo
        for tipo_mediador, lista_mediadores in self.mediadores.items():
            for mediador in lista_mediadores:
                G.add_node(mediador, tipo="mediador", categoria=tipo_mediador)
                G.add_edge("insumos", mediador, weight=1.0, tipo="causal")

                # Conectar mediadores con resultados intermedios
                for resultado in self.resultados_intermedios:
                    G.add_node(resultado, tipo="resultado")
                    G.add_edge(mediador, resultado, weight=0.8, tipo="causal")
                    G.add_edge(resultado, "impactos", weight=0.9, tipo="causal")

        return G

    def calcular_coeficiente_causal(self) -> float:
        """
        Calculate causal robustness coefficient based on connectivity and paths.
        
        Computes a robustness measure for the causal model based on the ratio
        of valid causal paths to total possible paths in the graph structure.
        
        Returns:
            float: Causal robustness coefficient (0.0-1.0)
                  Higher values indicate stronger causal structure
                  
        Note:
            Returns 0.3 for minimal graphs, 0.5 as fallback for computation errors.

        """
        G = self.construir_grafo_causal()
        if len(G.nodes) < 3:
            return 0.3

        try:
            # Calcular robustez como promedio de paths válidos
            paths_validos = 0
            total_paths = 0

            for mediador in [n for n in G.nodes if G.nodes[n].get('tipo') == 'mediador']:
                for resultado in [n for n in G.nodes if G.nodes[n].get('tipo') == 'resultado']:
                    if nx.has_path(G, mediador, resultado) and nx.has_path(G, resultado, "impactos"):
                        paths_validos += 1
                    total_paths += 1

            return paths_validos / max(1, total_paths) if total_paths > 0 else 0.5
        except Exception:
            return 0.5


@dataclass(frozen=True)
class EslabonCadena:
    """
    Industrial value chain link model with quantitative metrics.
    
    Represents a single link in the value chain with comprehensive
    performance metrics, temporal constraints, and validation logic.
    
    Args:
        id (str): Unique identifier for the chain link
        tipo (TipoCadenaValor): Type of value chain component
        indicadores (List[str]): Performance indicators list
        capacidades_requeridas (List[str]): Required institutional capacities
        puntos_criticos (List[str]): Critical success factors
        ventana_temporal (Tuple[int, int]): Temporal window (min-max months)
        kpi_ponderacion (float, optional): KPI weighting factor. Defaults to 1.0.
        
    Methods:
        calcular_lead_time: Calculate expected lead time with confidence interval
        generar_hash: Generate unique hash for industrial traceability
        
    Raises:
        ValueError: When KPI weighting is outside valid range or temporal window invalid
        
    Example:
        >>> eslabon = EslabonCadena(
        ...     id="proc_1",
        ...     tipo=TipoCadenaValor.PROCESOS,
        ...     indicadores=["eficiencia", "calidad"],
        ...     capacidades_requeridas=["personal_capacitado"],
        ...     puntos_criticos=["coordinacion"],
        ...     ventana_temporal=(6, 12),
        ...     kpi_ponderacion=1.2
        ... )
        >>> lead_time = eslabon.calcular_lead_time()
        
    Note:
        Immutable dataclass with industrial validation for production environments.
        All temporal and performance constraints are validated at initialization.

    """
    id: str
    tipo: TipoCadenaValor
    indicadores: List[str]
    capacidades_requeridas: List[str]
    puntos_criticos: List[str]
    ventana_temporal: Tuple[int, int]  # Meses mínimo-máximo
    kpi_ponderacion: float = 1.0  # Ponderación para cálculo de KPI

    def __post_init__(self):
        """
        Validate industrial data constraints.
        
        Raises:
            ValueError: If KPI weighting or temporal window is invalid
        """

        if not (0 <= self.kpi_ponderacion <= 2.0):
            raise ValueError("KPI ponderación debe estar entre 0 y 2.0")
        if self.ventana_temporal[0] > self.ventana_temporal[1]:
            raise ValueError("Ventana temporal inválida")

    def calcular_lead_time(self) -> float:
        """
        Calculate expected lead time with confidence interval.
        
        Returns:
            float: Expected lead time in months (average of min-max window)

        """
        return (self.ventana_temporal[0] + self.ventana_temporal[1]) / 2.0

    def generar_hash(self) -> str:
        """
        Generate unique hash for industrial traceability.
        
        Creates MD5 hash from key components for tracking and validation
        in industrial processing environments.
        
        Returns:
            str: MD5 hash string for unique identification

        """
        data = f"{self.id}|{self.tipo.value}|{sorted(self.indicadores)}|{sorted(self.capacidades_requeridas)}"
        return hashlib.md5(data.encode('utf-8')).hexdigest()


# ==================== ONTOLOGÍA DE POLÍTICAS PÚBLICAS INDUSTRIAL ====================
@dataclass
class OntologiaPoliticas:
    """
    Industrial ontological system with cross-validation and traceability.
    
    Comprehensive policy ontology system with standardized dimensions,
    causal relationships, and ODS indicator mapping for systematic
    policy evaluation and analysis.
    
    Args:
        dimensiones (Dict[str, List[str]]): Policy dimensions by category
        relaciones_causales (Dict[str, List[str]]): Causal relationship mapping
        indicadores_ods (Dict[str, List[str]]): ODS indicators by goal
        fecha_creacion (str, optional): Creation timestamp in ISO format
        version (str, optional): Ontology version. Defaults to "2.0-industrial"
        
    Class Methods:
        cargar_estandar: Load standard ontology with robust validation
        
    Static Methods:
        _cargar_indicadores_ods: Load ODS indicators with fallback system
        
    Example:
        >>> ontologia = OntologiaPoliticas.cargar_estandar()
        >>> dimensiones_sociales = ontologia.dimensiones["social"]
        >>> print(f"Dimensiones sociales: {len(dimensiones_sociales)}")
        
    Note:
        Includes comprehensive fallback mechanisms and validation systems
        for production reliability. All data is validated against standards.

    """
    dimensiones: Dict[str, List[str]]
    relaciones_causales: Dict[str, List[str]]
    indicadores_ods: Dict[str, List[str]]
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0-industrial"

    @classmethod
    def cargar_estandar(cls) -> 'OntologiaPoliticas':
        """
        Load standard ontology with robust industrial validation and hierarchical fallback.
        
        Loads the standard policy ontology with comprehensive validation,
        including fallback mechanisms for missing or corrupted data sources.
        
        Returns:
            OntologiaPoliticas: Fully validated ontology instance
            
        Raises:
            SystemExit: When critical ontology loading fails requiring manual intervention
            
        Note:
            Includes automatic generation of missing indicator files and
            comprehensive validation of all ontology components.

        """
        try:
            dimensiones_industrial = {
                "social": ["salud", "educación", "vivienda", "protección_social", "equidad_genero", "inclusión"],
                "economico": ["empleo", "productividad", "innovación", "infraestructura", "competitividad",
                              "emprendimiento"],
                "ambiental": ["sostenibilidad", "biodiversidad", "cambio_climatico", "gestión_residuos", "agua",
                              "energía_limpia"],
                "institucional": ["gobernanza", "transparencia", "participación", "rendición_cuentas", "eficiencia",
                                  "innovación_gubernamental"]
            }

            relaciones_industrial = {
                "inversión_publica": ["crecimiento_economico", "empleo", "infraestructura"],
                "educación_calidad": ["productividad", "innovación", "reducción_pobreza"],
                "salud_acceso": ["productividad_laboral", "calidad_vida", "equidad_social"],
                "gobernanza": ["transparencia", "eficiencia", "confianza_ciudadana"],
                "sostenibilidad": ["medio_ambiente", "economía_circular", "resiliencia_climática"]
            }

            # Carga industrial de indicadores ODS con validación
            indicadores_ods_path = Path("indicadores_ods_industrial.json")
            indicadores_ods = cls._cargar_indicadores_ods(indicadores_ods_path)

            return cls(
                dimensiones=dimensiones_industrial,
                relaciones_causales=relaciones_industrial,
                indicadores_ods=indicadores_ods
            )
        except Exception as e:
            log_error_with_text(LOGGER, f"❌ Error crítico cargando ontología industrial: {e}")
            raise SystemExit("Fallo en carga de ontología - Requiere intervención manual")

    @staticmethod
    @staticmethod
    def _cargar_indicadores_ods(ruta: Path) -> Dict[str, List[str]]:
        """
        Load ODS indicators with industrial fallback system.
        
        Loads Sustainable Development Goals (ODS) indicators from file with
        comprehensive fallback to base indicators if file is missing or invalid.
        
        Args:
            ruta (Path): Path to ODS indicators JSON file
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping ODS goals to indicator lists
            
        Note:
            Automatically generates template file if missing and validates
            data structure integrity. Falls back to base indicators on any error.

        """
        indicadores_base = {
            "ods1": ["tasa_pobreza", "protección_social", "vulnerabilidad_económica"],
            "ods3": ["mortalidad_infantil", "acceso_salud", "cobertura_sanitaria"],
            "ods4": ["alfabetización", "matrícula_escolar", "calidad_educativa"],
            "ods5": ["equidad_genero", "participación_mujeres", "violencia_genero"],
            "ods8": ["empleo_decente", "crecimiento_económico", "productividad_laboral"],
            "ods11": ["vivienda_digna", "transporte_sostenible", "espacios_públicos"],
            "ods13": ["emisiones_co2", "adaptación_climática", "educación_ambiental"],
            "ods16": ["gobernanza", "transparencia", "acceso_justicia"]
        }

        if ruta.exists():
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and len(data) >= 5:
                    LOGGER.info("✅ Ontología ODS industrial cargada desde archivo")
                    return data
                else:
                    LOGGER.warning("⚠️  Ontología ODS inválida, usando base industrial")
            except Exception as e:
                LOGGER.warning(f"⚠️  Error leyendo {ruta}: {e}, usando base industrial")

        # Generar archivo industrial si no existe o es inválido
        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump(indicadores_base, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"✅ Template industrial de indicadores ODS generado: {ruta}")
        except Exception as e:
            LOGGER.error(f"❌ Error generando template ODS: {e}")

        return indicadores_base


# ==================== SISTEMA DE CARGA DINÁMICA DEL DECÁLOGO INDUSTRIAL ====================
def cargar_decalogo_industrial() -> List[DimensionDecalogo]:
    """
    Load complete industrial decalog from JSON with comprehensive schema validation.
    
    Loads and validates the complete 10-dimension policy evaluation framework
    from JSON file with exhaustive validation of structure, content, and
    theoretical consistency.
    
    Returns:
        List[DimensionDecalogo]: List of validated DimensionDecalogo instances
        
    Raises:
        SystemExit: When validation fails or critical errors require manual intervention
        ValueError: When dimension structure or theory of change is invalid
        
    Note:
        Performs comprehensive validation including:
        - Schema structure validation (exactly 10 dimensions)
        - Theory of change identification verification
        - Value chain link validation with industrial constraints
        - Causal graph consistency checks
        
    Example:
        >>> decalogo = cargar_decalogo_industrial()
        >>> print(f"Loaded {len(decalogo)} validated dimensions")
        >>> primera_dimension = decalogo[0]
        >>> if primera_dimension.teoria_cambio.verificar_identificabilidad():
        ...     print("Theory of change is identifiable")

    """
    json_path = Path("decalogo_industrial.json")

    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validación industrial de esquema
            if not isinstance(data, list) or len(data) != 10:
                raise ValueError("Decálogo debe contener exactamente 10 dimensiones")

            decalogos = []
            for i, item in enumerate(data):
                try:
                    # Validación exhaustiva de cada dimensión
                    if not all(k in item for k in ['id', 'nombre', 'cluster', 'teoria_cambio', 'eslabones']):
                        raise ValueError(f"Dimensión {i + 1} incompleta")

                    if item['id'] != i + 1:
                        raise ValueError(f"ID de dimensión incorrecto: esperado {i + 1}, encontrado {item['id']}")

                    # Reconstruir Teoría de Cambio con validación
                    tc_data = item["teoria_cambio"]
                    teoria_cambio = TeoriaCambio(
                        supuestos_causales=tc_data["supuestos_causales"],
                        mediadores=tc_data["mediadores"],
                        resultados_intermedios=tc_data["resultados_intermedios"],
                        precondiciones=tc_data["precondiciones"]
                    )

                    # Validar teoría de cambio
                    if not teoria_cambio.verificar_identificabilidad():
                        raise ValueError(f"Teoría de cambio no identificable en dimensión {i + 1}")

                    # Reconstruir Eslabones con validación industrial
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
                                kpi_ponderacion=float(eslabon_data.get("kpi_ponderacion", 1.0))
                            )
                            eslabones.append(eslabon)
                        except Exception as e:
                            raise ValueError(f"Error en eslabón {j + 1} de dimensión {i + 1}: {e}")

                    # Crear dimensión con todos los componentes validados
                    dimension = DimensionDecalogo(
                        id=item["id"],
                        nombre=item["nombre"],
                        cluster=item["cluster"],
                        teoria_cambio=teoria_cambio,
                        eslabones=eslabones
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
    LOGGER.info("⚙️  Generando template industrial de decálogo estructurado")
    template_industrial = [
        {
            "id": 1,
            "nombre": "Prevención de la violencia y protección frente al conflicto armado y GDO",
            "cluster": "Paz, Seguridad y Protección de Defensores",
            "teoria_cambio": {
                "supuestos_causales": ["Reducción violencia → Mejoramiento seguridad humana",
                                       "Fortalecimiento institucional → Disminución impunidad"],
                "mediadores": {
                    "institucionales": ["capacidad estatal", "coordinación interinstitucional",
                                        "presencia territorial"],
                    "comunitarios": ["organización social", "vigilancia ciudadana", "cultura de paz"]
                },
                "resultados_intermedios": ["disminución homicidios", "aumento percepción seguridad",
                                           "reducción desplazamientos"],
                "precondiciones": ["voluntad política", "recursos adecuados", "marco legal"]
            },
            "eslabones": [
                {
                    "id": "ins_1", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_asignado", "personal_capacitado", "infraestructura_disponible"],
                    "capacidades_requeridas": ["analistas_seguridad", "gestores_conflicto", "monitores_territoriales"],
                    "puntos_criticos": ["disponibilidad_financiera", "cobertura_territorial", "capacitación_personal"],
                    "ventana_temporal": [1, 3],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "proc_1", "tipo": "PROCESOS",
                    "indicadores": ["planes_implementados", "mesas_coordinación", "protocolos_activados"],
                    "capacidades_requeridas": ["coordinadores_interinstitucionales", "facilitadores_comunitarios",
                                               "analistas_datos"],
                    "puntos_criticos": ["coordinación_interinstitucional", "participación_comunitaria",
                                        "oportunidad_respuesta"],
                    "ventana_temporal": [6, 12],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "prod_1", "tipo": "PRODUCTOS",
                    "indicadores": ["sistemas_alerta_temprana", "rutas_protección", "centros_atención"],
                    "capacidades_requeridas": ["tecnicos_sistemas", "operadores_campo", "gestores_caso"],
                    "puntos_criticos": ["sostenibilidad_técnica", "cobertura_poblacional", "calidad_servicio"],
                    "ventana_temporal": [12, 24],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "res_1", "tipo": "RESULTADOS",
                    "indicadores": ["tasa_victimización", "percepción_seguridad", "confianza_institucional"],
                    "capacidades_requeridas": ["evaluadores_impacto", "investigadores_campo", "analistas_políticas"],
                    "puntos_criticos": ["calidad_datos", "atribución_causal", "representatividad_muestral"],
                    "ventana_temporal": [24, 36],
                    "kpi_ponderacion": 1.3
                },
                {
                    "id": "imp_1", "tipo": "IMPACTOS",
                    "indicadores": ["índice_paz_territorial", "desarrollo_humano", "cohesión_social"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "economistas_desarrollo", "sociólogos"],
                    "puntos_criticos": ["atribución_causal", "factores_exógenos", "sostenibilidad_temporal"],
                    "ventana_temporal": [36, 60],
                    "kpi_ponderacion": 1.5
                }
            ]
        },
        {
            "id": 2,
            "nombre": "Equidad de género en acceso a servicios y oportunidades",
            "cluster": "Equidad Social e Inclusión",
            "teoria_cambio": {
                "supuestos_causales": ["Políticas con enfoque de género → Reducción brechas estructurales",
                                       "Empoderamiento económico → Autonomía de mujeres"],
                "mediadores": {
                    "institucionales": ["presupuesto_sensible_género", "mecanismos_participación",
                                        "sistemas_monitoreo"],
                    "sociales": ["transformación_cultural", "educación_igualdad", "redes_apoyo"]
                },
                "resultados_intermedios": ["aumento_participación_femenina", "reducción_brecha_salarial",
                                           "disminución_violencia_genero"],
                "precondiciones": ["voluntad_política", "datos_desagregados", "marco_normativo"]
            },
            "eslabones": [
                {
                    "id": "ins_2", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_género", "personal_especializado", "infraestructura_adecuada"],
                    "capacidades_requeridas": ["expertos_género", "gestores_presupuesto", "facilitadores_comunitarios"],
                    "puntos_criticos": ["sensibilización_institucional", "disponibilidad_recursos",
                                        "capacitación_personal"],
                    "ventana_temporal": [1, 4],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "proc_2", "tipo": "PROCESOS",
                    "indicadores": ["políticas_implementadas", "mecanismos_participación", "protocolos_atención"],
                    "capacidades_requeridas": ["coordinadores_género", "facilitadores_diálogo", "analistas_datos"],
                    "puntos_criticos": ["articulación_institucional", "participación_significativa",
                                        "seguimiento_oportuno"],
                    "ventana_temporal": [6, 15],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "prod_2", "tipo": "PRODUCTOS",
                    "indicadores": ["servicios_especializados", "programas_capacitación", "sistemas_denuncia"],
                    "capacidades_requeridas": ["psicólogos", "abogados", "educadores"],
                    "puntos_criticos": ["calidad_servicio", "accesibilidad_geográfica", "sostenibilidad_financiera"],
                    "ventana_temporal": [12, 24],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "res_2", "tipo": "RESULTADOS",
                    "indicadores": ["índice_autonomía_económica", "tasa_participación_laboral",
                                    "percepción_seguridad_mujeres"],
                    "capacidades_requeridas": ["evaluadores_género", "investigadores_campo", "analistas_políticas"],
                    "puntos_criticos": ["medición_precisa", "atribución_causal", "factores_culturales"],
                    "ventana_temporal": [24, 48],
                    "kpi_ponderacion": 1.4
                },
                {
                    "id": "imp_2", "tipo": "IMPACTOS",
                    "indicadores": ["índice_equidad_género", "desarrollo_humano_mujeres", "transformación_cultural"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "sociólogos", "economistas"],
                    "puntos_criticos": ["cambio_estructural", "sostenibilidad_temporal", "factores_exógenos"],
                    "ventana_temporal": [48, 72],
                    "kpi_ponderacion": 1.6
                }
            ]
        },
        {
            "id": 3,
            "nombre": "Calidad educativa y cobertura universal con pertinencia territorial",
            "cluster": "Educación y Desarrollo Humano",
            "teoria_cambio": {
                "supuestos_causales": ["Inversión educativa → Mejora calidad educativa",
                                       "Formación docente → Resultados aprendizaje"],
                "mediadores": {
                    "institucionales": ["infraestructura_escolar", "material_didáctico", "sistemas_evaluación"],
                    "pedagógicos": ["metodologías_activas", "evaluación_formativa", "acompañamiento_docente"]
                },
                "resultados_intermedios": ["mejora_puntajes_pruebas", "reducción_deserción", "aumento_cobertura"],
                "precondiciones": ["voluntad_política", "presupuesto_adecuado", "marco_curricular"]
            },
            "eslabones": [
                {
                    "id": "ins_3", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_educación", "docentes_calificados", "infraestructura_escolar"],
                    "capacidades_requeridas": ["gestores_educativos", "planificadores", "supervisores"],
                    "puntos_criticos": ["distribución_recursos", "formación_docente", "mantenimiento_infraestructura"],
                    "ventana_temporal": [1, 3],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "proc_3", "tipo": "PROCESOS",
                    "indicadores": ["planes_mejora_implementados", "programas_formación", "sistemas_monitoreo"],
                    "capacidades_requeridas": ["coordinadores_pedagógicos", "facilitadores_formación",
                                               "analistas_datos"],
                    "puntos_criticos": ["articulación_niveles", "pertinencia_territorial", "seguimiento_continuo"],
                    "ventana_temporal": [6, 18],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "prod_3", "tipo": "PRODUCTOS",
                    "indicadores": ["materiales_didácticos", "plataformas_digitales", "programas_especiales"],
                    "capacidades_requeridas": ["diseñadores_curriculares", "desarrolladores_tecnológicos",
                                               "especialistas_contenido"],
                    "puntos_criticos": ["calidad_contenido", "accesibilidad_tecnológica", "actualización_periódica"],
                    "ventana_temporal": [12, 24],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "res_3", "tipo": "RESULTADOS",
                    "indicadores": ["puntajes_pruebas_estandarizadas", "tasa_retención", "satisfacción_estudiantes"],
                    "capacidades_requeridas": ["evaluadores_educativos", "investigadores_aprendizaje",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_confiable", "atribución_causal", "factores_contextuales"],
                    "ventana_temporal": [24, 36],
                    "kpi_ponderacion": 1.3
                },
                {
                    "id": "imp_3", "tipo": "IMPACTOS",
                    "indicadores": ["índice_desarrollo_humano", "productividad_laboral", "cohesión_social"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "economistas_educación", "sociólogos"],
                    "puntos_criticos": ["impacto_estructural", "sostenibilidad_temporal", "factores_exógenos"],
                    "ventana_temporal": [36, 60],
                    "kpi_ponderacion": 1.5
                }
            ]
        },
        {
            "id": 4,
            "nombre": "Salud preventiva, acceso universal y atención de calidad",
            "cluster": "Salud Pública y Bienestar",
            "teoria_cambio": {
                "supuestos_causales": ["Inversión salud → Mejora acceso servicios",
                                       "Prevención → Reducción enfermedades"],
                "mediadores": {
                    "institucionales": ["infraestructura_salud", "talento_humano", "sistemas_información"],
                    "comunitarios": ["promoción_salud", "prevención_comunitaria", "autocuidado"]
                },
                "resultados_intermedios": ["aumento_cobertura", "reducción_mortalidad", "mejora_percepción_calidad"],
                "precondiciones": ["voluntad_política", "presupuesto_adecuado", "marco_normativo"]
            },
            "eslabones": [
                {
                    "id": "ins_4", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_salud", "personal_calificado", "infraestructura_salud"],
                    "capacidades_requeridas": ["gestores_salud", "planificadores", "supervisores"],
                    "puntos_criticos": ["distribución_recursos", "formación_personal", "mantenimiento_infraestructura"],
                    "ventana_temporal": [1, 4],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "proc_4", "tipo": "PROCESOS",
                    "indicadores": ["programas_implementados", "campañas_prevención", "sistemas_referencia"],
                    "capacidades_requeridas": ["coordinadores_programas", "facilitadores_comunitarios",
                                               "analistas_datos"],
                    "puntos_criticos": ["articulación_niveles", "cobertura_territorial", "oportunidad_intervención"],
                    "ventana_temporal": [6, 18],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "prod_4", "tipo": "PRODUCTOS",
                    "indicadores": ["servicios_salud", "programas_especiales", "sistemas_información"],
                    "capacidades_requeridas": ["médicos", "enfermeros", "tecnólogos"],
                    "puntos_criticos": ["calidad_servicio", "accesibilidad_geográfica", "sostenibilidad_financiera"],
                    "ventana_temporal": [12, 30],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "res_4", "tipo": "RESULTADOS",
                    "indicadores": ["tasa_mortalidad", "cobertura_servicios", "satisfacción_usuarios"],
                    "capacidades_requeridas": ["evaluadores_salud", "investigadores_epidemiología",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_precisa", "atribución_causal", "factores_socioeconómicos"],
                    "ventana_temporal": [24, 48],
                    "kpi_ponderacion": 1.4
                },
                {
                    "id": "imp_4", "tipo": "IMPACTOS",
                    "indicadores": ["esperanza_vida", "índice_salud_poblacional", "productividad_laboral"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "epidemiólogos", "economistas_salud"],
                    "puntos_criticos": ["impacto_estructural", "sostenibilidad_temporal", "determinantes_sociales"],
                    "ventana_temporal": [48, 84],
                    "kpi_ponderacion": 1.6
                }
            ]
        },
        {
            "id": 5,
            "nombre": "Desarrollo económico productivo con inclusión y sostenibilidad",
            "cluster": "Economía y Empleo",
            "teoria_cambio": {
                "supuestos_causales": ["Inversión productiva → Crecimiento económico", "Capacitación → Empleo decente"],
                "mediadores": {
                    "institucionales": ["infraestructura_productiva", "apoyo_financiero", "sistemas_innovación"],
                    "empresariales": ["formación_empresarial", "acceso_mercados", "alianzas_productivas"]
                },
                "resultados_intermedios": ["aumento_empleo", "crecimiento_PIB", "diversificación_productiva"],
                "precondiciones": ["estabilidad_macro", "marco_legal", "capital_humano"]
            },
            "eslabones": [
                {
                    "id": "ins_5", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_desarrollo", "infraestructura_productiva", "talento_humano"],
                    "capacidades_requeridas": ["gestores_desarrollo", "planificadores", "especialistas_sectoriales"],
                    "puntos_criticos": ["distribución_recursos", "formación_técnica", "mantenimiento_infraestructura"],
                    "ventana_temporal": [1, 6],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "proc_5", "tipo": "PROCESOS",
                    "indicadores": ["programas_implementados", "alianzas_productivas", "sistemas_financiamiento"],
                    "capacidades_requeridas": ["coordinadores_programas", "facilitadores_negocios",
                                               "analistas_mercado"],
                    "puntos_criticos": ["articulación_actores", "pertinencia_territorial", "seguimiento_resultados"],
                    "ventana_temporal": [6, 24],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "prod_5", "tipo": "PRODUCTOS",
                    "indicadores": ["empresas_formadas", "productos_nuevos", "mercados_accedidos"],
                    "capacidades_requeridas": ["emprendedores", "gestores_negocios", "especialistas_comercialización"],
                    "puntos_criticos": ["calidad_productos", "sostenibilidad_empresas", "acceso_financiamiento"],
                    "ventana_temporal": [18, 36],
                    "kpi_ponderacion": 1.3
                },
                {
                    "id": "res_5", "tipo": "RESULTADOS",
                    "indicadores": ["tasa_empleo", "crecimiento_PIB_local", "productividad_sectorial"],
                    "capacidades_requeridas": ["evaluadores_económicos", "investigadores_mercado",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_precisa", "atribución_causal", "factores_exógenos"],
                    "ventana_temporal": [36, 60],
                    "kpi_ponderacion": 1.5
                },
                {
                    "id": "imp_5", "tipo": "IMPACTOS",
                    "indicadores": ["índice_desarrollo_económico", "reducción_pobreza", "cohesión_social"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "economistas_desarrollo", "sociólogos"],
                    "puntos_criticos": ["impacto_estructural", "sostenibilidad_temporal", "equidad_territorial"],
                    "ventana_temporal": [60, 120],
                    "kpi_ponderacion": 1.8
                }
            ]
        },
        {
            "id": 6,
            "nombre": "Sostenibilidad ambiental y gestión del riesgo climático",
            "cluster": "Medio Ambiente y Cambio Climático",
            "teoria_cambio": {
                "supuestos_causales": ["Gestión ambiental → Conservación ecosistemas",
                                       "Adaptación climática → Reducción vulnerabilidad"],
                "mediadores": {
                    "institucionales": ["marco_normativo", "sistemas_monitoreo", "infraestructura_verde"],
                    "comunitarios": ["educación_ambiental", "prácticas_sostenibles", "vigilancia_ciudadana"]
                },
                "resultados_intermedios": ["reducción_emisiones", "conservación_biodiversidad",
                                           "resiliencia_comunitaria"],
                "precondiciones": ["voluntad_política", "recursos_técnicos", "participación_ciudadana"]
            },
            "eslabones": [
                {
                    "id": "ins_6", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_ambiental", "personal_especializado", "equipamiento_técnico"],
                    "capacidades_requeridas": ["gestores_ambientales", "técnicos_monitoreo", "educadores_ambientales"],
                    "puntos_criticos": ["disponibilidad_recursos", "capacitación_técnica", "cobertura_territorial"],
                    "ventana_temporal": [1, 6],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "proc_6", "tipo": "PROCESOS",
                    "indicadores": ["planes_implementados", "programas_monitoreo", "campañas_educación"],
                    "capacidades_requeridas": ["coordinadores_programas", "facilitadores_comunitarios",
                                               "analistas_datos"],
                    "puntos_criticos": ["articulación_institucional", "participación_significativa",
                                        "seguimiento_científico"],
                    "ventana_temporal": [6, 24],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "prod_6", "tipo": "PRODUCTOS",
                    "indicadores": ["áreas_protegidas", "sistemas_alerta", "tecnologías_limpia"],
                    "capacidades_requeridas": ["biólogos", "ingenieros_ambientales", "educadores"],
                    "puntos_criticos": ["calidad_técnica", "sostenibilidad_ecológica", "aceptación_social"],
                    "ventana_temporal": [18, 48],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "res_6", "tipo": "RESULTADOS",
                    "indicadores": ["índice_calidad_ambiental", "reducción_emisiones", "percepción_riesgo"],
                    "capacidades_requeridas": ["evaluadores_ambientales", "investigadores_cambio_climático",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_científica", "atribución_causal", "factores_globales"],
                    "ventana_temporal": [36, 72],
                    "kpi_ponderacion": 1.4
                },
                {
                    "id": "imp_6", "tipo": "IMPACTOS",
                    "indicadores": ["índice_sostenibilidad", "resiliencia_territorial", "calidad_vida_ambiental"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "científicos_clima",
                                               "economistas_ambientales"],
                    "puntos_criticos": ["impacto_global", "sostenibilidad_temporal", "factores_exógenos"],
                    "ventana_temporal": [72, 120],
                    "kpi_ponderacion": 1.7
                }
            ]
        },
        {
            "id": 7,
            "nombre": "Infraestructura básica, conectividad y movilidad sostenible",
            "cluster": "Infraestructura y Conectividad",
            "teoria_cambio": {
                "supuestos_causales": ["Inversión infraestructura → Mejora conectividad",
                                       "Movilidad sostenible → Calidad vida urbana"],
                "mediadores": {
                    "técnicos": ["diseño_ingenieril", "gestión_proyectos", "mantenimiento"],
                    "sociales": ["participación_ciudadana", "uso_eficiente", "cultura_movilidad"]
                },
                "resultados_intermedios": ["reducción_tiempo_desplazamiento", "aumento_cobertura_servicios",
                                           "mejora_accesibilidad"],
                "precondiciones": ["planificación_urbana", "recursos_financieros", "marco_normativo"]
            },
            "eslabones": [
                {
                    "id": "ins_7", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_infraestructura", "personal_técnico", "equipamiento"],
                    "capacidades_requeridas": ["ingenieros", "arquitectos", "gestores_proyectos"],
                    "puntos_criticos": ["disponibilidad_recursos", "capacitación_técnica", "planificación_integral"],
                    "ventana_temporal": [1, 12],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "proc_7", "tipo": "PROCESOS",
                    "indicadores": ["proyectos_ejecutados", "sistemas_gestión", "protocolos_mantenimiento"],
                    "capacidades_requeridas": ["supervisores_obras", "gestores_mantenimiento", "analistas_tráfico"],
                    "puntos_criticos": ["eficiencia_ejecución", "control_calidad", "gestión_riesgos"],
                    "ventana_temporal": [12, 36],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "prod_7", "tipo": "PRODUCTOS",
                    "indicadores": ["infraestructura_construida", "sistemas_inteligentes", "servicios_conectividad"],
                    "capacidades_requeridas": ["operadores_sistemas", "mantenimiento", "soporte_técnico"],
                    "puntos_criticos": ["calidad_construcción", "sostenibilidad_operativa", "accesibilidad_universal"],
                    "ventana_temporal": [24, 60],
                    "kpi_ponderacion": 1.3
                },
                {
                    "id": "res_7", "tipo": "RESULTADOS",
                    "indicadores": ["índice_movilidad", "tiempo_promedio_desplazamiento", "satisfacción_usuarios"],
                    "capacidades_requeridas": ["evaluadores_infraestructura", "investigadores_urbanismo",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_objetiva", "atribución_causal", "factores_demográficos"],
                    "ventana_temporal": [48, 84],
                    "kpi_ponderacion": 1.5
                },
                {
                    "id": "imp_7", "tipo": "IMPACTOS",
                    "indicadores": ["índice_desarrollo_urbano", "productividad_económica", "calidad_vida_ciudadana"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "urbanistas", "economistas"],
                    "puntos_criticos": ["impacto_estructural", "sostenibilidad_temporal", "equidad_territorial"],
                    "ventana_temporal": [84, 144],
                    "kpi_ponderacion": 1.8
                }
            ]
        },
        {
            "id": 8,
            "nombre": "Gobernanza, participación ciudadana y rendición de cuentas",
            "cluster": "Gobernanza y Democracia",
            "teoria_cambio": {
                "supuestos_causales": ["Participación ciudadana → Mejora decisiones públicas",
                                       "Transparencia → Confianza institucional"],
                "mediadores": {
                    "institucionales": ["mecanismos_participación", "sistemas_transparencia", "capacidades_gobierno"],
                    "ciudadanos": ["organización_social", "veedurías", "educación_cívica"]
                },
                "resultados_intermedios": ["aumento_participación", "mejora_percepción_transparencia",
                                           "reducción_corrupción"],
                "precondiciones": ["marco_legal", "voluntad_política", "cultura_democrática"]
            },
            "eslabones": [
                {
                    "id": "ins_8", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_gobernanza", "personal_especializado", "plataformas_digitales"],
                    "capacidades_requeridas": ["gestores_participación", "expertos_transparencia",
                                               "facilitadores_diálogo"],
                    "puntos_criticos": ["disponibilidad_recursos", "capacitación_funcionarios",
                                        "accesibilidad_ciudadana"],
                    "ventana_temporal": [1, 6],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "proc_8", "tipo": "PROCESOS",
                    "indicadores": ["mecanismos_implementados", "procesos_participativos", "sistemas_rendición"],
                    "capacidades_requeridas": ["coordinadores_participación", "auditores_sociales",
                                               "analistas_políticas"],
                    "puntos_criticos": ["inclusión_real", "impacto_decisiones", "seguimiento_compromisos"],
                    "ventana_temporal": [6, 24],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "prod_8", "tipo": "PRODUCTOS",
                    "indicadores": ["espacios_participación", "informes_transparencia", "sistemas_quejas"],
                    "capacidades_requeridas": ["facilitadores_diálogo", "gestores_información", "mediadores"],
                    "puntos_criticos": ["calidad_procesos", "sostenibilidad_institucional", "impacto_real"],
                    "ventana_temporal": [18, 48],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "res_8", "tipo": "RESULTADOS",
                    "indicadores": ["índice_participación", "percepción_transparencia", "confianza_institucional"],
                    "capacidades_requeridas": ["evaluadores_gobernanza", "investigadores_políticos",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_confiable", "atribución_causal", "factores_culturales"],
                    "ventana_temporal": [36, 72],
                    "kpi_ponderacion": 1.4
                },
                {
                    "id": "imp_8", "tipo": "IMPACTOS",
                    "indicadores": ["índice_democracia", "cohesión_social", "desarrollo_institucional"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "científicos_políticos", "sociólogos"],
                    "puntos_criticos": ["cambio_estructural", "sostenibilidad_temporal", "factores_históricos"],
                    "ventana_temporal": [72, 120],
                    "kpi_ponderacion": 1.7
                }
            ]
        },
        {
            "id": 9,
            "nombre": "Innovación, transformación digital y economía del conocimiento",
            "cluster": "Innovación y Tecnología",
            "teoria_cambio": {
                "supuestos_causales": ["Inversión I+D → Innovación tecnológica",
                                       "Transformación digital → Eficiencia institucional"],
                "mediadores": {
                    "tecnológicos": ["infraestructura_digital", "talento_tecnológico", "ecosistemas_innovación"],
                    "organizacionales": ["cultura_innovación", "procesos_ágiles", "gestión_conocimiento"]
                },
                "resultados_intermedios": ["aumento_patentes", "adopción_tecnológica", "mejora_eficiencia"],
                "precondiciones": ["marco_normativo", "inversión_privada", "capital_humano"]
            },
            "eslabones": [
                {
                    "id": "ins_9", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_I+D", "infraestructura_tecnológica", "talento_especializado"],
                    "capacidades_requeridas": ["gestores_innovación", "científicos", "ingenieros"],
                    "puntos_criticos": ["disponibilidad_recursos", "formación_talentos", "infraestructura_adecuada"],
                    "ventana_temporal": [1, 12],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "proc_9", "tipo": "PROCESOS",
                    "indicadores": ["proyectos_I+D", "programas_transformación", "alianzas_tecnológicas"],
                    "capacidades_requeridas": ["gestores_proyectos", "facilitadores_innovación",
                                               "analistas_tecnológicos"],
                    "puntos_criticos": ["articulación_actores", "gestión_riesgos", "seguimiento_resultados"],
                    "ventana_temporal": [12, 36],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "prod_9", "tipo": "PRODUCTOS",
                    "indicadores": ["patentes_registradas", "soluciones_digitales", "nuevos_productos"],
                    "capacidades_requeridas": ["desarrolladores", "diseñadores", "especialistas_mercado"],
                    "puntos_criticos": ["calidad_tecnológica", "propiedad_intelectual", "adopción_mercado"],
                    "ventana_temporal": [24, 60],
                    "kpi_ponderacion": 1.4
                },
                {
                    "id": "res_9", "tipo": "RESULTADOS",
                    "indicadores": ["índice_innovación", "productividad_tecnológica", "eficiencia_institucional"],
                    "capacidades_requeridas": ["evaluadores_tecnológicos", "investigadores_innovación",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_técnica", "atribución_causal", "factores_globales"],
                    "ventana_temporal": [48, 84],
                    "kpi_ponderacion": 1.6
                },
                {
                    "id": "imp_9", "tipo": "IMPACTOS",
                    "indicadores": ["índice_economía_conocimiento", "competitividad_global",
                                    "transformación_productiva"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "economistas_innovación",
                                               "estrategas_tecnológicos"],
                    "puntos_criticos": ["impacto_estructural", "sostenibilidad_temporal", "brecha_digital"],
                    "ventana_temporal": [84, 144],
                    "kpi_ponderacion": 2.0
                }
            ]
        },
        {
            "id": 10,
            "nombre": "Sostenibilidad fiscal, eficiencia del gasto y gestión del riesgo financiero",
            "cluster": "Finanzas Públicas y Gestión Fiscal",
            "teoria_cambio": {
                "supuestos_causales": ["Gestión fiscal eficiente → Sostenibilidad financiera",
                                       "Transparencia presupuestal → Confianza inversionistas"],
                "mediadores": {
                    "técnicos": ["planeación_financiera", "sistemas_control", "gestión_riesgos"],
                    "institucionales": ["marco_normativo", "capacidades_técnicas", "sistemas_información"]
                },
                "resultados_intermedios": ["equilibrio_fiscal", "eficiencia_gasto", "calificación_riesgo"],
                "precondiciones": ["marco_legal", "voluntad_política", "talento_técnico"]
            },
            "eslabones": [
                {
                    "id": "ins_10", "tipo": "INSUMOS",
                    "indicadores": ["presupuesto_gestión", "personal_calificado", "sistemas_información"],
                    "capacidades_requeridas": ["gestores_fiscales", "contadores", "analistas_riesgo"],
                    "puntos_criticos": ["disponibilidad_recursos", "formación_técnica", "infraestructura_tecnológica"],
                    "ventana_temporal": [1, 6],
                    "kpi_ponderacion": 1.1
                },
                {
                    "id": "proc_10", "tipo": "PROCESOS",
                    "indicadores": ["planes_financieros", "sistemas_control", "mecanismos_transparencia"],
                    "capacidades_requeridas": ["planificadores_financieros", "auditores", "analistas_presupuesto"],
                    "puntos_criticos": ["eficiencia_procesos", "control_calidad", "gestión_riesgos"],
                    "ventana_temporal": [6, 24],
                    "kpi_ponderacion": 1.0
                },
                {
                    "id": "prod_10", "tipo": "PRODUCTOS",
                    "indicadores": ["informes_financieros", "sistemas_presupuestales", "herramientas_análisis"],
                    "capacidades_requeridas": ["analistas_financieros", "desarrolladores_sistemas",
                                               "especialistas_riesgo"],
                    "puntos_criticos": ["calidad_técnica", "sostenibilidad_operativa", "actualización_normativa"],
                    "ventana_temporal": [18, 48],
                    "kpi_ponderacion": 1.2
                },
                {
                    "id": "res_10", "tipo": "RESULTADOS",
                    "indicadores": ["índice_sostenibilidad_fiscal", "eficiencia_gasto", "percepción_riesgo"],
                    "capacidades_requeridas": ["evaluadores_fiscales", "investigadores_economía",
                                               "analistas_políticas"],
                    "puntos_criticos": ["medición_precisa", "atribución_causal", "factores_exógenos"],
                    "ventana_temporal": [36, 72],
                    "kpi_ponderacion": 1.5
                },
                {
                    "id": "imp_10", "tipo": "IMPACTOS",
                    "indicadores": ["índice_estabilidad_económica", "confianza_inversionista", "desarrollo_sostenible"],
                    "capacidades_requeridas": ["investigadores_largo_plazo", "economistas_fiscales",
                                               "estrategas_financieros"],
                    "puntos_criticos": ["impacto_estructural", "sostenibilidad_temporal", "factores_globales"],
                    "ventana_temporal": [72, 120],
                    "kpi_ponderacion": 1.8
                }
            ]
        }
    ]

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(template_industrial, f, indent=2, ensure_ascii=False)
        LOGGER.info(f"✅ Template industrial de decálogo generado: {json_path}")
        LOGGER.warning("⚠️  COMPLETE Y VALIDA MANUALMENTE EL ARCHIVO decalogo_industrial.json")
    except Exception as e:
        LOGGER.error("❌ Error generando template industrial: {e}")
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
        # Validación industrial
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

        # Verificación de identificabilidad (peso 0.4)
        if self.teoria_cambio.verificar_identificabilidad():
            coherencia += 0.4
            peso_total += 0.4
        else:
            peso_total += 0.4

        # Verificación de estructura de cadena (peso 0.3)
        tipos_presentes = {eslabon.tipo for eslabon in self.eslabones}
        tipos_esenciales = {TipoCadenaValor.INSUMOS, TipoCadenaValor.PROCESOS, TipoCadenaValor.PRODUCTOS}
        if tipos_esenciales.issubset(tipos_presentes):
            coherencia += 0.3
            peso_total += 0.3
        else:
            peso_total += 0.3

        # Verificación de impactos (peso 0.3)
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
        "logica": (
            "Estos tres puntos comparten una matriz común centrada en la seguridad humana, la "
            "protección de la vida y la construcción de paz territorial. Abordan las dinámicas del "
            "conflicto armado, sus víctimas y quienes defienden derechos en contextos de riesgo."
        ),
    },
    "CLUSTER 2": {
        "titulo": "CLUSTER 2: DERECHOS DE GRUPOS POBLACIONALES",
        "puntos": [2, 6, 9],
        "logica": (
            "Agrupa derechos de poblaciones que enfrentan vulnerabilidades específicas y requieren "
            "enfoques diferenciales. Comparten la necesidad de políticas focalizadas, sistemas de "
            "protección especializados y transformación de patrones culturales discriminatorios."
        ),
    },
    "CLUSTER 3": {
        "titulo": "CLUSTER 3: TERRITORIO, AMBIENTE Y DESARROLLO SOSTENIBLE",
        "puntos": [3, 7],
        "logica": (
            "Ambos puntos abordan la relación sociedad-territorio desde una perspectiva de "
            "sostenibilidad, justicia ambiental y equidad en el acceso a recursos. Comparten la visión "
            "del territorio como base para el desarrollo y la vida digna."
        ),
    },
    "CLUSTER 4": {
        "titulo": "CLUSTER 4: DERECHOS SOCIALES FUNDAMENTALES Y CRISIS HUMANITARIAS",
        "puntos": [4, 10],
        "logica": (
            "Aunque el punto 10 es altamente específico territorialmente, comparte con los derechos "
            "sociales la dimensión de respuesta a necesidades básicas y dignidad humana. Ambos "
            "requieren capacidad institucional para garantizar mínimos vitales y atención humanitaria."
        ),
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
            (
                dimensiones_por_id[punto].cluster
                for punto in puntos
                if punto in dimensiones_por_id
            ),
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
                self.logger.info(
                    f"✅ Embeddings precomputados para {len(textos_validos)} segmentos - {self.nombre_plan}")
            except Exception as e:
                self.logger.error(f"❌ Error precomputando embeddings: {e}")
                self.embeddings_doc = torch.tensor([])
        else:
            self.embeddings_doc = torch.tensor([])
            self.logger.warning(f"⚠️  No hay textos suficientes para precomputar embeddings - {self.nombre_plan}")

    def buscar_evidencia_causal(self, query: str, conceptos_clave: List[str],
                                top_k: int = 5, umbral_certeza: float = 0.75) -> List[Dict[str, Any]]:
        """Búsqueda semántica industrial con filtrado por relaciones causales y umbral de certeza ajustable"""
        # Check if precomputed embeddings are available and properly initialized
        if not hasattr(self, 'embeddings_doc') or self.embeddings_doc is None or self.embeddings_doc.numel() == 0:
            self.logger.warning("⚠️  No hay embeddings precomputados disponibles, fallback a encoding en tiempo real")
            # Fallback to current encoding behavior if precomputed embeddings aren't available
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)

        try:
            # Reuse precomputed embeddings - encode only the query
            query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            similitudes = util.pytorch_cos_sim(query_embedding, self.embeddings_doc)[0]
            resultados = []

            # Obtener top 2*top_k resultados para filtrado posterior
            indices_top = torch.topk(similitudes, min(top_k * 2, len(self.textos_originales))).indices

            for idx in indices_top:
                # Encontrar el documento original correspondiente
                texto_original = None
                pagina_original = None
                for doc in self.documentos:
                    if doc[1] in self.textos_originales and self.textos_originales.index(doc[1]) == idx:
                        texto_original = doc[1]
                        pagina_original = doc[0]
                        break

                if not texto_original:
                    continue

                texto = texto_original
                pagina = pagina_original

                # Cálculo industrial de relevancia conceptual
                coincidencias_conceptuales = sum(1 for concepto in conceptos_clave
                                                 if concepto.lower() in texto.lower())
                relevancia_conceptual = coincidencias_conceptuales / max(1, len(conceptos_clave))

                # Detección avanzada de relaciones causales con patrones industriales
                patrones_causales_industriales = [
                    r"\b(porque|debido a|como consecuencia de|en razón de|a causa de)\b",
                    r"\b(genera|produce|causa|determina|influye en|afecta a)\b",
                    r"\b(impacto|efecto|resultado|consecuencia|repercusión)\b",
                    r"\b(mejora|aumenta|reduce|disminuye|fortalece|debilita)\b",
                    r"\b(siempre que|cuando|si)\b.*\b(entonces|por lo tanto|en consecuencia)\b"
                ]

                densidad_causal = 0.0
                for patron in patrones_causales_industriales:
                    matches = len(re.findall(patron, texto.lower(), re.IGNORECASE))
                    densidad_causal += matches * 0.2  # Ponderación por tipo de patrón

                densidad_causal = min(1.0, densidad_causal / max(1, len(texto.split()) / 100))

                # Cálculo de score final con ponderaciones industriales
                score_final = (
                        similitudes[idx].item() * 0.5 +  # Similaridad semántica
                        relevancia_conceptual * 0.3 +  # Relevancia conceptual
                        densidad_causal * 0.2  # Densidad causal
                )

                # Aplicar umbral de certeza industrial estricto
                if score_final >= umbral_certeza:
                    resultados.append({
                        "texto": texto,
                        "pagina": pagina,
                        "similitud_semantica": float(similitudes[idx].item()),
                        "relevancia_conceptual": relevancia_conceptual,
                        "densidad_causal": densidad_causal,
                        "score_final": score_final,
                        "hash_segmento": hashlib.md5(texto.encode('utf-8')).hexdigest()[:8],
                        "timestamp_extraccion": datetime.now().isoformat()
                    })

            # Ordenar por score_final y retornar top_k
            resultados_ordenados = sorted(resultados, key=lambda x: x["score_final"], reverse=True)
            return resultados_ordenados[:top_k]

        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda causal con embeddings precomputados: {e}")
            # Fallback to current encoding behavior on error
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)

    def _buscar_evidencia_fallback(self, query: str, conceptos_clave: List[str],
                                   top_k: int = 5, umbral_certeza: float = 0.75) -> List[Dict[str, Any]]:
        """Fallback method that performs document encoding at runtime when precomputed embeddings are not available"""
        if not self.textos_originales:
            self.logger.warning("⚠️  No hay textos disponibles para búsqueda")
            return []
        
        try:
            # Encode both query and documents at runtime (original behavior)
            query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            doc_embeddings = EMBEDDING_MODEL.encode(self.textos_originales, convert_to_tensor=True)
            similitudes = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            resultados = []

            # Obtener top 2*top_k resultados para filtrado posterior
            indices_top = torch.topk(similitudes, min(top_k * 2, len(self.textos_originales))).indices

            for idx in indices_top:
                # Encontrar el documento original correspondiente
                texto_original = None
                pagina_original = None
                for doc in self.documentos:
                    if doc[1] in self.textos_originales and self.textos_originales.index(doc[1]) == idx:
                        texto_original = doc[1]
                        pagina_original = doc[0]
                        break

                if not texto_original:
                    continue

                texto = texto_original
                pagina = pagina_original

                # Cálculo industrial de relevancia conceptual
                coincidencias_conceptuales = sum(1 for concepto in conceptos_clave
                                                 if concepto.lower() in texto.lower())
                relevancia_conceptual = coincidencias_conceptuales / max(1, len(conceptos_clave))

                # Detección avanzada de relaciones causales con patrones industriales
                patrones_causales_industriales = [
                    r"\b(porque|debido a|como consecuencia de|en razón de|a causa de)\b",
                    r"\b(genera|produce|causa|determina|influye en|afecta a)\b",
                    r"\b(impacto|efecto|resultado|consecuencia|repercusión)\b",
                    r"\b(mejora|aumenta|reduce|disminuye|fortalece|debilita)\b",
                    r"\b(siempre que|cuando|si)\b.*\b(entonces|por lo tanto|en consecuencia)\b"
                ]

                densidad_causal = 0.0
                for patron in patrones_causales_industriales:
                    matches = len(re.findall(patron, texto.lower(), re.IGNORECASE))
                    densidad_causal += matches * 0.2  # Ponderación por tipo de patrón

                densidad_causal = min(1.0, densidad_causal / max(1, len(texto.split()) / 100))

                # Cálculo de score final con ponderaciones industriales
                score_final = (
                        similitudes[idx].item() * 0.5 +  # Similaridad semántica
                        relevancia_conceptual * 0.3 +  # Relevancia conceptual
                        densidad_causal * 0.2  # Densidad causal
                )

                # Aplicar umbral de certeza industrial estricto
                if score_final >= umbral_certeza:
                    resultados.append({
                        "texto": texto,
                        "pagina": pagina,
                        "similitud_semantica": float(similitudes[idx].item()),
                        "relevancia_conceptual": relevancia_conceptual,
                        "densidad_causal": densidad_causal,
                        "score_final": score_final,
                        "hash_segmento": hashlib.md5(texto.encode('utf-8')).hexdigest()[:8],
                        "timestamp_extraccion": datetime.now().isoformat()
                    })

            # Ordenar por score_final y retornar top_k
            resultados_ordenados = sorted(resultados, key=lambda x: x["score_final"], reverse=True)
            return resultados_ordenados[:top_k]

        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda causal fallback: {e}")
            return []

    def buscar_segmentos_semanticos_global(self, queries: List[str], max_segmentos: int, 
                                         batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica global con selección top-k usando heap para optimización de memoria.
        Procesa múltiples queries en batches y mantiene solo los mejores max_segmentos globalmente.
        """
        if self.embeddings_doc.numel() == 0:
            self.logger.warning("⚠️  No hay embeddings disponibles para búsqueda semántica global")
            return []

        # Min-heap para mantener los top-k segmentos globalmente
        # Formato: (score_negativo, indice_doc, query_idx, datos_segmento)
        heap_global = []
        
        try:
            # Procesar queries en batches
            for batch_start in range(0, len(queries), batch_size):
                batch_queries = queries[batch_start:batch_start + batch_size]
                
                # Encode el batch de queries
                query_embeddings = EMBEDDING_MODEL.encode(batch_queries, convert_to_tensor=True)
                
                # Calcular similitudes para todo el batch contra todos los documentos
                similitudes_batch = util.pytorch_cos_sim(query_embeddings, self.embeddings_doc)
                
                # Procesar cada query del batch
                for query_idx_local, query in enumerate(batch_queries):
                    query_idx_global = batch_start + query_idx_local
                    similitudes = similitudes_batch[query_idx_local]
                    
                    # Procesar todos los segmentos para esta query
                    for doc_idx, similitud in enumerate(similitudes):
                        score = float(similitud.item())
                        
                        # Encontrar el documento original correspondiente
                        if doc_idx < len(self.textos_originales):
                            texto_seg = self.textos_originales[doc_idx]
                            
                            # Encontrar la página correspondiente
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
                                "hash_segmento": hashlib.md5(texto_seg.encode('utf-8')).hexdigest()[:8],
                                "timestamp_extraccion": datetime.now().isoformat()
                            }
                            
                            # Usar heap para mantener top-k global eficientemente
                            # Como usamos un min-heap con scores negativos, el elemento en heap[0] es el de MENOR score
                            if len(heap_global) < max_segmentos:
                                # Heap no está lleno, agregar directamente
                                heapq.heappush(heap_global, (score, doc_idx, query_idx_global, datos_segmento))
                            elif score > heap_global[0][0]:  # score > min_score_in_heap
                                # Reemplazar el peor elemento con este mejor elemento
                                heapq.heappushpop(heap_global, (score, doc_idx, query_idx_global, datos_segmento))

            # Extraer los resultados finales del heap y ordenar por score descendente
            resultados_finales = []
            while heap_global:
                score, _, _, datos_segmento = heapq.heappop(heap_global)
                resultados_finales.append(datos_segmento)
            
            # Ordenar por score final descendente para tener los mejores primero
            resultados_finales.sort(key=lambda x: x['score_final'], reverse=True)
            
            self.logger.info(f"✅ Búsqueda semántica global completada: {len(resultados_finales)} segmentos "
                           f"seleccionados de {len(self.textos_originales)} totales con {len(queries)} queries")
            
            return resultados_finales
            
        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda semántica global: {e}")
            return []

    def extraer_variables_operativas(self, dimension: DimensionDecalogo) -> Dict[str, List[str]]:
        """Extrae variables operativas específicas para cada dimensión con trazabilidad industrial"""
        variables = {
            "indicadores": [],
            "metas": [],
            "recursos": [],
            "responsables": [],
            "plazos": [],
            "riesgos": []
        }

        try:
            # Extracción de indicadores por eslabón
            for eslabon in dimension.eslabones:
                for indicador in eslabon.indicadores:
                    # Búsqueda semántica mejorada
                    resultados = self.buscar_evidencia_causal(
                        f"indicador {indicador} meta objetivo {dimension.nombre}",
                        [indicador, "meta", "objetivo", "línea base", "indicador"],
                        top_k=3,
                        umbral_certeza=0.7
                    )
                    if resultados:
                        for resultado in resultados:
                            resultado["eslabon_origen"] = eslabon.id
                            resultado["tipo_variable"] = "indicador"
                        variables["indicadores"].extend(resultados)

            # Detección de asignación de recursos con patrones industriales
            patrones_recursos = [
                "presupuesto", "financiación", "recursos", "inversión", "asignación",
                "fondo", "subsidio", "transferencia", "cofinanciación", "contrapartida"
            ]

            resultados_recursos = self.buscar_evidencia_causal(
                f"presupuesto financiación recursos para {dimension.nombre}",
                patrones_recursos,
                top_k=5,
                umbral_certeza=0.65
            )

            if resultados_recursos:
                for resultado in resultados_recursos:
                    resultado["tipo_variable"] = "recurso"
                variables["recursos"].extend(resultados_recursos)

            # Detección de responsables y plazos
            patrones_responsables = ["responsable", "encargado", "lidera", "coordina", "gestiona"]
            patrones_plazos = ["plazo", "fecha", "cronograma", "tiempo", "duración", "inicio", "finalización"]

            resultados_responsables = self.buscar_evidencia_causal(
                f"responsable encargado de {dimension.nombre}",
                patrones_responsables,
                top_k=3,
                umbral_certeza=0.6
            )

            resultados_plazos = self.buscar_evidencia_causal(
                f"plazo fecha cronograma para {dimension.nombre}",
                patrones_plazos,
                top_k=3,
                umbral_certeza=0.6
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
                f"✅ Extracción completada para dimensión {dimension.id}: {sum(len(v) for v in variables.values())} variables encontradas")

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
                    data.append({
                        "dimension_id": dimension.id,
                        "dimension_nombre": dimension.nombre,
                        "tipo_variable": tipo_variable,
                        "texto_evidencia": resultado.get("texto", "")[:200] + "...",
                        "pagina": resultado.get("pagina", 0),
                        "score_confianza": resultado.get("score_final", 0.0),
                        "hash_evidencia": resultado.get("hash_segmento", "N/A"),
                        "timestamp": resultado.get("timestamp_extraccion", "")
                    })

            if data:
                df = pd.DataFrame(data)
                return df
            else:
                return pd.DataFrame(columns=[
                    "dimension_id", "dimension_nombre", "tipo_variable",
                    "texto_evidencia", "pagina", "score_confianza",
                    "hash_evidencia", "timestamp"
                ])

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
        # Validación industrial de rangos
        for field_name, value in self.__dict__.items():
            if isinstance(value, float) and not (0.0 <= value <= 1.0):
                if field_name not in ['evidencia_soporte', 'brechas_criticas']:
                    raise ValueError(f"Campo {field_name} fuera de rango [0,1]: {value}")

    @property
    def puntaje_global(self) -> float:
        """Cálculo industrial del puntaje global con ponderaciones estratégicas"""
        # Ponderaciones industriales basadas en importancia estratégica
        return (
                self.consistencia_logica * 0.25 +
                self.identificabilidad_causal * 0.20 +
                self.factibilidad_operativa * 0.20 +
                self.certeza_probabilistica * 0.15 +
                self.robustez_causal * 0.20
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
                "version_sistema": "8.0-industrial"
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
                    "robustez_causal": self.evaluacion_causal.robustez_causal
                }
            },
            "diagnostico": {
                "brechas_criticas": len(self.brechas_identificadas),
                "riesgos_principales": self.evaluacion_causal.riesgos_implementacion[:5],
                "evidencia_disponible": sum(len(v) for v in self.evidencia.values()),
                "nivel_madurez": self.nivel_madurez
            },
            "recomendaciones": self.recomendaciones[:10],
            "trazabilidad": self.matriz_trazabilidad.to_dict() if self.matriz_trazabilidad is not None else {}
        }


# ==================== SISTEMA COMPLETO INDUSTRIAL CON PROCESAMIENTO PARALELO ====================
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
        
        # Verificar disponibilidad de pdfplumber
        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("❌ pdfplumber no está instalado. Por favor, instale con: pip install pdfplumber")

    def calcular_hash_documento(self) -> str:
        """Calcula hash industrial del documento para trazabilidad"""
        if self.paginas:
            contenido_completo = " ".join(self.paginas)
            return hashlib.sha256(contenido_completo.encode('utf-8')).hexdigest()
        return ""

    def extraer_metadata_pdf(self) -> Dict[str, Any]:
        """Extrae metadata industrial del PDF"""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    return {
                        "author": pdf.metadata.get('Author', 'Desconocido'),
                        "title": pdf.metadata.get('Title', self.nombre_plan),
                        "creation_date": pdf.metadata.get('CreationDate', ''),
                        "modification_date": pdf.metadata.get('ModDate', ''),
                        "producer": pdf.metadata.get('Producer', ''),
                        "page_count": len(pdf.pages)
                    }
        except Exception as e:
            self.logger.warning(f"⚠️  Error extrayendo metadata: {e}")
        return {"page_count": len(self.paginas) if self.paginas else 0}

    def cargar(self) -> bool:
        """Carga el documento PDF con manejo de errores industrial"""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, pagina in enumerate(pdf.pages, start=1):
                    try:
                        texto = pagina.extract_text() or ""
                        texto = re.sub(r'\s+', ' ', texto).strip()
                        if len(texto) > 10:  # Filtrar páginas vacías
                            self.paginas.append(texto)
                    except Exception as e:
                        self.logger.warning(f"⚠️  Error procesando página {i}: {e}")
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
                        # Filtrar sentencias muy cortas o sin sustantivos/verbos
                        if len(texto) >= 20 and self._tiene_contenido_sustancial(sentencia):
                            buffer.append(texto)
                            # Crear segmentos de 2-4 sentencias con coherencia temática
                            if len(buffer) >= 3 or (len(buffer) >= 2 and self._detectar_cambio_tematico(buffer)):
                                segmento = " ".join(buffer)
                                self.segmentos.append((i, segmento))
                                total_segmentos += 1
                                buffer = []

                    # Procesar buffer restante
                    if buffer:
                        segmento = " ".join(buffer)
                        if len(segmento) >= 30:
                            self.segmentos.append((i, segmento))
                            total_segmentos += 1

                except Exception as e:
                    self.logger.warning(f"⚠️  Error procesando página {i}: {e}")
                    continue

            self.logger.info(f"✅ Segmentación completada: {len(self.segmentos)} segmentos - {self.nombre_plan}")
            
            # Fallback: si no se generaron segmentos, crear segmentos básicos por párrafo
            if len(self.segmentos) == 0:
                self.logger.warning("⚠️  No se generaron segmentos con método NLP, usando fallback básico")
                for i, pagina in enumerate(self.paginas, start=1):
                    paragrafos = [p.strip() for p in pagina.split('\n\n') if len(p.strip()) > 20]
                    for paragrafo in paragrafos:
                        if len(paragrafo) > 30:
                            self.segmentos.append((i, paragrafo))
                
                # Si aún no hay segmentos, crear uno por página completa
                if len(self.segmentos) == 0:
                    for i, pagina in enumerate(self.paginas, start=1):
                        if len(pagina.strip()) > 30:
                            self.segmentos.append((i, pagina.strip()))
                
                self.logger.info(f"✅ Segmentación fallback completada: {len(self.segmentos)} segmentos")
            
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
            # Calcular similitud semántica entre últimas sentencias
            if len(buffer) >= 2:
                embeddings = EMBEDDING_MODEL.encode(buffer[-2:], convert_to_tensor=True)
                if len(embeddings) == 2:
                    similitud = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                    return similitud < 0.6  # Umbral para cambio temático
        except Exception:
            pass

        return False


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

# ==================== INTEGRACIÓN CON EVALUADOR DEL DECÁLOGO ====================

def integrar_evaluador_decatalogo(sistema: SistemaEvaluacionIndustrial, 
                                 dimension: DimensionDecalogo) -> Optional[ResultadoDimensionIndustrial]:
    """
    Integra el evaluador del Decálogo con el sistema industrial principal.
    
    Args:
        sistema: Sistema de evaluación industrial principal
        dimension: Dimensión del decálogo a evaluar
        
    Returns:
        Resultado de la evaluación industrial o None si hay error
    """
    if not sistema.extractor:
        raise ValueError("Extractor no inicializado - Error industrial crítico")

    try:
        # >>>>>>>> INTEGRACIÓN INDUSTRIAL DEL EVALUADOR DEL DECÁLOGO <<<<<<<<
        from Decatalogo_evaluador import integrar_evaluador_decatalogo as integrar_evaluador
        resultado_decatalogo = integrar_evaluador(sistema, dimension)
        
        if resultado_decatalogo:
            # Registrar advertencias si el puntaje es bajo
            if resultado_decatalogo.puntaje_final < 60:
                LOGGER.warning(f"⚠️  [DECÁLOGO] Dimensión {dimension.id} tiene baja calidad: {resultado_decatalogo.puntaje_final:.1f}")
            
            return resultado_decatalogo
        # >>>>>>>> FIN DE LA INTEGRACIÓN <<<<<<<<

        return None

    except ImportError as e:
        LOGGER.warning(f"⚠️  No se pudo importar el evaluador del decálogo: {e}")
        return None
    except Exception as e:
        LOGGER.error(f"❌ Error en integración del evaluador del decálogo: {e}")
        return None


class SistemaEvaluacionIndustrial:
    def __init__(self, pdf_path: Path, logger: logging.Logger):
        self.pdf_path = pdf_path
        self.logger = logger
        self.loader = LoaderDocumento(pdf_path)
        self.extractor = None
        self.metadata_plan = None
        self.hash_evaluacion = None

    def inicializar(self) -> bool:
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
                f"{self.loader.hash_documento}_{datetime.now().isoformat()}".encode('utf-8')).hexdigest()

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
            self.logger.info(f"🔍 Iniciando evaluación industrial de dimensión {dimension.id}: {dimension.nombre}")

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

            # >>>>>>>> INTEGRACIÓN CON EVALUADOR DEL DECÁLOGO <<<<<<<<
            # Importar e integrar el evaluador del decálogo
            try:
                from Decatalogo_evaluador import integrar_evaluador_decatalogo
                resultado_decatalogo = integrar_evaluador_decatalogo(self, dimension)
                
                if resultado_decatalogo:
                    # Combinar resultados del evaluador del decálogo con los resultados principales
                    # Aquí podrías decidir cómo combinar los resultados, por ejemplo:
                    # 1. Usar el resultado del decálogo como base
                    # 2. Combinar puntajes
                    # 3. Enriquecer el resultado con información adicional del decálogo
                    
                    # Por ahora, simplemente usamos el resultado del decálogo como base
                    # y añadimos información adicional del sistema principal
                    resultado_decatalogo.evidencia = evidencia
                    resultado_decatalogo.matriz_trazabilidad = matriz_trazabilidad
                    resultado_decatalogo.brechas_identificadas = brechas
                    resultado_decatalogo.recomendaciones = recomendaciones
                    
                    self.logger.info(
                        f"✅ Evaluación completada para dimensión {dimension.id} usando evaluador del decálogo: {resultado_decatalogo.puntaje_final:.1f}/100")
                    return resultado_decatalogo
            except ImportError as e:
                self.logger.warning(f"⚠️  No se pudo importar el evaluador del decálogo: {e}")
            except Exception as e:
                self.logger.error(f"❌ Error en evaluador del decálogo: {e}")
            # >>>>>>>> FIN DE LA INTEGRACIÓN CON EVALUADOR DEL DECÁLOGO <<<<<<<<

            resultado = ResultadoDimensionIndustrial(
                dimension=dimension,
                evaluacion_causal=evaluacion_causal,
                evidencia=evidencia,
                brechas_identificadas=brechas,
                recomendaciones=recomendaciones,
                matriz_trazabilidad=matriz_trazabilidad,
                timestamp_evaluacion=datetime.now().isoformat()
            )

            self.logger.info(
                f"✅ Evaluación completada para dimensión {dimension.id}: {resultado.puntaje_final:.1f}/100")
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
                brechas_criticas=5
            )

            return ResultadoDimensionIndustrial(
                dimension=dimension,
                evaluacion_causal=evaluacion_fallback,
                evidencia={},
                brechas_identificadas=[f"Error crítico en evaluación: {str(e)}"],
                recomendaciones=["Requiere revisión manual urgente"],
                timestamp_evaluacion=datetime.now().isoformat()
            )

    def _evaluar_coherencia_causal_industrial(self, dimension: DimensionDecalogo,
                                              evidencia: Dict) -> EvaluacionCausalIndustrial:
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
            for _ in range(200):  # 200 iteraciones para mayor precisión
                try:
                    if len(G.nodes) > 2:
                        # Muestreo estratificado de nodos
                        nodos_base = ["insumos", "impactos"]
                        nodos_mediadores = [n for n in G.nodes if G.nodes[n].get('tipo') == 'mediador']
                        nodos_resultados = [n for n in G.nodes if G.nodes[n].get('tipo') == 'resultado']

                        # Muestreo proporcional
                        nodos_sample = nodos_base.copy()
                        if nodos_mediadores:
                            nodos_sample.extend(np.random.choice(nodos_mediadores,
                                                                 size=min(3, len(nodos_mediadores)),
                                                                 replace=False).tolist())
                        if nodos_resultados:
                            nodos_sample.extend(np.random.choice(nodos_resultados,
                                                                 size=min(2, len(nodos_resultados)),
                                                                 replace=False).tolist())

                        subsample = G.subgraph(nodos_sample)

                        if (nx.is_directed_acyclic_graph(subsample) and
                                nx.has_path(subsample, "insumos", "impactos") and
                                len(subsample.edges) >= 2):
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
                riesgos.append("⚠️  Baja certeza causal: Requiere fortalecimiento de marco teórico")
            if robustez < 0.6:
                riesgos.append("⚠️  Baja robustez causal: Requiere simplificación de relaciones causales")

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
                brechas_criticas=brechas_criticas
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
                brechas_criticas=5
            )

    def _calcular_factibilidad_industrial(self, dimension: DimensionDecalogo, evidencia: Dict) -> float:
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

    def _identificar_brechas_industrial(self, dimension: DimensionDecalogo, evidencia: Dict) -> List[str]:
        """Identifica brechas industriales en la implementación con diagnóstico preciso"""
        brechas = []

        # Brecha 1: Falta de indicadores por eslabón
        for eslabon in dimension.eslabones:
            indicadores_encontrados = any(
                any(ind.lower() in ev["texto"].lower()
                    for ind in eslabon.indicadores
                    for ev in evidencia.get("indicadores", []))
            )
            if not indicadores_encontrados:
                brechas.append(
                    f"🔴 BRECHA CRÍTICA: Falta especificación de indicadores para eslabón {eslabon.id} ({eslabon.tipo.value})")

        # Brecha 2: Falta de recursos
        if not evidencia.get("recursos", []):
            brechas.append("🔴 BRECHA CRÍTICA: No se encontró especificación presupuestal o de recursos")

        # Brecha 3: Falta de responsables
        if not evidencia.get("responsables", []):
            brechas.append("🟠 BRECHA IMPORTANTE: No se identificaron responsables claros para la implementación")

        # Brecha 4: Falta de plazos
        if not evidencia.get("plazos", []):
            brechas.append("🟠 BRECHA IMPORTANTE: No se encontraron plazos o cronogramas definidos")

        # Brecha 5: Complejidad causal excesiva
        if len(dimension.teoria_cambio.supuestos_causales) > 5:
            brechas.append("🟡 BRECHA MODERADA: Alta complejidad causal puede dificultar la implementación y medición")

        return brechas

    def _identificar_riesgos_industrial(self, dimension: DimensionDecalogo, evidencia: Dict) -> List[str]:
        """Identifica riesgos industriales de implementación con clasificación por severidad"""
        riesgos = []

        # Riesgo 1: Falta de recursos (ALTO)
        if not evidencia.get("recursos", []):
            riesgos.append("🔴 ALTO: Falta de especificación presupuestal - Riesgo de inviabilidad operativa")

        # Riesgo 2: Complejidad causal (MEDIO-ALTO)
        if len(dimension.teoria_cambio.supuestos_causales) > 4:
            riesgos.append(
                "🟠 MEDIO-ALTO: Alta complejidad causal - Riesgo de dificultad en implementación y atribución")

        # Riesgo 3: Falta de indicadores (ALTO)
        indicadores_encontrados = len(evidencia.get("indicadores", []))
        indicadores_requeridos = len(dimension.eslabones)
        if indicadores_encontrados < indicadores_requeridos * 0.5:
            riesgos.append(
                "🔴 ALTO: Cobertura insuficiente de indicadores - Riesgo de imposibilidad de medición y evaluación")

        # Riesgo 4: Ventanas temporales (MEDIO)
        for eslabon in dimension.eslabones:
            if eslabon.ventana_temporal[1] - eslabon.ventana_temporal[0] > 36:
                riesgos.append(
                    f"🟠 MEDIO: Ventana temporal excesivamente amplia en eslabón {eslabon.id} - Riesgo de desfase en resultados")
                break

        # Riesgo 5: Falta de responsables (MEDIO)
        if not evidencia.get("responsables", []):
            riesgos.append("🟠 MEDIO: Ausencia de responsables definidos - Riesgo de falta de rendición de cuentas")

        return riesgos

    def _generar_recomendaciones_industrial(self, dimension: DimensionDecalogo,
                                            evaluacion: EvaluacionCausalIndustrial,
                                            brechas: List[str]) -> List[str]:
        """Genera recomendaciones industriales específicas basadas en el análisis con enfoque estratégico"""
        recomendaciones = []

        # Recomendaciones basadas en evaluación causal
        if evaluacion.consistencia_logica < 0.7:
            recomendaciones.append("🔧 FORTALECER: Revisar y fortalecer la coherencia lógica de la teoría de cambio")

        if evaluacion.factibilidad_operativa < 0.6:
            recomendaciones.append("🔧 FORTALECER: Especificar mejor los mecanismos de implementación operativa")

        if evaluacion.certeza_probabilistica < 0.7:
            recomendaciones.append(
                "📊 EVIDENCIA: Incorporar mayor evidencia empírica para sustentar las relaciones causales")

        if evaluacion.robustez_causal < 0.6:
            recomendaciones.append("🧩 SIMPLIFICAR: Reducir la complejidad del modelo causal para mejorar su robustez")

        # Recomendaciones basadas en brechas
        for brecha in brechas:
            if "BRECHA CRÍTICA" in brecha:
                recomendaciones.append(f"🚨 ACCIÓN INMEDIATA: {brecha.replace('🔴 BRECHA CRÍTICA: ', '')}")
            elif "BRECHA IMPORTANTE" in brecha:
                recomendaciones.append(f"⚠️  PRIORIDAD ALTA: {brecha.replace('🟠 BRECHA IMPORTANTE: ', '')}")
            elif "BRECHA MODERADA" in brecha:
                recomendaciones.append(f"🔧 MEJORA CONTINUA: {brecha.replace('🟡 BRECHA MODERADA: ', '')}")

        # Recomendaciones estratégicas adicionales
        if evaluacion.evidencia_soporte < 5:
            recomendaciones.append(
                "📚 INVESTIGACIÓN: Realizar estudio de línea base para fortalecer la evidencia disponible")

        if len(evaluacion.riesgos_implementacion) > 3:
            recomendaciones.append("🛡️  GESTIÓN DE RIESGOS: Desarrollar plan integral de mitigación de riesgos")

        # Recomendación de monitoreo
        recomendaciones.append("📈 MONITOREO: Establecer sistema de monitoreo y evaluación con indicadores SMART")

        return recomendaciones[:15]  # Limitar a 15 recomendaciones principales

    def generar_reporte_tecnico_completo(self, resultados: List[ResultadoDimensionIndustrial]) -> Dict[str, Any]:
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
                "nivel_madurez_predominante": max(set(niveles_madurez),
                                                  key=niveles_madurez.count) if niveles_madurez else "N/A",
                "certidumbre_predominante": max(set(certidumbres), key=certidumbres.count) if certidumbres else "N/A",
                "recomendacion_estrategica_global": self._generar_recomendacion_global(resultados),
                "riesgos_sistemicos": self._identificar_riesgos_sistemicos(resultados)
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
                    "version_sistema": "8.0-industrial",
                    "total_dimensiones": len(DECALOGO_INDUSTRIAL)
                },
                "analisis_agregado": analisis_agregado,
                "resultados_por_dimension": resultados_detalles,
                "timestamp_generacion": datetime.now().isoformat()
            }

            return reporte_completo

        except Exception as e:
            self.logger.error(f"❌ Error generando reporte técnico completo: {e}")
            return {
                "metadata": {
                    "nombre_plan": self.pdf_path.stem,
                    "error": str(e)
                },
                "analisis_agregado": {},
                "resultados_por_dimension": []
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
            riesgos_sistemicos.append("🔴 DESCOHERENCIA ESTRATÉGICA: Alta variabilidad en madurez entre dimensiones")

        # Riesgo de evidencia global
        evidencia_total = sum(r.evaluacion_causal.evidencia_soporte for r in resultados)
        if evidencia_total < len(resultados) * 3:
            riesgos_sistemicos.append("🟠 DÉFICIT DE EVIDENCIA: Bajo soporte empírico para el conjunto del plan")

        # Riesgo de implementación global
        riesgos_criticos = sum(len(r.evaluacion_causal.riesgos_implementacion) for r in resultados)
        if riesgos_criticos > len(resultados) * 3:
            riesgos_sistemicos.append("🔴 SOBRECARGA DE RIESGOS: Alta concentración de riesgos de implementación")

        # Riesgo de certeza global
        certezas_bajas = sum(1 for r in resultados if r.evaluacion_causal.certeza_probabilistica < 0.6)
        if certezas_bajas > len(resultados) * 0.4:
            riesgos_sistemicos.append("🟠 INCERTIDUMBRE SISTÉMICA: Baja certeza causal en múltiples dimensiones")

        return riesgos_sistemicos if riesgos_sistemicos else ["✅ SIN RIESGOS SISTÉMICOS IDENTIFICADOS"]


# ==================== GENERADOR DE REPORTES INDUSTRIAL ====================
class GeneradorReporteIndustrial:
    """Genera reportes industriales en múltiples formatos con estándares profesionales"""

    @staticmethod
    def generar_reporte_markdown(resultados: List[ResultadoDimensionIndustrial],
                                 nombre_plan: str,
                                 metadata: Dict = None) -> str:
        """Genera reporte industrial en formato markdown con análisis profundo"""
        reporte = []

        # Encabezado industrial
        reporte.append("# 🏭 EVALUACIÓN INDUSTRIAL DE POLÍTICAS PÚBLICAS")
        reporte.append(f"## 📄 Plan de Desarrollo Municipal: {nombre_plan}")
        reporte.append("### 🎯 Análisis Multinivel con Enfoque Causal y Certificación de Rigor")
        reporte.append(f"### 📊 Fecha de evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if metadata and metadata.get('hash_evaluacion'):
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
            reporte.append(f"**⚠️  Dimensiones Deficientes (<70):** {deficientes}/{len(resultados)}")

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
            reporte.append(f"### 💡 Recomendación Estratégica: {resultado.evaluacion_causal.recomendacion_estrategica}")
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
                reporte.append("### ⚠️  RIESGOS DE IMPLEMENTACIÓN")
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
                for recomendacion in resultado.recomendaciones[:8]:  # Top 8 recomendaciones
                    reporte.append(f"- {recomendacion}")
                reporte.append("")

            # Evidencia clave
            evidencia_total = sum(len(v) for v in resultado.evidencia.values())
            if evidencia_total > 0:
                reporte.append("### 📚 EVIDENCIA CLAVE")
                for tipo, evidencias in resultado.evidencia.items():
                    if evidencias:
                        reporte.append(f"**{tipo.upper()}:** ({len(evidencias)} elementos)")
                        for ev in evidencias[:2]:  # Top 2 evidencias por tipo
                            texto_resumido = ev['texto'][:250] + "..." if len(ev['texto']) > 250 else ev['texto']
                            reporte.append(f"- Pág {ev['pagina']} (Score: {ev['score_final']:.3f}): {texto_resumido}")
                        reporte.append("")

        # Recomendación final
        reporte.append("## 🎯 RECOMENDACIÓN FINAL INDUSTRIAL")
        if puntajes:
            promedio = statistics.mean(puntajes)
            if promedio >= 85:
                reporte.append("✅ **APROBADO PARA IMPLEMENTACIÓN INTEGRAL**")
                reporte.append("   - El plan demuestra alto nivel de madurez y coherencia causal")
                reporte.append("   - Se recomienda monitoreo continuo y ajustes menores")
            elif promedio >= 70:
                reporte.append("⚠️  **APROBADO CON RECOMENDACIONES**")
                reporte.append("   - El plan es viable pero requiere mejoras en dimensiones específicas")
                reporte.append("   - Priorizar implementación de recomendaciones en dimensiones deficientes")
            elif promedio >= 50:
                reporte.append("🛠️  **REQUIERE REDISEÑO PARCIAL**")
                reporte.append("   - El plan tiene fundamentos válidos pero necesita mejoras sustanciales")
                reporte.append("   - Priorizar rediseño de dimensiones con puntajes más bajos")
            else:
                reporte.append("❌ **REQUIERE REDISEÑO INTEGRAL**")
                reporte.append("   - El plan presenta deficiencias estructurales significativas")
                reporte.append("   - Se recomienda revisión estratégica fundamental antes de implementación")

        reporte.append("")
        reporte.append("---")
        reporte.append("*Reporte generado por Sistema Industrial de Evaluación de Políticas Públicas v8.0*")
        reporte.append(f"*Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(reporte)

    @staticmethod
    def generar_reporte_json(resultados: List[ResultadoDimensionIndustrial],
                             nombre_plan: str,
                             sistema: SistemaEvaluacionIndustrial = None) -> Dict[str, Any]:
        """Genera reporte industrial en formato JSON con estructura estandarizada"""
        if sistema:
            return sistema.generar_reporte_tecnico_completo(resultados)

        # Estructura básica si no se proporciona sistema
        puntajes = [r.puntaje_final for r in resultados]
        return {
            "nombre_plan": nombre_plan,
            "fecha_evaluacion": datetime.now().isoformat(),
            "version_sistema": "8.0-industrial",
            "resumen_ejecutivo": {
                "puntaje_global": statistics.mean(puntajes) if puntajes else 0,
                "dimensiones_evaluadas": len(resultados),
                "nivel_madurez_global": max([r.nivel_madurez for r in resultados]) if resultados else "N/A"
            },
            "dimensiones": [r.generar_reporte_tecnico() for r in resultados],
            "timestamp_generacion": datetime.now().isoformat()
        }


# ==================== CONFIGURACIÓN DE REPRODUCIBILIDAD DETERMINISTA ====================
def configurar_semillas_deterministas(base_seed: int, plan_name: str) -> None:
    """
    Configura semillas deterministas para reproducibilidad en el procesamiento de cada plan.
    
    Args:
        base_seed: Semilla base proporcionada por el usuario
        plan_name: Nombre del plan para generar semilla específica
    """
    # Crear semilla específica para el plan combinando base_seed con hash del nombre
    plan_hash = hashlib.md5(plan_name.encode('utf-8')).hexdigest()
    plan_seed = base_seed ^ int(plan_hash[:8], 16)  # XOR con primeros 8 dígitos hex del hash
    
    # Configurar semillas para todas las librerías de números aleatorios
    random.seed(plan_seed)
    np.random.seed(plan_seed % (2**32))  # numpy requiere seed de 32 bits
    torch.manual_seed(plan_seed)
    
    # Configurar generadores adicionales de PyTorch si están disponibles
    if torch.cuda.is_available():
        torch.cuda.manual_seed(plan_seed)
        torch.cuda.manual_seed_all(plan_seed)
    
    # Configurar comportamiento determinista adicional para PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    LOGGER.debug(f"🎲 Semillas configuradas para '{plan_name}': base={base_seed}, plan_specific={plan_seed}")


# ==================== PROCESAMIENTO PARALELO INDUSTRIAL PARA 170+ PLANES ====================
def procesar_plan_industrial(pdf_path: Path, base_seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """Worker industrial para procesamiento paralelo de planes de desarrollo"""
    nombre_plan = pdf_path.stem
    logger_worker = logging.getLogger(f"Worker_{nombre_plan}")
    logger_worker.info(f"🔄 Iniciando procesamiento industrial de: {nombre_plan}")

    if not pdf_path.exists() or pdf_path.suffix.lower() not in ['.pdf', '.PDF']:
        logger_worker.error(f"❌ Archivo inválido o no encontrado: {pdf_path}")
        return nombre_plan, {"error": "Archivo inválido o no encontrado", "status": "failed"}

    try:
        # Configurar reproducibilidad determinista si se proporciona semilla
        if base_seed is not None:
            configurar_semillas_deterministas(base_seed, nombre_plan)
            logger_worker.info(f"🎲 Configuración determinista activada para: {nombre_plan}")
        
        # Inicializar sistema industrial
        sistema = SistemaEvaluacionIndustrial(pdf_path)

        # Cargar y procesar documento
        if not sistema.cargar_y_procesar():
            logger_worker.error(f"❌ Falló la carga y procesamiento de: {nombre_plan}")
            return nombre_plan, {"error": "Falló carga y procesamiento", "status": "failed"}

        # Aplicar límite global de segmentos si está especificado
        if max_segmentos and sistema.extractor:
            logger_worker.info(f"🔍 Aplicando selección global de top-{max_segmentos} segmentos con batch_size={batch_size}")
            
            # Generar queries para búsqueda semántica global
            queries_semanticas = []
            for dimension in DECALOGO_INDUSTRIAL:
                # Query principal basada en el nombre de la dimensión
                queries_semanticas.append(f"{dimension.nombre} objetivo meta resultado")
                
                # Queries específicas por eslabón
                for eslabon in dimension.eslabones:
                    for indicador in eslabon.indicadores[:2]:  # Top 2 indicadores por eslabón
                        queries_semanticas.append(f"{indicador} {dimension.nombre}")
                
                # Queries de teoría de cambio
                for supuesto in dimension.teoria_cambio.supuestos_causales[:1]:  # Top 1 supuesto
                    queries_semanticas.append(supuesto)
            
            # Limitar número de queries para evitar sobrecarga
            queries_semanticas = queries_semanticas[:100]  # Máximo 100 queries
            
            # Ejecutar búsqueda semántica global con heap
            segmentos_seleccionados = sistema.extractor.buscar_segmentos_semanticos_global(
                queries_semanticas, max_segmentos, batch_size
            )
            
            # Reemplazar los documentos originales con los segmentos seleccionados
            if segmentos_seleccionados:
                documentos_filtrados = []
                for i, seg in enumerate(segmentos_seleccionados):
                    documentos_filtrados.append((seg["pagina"], seg["texto"]))
                
                # Actualizar el extractor con los segmentos seleccionados
                sistema.extractor.documentos = documentos_filtrados
                sistema.extractor.textos_originales = [doc[1] for doc in documentos_filtrados]
                sistema.extractor._precomputar_embeddings()
                
                logger_worker.info(f"✅ Segmentos filtrados: {len(documentos_filtrados)} de {len(sistema.loader.segmentos)} originales")

        # Evaluar todas las dimensiones industriales
        resultados = []
        segmentos_totales = len(sistema.loader.segmentos) if hasattr(sistema.loader, 'segmentos') else 0
        segmentos_procesados = len(sistema.extractor.textos_originales) if sistema.extractor else 0
        
        for dimension in DECALOGO_INDUSTRIAL:
            try:
                resultado = sistema.evaluar_dimension(dimension)
                resultados.append(resultado)
                logger_worker.info(
                    f"✅ Evaluada dimensión {dimension.id} para {nombre_plan}: {resultado.puntaje_final:.1f}/100")
            except Exception as e:
                logger_worker.error(f"❌ Error evaluando dimensión {dimension.id}: {e}")
                continue

        if not resultados:
            logger_worker.error(f"❌ No se pudo evaluar ninguna dimensión para: {nombre_plan}")
            return nombre_plan, {"error": "No se pudo evaluar dimensiones", "status": "failed"}

        # Generar reportes industriales
        output_dir = Path("resultados_evaluacion_industrial") / nombre_plan
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reporte Markdown
        reporte_md = GeneradorReporteIndustrial.generar_reporte_markdown(
            resultados, nombre_plan, {"hash_evaluacion": sistema.hash_evaluacion}
        )
        reporte_md_path = output_dir / f"{nombre_plan}_evaluacion_industrial.md"
        reporte_md_path.write_text(reporte_md, encoding='utf-8')

        # Reporte JSON
        reporte_json = GeneradorReporteIndustrial.generar_reporte_json(
            resultados, nombre_plan, sistema
        )
        reporte_json_path = output_dir / f"{nombre_plan}_evaluacion_industrial.json"
        with open(reporte_json_path, 'w', encoding='utf-8') as f:
            json.dump(reporte_json, f, indent=2, ensure_ascii=False)

        # Calcular métricas agregadas industriales
        puntajes = [r.puntaje_final for r in resultados]
        metrics = {
            "puntaje_promedio": statistics.mean(puntajes) if puntajes else 0,
            "desviacion_estandar": statistics.stdev(puntajes) if len(puntajes) > 1 else 0,
            "dimensiones_evaluadas": len(puntajes),
            "dimensiones_excelentes": len([p for p in puntajes if p >= 85]),
            "dimensiones_deficientes": len([p for p in puntajes if p < 70]),
            "nivel_madurez_predominante": max([r.nivel_madurez for r in resultados]) if resultados else "N/A",
            "hash_evaluacion": sistema.hash_evaluacion[:12] + "...",
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "segmentos_totales": segmentos_totales,
            "segmentos_procesados": segmentos_procesados,
            "segmentos_filtrados": max_segmentos is not None,
            "batch_size_usado": batch_size,
            "reportes_generados": {
                "markdown": str(reporte_md_path),
                "json": str(reporte_json_path)
            }
        }

        logger_worker.info(f"✅ Procesamiento completado para {nombre_plan}")
        return nombre_plan, metrics

    except Exception as e:
        logger_worker.error(f"❌ Error crítico procesando {nombre_plan}: {e}")
        return nombre_plan, {"error": str(e), "status": "failed"}


# ==================== SISTEMA DE MONITOREO Y CONTROL INDUSTRIAL ====================
class SistemaMonitoreoIndustrial:
    """Sistema de monitoreo y control industrial para procesamiento batch con manejo de señales"""

    def __init__(self):
        self.logger = logging.getLogger("MonitoreoIndustrial")
        self.ejecuciones = []
        self.tiempo_inicio = None
        self.tiempo_fin = None
        self.trabajadores_activos = set()
        self.lock = threading.RLock()
        self.interrumpido = False

    def iniciar_monitoreo(self):
        """Inicia el monitoreo industrial"""
        self.tiempo_inicio = datetime.now()
        self.logger.info("🚀 Iniciando sistema de monitoreo industrial")

    def registrar_ejecucion(self, nombre_plan: str, resultado: Dict[str, Any]):
        """Registra una ejecución industrial de forma thread-safe"""
        with self.lock:
            ejecucion = {
                "nombre_plan": nombre_plan,
                "resultado": resultado,
                "timestamp": datetime.now().isoformat(),
                "duracion_estimada": self._calcular_duracion_estimada()
            }
            self.ejecuciones.append(ejecucion)

            if resultado.get("status") == "completed":
                self.logger.info(f"✅ Plan completado: {nombre_plan} - Puntaje: {resultado.get('puntaje_promedio', 0):.1f}")
            else:
                self.logger.error(f"❌ Plan fallido: {nombre_plan} - Error: {resultado.get('error', 'Desconocido')}")

    def registrar_trabajador(self, trabajador_id: str):
        """Registra un trabajador activo de forma thread-safe"""
        with self.lock:
            self.trabajadores_activos.add(trabajador_id)
            self.logger.debug(f"Trabajador registrado: {trabajador_id}")

    def desregistrar_trabajador(self, trabajador_id: str):
        """Desregistra un trabajador activo de forma thread-safe"""
        with self.lock:
            self.trabajadores_activos.discard(trabajador_id)
            self.logger.debug(f"Trabajador desregistrado: {trabajador_id}")

    def terminar_trabajadores(self):
        """Termina todos los trabajadores activos graciosamente"""
        with self.lock:
            self.interrumpido = True
            trabajadores_activos_copia = self.trabajadores_activos.copy()
        
        self.logger.warning(f"Terminando {len(trabajadores_activos_copia)} trabajadores activos...")
        # Note: En joblib, los trabajadores se terminan automáticamente cuando se interrumpe el proceso padre

    def _calcular_duracion_estimada(self) -> str:
        """Calcula duración estimada del proceso"""
        if not self.tiempo_inicio:
            return "N/A"

        tiempo_transcurrido = datetime.now() - self.tiempo_inicio
        return str(tiempo_transcurrido)

    def generar_reporte_monitoreo(self) -> Dict[str, Any]:
        """Genera reporte de monitoreo industrial"""
        if not self.ejecuciones:
            return {"error": "No hay ejecuciones registradas"}

        completados = [e for e in self.ejecuciones if e["resultado"].get("status") == "completed"]
        fallidos = [e for e in self.ejecuciones if e["resultado"].get("status") == "failed"]

        puntajes = [e["resultado"].get("puntaje_promedio", 0) for e in completados if
                    e["resultado"].get("puntaje_promedio")]

        reporte = {
            "metadata": {
                "total_planes": len(self.ejecuciones),
                "planes_completados": len(completados),
                "planes_fallidos": len(fallidos),
                "tasa_exito": len(completados) / len(self.ejecuciones) if self.ejecuciones else 0,
                "tiempo_total": str(datetime.now() - self.tiempo_inicio) if self.tiempo_inicio else "N/A",
                "timestamp": datetime.now().isoformat()
            },
            "analisis_rendimiento": {
                "puntaje_promedio_global": statistics.mean(puntajes) if puntajes else 0,
                "desviacion_estandar_global": statistics.stdev(puntajes) if len(puntajes) > 1 else 0,
                "mejor_plan": max(completados, key=lambda x: x["resultado"].get("puntaje_promedio", 0))[
                    "nombre_plan"] if completados else "N/A",
                "peor_plan": min(completados, key=lambda x: x["resultado"].get("puntaje_promedio", 100))[
                    "nombre_plan"] if completados else "N/A"
            },
            "diagnostico_sistema": self._generar_diagnostico_sistema(completados, fallidos),
            "recomendaciones_operativas": self._generar_recomendaciones_operativas(fallidos)
        }

        return reporte

    def _generar_diagnostico_sistema(self, completados: List[Dict], fallidos: List[Dict]) -> Dict[str, str]:
        """Genera diagnóstico del sistema de evaluación"""
        tasa_exito = len(completados) / (len(completados) + len(fallidos)) if (len(completados) + len(
            fallidos)) > 0 else 0

        if tasa_exito >= 0.95:
            estado = "ÓPTIMO"
            descripcion = "Sistema operando con máxima eficiencia y estabilidad"
        elif tasa_exito >= 0.85:
            estado = "BUENO"
            descripcion = "Sistema operando con buen nivel de eficiencia, requiere monitoreo"
        elif tasa_exito >= 0.70:
            estado = "ADECUADO"
            descripcion = "Sistema operando a nivel aceptable, requiere mejoras puntuales"
        else:
            estado = "CRÍTICO"
            descripcion = "Sistema con problemas significativos, requiere intervención urgente"

        return {
            "estado_sistema": estado,
            "descripcion_estado": descripcion,
            "tasa_exito": tasa_exito,
            "planes_analizados": len(completados) + len(fallidos)
        }

    def _generar_recomendaciones_operativas(self, fallidos: List[Dict]) -> List[str]:
        """Genera recomendaciones operativas basadas en fallos"""
        recomendaciones = []

        if len(fallidos) > 0:
            recomendaciones.append("🔧 ANALIZAR PLANES FALLIDOS: Revisar logs de error para identificar patrones")

            # Contar tipos de error
            errores = [f["resultado"].get("error", "") for f in fallidos]
            if any("PDF" in str(error) for error in errores):
                recomendaciones.append("📄 VERIFICAR FORMATO PDF: Algunos archivos pueden estar corruptos o protegidos")

            if any("memoria" in str(error).lower() or "memory" in str(error).lower() for error in errores):
                recomendaciones.append("🧠 OPTIMIZAR MEMORIA: Reducir tamaño de batch o aumentar recursos del sistema")

        return recomendaciones

    def generar_dump_emergencia(self, output_dir: Path) -> Path:
        """Genera dump de emergencia con estado actual del monitoreo de forma thread-safe"""
        with self.lock:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dump_path = output_dir / f"dump_emergencia_monitoreo_{timestamp}.json"
            
            estado_emergencia = {
                "timestamp_dump": datetime.now().isoformat(),
                "sistema_interrumpido": self.interrumpido,
                "trabajadores_activos": list(self.trabajadores_activos),
                "tiempo_inicio": self.tiempo_inicio.isoformat() if self.tiempo_inicio else None,
                "ejecuciones_completadas": len([e for e in self.ejecuciones if e["resultado"].get("status") == "completed"]),
                "ejecuciones_fallidas": len([e for e in self.ejecuciones if e["resultado"].get("status") == "failed"]),
                "ejecuciones_parciales": self.ejecuciones,
                "estadisticas_parciales": self._calcular_estadisticas_parciales(),
                "recomendaciones_recuperacion": self._generar_recomendaciones_recuperacion()
            }
            
            try:
                with open(dump_path, 'w', encoding='utf-8') as f:
                    json.dump(estado_emergencia, f, indent=2, ensure_ascii=False)
                self.logger.info(f"🚨 Dump de emergencia guardado: {dump_path}")
                return dump_path
            except Exception as e:
                self.logger.error(f"❌ Error guardando dump de emergencia: {e}")
                raise

    def _calcular_estadisticas_parciales(self) -> Dict[str, Any]:
        """Calcula estadísticas parciales de las ejecuciones completadas hasta el momento"""
        completados = [e for e in self.ejecuciones if e["resultado"].get("status") == "completed"]
        puntajes = [e["resultado"].get("puntaje_promedio", 0) for e in completados if e["resultado"].get("puntaje_promedio")]
        
        if not puntajes:
            return {"mensaje": "No hay ejecuciones completadas para estadísticas"}
        
        return {
            "total_completados": len(completados),
            "puntaje_promedio": statistics.mean(puntajes),
            "puntaje_mediana": statistics.median(puntajes),
            "desviacion_estandar": statistics.stdev(puntajes) if len(puntajes) > 1 else 0,
            "puntaje_maximo": max(puntajes),
            "puntaje_minimo": min(puntajes),
            "planes_excelentes": len([p for p in puntajes if p >= 85]),
            "planes_aceptables": len([p for p in puntajes if 70 <= p < 85]),
            "planes_deficientes": len([p for p in puntajes if p < 70])
        }

    def _generar_recomendaciones_recuperacion(self) -> List[str]:
        """Genera recomendaciones para recuperar el procesamiento después de una interrupción"""
        recomendaciones = [
            "Revisar el dump de emergencia para identificar planes completados",
            "Verificar la integridad de los archivos de salida generados",
            "Reanudar procesamiento solo con planes no completados",
            "Considerar reducir paralelismo si se detectaron problemas de recursos"
        ]
        
        fallidos = [e for e in self.ejecuciones if e["resultado"].get("status") == "failed"]
        if len(fallidos) > len(self.ejecuciones) * 0.2:
            recomendaciones.append("Alta tasa de fallos detectada - revisar configuración del sistema")
        
        return recomendaciones


# ==================== GESTIÓN DE SEÑALES Y TERMINACIÓN GRACIOSA ====================

# Variables globales para manejo de señales thread-safe
_sistema_monitoreo_global = None
_output_dir_global = None
_signal_handler_lock = threading.RLock()

def signal_handler(signum, frame):
    """Manejador de señales thread-safe para terminación graciosa"""
    global _sistema_monitoreo_global, _output_dir_global
    
    with _signal_handler_lock:
        LOGGER.warning(f"🚨 Señal {signum} recibida - Iniciando terminación graciosa...")
        
        if _sistema_monitoreo_global:
            try:
                # Terminar trabajadores activos
                _sistema_monitoreo_global.terminar_trabajadores()
                
                # Generar dump de emergencia
                if _output_dir_global:
                    dump_path = _sistema_monitoreo_global.generar_dump_emergencia(_output_dir_global)
                    LOGGER.warning(f"📊 Estado del monitoreo guardado en: {dump_path}")
                else:
                    # Fallback al directorio actual si no hay output_dir configurado
                    output_fallback = Path("resultados_evaluacion_industrial")
                    output_fallback.mkdir(exist_ok=True)
                    dump_path = _sistema_monitoreo_global.generar_dump_emergencia(output_fallback)
                    LOGGER.warning(f"📊 Estado del monitoreo guardado en: {dump_path}")
                    
            except Exception as e:
                LOGGER.error(f"❌ Error durante terminación graciosa: {e}")
        
        LOGGER.warning("🛑 Terminación graciosa completada - Saliendo del sistema...")
        sys.exit(1)

def atexit_handler():
    """Manejador atexit thread-safe para terminación inesperada"""
    global _sistema_monitoreo_global, _output_dir_global
    
    with _signal_handler_lock:
        LOGGER.warning("🚨 Terminación inesperada detectada - Guardando estado...")
        
        if _sistema_monitoreo_global:
            try:
                if _output_dir_global:
                    dump_path = _sistema_monitoreo_global.generar_dump_emergencia(_output_dir_global)
                    LOGGER.warning(f"📊 Estado del monitoreo guardado en: {dump_path}")
                else:
                    # Fallback al directorio actual
                    output_fallback = Path("resultados_evaluacion_industrial")
                    output_fallback.mkdir(exist_ok=True)
                    dump_path = _sistema_monitoreo_global.generar_dump_emergencia(output_fallback)
                    LOGGER.warning(f"📊 Estado del monitoreo guardado en: {dump_path}")
                    
                # Generar estadísticas finales
                if _sistema_monitoreo_global.ejecuciones:
                    reporte_final = _sistema_monitoreo_global.generar_reporte_monitoreo()
                    LOGGER.warning(f"📈 Tasa de éxito parcial: {reporte_final.get('metadata', {}).get('tasa_exito', 0):.1%}")
                    
            except Exception as e:
                LOGGER.error(f"❌ Error en handler atexit: {e}")

def procesar_plan_industrial_con_monitoreo(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    """Wrapper thread-safe para procesar plan con registro de trabajador"""
    global _sistema_monitoreo_global
    
    trabajador_id = f"worker_{threading.current_thread().ident}_{pdf_path.stem}"
    
    try:
        # Registrar trabajador activo
        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.registrar_trabajador(trabajador_id)
        
        # Procesar plan normal
        resultado = procesar_plan_industrial(pdf_path)
        
        return resultado
        
    except Exception as e:
        LOGGER.error(f"❌ Error en trabajador {trabajador_id}: {e}")
        return pdf_path.stem, {"status": "failed", "error": str(e)}
        
    finally:
        # Desregistrar trabajador
        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.desregistrar_trabajador(trabajador_id)

# ==================== FUNCIÓN PRINCIPAL INDUSTRIAL CON ARGUMENTOS ====================
def main():
    """Función principal industrial con procesamiento batch, monitoreo y manejo de señales"""
    global _sistema_monitoreo_global, _output_dir_global
    
    parser = argparse.ArgumentParser(
        description="Sistema Industrial de Evaluación de Políticas Públicas v8.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python Decatalogo_principal.py ./planes_desarrollo/
  python Decatalogo_principal.py ./plan.pdf --max-segmentos 1000
  python Decatalogo_principal.py ./planes/ --max-segmentos 500 --batch-size 64
        """
    )
    
    # Configure device
    parser = add_device_args(parser)
    
    parser.add_argument(
        "input_path",
        help="Directorio con archivos PDF o archivo PDF individual"
    )
    
    parser.add_argument(
        "--max-segmentos",
        type=int,
        default=None,
        help="Límite total de segmentos de texto procesados globalmente (default: sin límite)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int, 
        default=32,
        help="Tamaño de batch para procesamiento de embeddings (default: 32)"
    )

# ==================== FUNCIÓN PRINCIPAL INDUSTRIAL ====================
def main():
    """Función principal industrial con procesamiento batch y monitoreo"""
    parser = argparse.ArgumentParser(
        description="Sistema Industrial de Evaluación de Políticas Públicas v8.0",
        epilog="Ejemplo: python Decatalogo_principal.py --seed 42 ./planes_desarrollo/"
    )
    parser.add_argument('input_path', 
                       help='Directorio con archivos PDF o archivo PDF individual')
    parser.add_argument('--seed', 
                       type=int, 
                       help='Semilla para reproducibilidad determinista (opcional)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path).expanduser()
    output_dir = Path("resultados_evaluacion_industrial")
    output_dir.mkdir(exist_ok=True)
    
    # Configure device
    device_config = configure_device_from_args(args)
    LOGGER.info(f"🔧 Dispositivo configurado: {device_config.get_device()}")
    
    # Get device information
    device_info = device_config.get_device_info()
    base_seed = args.seed

    # Inicializar sistema de monitoreo industrial
    sistema_monitoreo = SistemaMonitoreoIndustrial()
    _sistema_monitoreo_global = sistema_monitoreo
    sistema_monitoreo.iniciar_monitoreo()
    
    # Configurar manejadores de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(atexit_handler)

    LOGGER.info(f"🚀 Iniciando sistema industrial de evaluación de políticas públicas v8.1")
    LOGGER.info(f"📁 Directorio de entrada: {input_path}")
    if args.max_segmentos:
    
    if base_seed is not None:
        LOGGER.info(f"🎲 Modo determinista activado con semilla base: {base_seed}")
        LOGGER.warning("⚠️  NOTA: La extracción de texto PDF puede contener elementos no deterministas debido a comportamientos subyacentes de las librerías, pero las operaciones de análisis numérico serán reproducibles.")

    if input_path.is_dir():
        # Procesamiento batch industrial paralelo
        pdf_paths = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))

        if not pdf_paths:
            LOGGER.error("❌ No se encontraron archivos PDF en el directorio especificado")
            sys.exit(1)

        LOGGER.info(f"🏭 Procesando {len(pdf_paths)} planes de desarrollo en paralelo...")
        LOGGER.info(f"⚙️  Utilizando {os.cpu_count()} núcleos disponibles")

        # Función de procesamiento con argumentos adicionales y monitoreo
        def procesar_con_args(pdf_path):
            return procesar_plan_industrial_con_monitoreo(pdf_path)

        # Procesamiento paralelo industrial con monitoreo
        try:
            resultados_batch = Parallel(n_jobs=-1, backend='threading', verbose=10)(
                delayed(procesar_con_args)(pdf_path) for pdf_path in pdf_paths
            )
        # Procesamiento paralelo industrial
        resultados_batch = Parallel(n_jobs=-1, backend='threading', verbose=10)(
            delayed(procesar_plan_industrial)(pdf_path, base_seed) for pdf_path in pdf_paths
        )

        # Registrar resultados en sistema de monitoreo
        for nombre_plan, metrics in resultados_batch:
            sistema_monitoreo.registrar_ejecucion(nombre_plan, metrics)

        # Generar meta-reporte industrial agregado
        nombres, metrics_list = zip(*resultados_batch) if resultados_batch else ([], [])
        puntajes_validos = [m["puntaje_promedio"] for m in metrics_list if "error" not in m and "puntaje_promedio" in m]

        if puntajes_validos:
            meta_reporte = [
                "# 🏭 META-REPORTE INDUSTRIAL: EVALUACIÓN DE PLANES DE DESARROLLO",
                f"## 📊 Total de planes procesados: {len(puntajes_validos)}",
                f"## 🎯 Puntaje promedio global: {statistics.mean(puntajes_validos):.1f}/100",
                f"## 📈 Desviación estándar: {statistics.stdev(puntajes_validos):.1f}" if len(
                    puntajes_validos) > 1 else "",
                f"## 🕒 Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "## 📊 ESTADÍSTICAS DETALLADAS",
                f"- Planes excelentes (≥85): {len([p for p in puntajes_validos if p >= 85])}",
                f"- Planes aceptables (70-84): {len([p for p in puntajes_validos if 70 <= p < 85])}",
                f"- Planes deficientes (<70): {len([p for p in puntajes_validos if p < 70])}",
                "",
                "## 🏆 TOP 5 PLANES POR PUNTAJE",
            ]

            # Top 5 planes
            planes_ordenados = sorted(
                [(n, m) for n, m in zip(nombres, metrics_list) if "error" not in m],
                key=lambda x: x[1].get("puntaje_promedio", 0),
                reverse=True
            )[:5]

            for i, (nombre, metricas) in enumerate(planes_ordenados, 1):
                meta_reporte.append(f"{i}. {nombre}: {metricas.get('puntaje_promedio', 0):.1f}/100")

            meta_reporte.extend([
                "",
                "## 📉 PLANES QUE REQUIEREN ATENCIÓN PRIORITARIA",
            ])

            # Bottom 5 planes
            planes_deficientes = sorted(
                [(n, m) for n, m in zip(nombres, metrics_list) if
                 "error" not in m and m.get("puntaje_promedio", 100) < 70],
                key=lambda x: x[1].get("puntaje_promedio", 0)
            )[:5]

            if planes_deficientes:
                for i, (nombre, metricas) in enumerate(planes_deficientes, 1):
                    meta_reporte.append(f"{i}. {nombre}: {metricas.get('puntaje_promedio', 0):.1f}/100")
            else:
                meta_reporte.append("✅ Todos los planes superan el umbral mínimo de 70/100")

            # Generar reporte de monitoreo
            reporte_monitoreo = sistema_monitoreo.generar_reporte_monitoreo()

            meta_reporte.extend([
                "",
                "## 🛡️  DIAGNÓSTICO DEL SISTEMA INDUSTRIAL",
                f"- Estado del sistema: {reporte_monitoreo['diagnostico_sistema']['estado_sistema']}",
                f"- Descripción: {reporte_monitoreo['diagnostico_sistema']['descripcion_estado']}",
                f"- Tasa de éxito: {reporte_monitoreo['diagnostico_sistema']['tasa_exito']:.1%}",
                f"- Planes analizados: {reporte_monitoreo['diagnostico_sistema']['planes_analizados']}",
                "",
                "## 💡 RECOMENDACIONES OPERATIVAS",
            ])

            for i, recomendacion in enumerate(reporte_monitoreo['recomendaciones_operativas'], 1):
                meta_reporte.append(f"{i}. {recomendacion}")

            meta_reporte.append("")
            meta_reporte.append("---")
            meta_reporte.append("*Generado por Sistema Industrial de Evaluación de Políticas Públicas v8.0*")

            # Guardar meta-reporte
            meta_path = output_dir / "meta_evaluacion_industrial.md"
            meta_path.write_text("\n".join(meta_reporte), encoding='utf-8')

            # Guardar reporte de monitoreo en JSON
            monitoreo_path = output_dir / "reporte_monitoreo_industrial.json"
            with open(monitoreo_path, 'w', encoding='utf-8') as f:
                json.dump(reporte_monitoreo, f, indent=2, ensure_ascii=False)

            LOGGER.info(f"✅✅✅ META-REPORTE INDUSTRIAL GENERADO: {meta_path}")
            LOGGER.info(f"📊 REPORTE DE MONITOREO: {monitoreo_path}")

        else:
            LOGGER.error("❌❌❌ NO SE PROCESARON PLANES VÁLIDOS")
            sys.exit(1)

    else:
        # Modo single-file industrial
        LOGGER.info(f"📄 Procesando archivo individual: {input_path.name}")
        nombre_plan, metrics = procesar_plan_industrial(input_path, base_seed)

            if "error" not in metrics:
                LOGGER.info(f"✅✅✅ EVALUACIÓN COMPLETADA PARA {nombre_plan}")
                LOGGER.info(f"📊 PUNTAJE FINAL: {metrics.get('puntaje_promedio', 0):.1f}/100")
                LOGGER.info(f"🏭 NIVEL DE MADUREZ: {metrics.get('nivel_madurez_predominante', 'N/A')}")
                if args.max_segmentos:
                    LOGGER.info(f"📊 SEGMENTOS PROCESADOS: {metrics.get('segmentos_procesados', 'N/A')}/{metrics.get('segmentos_totales', 'N/A')}")
            else:
                LOGGER.error(f"❌❌❌ ERROR PROCESANDO {nombre_plan}: {metrics.get('error', 'Desconocido')}")
                sys.exit(1)
        except KeyboardInterrupt:
            LOGGER.warning("🚨 Procesamiento interrumpido por el usuario")
            # El signal handler se encargará de la limpieza
            raise

    # Mostrar resumen final
    print(f"\n{'=' * 80}")
    print("🏭 EVALUACIÓN INDUSTRIAL COMPLETADA")
    print(f"{'=' * 80}")
    print(f"📁 Resultados disponibles en: {output_dir.absolute()}")
    if input_path.is_dir():
        print(f"📊 Meta-reporte: {output_dir / 'meta_evaluacion_industrial.md'}")
    print(
        f"⏱️  Tiempo total de ejecución: {datetime.now() - sistema_monitoreo.tiempo_inicio if sistema_monitoreo.tiempo_inicio else 'N/A'}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # El signal handler ya se encargó de la limpieza
        LOGGER.info("🛑 Ejecución interrumpida - Limpieza completada por signal handler")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"❌❌❌ ERROR CRÍTICO EN EJECUCIÓN INDUSTRIAL: {e}")
        # El atexit handler se encargará de guardar el estado
        sys.exit(1)
