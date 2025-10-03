"""
Cluster Analyzer Module

Analyzes decalogo points grouped by policy clusters, providing a thematic
analysis that connects related policy domains beyond individual evaluations.

This module ensures that the system not only evaluates individual questions
but also produces integrated analysis across thematically related policy areas.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union

import numpy as np
import networkx as nx

from Decatalogo_principal import ResultadoDimensionIndustrial
from Decatalogo_evaluador import EvaluacionPuntoCompleto
from dag_validation import AdvancedDAGValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterMetadata:
    """Metadata describing a policy cluster."""
    cluster_id: str
    nombre: str
    descripcion: str
    puntos_decalogo: List[int]
    peso_cluster: float
    areas_tematicas: List[str]
    objetivos_desarrollo: List[str]
    indicadores_clave: List[str]


@dataclass
class ClusterEvaluacion:
    """Complete evaluation of a policy cluster across all dimensions."""
    cluster_id: str
    nombre: str 
    puntos_evaluados: List[int]
    puntaje_promedio: float
    puntaje_ponderado: float
    clasificacion: str
    evaluacion_por_dimension: Dict[str, float]
    evaluacion_coherencia: float
    puntos_fuertes: List[str]
    puntos_debiles: List[str]
    recomendaciones: List[str]
    integracion_tematica: float


@dataclass
class AnalisisMeso:
    """Meso-level analysis connecting clusters and their interdependencies."""
    clusters_evaluados: List[ClusterEvaluacion]
    interdependencias: Dict[str, float]
    coherencia_global: float
    brechas_entre_clusters: List[Dict[str, Any]]
    puntaje_global: float
    recomendaciones_integracion: List[str]
    matriz_coherencia: Dict[str, Dict[str, float]]
    red_interdependencia: Any  # NetworkX graph


class ClusterAnalyzer:
    """
    Analyzer for policy clusters that provides thematic integration.
    
    This class ensures that the evaluation includes not just answers to individual
    questions but also thematic integration across related policy domains.
    """
    
    def __init__(self, decalogo_full_path: Optional[str] = None, dnp_standards_path: Optional[str] = None):
        """Initialize the cluster analyzer with configuration files."""
        # Set default paths if not provided
        if decalogo_full_path is None:
            decalogo_full_path = os.path.join(os.path.dirname(__file__), "DECALOGO_FULL.json")
        if dnp_standards_path is None:
            dnp_standards_path = os.path.join(os.path.dirname(__file__), "DNP_STANDARDS.json")
        
        # Load configuration
        self.decalogo_full = self._load_json(decalogo_full_path)
        self.dnp_standards = self._load_json(dnp_standards_path)
        
        # Extract cluster metadata
        self.clusters = self._extract_cluster_metadata()
        
        # Extract evaluation criteria
        self.criterios_cluster = self.decalogo_full.get("criterios_evaluacion_cluster", {})
        self.cluster_standards = self.dnp_standards.get("cluster_standards", {})
        self.evaluation_criteria = self.dnp_standards.get("cluster_evaluation_criteria", {})
        
        # Initialize interdependency graph
        self.interdependency_graph = self._build_interdependency_graph()
        
    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON file with error handling."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}
    
    def _extract_cluster_metadata(self) -> Dict[str, ClusterMetadata]:
        """Extract cluster metadata from DECALOGO_FULL."""
        clusters = {}
        
        for cluster in self.decalogo_full.get("clusters_politica", []):
            clusters[cluster["id"]] = ClusterMetadata(
                cluster_id=cluster["id"],
                nombre=cluster["nombre"],
                descripcion=cluster["descripcion"],
                puntos_decalogo=cluster["puntos_decalogo"],
                peso_cluster=cluster["peso_cluster"],
                areas_tematicas=cluster["areas_tematicas"],
                objetivos_desarrollo=cluster.get("objetivos_desarrollo_relacionados", []),
                indicadores_clave=cluster.get("indicadores_clave", [])
            )
        
        return clusters
    
    def _build_interdependency_graph(self) -> nx.DiGraph:
        """Build a graph representing interdependencies between clusters."""
        G = nx.DiGraph()
        
        # Add nodes (clusters)
        for cluster_id in self.clusters.keys():
            G.add_node(cluster_id)
        
        # Add edges (interdependencies)
        interdependencies = self.evaluation_criteria.get("cluster_interdependencies", {})
        for relation, weight in interdependencies.items():
            try:
                source, target = relation.split("_")
                G.add_edge(source, target, weight=weight)
                G.add_edge(target, source, weight=weight)  # Bidirectional
            except ValueError:
                logger.warning(f"Invalid interdependency format: {relation}")
        
        return G
        
    def analyze_cluster(
        self, 
        cluster_id: str, 
        punto_evaluaciones: Dict[int, EvaluacionPuntoCompleto]
    ) -> ClusterEvaluacion:
        """
        Analyze a specific policy cluster based on its points' evaluations.
        
        Args:
            cluster_id: ID of the cluster to analyze
            punto_evaluaciones: Dictionary mapping point IDs to their evaluations
            
        Returns:
            ClusterEvaluacion object with cluster analysis
        """
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            raise ValueError(f"Unknown cluster ID: {cluster_id}")
        
        # Get evaluations for points in this cluster
        evaluaciones_relevantes = {}
        for punto_id in cluster.puntos_decalogo:
            if punto_id in punto_evaluaciones:
                evaluaciones_relevantes[punto_id] = punto_evaluaciones[punto_id]
        
        if not evaluaciones_relevantes:
            logger.warning(f"No evaluations found for cluster {cluster_id}")
            return self._create_empty_cluster_evaluation(cluster)
        
        # Calculate average scores by dimension
        dimension_scores = {"DE-1": [], "DE-2": [], "DE-3": [], "DE-4": []}
        
        for eval_punto in evaluaciones_relevantes.values():
            for eval_dim in eval_punto.evaluaciones_dimensiones:
                dimension_id = eval_dim.dimension
                dimension_scores[dimension_id].append(eval_dim.puntaje_dimension)
        
        # Calculate average dimension scores
        evaluacion_por_dimension = {}
        for dim_id, scores in dimension_scores.items():
            if scores:
                evaluacion_por_dimension[dim_id] = sum(scores) / len(scores)
            else:
                evaluacion_por_dimension[dim_id] = 0.0
        
        # Calculate weighted dimension scores based on cluster-specific weights
        dimension_weights = self.evaluation_criteria.get("dimension_weights_by_cluster", {}).get(cluster_id)
        if not dimension_weights:
            dimension_weights = {dim: 0.25 for dim in ["DE-1", "DE-2", "DE-3", "DE-4"]}
        
        puntaje_ponderado = sum(
            evaluacion_por_dimension.get(dim_id, 0) * weight 
            for dim_id, weight in dimension_weights.items()
        )
        
        # Calculate simple average
        puntaje_promedio = sum(
            eval_punto.puntaje_agregado_punto 
            for eval_punto in evaluaciones_relevantes.values()
        ) / len(evaluaciones_relevantes)
        
        # Calculate thematic integration score
        integracion_tematica = self._calculate_thematic_integration(
            cluster_id, evaluaciones_relevantes
        )
        
        # Calculate coherence evaluation
        evaluacion_coherencia = self._calculate_coherence_evaluation(
            cluster_id, evaluaciones_relevantes
        )
        
        # Determine classification
        clasificacion = self._determine_cluster_classification(puntaje_ponderado)
        
        # Generate strengths, weaknesses, and recommendations
        puntos_fuertes = self._identify_cluster_strengths(cluster_id, evaluaciones_relevantes)
        puntos_debiles = self._identify_cluster_weaknesses(cluster_id, evaluaciones_relevantes)
        recomendaciones = self._generate_cluster_recommendations(
            cluster_id, puntos_debiles, evaluacion_por_dimension
        )
        
        return ClusterEvaluacion(
            cluster_id=cluster_id,
            nombre=cluster.nombre,
            puntos_evaluados=list(evaluaciones_relevantes.keys()),
            puntaje_promedio=puntaje_promedio,
            puntaje_ponderado=puntaje_ponderado,
            clasificacion=clasificacion,
            evaluacion_por_dimension=evaluacion_por_dimension,
            evaluacion_coherencia=evaluacion_coherencia,
            puntos_fuertes=puntos_fuertes,
            puntos_debiles=puntos_debiles,
            recomendaciones=recomendaciones,
            integracion_tematica=integracion_tematica
        )
    
    def _calculate_thematic_integration(
        self, 
        cluster_id: str,
        evaluaciones: Dict[int, EvaluacionPuntoCompleto]
    ) -> float:
        """
        Calculate thematic integration score for a cluster.
        This measures how well the different points in the cluster relate to each other.
        """
        # If only one point, integration is perfect by definition
        if len(evaluaciones) <= 1:
            return 1.0
        
        # Extract common themes and areas
        areas_por_punto = {}
        for punto_id, evaluacion in evaluaciones.items():
            # Extract areas from evaluation texts
            areas = set()
            
            # From fortalezas
            for fortaleza in evaluacion.fortalezas_identificadas:
                areas.update(self._extract_keywords(fortaleza))
            
            # From brechas
            for brecha in evaluacion.brechas_identificadas:
                areas.update(self._extract_keywords(brecha))
                
            areas_por_punto[punto_id] = areas
        
        # Calculate area overlap between points
        overlaps = []
        point_ids = list(areas_por_punto.keys())
        
        for i in range(len(point_ids)):
            for j in range(i + 1, len(point_ids)):
                point1 = point_ids[i]
                point2 = point_ids[j]
                
                areas1 = areas_por_punto[point1]
                areas2 = areas_por_punto[point2]
                
                if not areas1 or not areas2:
                    continue
                    
                # Calculate Jaccard similarity
                overlap = len(areas1.intersection(areas2)) / len(areas1.union(areas2))
                overlaps.append(overlap)
        
        # Average overlap as integration score
        return sum(overlaps) / len(overlaps) if overlaps else 0.5
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract thematic keywords from text."""
        # Simple extraction based on word presence
        # In a real system, this would use NLP techniques
        keywords = set()
        
        # Check each cluster's thematic areas
        for cluster in self.clusters.values():
            for area in cluster.areas_tematicas:
                if area.lower() in text.lower():
                    keywords.add(area.lower())
        
        return keywords
    
    def _calculate_coherence_evaluation(
        self, 
        cluster_id: str,
        evaluaciones: Dict[int, EvaluacionPuntoCompleto]
    ) -> float:
        """
        Calculate coherence evaluation for a cluster.
        This measures how consistently the different points perform across dimensions.
        """
        # Extract dimension scores for all points
        dimension_scores = {dim: [] for dim in ["DE-1", "DE-2", "DE-3", "DE-4"]}
        
        for evaluacion in evaluaciones.values():
            for eval_dim in evaluacion.evaluaciones_dimensiones:
                dimension_id = eval_dim.dimension
                dimension_scores[dimension_id].append(eval_dim.puntaje_dimension)
        
        # Calculate standard deviation for each dimension
        std_devs = []
        for dim, scores in dimension_scores.items():
            if len(scores) >= 2:  # Need at least 2 points to calculate std dev
                std_devs.append(np.std(scores))
        
        if not std_devs:
            return 1.0  # Perfect coherence if only one point
        
        # Convert standard deviation to coherence score (lower std dev = higher coherence)
        # Normalize to 0-1 range where 1 is perfect coherence
        avg_std = sum(std_devs) / len(std_devs)
        coherence = 1 - min(avg_std / 20, 1)  # 20 is max reasonable std dev
        
        return coherence
    
    def _determine_cluster_classification(self, score: float) -> str:
        """Determine classification based on cluster score."""
        for classification, criteria in sorted(
            self.criterios_cluster.items(),
            key=lambda x: x[1].get("umbral", 0),
            reverse=True
        ):
            if score >= criteria.get("umbral", 0):
                return classification
        
        return "impacto_insuficiente"  # Default
    
    def _identify_cluster_strengths(
        self, 
        cluster_id: str,
        evaluaciones: Dict[int, EvaluacionPuntoCompleto]
    ) -> List[str]:
        """Identify key strengths of a cluster."""
        # Extract top strengths from individual points
        all_strengths = []
        for evaluacion in evaluaciones.values():
            all_strengths.extend(evaluacion.fortalezas_identificadas[:2])  # Top 2 strengths
        
        # Prioritize strengths related to cluster themes
        cluster = self.clusters.get(cluster_id)
        if cluster:
            prioritized_strengths = []
            
            for strength in all_strengths:
                for area in cluster.areas_tematicas:
                    if area.lower() in strength.lower():
                        prioritized_strengths.append(strength)
                        break
            
            # Add some non-prioritized if needed
            remaining = [s for s in all_strengths if s not in prioritized_strengths]
            prioritized_strengths.extend(remaining[:3 - len(prioritized_strengths)])
            
            return prioritized_strengths[:3]  # Return top 3 strengths
        
        return all_strengths[:3]  # Return top 3 if no cluster
    
    def _identify_cluster_weaknesses(
        self, 
        cluster_id: str,
        evaluaciones: Dict[int, EvaluacionPuntoCompleto]
    ) -> List[str]:
        """Identify key weaknesses of a cluster."""
        # Similar to strengths but for weaknesses
        all_weaknesses = []
        for evaluacion in evaluaciones.values():
            all_weaknesses.extend(evaluacion.brechas_identificadas[:2])  # Top 2 weaknesses
        
        # Prioritize weaknesses related to cluster themes
        cluster = self.clusters.get(cluster_id)
        if cluster:
            prioritized_weaknesses = []
            
            for weakness in all_weaknesses:
                for area in cluster.areas_tematicas:
                    if area.lower() in weakness.lower():
                        prioritized_weaknesses.append(weakness)
                        break
            
            # Add some non-prioritized if needed
            remaining = [w for w in all_weaknesses if w not in prioritized_weaknesses]
            prioritized_weaknesses.extend(remaining[:3 - len(prioritized_weaknesses)])
            
            return prioritized_weaknesses[:3]  # Return top 3 weaknesses
        
        return all_weaknesses[:3]  # Return top 3 if no cluster
    
    def _generate_cluster_recommendations(
        self, 
        cluster_id: str,
        weaknesses: List[str],
        dimension_scores: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for a cluster."""
        recommendations = []
        
        # Get lowest-performing dimension
        lowest_dim = min(dimension_scores.items(), key=lambda x: x[1], default=("", 0))
        
        # Add dimension-specific recommendation
        if lowest_dim[0]:
            dim_name = {
                "DE-1": "Marco Lógico de Intervención",
                "DE-2": "Inclusión Temática",
                "DE-3": "Participación y Gobernanza",
                "DE-4": "Orientación a Resultados"
            }.get(lowest_dim[0], lowest_dim[0])
            
            recommendations.append(
                f"Fortalecer la dimensión de {dim_name} para mejorar el desempeño integral del cluster."
            )
        
        # Add cluster-specific recommendations
        cluster = self.clusters.get(cluster_id)
        if cluster:
            # Add indicator-based recommendation
            recommendations.append(
                f"Implementar sistema de seguimiento para los indicadores clave del cluster: "
                f"{', '.join(cluster.indicadores_clave[:3])}."
            )
            
            # Add thematic recommendation
            recommendations.append(
                f"Desarrollar estrategias de articulación en torno a las áreas temáticas de "
                f"{', '.join(cluster.areas_tematicas[:3])}."
            )
            
            # Add weakness-based recommendation if available
            if weaknesses:
                recommendations.append(
                    f"Atender prioritariamente: {weaknesses[0]}"
                )
        
        return recommendations
    
    def _create_empty_cluster_evaluation(self, cluster: ClusterMetadata) -> ClusterEvaluacion:
        """Create an empty cluster evaluation when no point evaluations are available."""
        return ClusterEvaluacion(
            cluster_id=cluster.cluster_id,
            nombre=cluster.nombre,
            puntos_evaluados=[],
            puntaje_promedio=0.0,
            puntaje_ponderado=0.0,
            clasificacion="impacto_insuficiente",
            evaluacion_por_dimension={dim: 0.0 for dim in ["DE-1", "DE-2", "DE-3", "DE-4"]},
            evaluacion_coherencia=0.0,
            puntos_fuertes=[],
            puntos_debiles=[f"No se encontraron evaluaciones para los puntos del cluster {cluster.nombre}"],
            recomendaciones=[f"Realizar evaluación completa de los puntos del Decálogo en el cluster {cluster.nombre}"],
            integracion_tematica=0.0
        )
    
    def perform_meso_analysis(
        self, 
        cluster_evaluaciones: List[ClusterEvaluacion]
    ) -> AnalisisMeso:
        """
        Perform meso-level analysis connecting clusters and their interdependencies.
        
        Args:
            cluster_evaluaciones: List of cluster evaluations
            
        Returns:
            AnalisisMeso object with meso-level analysis
        """
        if not cluster_evaluaciones:
            return self._create_empty_meso_analysis()
        
        # Create dictionary for quick lookup
        eval_dict = {eval.cluster_id: eval for eval in cluster_evaluaciones}
        
        # Calculate interdependencies
        interdependencies = {}
        for edge in self.interdependency_graph.edges(data=True):
            source, target, data = edge
            
            if source in eval_dict and target in eval_dict:
                weight = data.get("weight", 0.5)
                source_score = eval_dict[source].puntaje_ponderado
                target_score = eval_dict[target].puntaje_ponderado
                
                # Calculate interdependency strength
                strength = weight * (source_score + target_score) / 200
                interdependencies[f"{source}_{target}"] = strength
        
        # Calculate coherence matrix
        coherence_matrix = {}
        for c1 in eval_dict.keys():
            coherence_matrix[c1] = {}
            for c2 in eval_dict.keys():
                if c1 == c2:
                    coherence_matrix[c1][c2] = 1.0
                else:
                    interdep_key = f"{c1}_{c2}"
                    rev_key = f"{c2}_{c1}"
                    coherence_matrix[c1][c2] = interdependencies.get(interdep_key, 
                                             interdependencies.get(rev_key, 0.3))
        
        # Calculate global coherence
        coherence_values = []
        for row in coherence_matrix.values():
            coherence_values.extend([v for k, v in row.items() if k != row])
        
        coherencia_global = sum(coherence_values) / len(coherence_values) if coherence_values else 0.5
        
        # Calculate global score
        cluster_scores = []
        for eval in cluster_evaluaciones:
            weight = self.clusters.get(eval.cluster_id, ClusterMetadata(
                cluster_id=eval.cluster_id,
                nombre=eval.nombre,
                descripcion="",
                puntos_decalogo=[],
                peso_cluster=0.2,
                areas_tematicas=[],
                objetivos_desarrollo=[],
                indicadores_clave=[]
            )).peso_cluster
            
            cluster_scores.append((eval.puntaje_ponderado, weight))
        
        puntaje_global = sum(score * weight for score, weight in cluster_scores) / sum(weight for _, weight in cluster_scores)
        
        # Identify gaps between clusters
        brechas = []
        for i, eval1 in enumerate(cluster_evaluaciones):
            for j, eval2 in enumerate(cluster_evaluaciones):
                if i >= j:
                    continue
                
                # Check for significant score difference
                score_diff = abs(eval1.puntaje_ponderado - eval2.puntaje_ponderado)
                if score_diff > 15:  # Significant gap threshold
                    # Check if there's supposed to be strong interdependency
                    interdep_key = f"{eval1.cluster_id}_{eval2.cluster_id}"
                    rev_key = f"{eval2.cluster_id}_{eval1.cluster_id}"
                    interdep = interdependencies.get(interdep_key, interdependencies.get(rev_key, 0))
                    
                    if interdep > 0.4:  # Strong interdependency threshold
                        stronger = eval1 if eval1.puntaje_ponderado > eval2.puntaje_ponderado else eval2
                        weaker = eval2 if eval1.puntaje_ponderado > eval2.puntaje_ponderado else eval1
                        
                        brechas.append({
                            "cluster_fuerte": stronger.cluster_id,
                            "nombre_fuerte": stronger.nombre,
                            "puntaje_fuerte": stronger.puntaje_ponderado,
                            "cluster_debil": weaker.cluster_id,
                            "nombre_debil": weaker.nombre,
                            "puntaje_debil": weaker.puntaje_ponderado,
                            "diferencia": score_diff,
                            "interdependencia": interdep
                        })
        
        # Generate integration recommendations
        recomendaciones = self._generate_integration_recommendations(
            cluster_evaluaciones, brechas, coherencia_global
        )
        
        return AnalisisMeso(
            clusters_evaluados=cluster_evaluaciones,
            interdependencias=interdependencies,
            coherencia_global=coherencia_global,
            brechas_entre_clusters=brechas,
            puntaje_global=puntaje_global,
            recomendaciones_integracion=recomendaciones,
            matriz_coherencia=coherence_matrix,
            red_interdependencia=self.interdependency_graph
        )
    
    def _generate_integration_recommendations(
        self,
        evaluaciones: List[ClusterEvaluacion],
        brechas: List[Dict[str, Any]],
        coherencia_global: float
    ) -> List[str]:
        """Generate recommendations for improving integration between clusters."""
        recommendations = []
        
        # Recommendation based on global coherence
        if coherencia_global < 0.4:
            recommendations.append(
                "Desarrollar una estrategia integral de articulación entre clusters de política "
                "para mejorar la coherencia global del plan."
            )
        elif coherencia_global < 0.6:
            recommendations.append(
                "Fortalecer mecanismos de coordinación entre áreas temáticas relacionadas "
                "para mejorar la coherencia entre clusters."
            )
        
        # Recommendations based on gaps
        if brechas:
            for i, brecha in enumerate(brechas[:2]):  # Focus on top 2 gaps
                recommendations.append(
                    f"Atender la brecha entre {brecha['nombre_debil']} y {brecha['nombre_fuerte']} "
                    f"dado su alto nivel de interdependencia ({brecha['interdependencia']:.2f})."
                )
        
        # Identify weakest cluster
        if evaluaciones:
            weakest = min(evaluaciones, key=lambda x: x.puntaje_ponderado)
            recommendations.append(
                f"Priorizar el fortalecimiento del cluster {weakest.nombre} para mejorar "
                f"el desempeño integral del plan."
            )
        
        # Add strategic recommendation
        recommendations.append(
            "Implementar un sistema de monitoreo que vincule indicadores entre clusters "
            "para promover una visión integrada del desarrollo."
        )
        
        return recommendations
    
    def _create_empty_meso_analysis(self) -> AnalisisMeso:
        """Create an empty meso analysis when no cluster evaluations are available."""
        return AnalisisMeso(
            clusters_evaluados=[],
            interdependencias={},
            coherencia_global=0.0,
            brechas_entre_clusters=[],
            puntaje_global=0.0,
            recomendaciones_integracion=[
                "Realizar evaluaciones de clusters para permitir un análisis meso."
            ],
            matriz_coherencia={},
            red_interdependencia=nx.DiGraph()
        )
    
    def evaluate_all_clusters(
        self, 
        punto_evaluaciones: Dict[int, EvaluacionPuntoCompleto]
    ) -> Tuple[List[ClusterEvaluacion], AnalisisMeso]:
        """
        Evaluate all clusters and perform meso-level analysis.
        
        Args:
            punto_evaluaciones: Dictionary mapping point IDs to their evaluations
            
        Returns:
            Tuple of (list of cluster evaluations, meso-level analysis)
        """
        cluster_evaluations = []
        
        # Evaluate each cluster
        for cluster_id in self.clusters.keys():
            evaluation = self.analyze_cluster(cluster_id, punto_evaluaciones)
            cluster_evaluations.append(evaluation)
        
        # Perform meso-level analysis
        meso_analysis = self.perform_meso_analysis(cluster_evaluations)
        
        return cluster_evaluations, meso_analysis


def create_cluster_analyzer() -> ClusterAnalyzer:
    """Factory function to create a cluster analyzer."""
    return ClusterAnalyzer()


# Standalone execution for testing
if __name__ == "__main__":
    # Example usage
    analyzer = create_cluster_analyzer()
    
    # Print cluster information
    for cluster_id, cluster in analyzer.clusters.items():
        print(f"Cluster {cluster_id}: {cluster.nombre}")
        print(f"  Points: {cluster.puntos_decalogo}")
        print(f"  Areas: {cluster.areas_tematicas}")
        print(f"  Weight: {cluster.peso_cluster}")
        print()
    
    # Print interdependency information
    print("Cluster interdependencies:")
    for edge in analyzer.interdependency_graph.edges(data=True):
        source, target, data = edge
        print(f"  {source} → {target}: {data.get('weight', 0.5)}")
