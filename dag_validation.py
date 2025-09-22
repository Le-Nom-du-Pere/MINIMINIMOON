# coding=utf-8
"""
Deterministic Monte Carlo sampling for DAG validation.

This module provides tools for validating Directed Acyclic Graphs (DAGs) using
Monte Carlo sampling with deterministic seeding based on plan names. The main
purpose is to assess the statistical significance of acyclicity in causal graphs.

Statistical Interpretation:
- p-value: Probability of observing acyclicity in random subgraphs under null hypothesis
- Lower p-values suggest stronger evidence against random acyclicity
- This is NOT a test of causal validity, only structural acyclicity
- Use with caution: statistical significance ≠ causal validity
- Results should be interpreted within domain knowledge context
"""

import hashlib
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools


@dataclass
class GraphNode:
    """Represents a node in the causal graph."""
    name: str
    dependencies: Set[str]


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo acyclicity testing."""
    plan_name: str
    seed: int
    total_iterations: int
    acyclic_count: int
    p_value: float
    subgraph_sizes: List[int]
    reproducible: bool


class DAGValidator:
    """Deterministic Monte Carlo sampling for DAG validation."""
    
    def __init__(self):
        self.graph_nodes: Dict[str, GraphNode] = {}
        self._rng = None
    
    def add_node(self, name: str, dependencies: Set[str] = None):
        """Add a node to the causal graph."""
        if dependencies is None:
            dependencies = set()
        self.graph_nodes[name] = GraphNode(name, dependencies)
    
    def add_edge(self, from_node: str, to_node: str):
        """Add a directed edge from one node to another."""
        if to_node not in self.graph_nodes:
            self.add_node(to_node)
        if from_node not in self.graph_nodes:
            self.add_node(from_node)
        
        self.graph_nodes[to_node].dependencies.add(from_node)
    
    def _create_seed_from_plan_name(self, plan_name: str) -> int:
        """Create a deterministic seed from plan name using hash function."""
        hash_obj = hashlib.sha256(plan_name.encode('utf-8'))
        # Use first 4 bytes of hash as seed to ensure deterministic behavior
        seed_bytes = hash_obj.digest()[:4]
        seed = int.from_bytes(seed_bytes, byteorder='big', signed=False)
        return seed
    
    def _initialize_rng(self, plan_name: str) -> int:
        """Initialize random number generator with plan name seed."""
        seed = self._create_seed_from_plan_name(plan_name)
        self._rng = random.Random(seed)
        return seed
    
    def _is_acyclic(self, nodes: Dict[str, GraphNode]) -> bool:
        """Check if a graph is acyclic using topological sort approach."""
        if not nodes:
            return True
        
        # Create adjacency representation
        in_degree = {name: 0 for name in nodes.keys()}
        adjacency = defaultdict(set)
        
        for node_name, node in nodes.items():
            for dep in node.dependencies:
                if dep in nodes:  # Only consider dependencies within subgraph
                    adjacency[dep].add(node_name)
                    in_degree[node_name] += 1
        
        # Kahn's algorithm for topological sorting
        queue = [name for name, degree in in_degree.items() if degree == 0]
        processed = 0
        
        while queue:
            current = queue.pop(0)
            processed += 1
            
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If we processed all nodes, the graph is acyclic
        return processed == len(nodes)
    
    def _generate_random_subgraph(self, min_size: int = 3, max_size: int = None) -> Dict[str, GraphNode]:
        """Generate a random subgraph from the main graph."""
        if not self.graph_nodes:
            return {}
        
        all_nodes = list(self.graph_nodes.keys())
        if max_size is None:
            max_size = len(all_nodes)
        
        max_size = min(max_size, len(all_nodes))
        min_size = min(min_size, max_size)
        
        subgraph_size = self._rng.randint(min_size, max_size)
        selected_nodes = self._rng.sample(all_nodes, subgraph_size)
        
        # Create subgraph with only selected nodes and their internal dependencies
        subgraph = {}
        for node_name in selected_nodes:
            original_node = self.graph_nodes[node_name]
            # Only keep dependencies that are also in the subgraph
            filtered_deps = original_node.dependencies.intersection(set(selected_nodes))
            subgraph[node_name] = GraphNode(node_name, filtered_deps)
        
        return subgraph
    
    def calculate_acyclicity_pvalue(
        self, 
        plan_name: str, 
        iterations: int = 1000,
        min_subgraph_size: int = 3,
        max_subgraph_size: int = None
    ) -> MonteCarloResult:
        """
        Calculate p-value for acyclicity using Monte Carlo sampling.
        
        Args:
            plan_name: Name of the plan (used for deterministic seeding)
            iterations: Number of Monte Carlo iterations
            min_subgraph_size: Minimum size of random subgraphs
            max_subgraph_size: Maximum size of random subgraphs
        
        Returns:
            MonteCarloResult with p-value and statistics
        
        Statistical Interpretation:
        The p-value represents the probability of observing acyclic structure
        in random subgraphs of the causal graph. This is a structural test,
        NOT a test of causal validity. Lower p-values suggest the observed
        acyclicity is unlikely to occur by chance alone.
        
        IMPORTANT: This does not validate causal relationships, only graph structure.
        Domain expertise is required for proper causal interpretation.
        """
        seed = self._initialize_rng(plan_name)
        
        if not self.graph_nodes:
            return MonteCarloResult(
                plan_name=plan_name,
                seed=seed,
                total_iterations=0,
                acyclic_count=0,
                p_value=1.0,
                subgraph_sizes=[],
                reproducible=True
            )
        
        acyclic_count = 0
        subgraph_sizes = []
        
        for _ in range(iterations):
            subgraph = self._generate_random_subgraph(min_subgraph_size, max_subgraph_size)
            subgraph_sizes.append(len(subgraph))
            
            if self._is_acyclic(subgraph):
                acyclic_count += 1
        
        p_value = acyclic_count / iterations if iterations > 0 else 1.0
        
        return MonteCarloResult(
            plan_name=plan_name,
            seed=seed,
            total_iterations=iterations,
            acyclic_count=acyclic_count,
            p_value=p_value,
            subgraph_sizes=subgraph_sizes,
            reproducible=True
        )
    
    def verify_reproducibility(self, plan_name: str, test_iterations: int = 100) -> bool:
        """
        Verify that identical plan names produce identical results across executions.
        
        Args:
            plan_name: Plan name to test
            test_iterations: Number of iterations for reproducibility test
        
        Returns:
            True if results are reproducible, False otherwise
        """
        # Run the same test twice
        result1 = self.calculate_acyclicity_pvalue(plan_name, test_iterations)
        result2 = self.calculate_acyclicity_pvalue(plan_name, test_iterations)
        
        # Check if results are identical
        reproducible = (
            result1.seed == result2.seed and
            result1.acyclic_count == result2.acyclic_count and
            result1.p_value == result2.p_value and
            result1.subgraph_sizes == result2.subgraph_sizes
        )
        
        return reproducible
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get basic statistics about the current graph."""
        total_nodes = len(self.graph_nodes)
        total_edges = sum(len(node.dependencies) for node in self.graph_nodes.values())
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "max_possible_edges": total_nodes * (total_nodes - 1)
        }


def create_sample_causal_graph() -> DAGValidator:
    """Create a sample causal graph for testing (teoria de cambio)."""
    validator = DAGValidator()
    
    # Sample nodes for a theory of change model
    validator.add_node("recursos_financieros")
    validator.add_node("capacitacion_personal")  
    validator.add_node("infraestructura")
    validator.add_node("programas_intervencion")
    validator.add_node("participacion_comunidad")
    validator.add_node("cambio_comportamiento")
    validator.add_node("mejora_indicadores")
    validator.add_node("impacto_social")
    
    # Add causal relationships (edges)
    validator.add_edge("recursos_financieros", "capacitacion_personal")
    validator.add_edge("recursos_financieros", "infraestructura")
    validator.add_edge("capacitacion_personal", "programas_intervencion")
    validator.add_edge("infraestructura", "programas_intervencion")
    validator.add_edge("programas_intervencion", "participacion_comunidad")
    validator.add_edge("participacion_comunidad", "cambio_comportamiento")
    validator.add_edge("cambio_comportamiento", "mejora_indicadores")
    validator.add_edge("mejora_indicadores", "impacto_social")
    
    return validator


if __name__ == "__main__":
    # Example usage
    validator = create_sample_causal_graph()
    
    # Test reproducibility
    plan_name = "teoria_cambio_educacion_2024"
    print(f"Testing reproducibility for plan: {plan_name}")
    is_reproducible = validator.verify_reproducibility(plan_name, 100)
    print(f"Reproducible: {is_reproducible}")
    
    # Calculate p-value
    result = validator.calculate_acyclicity_pvalue(plan_name, 1000)
    print(f"\nMonte Carlo Results:")
    print(f"Plan: {result.plan_name}")
    print(f"Seed: {result.seed}")
    print(f"Total iterations: {result.total_iterations}")
    print(f"Acyclic count: {result.acyclic_count}")
    print(f"P-value: {result.p_value:.4f}")
    avg_size = sum(result.subgraph_sizes) / len(result.subgraph_sizes) if result.subgraph_sizes else 0
    print(f"Average subgraph size: {avg_size:.1f}")
    
    # Graph statistics
    stats = validator.get_graph_stats()
    print(f"\nGraph Statistics:")
    print(f"Total nodes: {stats['total_nodes']}")
    print( f"Total edges: {stats['total_edges']}")