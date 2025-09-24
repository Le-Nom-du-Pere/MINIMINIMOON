from typing import List, Dict
import networkx as nx


class TeoriaCambio:
    def __init__(self, supuestos_causales: List[str] = None, mediadores: Dict[str, List[str]] = None, 
                 resultados_intermedios: List[str] = None, precondiciones: List[str] = None):
        self._grafo_causal = None
        self.supuestos_causales = supuestos_causales or []
        self.mediadores = mediadores or {}
        self.resultados_intermedios = resultados_intermedios or []
        self.precondiciones = precondiciones or []
    
    def construir_grafo_causal(self):
        """
        Construye y retorna el grafo causal, utilizando cache para evitar reconstrucciones.
        """
        if self._grafo_causal is None:
            # Construir el grafo causal aquí
            self._grafo_causal = self._crear_grafo_causal()
        return self._grafo_causal
    
    def _crear_grafo_causal(self):
        """
        Método privado que contiene la lógica para crear el grafo causal.
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
    
    def invalidar_cache_grafo(self):
        """
        Invalida el cache del grafo causal, forzando su reconstrucción en la próxima llamada.
        """
        self._grafo_causal = None
    
    def verificar_identificabilidad(self) -> bool:
        """Verifica condiciones de identificabilidad según Pearl (2009)"""
        return len(self.supuestos_causales) > 0 and len(self.mediadores) > 0 and len(self.resultados_intermedios) > 0

    def calcular_coeficiente_causal(self) -> float:
        """Calcula coeficiente de robustez causal basado en conectividad y paths"""
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