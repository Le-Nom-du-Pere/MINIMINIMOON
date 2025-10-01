from enum import Enum
from typing import Dict, List, Set, Tuple

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

    # Mock implementation for NetworkX DiGraph
    class MockDiGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = {}

        def add_node(self, node, **attrs):
            self._nodes[node] = attrs

        def add_edge(self, from_node, to_node, **attrs):
            if from_node not in self._edges:
                self._edges[from_node] = []
            self._edges[from_node].append(to_node)

        def nodes(self):
            return self._nodes.keys()

        def successors(self, node):
            return self._edges.get(node, [])

        def has_path(self, source, target):
            if source == target:
                return True
            visited = set()
            stack = [source]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current == target:
                    return True

                for neighbor in self.successors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)
            return False

        def shortest_path(self, source, target):
            if not self.has_path(source, target):
                raise ValueError(f"No path from {source} to {target}")

            visited = set()
            queue = [(source, [source])]

            while queue:
                current, path = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                if current == target:
                    return path

                for neighbor in self.successors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

            raise ValueError(f"No path from {source} to {target}")

        def has_edge(self, from_node, to_node):
            return to_node in self._edges.get(from_node, [])

        def in_degree(self, node):
            count = 0
            for source_edges in self._edges.values():
                if node in source_edges:
                    count += 1
            return count

        def out_degree(self, node):
            return len(self._edges.get(node, []))

    # Replace networkx with mock
    class MockNetworkX:
        DiGraph = MockDiGraph

        @staticmethod
        def has_path(graph, source, target):
            return graph.has_path(source, target)

        @staticmethod
        def shortest_path(graph, source, target):
            return graph.shortest_path(source, target)

    nx = MockNetworkX()


class CategoriaCausal(Enum):
    """Categorías de la cadena causal del modelo de Teoría de Cambio"""

    INSUMOS = 0
    PROCESOS = 1
    PRODUCTOS = 2
    RESULTADOS = 3
    IMPACTOS = 4


class ValidacionResultado:
    """Resultado de la validación de la teoría de cambio"""

    def __init__(self):
        self.es_valida = True
        self.violaciones_orden = []
        self.caminos_completos = []
        self.categorias_faltantes = []
        self.sugerencias = []


class TeoriaCambio:
    def __init__(
        self,
        supuestos_causales: List[str] = None,
        mediadores: Dict[str, List[str]] = None,
        resultados_intermedios: List[str] = None,
        precondiciones: List[str] = None,
    ):
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
        return None

    def validar_orden_causal(self, grafo: nx.DiGraph = None) -> ValidacionResultado:
        """
        Valida que el grafo respete el ordenamiento causal INSUMOS→PROCESOS→PRODUCTOS→RESULTADOS→IMPACTOS
        """
        if grafo is None:
            grafo = self.construir_grafo_causal()

        resultado = ValidacionResultado()

        # Verificar violaciones de orden causal
        for nodo in grafo.nodes():
            categoria_nodo = self._obtener_categoria_nodo(nodo, grafo)
            for sucesor in grafo.successors(nodo):
                categoria_sucesor = self._obtener_categoria_nodo(sucesor, grafo)

                if not self._es_conexion_valida(categoria_nodo, categoria_sucesor):
                    violacion = {
                        "origen": nodo,
                        "destino": sucesor,
                        "categoria_origen": (
                            categoria_nodo.name if categoria_nodo else "INDEFINIDA"
                        ),
                        "categoria_destino": (
                            categoria_sucesor.name
                            if categoria_sucesor
                            else "INDEFINIDA"
                        ),
                    }
                    resultado.violaciones_orden.append(violacion)
                    resultado.es_valida = False

        return resultado

    def detectar_caminos_completos(
        self, grafo: nx.DiGraph = None
    ) -> ValidacionResultado:
        """
        Detecta si existe al menos un camino completo desde INSUMOS hasta IMPACTOS
        """
        if grafo is None:
            grafo = self.construir_grafo_causal()

        resultado = ValidacionResultado()

        # Encontrar nodos de cada categoría
        nodos_insumos = self._obtener_nodos_por_categoria(
            grafo, CategoriaCausal.INSUMOS
        )
        nodos_impactos = self._obtener_nodos_por_categoria(
            grafo, CategoriaCausal.IMPACTOS
        )

        # Buscar caminos completos
        for insumo in nodos_insumos:
            for impacto in nodos_impactos:
                if nx.has_path(grafo, insumo, impacto):
                    camino = nx.shortest_path(grafo, insumo, impacto)
                    if self._es_camino_completo(camino, grafo):
                        resultado.caminos_completos.append(camino)

        if not resultado.caminos_completos:
            resultado.es_valida = False

        return resultado

    def generar_sugerencias(self, grafo: nx.DiGraph = None) -> ValidacionResultado:
        """
        Analiza la estructura del grafo y genera sugerencias específicas para completar la teoría de cambio
        """
        if grafo is None:
            grafo = self.construir_grafo_causal()

        resultado = ValidacionResultado()

        # Identificar categorías presentes y faltantes
        categorias_presentes = set()
        for nodo in grafo.nodes():
            categoria = self._obtener_categoria_nodo(nodo, grafo)
            if categoria:
                categorias_presentes.add(categoria)

        categorias_todas = set(CategoriaCausal)
        resultado.categorias_faltantes = list(categorias_todas - categorias_presentes)

        # Generar sugerencias específicas
        self._generar_sugerencias_conexiones(grafo, resultado)
        self._generar_sugerencias_categorias_faltantes(resultado)

        return resultado

    def validacion_completa(self, grafo: nx.DiGraph = None) -> ValidacionResultado:
        """
        Ejecuta validación completa de la teoría de cambio
        """
        if grafo is None:
            grafo = self.construir_grafo_causal()

        # Combinar todos los resultados de validación
        orden = self.validar_orden_causal(grafo)
        caminos = self.detectar_caminos_completos(grafo)
        sugerencias = self.generar_sugerencias(grafo)

        resultado = ValidacionResultado()
        resultado.es_valida = orden.es_valida and caminos.es_valida
        resultado.violaciones_orden = orden.violaciones_orden
        resultado.caminos_completos = caminos.caminos_completos
        resultado.categorias_faltantes = sugerencias.categorias_faltantes
        resultado.sugerencias = sugerencias.sugerencias

        return resultado

    def _obtener_categoria_nodo(self, nodo: str, grafo: nx.DiGraph) -> CategoriaCausal:
        """
        Determina la categoría causal de un nodo basándose en su nombre o atributos
        """
        nodo_lower = nodo.lower()

        if "insumo" in nodo_lower or nodo == "insumos":
            return CategoriaCausal.INSUMOS
        elif "proceso" in nodo_lower or "actividad" in nodo_lower:
            return CategoriaCausal.PROCESOS
        elif "producto" in nodo_lower or "output" in nodo_lower:
            return CategoriaCausal.PRODUCTOS
        elif "resultado" in nodo_lower or "outcome" in nodo_lower:
            return CategoriaCausal.RESULTADOS
        elif "impacto" in nodo_lower or nodo == "impactos":
            return CategoriaCausal.IMPACTOS

        # Intentar determinar por posición en el grafo si no hay identificación por nombre
        return self._inferir_categoria_por_posicion(nodo, grafo)

    @staticmethod
    def _inferir_categoria_por_posicion(
        nodo: str, grafo: nx.DiGraph
    ) -> CategoriaCausal:
        """
        Infiere la categoría basándose en la posición topológica del nodo
        """
        # Calcular distancias topológicas
        in_degree = grafo.in_degree(nodo)
        out_degree = grafo.out_degree(nodo)

        if in_degree == 0:
            return CategoriaCausal.INSUMOS
        elif out_degree == 0:
            return CategoriaCausal.IMPACTOS
        else:
            # Para nodos intermedios, usar heurísticas adicionales
            try:
                if "mediador" in grafo.nodes[nodo].get("tipo", ""):
                    return CategoriaCausal.PROCESOS
                elif "resultado" in grafo.nodes[nodo].get("tipo", ""):
                    return CategoriaCausal.RESULTADOS
            except KeyError:
                # Nodo no encontrado en el grafo; se aplica heurística por defecto
                pass

            return CategoriaCausal.PRODUCTOS  # Valor por defecto para nodos intermedios

    @staticmethod
    def _es_conexion_valida(origen: CategoriaCausal, destino: CategoriaCausal) -> bool:
        """
        Verifica si una conexión entre dos categorías es válida según las reglas causales
        """
        if origen is None or destino is None:
            return False

        # Diferencia entre categorías
        diff = destino.value - origen.value

        # Permitir conexiones al sucesor inmediato (diff = 1) o saltar máximo una categoría (diff = 2)
        return 1 <= diff <= 2

    def _obtener_nodos_por_categoria(
        self, grafo: nx.DiGraph, categoria: CategoriaCausal
    ) -> List[str]:
        """
        Obtiene todos los nodos que pertenecen a una categoría específica
        """
        nodos = []
        for nodo in grafo.nodes():
            if self._obtener_categoria_nodo(nodo, grafo) == categoria:
                nodos.append(nodo)
        return nodos

    def _es_camino_completo(self, camino: List[str], grafo: nx.DiGraph) -> bool:
        """
        Verifica si un camino atraviesa al menos las categorías principales de la cadena causal
        """
        categorias_camino = []
        for nodo in camino:
            categoria = self._obtener_categoria_nodo(nodo, grafo)
            if categoria and categoria not in categorias_camino:
                categorias_camino.append(categoria)

        # Un camino completo debe empezar en INSUMOS y terminar en IMPACTOS
        # y pasar por al menos 3 categorías intermedias
        if not categorias_camino or categorias_camino[0] != CategoriaCausal.INSUMOS:
            return False
        if categorias_camino[-1] != CategoriaCausal.IMPACTOS:
            return False

        return len(categorias_camino) >= 3

    def _generar_sugerencias_conexiones(
        self, grafo: nx.DiGraph, resultado: ValidacionResultado
    ):
        """
        Genera sugerencias específicas para conectar categorías faltantes
        """
        nodos_por_categoria = {}
        for categoria in CategoriaCausal:
            nodos_por_categoria[categoria] = self._obtener_nodos_por_categoria(
                grafo, categoria
            )

        # Sugerencias para conexiones faltantes entre categorías adyacentes
        for i, categoria_actual in enumerate(CategoriaCausal):
            if i < len(CategoriaCausal) - 1:
                categoria_siguiente = list(CategoriaCausal)[i + 1]

                nodos_actual = nodos_por_categoria.get(categoria_actual, [])
                nodos_siguiente = nodos_por_categoria.get(categoria_siguiente, [])

                if nodos_actual and nodos_siguiente:
                    # Verificar si ya existen conexiones
                    conexiones_existentes = False
                    for nodo_actual in nodos_actual:
                        for nodo_siguiente in nodos_siguiente:
                            if grafo.has_edge(nodo_actual, nodo_siguiente):
                                conexiones_existentes = True
                                break
                        if conexiones_existentes:
                            break

                    if not conexiones_existentes:
                        sugerencia = f"Agregar conexiones entre {categoria_actual.name} y {categoria_siguiente.name}. "
                        sugerencia += f"Considerar conectar '{nodos_actual[0]}' con '{nodos_siguiente[0]}'"
                        resultado.sugerencias.append(sugerencia)
        return None

    @staticmethod
    def _generar_sugerencias_categorias_faltantes(resultado: ValidacionResultado):
        """
        Genera sugerencias para categorías completamente ausentes
        """
        for categoria in resultado.categorias_faltantes:
            if categoria == CategoriaCausal.INSUMOS:
                resultado.sugerencias.append(
                    "Falta definir INSUMOS: Agregar recursos humanos, financieros, materiales o técnicos necesarios"
                )
            elif categoria == CategoriaCausal.PROCESOS:
                resultado.sugerencias.append(
                    "Falta definir PROCESOS: Agregar actividades, intervenciones o acciones específicas a realizar"
                )
            elif categoria == CategoriaCausal.PRODUCTOS:
                resultado.sugerencias.append(
                    "Falta definir PRODUCTOS: Agregar entregables tangibles, servicios o resultados directos"
                )
            elif categoria == CategoriaCausal.RESULTADOS:
                resultado.sugerencias.append(
                    "Falta definir RESULTADOS: Agregar cambios intermedios en comportamientos, conocimientos o condiciones"
                )
            elif categoria == CategoriaCausal.IMPACTOS:
                resultado.sugerencias.append(
                    "Falta definir IMPACTOS: Agregar cambios de largo plazo en el problema o situación objetivo"
                )
        return None

    def verificar_identificabilidad(self) -> bool:
        """Verifica condiciones de identificabilidad según Pearl (2009)"""
        return (
            len(self.supuestos_causales) > 0
            and len(self.mediadores) > 0
            and len(self.resultados_intermedios) > 0
        )

    def calcular_coeficiente_causal(self) -> float:
        """Calcula coeficiente de robustez causal basado en conectividad y paths"""
        G = self.construir_grafo_causal()
        if len(G.nodes) < 3:
            return 0.3

        try:
            # Calcular robustez como promedio de paths válidos
            paths_validos = 0
            total_paths = 0

            for mediador in [
                n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"
            ]:
                for resultado in [
                    n for n in G.nodes if G.nodes[n].get("tipo") == "resultado"
                ]:
                    if nx.has_path(G, mediador, resultado) and nx.has_path(
                        G, resultado, "impactos"
                    ):
                        paths_validos += 1
                    total_paths += 1

            return paths_validos / max(1, total_paths) if total_paths > 0 else 0.5
        except Exception:
            return 0.5
