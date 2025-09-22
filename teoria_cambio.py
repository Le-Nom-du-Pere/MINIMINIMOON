class TeoriaCambio:
    def __init__(self):
        self._grafo_causal = None
    
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
        # Implementación de la construcción del grafo causal
        grafo = {}  # Placeholder - implementar según necesidades específicas
        return grafo
    
    def invalidar_cache_grafo(self):
        """
        Invalida el cache del grafo causal, forzando su reconstrucción en la próxima llamada.
        """
        self._grafo_causal = None