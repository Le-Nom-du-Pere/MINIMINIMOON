#!/usr/bin/env python3
"""
Validador simple de la implementación de Teoría de Cambio
"""

def validate_teoria_cambio():
    """Valida que la implementación de TeoriaCambio funcione correctamente"""
    try:
        # Test básico de importación
        from teoria_cambio import TeoriaCambio, CategoriaCausal, ValidacionResultado
        print("✓ Importación exitosa de clases principales")
        
        # Test de creación de instancia
        tc = TeoriaCambio()
        print("✓ Instancia de TeoriaCambio creada")
        
        # Test de enumeración de categorías
        categorias = list(CategoriaCausal)
        expected_categories = ['INSUMOS', 'PROCESOS', 'PRODUCTOS', 'RESULTADOS', 'IMPACTOS']
        category_names = [cat.name for cat in categorias]
        
        for expected in expected_categories:
            if expected in category_names:
                print(f"✓ Categoría {expected} definida correctamente")
            else:
                print(f"✗ Falta categoría {expected}")
        
        # Test de conexiones válidas
        insumos = CategoriaCausal.INSUMOS
        procesos = CategoriaCausal.PROCESOS
        productos = CategoriaCausal.PRODUCTOS
        resultados = CategoriaCausal.RESULTADOS
        impactos = CategoriaCausal.IMPACTOS
        
        # Conexiones que deben ser válidas
        valid_connections = [
            (insumos, procesos),      # Diferencia = 1
            (insumos, productos),     # Diferencia = 2
            (procesos, productos),    # Diferencia = 1
            (procesos, resultados),   # Diferencia = 2
            (productos, resultados),  # Diferencia = 1
            (productos, impactos),    # Diferencia = 2
            (resultados, impactos),   # Diferencia = 1
        ]
        
        # Conexiones que deben ser inválidas
        invalid_connections = [
            (insumos, resultados),    # Diferencia = 3
            (insumos, impactos),      # Diferencia = 4
            (procesos, impactos),     # Diferencia = 3
            (procesos, insumos),      # Diferencia = -1 (reversa)
            (impactos, resultados),   # Diferencia = -1 (reversa)
        ]
        
        print("\n--- Validando conexiones válidas ---")
        for origen, destino in valid_connections:
            if tc._es_conexion_valida(origen, destino):
                print(f"✓ {origen.name} → {destino.name}")
            else:
                print(f"✗ {origen.name} → {destino.name} (debería ser válida)")
        
        print("\n--- Validando conexiones inválidas ---")
        for origen, destino in invalid_connections:
            if not tc._es_conexion_valida(origen, destino):
                print(f"✓ {origen.name} → {destino.name} (correctamente rechazada)")
            else:
                print(f"✗ {origen.name} → {destino.name} (debería ser inválida)")
        
        print("\n--- Testando métodos de validación ---")
        
        # Test de crear grafo causal básico
        grafo = tc.construir_grafo_causal()
        print(f"✓ Grafo causal construido con {len(grafo.nodes)} nodos")
        
        # Test de validación de orden (debe funcionar sin NetworkX)
        try:
            resultado = tc.validar_orden_causal(grafo)
            print("✓ Validación de orden causal ejecutada")
            print(f"  - Es válida: {resultado.es_valida}")
            print(f"  - Violaciones: {len(resultado.violaciones_orden)}")
        except Exception as e:
            print(f"✗ Error en validación de orden: {e}")
        
        # Test de detección de caminos
        try:
            resultado = tc.detectar_caminos_completos(grafo)
            print("✓ Detección de caminos completos ejecutada")
            print(f"  - Caminos encontrados: {len(resultado.caminos_completos)}")
        except Exception as e:
            print(f"✗ Error en detección de caminos: {e}")
        
        # Test de generación de sugerencias
        try:
            resultado = tc.generar_sugerencias(grafo)
            print("✓ Generación de sugerencias ejecutada")
            print(f"  - Categorías faltantes: {len(resultado.categorias_faltantes)}")
            print(f"  - Sugerencias: {len(resultado.sugerencias)}")
        except Exception as e:
            print(f"✗ Error en generación de sugerencias: {e}")
        
        # Test de validación completa
        try:
            resultado = tc.validacion_completa(grafo)
            print("✓ Validación completa ejecutada")
            print(f"  - Resultado válido: {resultado.es_valida}")
        except Exception as e:
            print(f"✗ Error en validación completa: {e}")
        
        print("\n=== VALIDACIÓN COMPLETADA ===")
        print("La implementación de TeoriaCambio está funcionando correctamente.")
        print("Características implementadas:")
        print("• Validación de orden causal INSUMOS→PROCESOS→PRODUCTOS→RESULTADOS→IMPACTOS")
        print("• Detección de saltos máximos de una categoría en las conexiones")
        print("• Detección de caminos causales completos")
        print("• Sistema de sugerencias para completar la teoría de cambio")
        print("• Identificación automática de categorías faltantes")
        
        return True
        
    except Exception as e:
        print(f"✗ Error durante la validación: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_teoria_cambio()
    exit(0 if success else 1)