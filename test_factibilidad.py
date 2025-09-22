#!/usr/bin/env python3
"""
Test script for the factibilidad scoring module.
"""

from factibilidad import PatternDetector, FactibilidadScorer


def test_pattern_detection():
    """Test basic pattern detection functionality."""
    detector = PatternDetector()
    
    # Test text with all three pattern types
    test_text = """
    La línea base actual muestra que tenemos 100 usuarios registrados.
    Nuestro objetivo es alcanzar 500 usuarios para diciembre de 2024.
    Esta meta representa un crecimiento del 400% en 6 meses.
    """
    
    matches = detector.detect_patterns(test_text)
    
    print("=== Pattern Detection Test ===")
    for pattern_type, pattern_matches in matches.items():
        print(f"\n{pattern_type.upper()} patterns found: {len(pattern_matches)}")
        for match in pattern_matches:
            print(f"  - '{match.text}' at position {match.start}-{match.end}")
    
    clusters = detector.find_pattern_clusters(test_text)
    print(f"\nClusters found: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: span={cluster['span']} chars")
        print(f"    Text: {cluster['text'][:100]}...")


def test_scoring():
    """Test the factibilidad scoring functionality."""
    scorer = FactibilidadScorer()
    
    # Test texts with different levels of completeness
    test_texts = [
        {
            'name': 'Complete text',
            'text': """
            Actualmente tenemos una línea base de 50 proyectos completados.
            Nuestro objetivo es alcanzar 120 proyectos para el año 2025.
            Esta meta debe lograrse en un plazo de 18 meses.
            """
        },
        {
            'name': 'Missing baseline',
            'text': """
            El objetivo principal es conseguir 200 nuevos clientes.
            Esta meta debe cumplirse antes de diciembre de 2024.
            """
        },
        {
            'name': 'Missing timeframe',
            'text': """
            Partiendo de la situación inicial de 30% de satisfacción,
            buscamos alcanzar un 85% de satisfacción del cliente.
            """
        },
        {
            'name': 'Scattered patterns',
            'text': """
            La empresa tiene como propósito expandir su mercado.
            
            En el estado actual, contamos con 5 oficinas.
            
            Para el próximo año esperamos resultados positivos.
            """
        }
    ]
    
    print("\n=== Scoring Test ===")
    for test_case in test_texts:
        print(f"\n--- {test_case['name']} ---")
        result = scorer.score_text(test_case['text'])
        
        print(f"Total Score: {result['total_score']:.1f}")
        print(f"Clusters found: {result['cluster_scores']['count']}")
        
        analysis = result['analysis']
        print(f"Has baseline: {analysis['has_baseline']}")
        print(f"Has target: {analysis['has_target']}")
        print(f"Has timeframe: {analysis['has_timeframe']}")
        
        if analysis['strengths']:
            print("Strengths:")
            for strength in analysis['strengths']:
                print(f"  - {strength}")
        
        if analysis['weaknesses']:
            print("Weaknesses:")
            for weakness in analysis['weaknesses']:
                print(f"  - {weakness}")


def test_specific_patterns():
    """Test specific pattern recognition."""
    detector = PatternDetector()
    
    pattern_tests = [
        ("Baseline patterns", [
            "línea base establecida",
            "situación inicial crítica", 
            "punto de partida claro",
            "estado actual preocupante",
            "valor inicial de referencia"
        ]),
        ("Target patterns", [
            "meta ambiciosa",
            "objetivo principal",
            "alcanzar resultados",
            "conseguir la mejora",
            "lograr el cambio"
        ]),
        ("Timeframe patterns", [
            "al 2025",
            "para diciembre de 2024",
            "en 6 meses",
            "primer trimestre",
            "próximo año",
            "2023-2025"
        ])
    ]
    
    print("\n=== Specific Pattern Tests ===")
    for category, test_phrases in pattern_tests:
        print(f"\n--- {category} ---")
        for phrase in test_phrases:
            matches = detector.detect_patterns(phrase)
            found_types = [pt for pt, matches in matches.items() if matches]
            print(f"'{phrase}' -> {found_types}")


if __name__ == '__main__':
    test_pattern_detection()
    test_scoring()
    test_specific_patterns()