#!/usr/bin/env python3
# coding=utf-8
"""
Demonstration script showing the feasibility scorer in action.
"""

from feasibility_scorer import FeasibilityScorer, ComponentType


def main():
    scorer = FeasibilityScorer()
    
    print("=" * 60)
    print("FEASIBILITY SCORER DEMONSTRATION")
    print("=" * 60)
    
    # Test indicators of different quality levels
    test_indicators = [
        {
            'category': 'HIGH QUALITY',
            'indicators': [
                'Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025',
                'Reduce from baseline of 15.3 million people in poverty to target of 8 million by December 2024',
                'Aumentar el valor inicial de 2.5 millones de beneficiarios hasta alcanzar el objetivo de 4 millones en el horizonte temporal 2020-2025'
            ]
        },
        {
            'category': 'MEDIUM QUALITY', 
            'indicators': [
                'Mejorar desde la situación inicial hasta el objetivo propuesto con incremento del 20%',
                'Partir del nivel base actual para lograr la meta establecida en los próximos años',
                'Achieve target improvement from current baseline within the established timeframe'
            ]
        },
        {
            'category': 'LOW QUALITY',
            'indicators': [
                'Partir de la línea base para alcanzar el objetivo',
                'Improve from baseline to reach established goal'
            ]
        },
        {
            'category': 'INSUFFICIENT QUALITY',
            'indicators': [
                'Aumentar el acceso a servicios de salud en la región',
                'Mejorar la calidad educativa mediante nuevas estrategias',
                'La meta es fortalecer las instituciones públicas'
            ]
        }
    ]
    
    for category_data in test_indicators:
        category = category_data['category']
        indicators = category_data['indicators']
        
        print(f"\n{category}")
        print("-" * len(category))
        
        for i, indicator in enumerate(indicators, 1):
            result = scorer.calculate_feasibility_score(indicator)
            
            print(f"\n{i}. \"{indicator}\"")
            print(f"   Score: {result.feasibility_score:.2f}")
            print(f"   Quality Tier: {result.quality_tier}")
            print(f"   Components: {[c.value for c in result.components_detected]}")
            print(f"   Quantitative Baseline: {result.has_quantitative_baseline}")
            print(f"   Quantitative Target: {result.has_quantitative_target}")
            
            if result.detailed_matches:
                print(f"   Detected Patterns:")
                for match in result.detailed_matches:
                    print(f"     - {match.component_type.value}: '{match.matched_text}' (confidence: {match.confidence:.2f})")
    
    print(f"\n{'='*60}")
    print("BATCH SCORING EXAMPLE")
    print("=" * 60)
    
    batch_indicators = [
        "línea base 50% meta 80% año 2025",
        "situación actual mejorar objetivo general", 
        "aumentar servicios salud región",
        "baseline 30% target 60% by 2024"
    ]
    
    batch_results = scorer.batch_score(batch_indicators)
    
    print("\nBatch Results (sorted by score):")
    scored_indicators = list(zip(batch_indicators, batch_results))
    scored_indicators.sort(key=lambda x: x[1].feasibility_score, reverse=True)
    
    for indicator, result in scored_indicators:
        print(f"- {result.feasibility_score:.2f} | {result.quality_tier:>12} | \"{indicator}\"")
    
    print(f"\n{'='*60}")
    print("EVIDENCE QUALITY ANALYSIS")
    print("=" * 60)
    
    # Test the new calcular_calidad_evidencia method
    evidence_fragments = [
        'Línea base: COP $5.2 millones en 2023, meta $8.5 millones para Q4 2025 con monitoreo trimestral',
        'Investment of $2.3 million USD baseline for 2024 target achievement',  
        'Indicador de desempeño con periodicidad anual desde enero 2024',
        '• Mejora del sistema educativo',
        'Presupuesto de 15,000 millones de pesos para el baseline del proyecto',
        'Evaluación trimestral Q1 2024 con metas específicas',
        'Simple text without specific indicators'
    ]
    
    print("\nEvidence Quality Scores:")
    for i, fragment in enumerate(evidence_fragments, 1):
        score = scorer.calcular_calidad_evidencia(fragment)
        quality_level = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
        
        display_text = fragment[:50] + "..." if len(fragment) > 50 else fragment
        print(f"{i}. Score: {score:.3f} ({quality_level:>6}) | {display_text}")

    print(f"\n{'='*60}")
    print("DOCUMENTATION EXAMPLE")
    print("=" * 60)
    
    # Show a portion of the documentation
    docs = scorer.get_detection_rules_documentation()
    doc_lines = docs.split('\n')
    
    # Show first 30 lines of documentation
    for line in doc_lines[:30]:
        print(line)
    
    print("\n[... documentation continues ...]")
    print(f"\nTotal documentation length: {len(docs)} characters")


if __name__ == "__main__":
    main()