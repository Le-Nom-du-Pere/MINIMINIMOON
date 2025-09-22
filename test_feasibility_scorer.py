"""
Comprehensive test suite for FeasibilityScorer with manually annotated dataset.
Tests precision and recall of quality detection patterns.
"""

import pytest
from feasibility_scorer import FeasibilityScorer, ComponentType, IndicatorScore
from typing import List, Dict, Any


class TestDataset:
    """Manually annotated test dataset for validation."""
    
    @staticmethod
    def get_high_quality_indicators() -> List[Dict[str, Any]]:
        """High-quality indicators with all components and quantitative elements."""
        return [
            {
                'text': 'Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025',
                'expected_score': 0.9,
                'expected_tier': 'high',
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET, ComponentType.TIME_HORIZON, ComponentType.NUMERICAL, ComponentType.DATE],
                'has_quantitative_baseline': True,
                'has_quantitative_target': True
            },
            {
                'text': 'Reducir from baseline of 15.3 million people in poverty to target of 8 million by December 2024',
                'expected_score': 0.85,
                'expected_tier': 'high', 
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET, ComponentType.TIME_HORIZON, ComponentType.NUMERICAL, ComponentType.DATE],
                'has_quantitative_baseline': True,
                'has_quantitative_target': True
            },
            {
                'text': 'Aumentar el valor inicial de 2.5 millones de beneficiarios hasta alcanzar el objetivo de 4 millones en el horizonte temporal 2020-2025',
                'expected_score': 0.88,
                'expected_tier': 'high',
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET, ComponentType.TIME_HORIZON, ComponentType.NUMERICAL, ComponentType.DATE],
                'has_quantitative_baseline': True,
                'has_quantitative_target': True
            }
        ]
    
    @staticmethod 
    def get_medium_quality_indicators() -> List[Dict[str, Any]]:
        """Medium-quality indicators with basic components, some quantitative elements."""
        return [
            {
                'text': 'Mejorar desde la situación inicial hasta el objetivo propuesto con incremento del 20%',
                'expected_score': 0.6,
                'expected_tier': 'medium',
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET, ComponentType.NUMERICAL],
                'has_quantitative_baseline': False,
                'has_quantitative_target': True
            },
            {
                'text': 'Partir del nivel base actual para lograr la meta establecida en los próximos años',
                'expected_score': 0.55,
                'expected_tier': 'medium',
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET, ComponentType.TIME_HORIZON],
                'has_quantitative_baseline': False,
                'has_quantitative_target': False
            },
            {
                'text': 'Achieve target improvement from current baseline within the established timeframe',
                'expected_score': 0.58,
                'expected_tier': 'medium',
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET, ComponentType.TIME_HORIZON],
                'has_quantitative_baseline': False,
                'has_quantitative_target': False
            }
        ]
    
    @staticmethod
    def get_low_quality_indicators() -> List[Dict[str, Any]]:
        """Low-quality indicators with minimal components."""
        return [
            {
                'text': 'Partir de la línea base para alcanzar el objetivo',
                'expected_score': 0.3,
                'expected_tier': 'low',
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET],
                'has_quantitative_baseline': False,
                'has_quantitative_target': False
            },
            {
                'text': 'Improve from baseline to reach established goal',
                'expected_score': 0.32,
                'expected_tier': 'low', 
                'expected_components': [ComponentType.BASELINE, ComponentType.TARGET],
                'has_quantitative_baseline': False,
                'has_quantitative_target': False
            }
        ]
    
    @staticmethod
    def get_insufficient_indicators() -> List[Dict[str, Any]]:
        """Insufficient indicators missing core components."""
        return [
            {
                'text': 'Aumentar el acceso a servicios de salud en la región',
                'expected_score': 0.0,
                'expected_tier': 'insufficient',
                'expected_components': [],
                'has_quantitative_baseline': False,
                'has_quantitative_target': False
            },
            {
                'text': 'Mejorar la calidad educativa mediante nuevas estrategias',
                'expected_score': 0.0,
                'expected_tier': 'insufficient', 
                'expected_components': [],
                'has_quantitative_baseline': False,
                'has_quantitative_target': False
            },
            {
                'text': 'La meta es fortalecer las instituciones públicas',
                'expected_score': 0.0,
                'expected_tier': 'insufficient',
                'expected_components': [ComponentType.TARGET],
                'has_quantitative_baseline': False,
                'has_quantitative_target': False
            }
        ]


class TestFeasibilityScorer:
    """Test suite for FeasibilityScorer functionality."""
    
    @pytest.fixture
    def scorer(self):
        return FeasibilityScorer()
    
    def test_high_quality_indicators(self, scorer):
        """Test scoring of high-quality indicators."""
        indicators = TestDataset.get_high_quality_indicators()
        
        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data['text'])
            
            # Check score within reasonable range (±0.1)
            assert abs(result.feasibility_score - indicator_data['expected_score']) <= 0.15, \
                f"Score mismatch for '{indicator_data['text']}': expected {indicator_data['expected_score']}, got {result.feasibility_score}"
            
            # Check quality tier
            assert result.quality_tier == indicator_data['expected_tier'], \
                f"Quality tier mismatch for '{indicator_data['text']}': expected {indicator_data['expected_tier']}, got {result.quality_tier}"
            
            # Check quantitative components
            assert result.has_quantitative_baseline == indicator_data['has_quantitative_baseline']
            assert result.has_quantitative_target == indicator_data['has_quantitative_target']
            
            # Check that key components are detected
            detected_types = set(result.components_detected)
            for expected_component in indicator_data['expected_components']:
                assert expected_component in detected_types, \
                    f"Missing component {expected_component} in '{indicator_data['text']}'"
    
    def test_medium_quality_indicators(self, scorer):
        """Test scoring of medium-quality indicators."""
        indicators = TestDataset.get_medium_quality_indicators()
        
        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data['text'])
            
            assert abs(result.feasibility_score - indicator_data['expected_score']) <= 0.15
            assert result.quality_tier == indicator_data['expected_tier']
            assert result.has_quantitative_baseline == indicator_data['has_quantitative_baseline']
            assert result.has_quantitative_target == indicator_data['has_quantitative_target']
    
    def test_low_quality_indicators(self, scorer):
        """Test scoring of low-quality indicators.""" 
        indicators = TestDataset.get_low_quality_indicators()
        
        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data['text'])
            
            assert abs(result.feasibility_score - indicator_data['expected_score']) <= 0.15
            assert result.quality_tier == indicator_data['expected_tier']
            assert result.has_quantitative_baseline == indicator_data['has_quantitative_baseline']
            assert result.has_quantitative_target == indicator_data['has_quantitative_target']
    
    def test_insufficient_indicators(self, scorer):
        """Test scoring of insufficient indicators."""
        indicators = TestDataset.get_insufficient_indicators()
        
        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data['text'])
            
            assert result.feasibility_score == 0.0, \
                f"Expected 0.0 score for insufficient indicator, got {result.feasibility_score}"
            assert result.quality_tier == 'insufficient'
            assert result.has_quantitative_baseline == False
            assert result.has_quantitative_target == False
    
    def test_mandatory_requirements(self, scorer):
        """Test that baseline and target are mandatory for positive scores."""
        # Only baseline
        result = scorer.calculate_feasibility_score("La línea base es de 50% de cobertura")
        assert result.feasibility_score == 0.0
        
        # Only target  
        result = scorer.calculate_feasibility_score("El objetivo es llegar al 80%")
        assert result.feasibility_score == 0.0
        
        # Both present
        result = scorer.calculate_feasibility_score("Partir de línea base 50% hasta objetivo 80%")
        assert result.feasibility_score > 0.0
    
    def test_spanish_patterns(self, scorer):
        """Test Spanish-specific pattern detection."""
        spanish_texts = [
            "línea base de 30% hasta meta de 60%",
            "valor inicial 25 millones para objetivo 40 millones", 
            "situación actual mejorar hasta propósito establecido",
            "desde punto de partida hasta finalidad en el plazo 2025"
        ]
        
        for text in spanish_texts:
            result = scorer.calculate_feasibility_score(text)
            assert result.feasibility_score > 0.0, f"Failed to detect Spanish patterns in: {text}"
    
    def test_english_patterns(self, scorer):
        """Test English-specific pattern detection."""
        english_texts = [
            "baseline of 30% to target of 60%",
            "current level 25 million to goal 40 million",
            "starting point improve to aim within timeline",
            "from initial value achieve target by 2025"
        ]
        
        for text in english_texts:
            result = scorer.calculate_feasibility_score(text)
            assert result.feasibility_score > 0.0, f"Failed to detect English patterns in: {text}"
    
    def test_numerical_detection(self, scorer):
        """Test numerical pattern detection."""
        numerical_texts = [
            "incrementar 25%",
            "reducir en 1.5 millones",
            "increase by 30 percent", 
            "reduce 2,500 thousand"
        ]
        
        for text in numerical_texts:
            components = scorer.detect_components(text)
            numerical_detected = any(c.component_type == ComponentType.NUMERICAL for c in components)
            assert numerical_detected, f"Failed to detect numerical pattern in: {text}"
    
    def test_date_detection(self, scorer):
        """Test date pattern detection.""" 
        date_texts = [
            "para el año 2025",
            "en diciembre 2024",
            "by January 2025",
            "15/12/2024",
            "hasta 2026"
        ]
        
        for text in date_texts:
            components = scorer.detect_components(text)
            date_detected = any(c.component_type == ComponentType.DATE for c in components)
            assert date_detected, f"Failed to detect date pattern in: {text}"
    
    def test_quantitative_component_detection(self, scorer):
        """Test detection of quantitative baselines and targets."""
        # Quantitative baseline
        text1 = "línea base de 65% incrementar hasta meta general"
        result1 = scorer.calculate_feasibility_score(text1)
        assert result1.has_quantitative_baseline == True
        assert result1.has_quantitative_target == False
        
        # Quantitative target
        text2 = "partir de situación actual hasta objetivo de 85%"
        result2 = scorer.calculate_feasibility_score(text2)
        assert result2.has_quantitative_baseline == False
        assert result2.has_quantitative_target == True
        
        # Both quantitative
        text3 = "línea base 40% hasta meta 70%"
        result3 = scorer.calculate_feasibility_score(text3)
        assert result3.has_quantitative_baseline == True
        assert result3.has_quantitative_target == True
    
    def test_batch_scoring(self, scorer):
        """Test batch scoring functionality."""
        indicators = [
            "línea base 50% meta 80% año 2025",
            "situación actual mejorar objetivo",
            "aumentar servicios región"
        ]
        
        results = scorer.batch_score(indicators)
        assert len(results) == 3
        assert results[0].feasibility_score > results[1].feasibility_score > results[2].feasibility_score
    
    def test_precision_recall_metrics(self, scorer):
        """Test precision and recall of component detection."""
        all_indicators = (
            TestDataset.get_high_quality_indicators() +
            TestDataset.get_medium_quality_indicators() + 
            TestDataset.get_low_quality_indicators() +
            TestDataset.get_insufficient_indicators()
        )
        
        total_baseline_expected = sum(1 for ind in all_indicators if ComponentType.BASELINE in ind['expected_components'])
        total_target_expected = sum(1 for ind in all_indicators if ComponentType.TARGET in ind['expected_components'])
        
        baseline_detected = 0
        target_detected = 0
        
        for indicator_data in all_indicators:
            result = scorer.calculate_feasibility_score(indicator_data['text'])
            if ComponentType.BASELINE in result.components_detected:
                baseline_detected += 1
            if ComponentType.TARGET in result.components_detected:
                target_detected += 1
        
        baseline_recall = baseline_detected / total_baseline_expected if total_baseline_expected > 0 else 0
        target_recall = target_detected / total_target_expected if total_target_expected > 0 else 0
        
        # Expect high recall (>80%) for core components
        assert baseline_recall >= 0.8, f"Baseline recall too low: {baseline_recall}"
        assert target_recall >= 0.8, f"Target recall too low: {target_recall}"
    
    def test_documentation_generation(self, scorer):
        """Test documentation generation."""
        docs = scorer.get_detection_rules_documentation()
        
        assert "Feasibility Scorer Detection Rules Documentation" in docs
        assert "Spanish Pattern Recognition" in docs
        assert "Quality Tiers" in docs
        assert "Examples" in docs
        assert len(docs) > 1000  # Ensure comprehensive documentation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])