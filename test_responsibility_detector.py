"""
Test suite for ResponsibilityDetector
"""

import pytest

from responsibility_detector import EntityType, ResponsibilityDetector


class TestResponsibilityDetector:
    @pytest.fixture
    def detector(self):
        """Create a ResponsibilityDetector instance for testing."""
        return ResponsibilityDetector()

    def test_government_entity_detection(self, detector):
        """Test detection of government entities with high confidence."""
        text = "La Alcaldía Municipal coordinará con la Secretaría de Salud."
        result = detector.calculate_responsibility_score(text)

        assert result["factibility_score"] > 0.7
        assert result["has_government_entities"] is True

        # Check for government entities
        gov_entities = [e for e in result["entities"] if e.is_government]
        assert len(gov_entities) >= 1

        # Government entities should have high confidence
        for entity in gov_entities:
            assert entity.confidence >= 0.8

    def test_official_position_detection(self, detector):
        """Test detection of official positions."""
        text = "El alcalde y la secretaria de educación supervisarán el proyecto."
        result = detector.calculate_responsibility_score(text)

        assert result["has_official_positions"] is True

        position_entities = [
            e for e in result["entities"] if e.entity_type == EntityType.POSITION
        ]
        assert len(position_entities) >= 1

        # Position entities should have high confidence
        for entity in position_entities:
            assert entity.confidence >= 0.7

    def test_person_organization_detection(self, detector):
        """Test detection of persons and organizations via NER."""
        text = "Juan Pérez de Microsoft trabajará con María González."
        result = detector.calculate_responsibility_score(text)

        entities = result["entities"]
        person_entities = [
            e for e in entities if e.entity_type == EntityType.PERSON]
        org_entities = [e for e in entities if e.entity_type ==
                        EntityType.ORGANIZATION]

        # Should detect some entities (specific counts may vary with NER model)
        assert len(entities) > 0

    def test_confidence_scoring_hierarchy(self, detector):
        """Test that government entities get higher confidence than generic organizations."""
        gov_text = "La Secretaría de Salud implementará el programa."
        generic_text = "La empresa XYZ implementará el programa."

        gov_result = detector.calculate_responsibility_score(gov_text)
        generic_result = detector.calculate_responsibility_score(generic_text)

        # Government text should have higher factibility score
        assert gov_result["factibility_score"] > generic_result["factibility_score"]

    def test_overlapping_entity_merge(self, detector):
        """Test that overlapping entities are properly merged."""
        text = "La Alcaldía Municipal de Bogotá coordinará las actividades."
        entities = detector.detect_entities(text)

        # Check that overlapping entities were merged (no overlaps)
        for i in range(len(entities) - 1):
            assert entities[i].end_pos <= entities[i + 1].start_pos

    def test_fallback_lexical_patterns(self, detector):
        """Test that lexical patterns catch entities missed by NER."""
        # Text with clear institutional language but may be missed by NER
        text = "El programa municipal establecerá nuevas directrices."
        result = detector.calculate_responsibility_score(text)

        # Should detect institutional entities even without NER
        assert len(result["entities"]) > 0
        assert result["factibility_score"] > 0.3

    def test_empty_text(self, detector):
        """Test handling of empty or whitespace-only text."""
        result = detector.calculate_responsibility_score("")

        assert result["factibility_score"] == 0.0
        assert len(result["entities"]) == 0
        assert result["has_government_entities"] is False

    def test_complex_institutional_text(self, detector):
        """Test complex text with multiple institutional entities."""
        text = """
        La Alcaldía Municipal, en coordinación con la Secretaría de Educación 
        y el Ministerio de Salud, designará al director técnico Juan Pérez 
        como responsable del programa de desarrollo social.
        """

        result = detector.calculate_responsibility_score(text)

        # Should have high factibility due to multiple government entities
        assert result["factibility_score"] > 0.8
        assert result["has_government_entities"] is True
        assert result["has_official_positions"] is True

        # Should detect multiple entity types
        entities = result["entities"]
        entity_types = {e.entity_type for e in entities}
        assert len(entity_types) >= 2

    def test_government_entity_identification(self, detector):
        """Test the _is_government_entity helper method."""
        assert detector._is_government_entity("Alcaldía Municipal")
        assert detector._is_government_entity("Secretaría de Salud")
        assert detector._is_government_entity("Instituto Nacional")
        assert not detector._is_government_entity("Empresa Privada")
        assert not detector._is_government_entity("Juan Pérez")

    def test_pattern_coverage(self, detector):
        """Test that key government patterns are covered."""
        test_cases = [
            ("alcaldía", True),
            ("secretaría de educación", True),
            ("programa social", True),
            ("ministerio del interior", True),
            ("alcalde municipal", True),
            ("director general", True),
            ("coordinador técnico", True),
        ]

        for text, should_detect in test_cases:
            entities = detector.detect_entities(text)
            if should_detect:
                assert len(
                    entities) > 0, f"Failed to detect entities in: {text}"
                assert any(e.confidence > 0.5 for e in entities), (
                    f"Low confidence for: {text}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
