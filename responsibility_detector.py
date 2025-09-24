"""
Responsibility Detection Module using spaCy NER and Lexical Rules
"""
import spacy
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    GOVERNMENT = "GOVERNMENT"
    POSITION = "POSITION"


@dataclass
class ResponsibilityEntity:
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    role: Optional[str] = None
    is_government: bool = False


class ResponsibilityDetector:
    """Detects responsibility entities in document segments using spaCy NER and lexical rules."""
    
    def __init__(self, model_name: str = "es_core_news_sm"):
        """
        Initialize the responsibility detector.
        
        Args:
            model_name: spaCy model name (Spanish model recommended)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Fallback to basic model if specialized model not available
            self.nlp = spacy.load("es_core_news_sm")
        
        # Government institution patterns (high priority)
        self.government_patterns = [
            r'alcaldía(?:\s+(?:de|del|municipal))?',
            r'secretaría(?:\s+(?:de|del))?',
            r'programa(?:\s+(?:de|del))?',
            r'ministerio(?:\s+(?:de|del))?',
            r'gobernación(?:\s+(?:de|del))?',
            r'municipalidad',
            r'ayuntamiento',
            r'concejalía',
            r'departamento\s+administrativo',
            r'instituto\s+(?:nacional|municipal|departamental)',
            r'dirección\s+(?:nacional|municipal|departamental)',
        ]
        
        # Official position patterns
        self.position_patterns = [
            r'alcalde(?:sa)?',
            r'secretari[oa](?:\s+(?:de|del))?',
            r'ministr[oa](?:\s+(?:de|del))?',
            r'director(?:a)?\s+(?:general|ejecutiv[oa]|técnic[oa])?',
            r'coordinador(?:a)?(?:\s+(?:de|del))?',
            r'jefe(?:\s+(?:de|del))?',
            r'responsable(?:\s+(?:de|del))?',
            r'encargad[oa](?:\s+(?:de|del))?',
            r'gerente(?:\s+(?:de|del))?',
            r'presidente(?:a)?(?:\s+(?:de|del))?',
            r'gobernador(?:a)?',
            r'concejal(?:a)?',
            r'comisionad[oa]',
        ]
        
        # Generic institutional terms
        self.institutional_patterns = [
            r'entidad(?:\s+(?:pública|gubernamental))?',
            r'organización(?:\s+(?:pública|gubernamental))?',
            r'institución(?:\s+(?:pública|gubernamental))?',
            r'dependencia(?:\s+(?:pública|gubernamental))?',
            r'oficina(?:\s+(?:pública|gubernamental))?',
            r'unidad(?:\s+(?:administrativa|técnica))?',
        ]
        
        # Compile patterns
        self.compiled_gov_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.government_patterns]
        self.compiled_pos_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.position_patterns]
        self.compiled_inst_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.institutional_patterns]
    
    def detect_entities(self, text: str) -> List[ResponsibilityEntity]:
        """
        Detect responsibility entities in text using NER and lexical rules.
        
        Args:
            text: Input text segment
            
        Returns:
            List of ResponsibilityEntity objects with confidence scores
        """
        entities = []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract NER entities
        ner_entities = self._extract_ner_entities(doc)
        entities.extend(ner_entities)
        
        # Extract pattern-based entities
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)
        
        # Remove duplicates and merge overlapping entities
        entities = self._merge_overlapping_entities(entities)
        
        # Calculate final confidence scores
        entities = self._calculate_final_scores(entities)
        
        return sorted(entities, key=lambda x: x.confidence, reverse=True)
    
    def _extract_ner_entities(self, doc) -> List[ResponsibilityEntity]:
        """Extract PERSON and ORG entities from spaCy NER."""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "PER"]:
                entity = ResponsibilityEntity(
                    text=ent.text,
                    entity_type=EntityType.PERSON,
                    confidence=0.7,  # Base NER confidence for persons
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                )
                entities.append(entity)
            
            elif ent.label_ in ["ORG"]:
                # Check if it's a government organization
                is_gov = self._is_government_entity(ent.text)
                entity = ResponsibilityEntity(
                    text=ent.text,
                    entity_type=EntityType.GOVERNMENT if is_gov else EntityType.ORGANIZATION,
                    confidence=0.8 if is_gov else 0.6,  # Higher confidence for government entities
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    is_government=is_gov
                )
                entities.append(entity)
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> List[ResponsibilityEntity]:
        """Extract entities using lexical patterns."""
        entities = []
        
        # Government institutions (highest priority)
        for pattern in self.compiled_gov_patterns:
            for match in pattern.finditer(text):
                entity = ResponsibilityEntity(
                    text=match.group(),
                    entity_type=EntityType.GOVERNMENT,
                    confidence=0.9,  # High confidence for government patterns
                    start_pos=match.start(),
                    end_pos=match.end(),
                    is_government=True
                )
                entities.append(entity)
        
        # Official positions
        for pattern in self.compiled_pos_patterns:
            for match in pattern.finditer(text):
                entity = ResponsibilityEntity(
                    text=match.group(),
                    entity_type=EntityType.POSITION,
                    confidence=0.8,  # High confidence for official positions
                    start_pos=match.start(),
                    end_pos=match.end(),
                    role=match.group(),
                    is_government=True
                )
                entities.append(entity)
        
        # Generic institutional terms (lower priority)
        for pattern in self.compiled_inst_patterns:
            for match in pattern.finditer(text):
                entity = ResponsibilityEntity(
                    text=match.group(),
                    entity_type=EntityType.ORGANIZATION,
                    confidence=0.5,  # Lower confidence for generic terms
                    start_pos=match.start(),
                    end_pos=match.end(),
                    is_government=False
                )
                entities.append(entity)
        
        return entities
    
    @staticmethod
    def _is_government_entity(text: str) -> bool:
        """Check if an entity is likely a government organization."""
        text_lower = text.lower()
        government_keywords = [
            'alcaldía', 'secretaría', 'programa', 'ministerio', 'gobernación',
            'municipalidad', 'ayuntamiento', 'concejalía', 'instituto',
            'dirección', 'departamento', 'nacional', 'municipal', 'departamental'
        ]
        
        return any(keyword in text_lower for keyword in government_keywords)
    
    @staticmethod
    def _merge_overlapping_entities(entities: List[ResponsibilityEntity]) -> List[ResponsibilityEntity]:
        """Merge overlapping entities, keeping the highest confidence one."""
        if not entities:
            return entities
        
        # Sort by position
        entities.sort(key=lambda x: x.start_pos)
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap
            if next_entity.start_pos <= current.end_pos:
                # Keep entity with higher confidence
                if next_entity.confidence > current.confidence:
                    current = next_entity
                # If same confidence, prefer government entities
                elif (next_entity.confidence == current.confidence and 
                      next_entity.is_government and not current.is_government):
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    @staticmethod
    def _calculate_final_scores(entities: List[ResponsibilityEntity]) -> List[ResponsibilityEntity]:
        """Calculate final confidence scores considering entity types and patterns."""
        for entity in entities:
            base_score = entity.confidence
            
            # Boost government entities
            if entity.is_government:
                entity.confidence = min(1.0, base_score * 1.2)
            
            # Boost official positions
            if entity.entity_type == EntityType.POSITION:
                entity.confidence = min(1.0, entity.confidence * 1.1)
            
            # Slight penalty for generic organizations
            if (entity.entity_type == EntityType.ORGANIZATION and 
                not entity.is_government):
                entity.confidence = max(0.1, entity.confidence * 0.9)
        
        return entities
    
    def calculate_responsibility_score(self, text: str) -> Dict:
        """
        Calculate overall responsibility score for a document segment.
        
        Args:
            text: Document segment text
            
        Returns:
            Dictionary with responsibility assessment
        """
        entities = self.detect_entities(text)
        
        # Calculate aggregate scores
        gov_score = sum(e.confidence for e in entities if e.is_government)
        person_score = sum(e.confidence for e in entities if e.entity_type == EntityType.PERSON)
        position_score = sum(e.confidence for e in entities if e.entity_type == EntityType.POSITION)
        org_score = sum(e.confidence for e in entities if e.entity_type == EntityType.ORGANIZATION and not e.is_government)
        
        # Calculate final factibility score
        total_score = gov_score * 2.0 + position_score * 1.5 + person_score * 1.0 + org_score * 0.8
        max_possible = len(entities) * 2.0 if entities else 1
        factibility_score = min(1.0, total_score / max_possible)
        
        return {
            'factibility_score': factibility_score,
            'entities': entities,
            'breakdown': {
                'government_score': gov_score,
                'position_score': position_score,
                'person_score': person_score,
                'organization_score': org_score,
                'total_entities': len(entities)
            },
            'has_government_entities': any(e.is_government for e in entities),
            'has_official_positions': any(e.entity_type == EntityType.POSITION for e in entities)
        }


# Example usage and testing
if __name__ == "__main__":
    detector = ResponsibilityDetector()
    
    # Test samples
    test_texts = [
        "La Alcaldía Municipal coordinará con la Secretaría de Salud el programa de vacunación.",
        "Juan Pérez, director de la oficina regional, supervisará el proyecto.",
        "El Ministerio de Educación establecerá nuevas directrices para las instituciones educativas.",
        "La empresa XYZ trabajará con el equipo técnico en la implementación."
    ]
    
    for text in test_texts:
        print(f"\nTexto: {text}")
        result = detector.calculate_responsibility_score(text)
        print(f"Factibilidad: {result['factibility_score']:.3f}")
        print("Entidades encontradas:")
        for entity in result['entities']:
            print(f"  - {entity.text} ({entity.entity_type.value}, conf: {entity.confidence:.3f})")