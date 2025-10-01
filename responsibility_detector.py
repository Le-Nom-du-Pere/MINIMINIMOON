"""
Responsibility Detection Module using spaCy NER and Lexical Rules
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import spacy


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
        self.nlp = self._load_spacy_pipeline(model_name)

        # Government institution patterns (high priority)
        self.government_patterns = [
            r"alcald[ií]a(?:\s+(?:de|del|municipal))?",
            r"secretar[ií]a(?:\s+(?:de|del))?",
            r"programa(?:\s+(?:de|del))?",
            r"ministerio(?:\s+(?:de|del))?",
            r"gobernaci[oó]n(?:\s+(?:de|del))?",
            r"municipalidad",
            r"ayuntamiento",
            r"concejal[ií]a",
            r"departamento\s+administrativo",
            r"instituto\s+(?:nacional|municipal|departamental)",
            r"direcci[oó]n\s+(?:nacional|municipal|departamental)",
            r"superintendencia",
            r"procuradur[ií]a",
            r"defensor[ií]a\s+del\s+pueblo",
            r"contralor[ií]a",
        ]

        # Official position patterns
        self.position_patterns = [
            r"alcalde(?:sa)?",
            r"secretari[oa](?:\s+(?:de|del))?",
            r"ministr[oa](?:\s+(?:de|del))?",
            r"director(?:a)?(?:\s+(?:general|ejecutiv[oa]|t[eé]cnic[oa]))?",
            r"coordinador(?:a)?(?:\s+(?:de|del))?",
            r"jefe(?:\s+(?:de|del))?",
            r"responsable(?:\s+(?:de|del))?",
            r"encargad[oa](?:\s+(?:de|del))?",
            r"gerente(?:\s+(?:de|del))?",
            r"presidente(?:a)?(?:\s+(?:de|del))?",
            r"gobernador(?:a)?",
            r"concejal(?:a)?",
            r"comisionad[oa]",
            r"subdirector(?:a)?",
            r"viceministr[oa]",
        ]

        # Generic institutional terms
        self.institutional_patterns = [
            r"entidad(?:\s+(?:p[úu]blica|gubernamental))?",
            r"organizaci[oó]n(?:\s+(?:p[úu]blica|gubernamental))?",
            r"instituci[oó]n(?:\s+(?:p[úu]blica|gubernamental))?",
            r"dependencia(?:\s+(?:p[úu]blica|gubernamental))?",
            r"oficina(?:\s+(?:p[úu]blica|gubernamental))?",
            r"unidad(?:\s+(?:administrativa|t[eé]cnica))?",
        ]

        # Contextual responsibility cues (increase confidence if in proximity)
        self.context_cues = re.compile(
            r"\b(responsable(?:s)?\s+de|a\s+cargo\s+de|competencia\s+de|deber(?:es)?\s+de|obligaci[oó]n\s+de|encargad[oa]\s+de|lidera(?:r|do)|coordina(?:r|ci[oó]n)|supervisa(?:r|ci[oó]n))\b",
            re.IGNORECASE,
        )

        # Compile patterns
        self.compiled_gov_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.government_patterns
        ]
        self.compiled_pos_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.position_patterns
        ]
        self.compiled_inst_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.institutional_patterns
        ]

        # Pre-compile normalization for government keyword check
        self.gov_keyword_re = re.compile(
            r"\b(alcald[ií]a|secretar[ií]a|programa|ministerio|gobernaci[oó]n|municipalidad|ayuntamiento|concejal[ií]a|instituto|direcci[oó]n|departamento|nacional|municipal|departamental|superintendencia|procuradur[ií]a|defensor[ií]a\s+del\s+pueblo|contralor[ií]a)\b",
            re.IGNORECASE,
        )

    # ---------- Public API ----------

    def detect_entities(self, text: str) -> List[ResponsibilityEntity]:
        """
        Detect responsibility entities in text using NER and lexical rules.

        Args:
            text: Input text segment

        Returns:
            List of ResponsibilityEntity objects with confidence scores
        """
        safe_text = text if isinstance(text, str) else ""
        doc = self.nlp(safe_text)

        entities: List[ResponsibilityEntity] = []
        entities.extend(self._extract_ner_entities(doc))
        entities.extend(self._extract_pattern_entities(safe_text))

        entities = self._merge_overlapping_entities(
            self._dedupe_entities(entities))
        entities = self._calculate_final_scores(safe_text, entities)

        return sorted(entities, key=lambda x: x.confidence, reverse=True)

    def calculate_responsibility_score(self, text: str) -> Dict:
        """
        Calculate overall responsibility score for a document segment.

        Args:
            text: Document segment text

        Returns:
            Dictionary with responsibility assessment
        """
        entities = self.detect_entities(text)

        gov_score = sum(e.confidence for e in entities if e.is_government)
        person_score = sum(
            e.confidence for e in entities if e.entity_type == EntityType.PERSON
        )
        position_score = sum(
            e.confidence for e in entities if e.entity_type == EntityType.POSITION
        )
        org_score = sum(
            e.confidence
            for e in entities
            if e.entity_type == EntityType.ORGANIZATION and not e.is_government
        )

        # Weighted linear combination; cap by theoretical max (2.0 per entity as en el diseño original)
        total_score = (
            gov_score * 2.0
            + position_score * 1.5
            + person_score * 1.0
            + org_score * 0.8
        )
        max_possible = len(entities) * 2.0 if entities else 1.0
        factibility_score = min(1.0, total_score / max_possible)

        return {
            "factibility_score": factibility_score,
            "entities": entities,
            "breakdown": {
                "government_score": gov_score,
                "position_score": position_score,
                "person_score": person_score,
                "organization_score": org_score,
                "total_entities": len(entities),
            },
            "has_government_entities": any(e.is_government for e in entities),
            "has_official_positions": any(
                e.entity_type == EntityType.POSITION for e in entities
            ),
        }

    # ---------- Internal helpers ----------

    @staticmethod
    def _load_spacy_pipeline(preferred: str):
        """
        Load spaCy pipeline with robust fallbacks while preserving the same constructor signature.
        """
        tried = []
        candidates = [
            preferred,
            "es_core_news_md",
            "es_core_news_lg",
            "xx_ent_wiki_sm",
            "xx_sent_ud_sm",
        ]
        for name in candidates:
            try:
                nlp = spacy.load(name)
                # Disable unnecessary components if present to speed up (keeps NER)
                for pipe in [
                    "lemmatizer",
                    "morphologizer",
                    "attribute_ruler",
                    "tagger",
                    "parser",
                    "senter",
                ]:
                    if pipe in nlp.pipe_names and pipe != "ner":
                        try:
                            nlp.disable_pipe(pipe)
                        except Exception:
                            pass
                return nlp
            except Exception as e:
                tried.append(f"{name}: {type(e).__name__}")
                continue
        # Ultimate fallback: blank Spanish with an empty NER (will rely on rules)
        nlp = spacy.blank("es")
        return nlp

    def _extract_ner_entities(self, doc) -> List[ResponsibilityEntity]:
        """Extract PERSON and ORG entities from spaCy NER."""
        entities: List[ResponsibilityEntity] = []
        if not hasattr(doc, "ents"):
            return entities

        for ent in doc.ents:
            label = ent.label_
            if label in ("PERSON", "PER"):
                entities.append(
                    ResponsibilityEntity(
                        text=ent.text,
                        entity_type=EntityType.PERSON,
                        confidence=0.72,  # slightly tuned base
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                    )
                )
            elif label == "ORG":
                text = ent.text
                is_gov = self._is_government_entity(text)
                entities.append(
                    ResponsibilityEntity(
                        text=text,
                        entity_type=(
                            EntityType.GOVERNMENT if is_gov else EntityType.ORGANIZATION
                        ),
                        confidence=0.82 if is_gov else 0.62,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        is_government=is_gov,
                    )
                )
        return entities

    def _extract_pattern_entities(self, text: str) -> List[ResponsibilityEntity]:
        """Extract entities using lexical patterns."""
        entities: List[ResponsibilityEntity] = []

        # Government institutions (highest priority)
        for pattern in self.compiled_gov_patterns:
            for m in pattern.finditer(text):
                entities.append(
                    ResponsibilityEntity(
                        text=m.group(),
                        entity_type=EntityType.GOVERNMENT,
                        confidence=0.90,
                        start_pos=m.start(),
                        end_pos=m.end(),
                        is_government=True,
                    )
                )

        # Official positions
        for pattern in self.compiled_pos_patterns:
            for m in pattern.finditer(text):
                entities.append(
                    ResponsibilityEntity(
                        text=m.group(),
                        entity_type=EntityType.POSITION,
                        confidence=0.80,
                        start_pos=m.start(),
                        end_pos=m.end(),
                        role=m.group(),
                        is_government=True,  # cargos públicos implican poder/deber institucional
                    )
                )

        # Generic institutional terms (lower priority)
        for pattern in self.compiled_inst_patterns:
            for m in pattern.finditer(text):
                entities.append(
                    ResponsibilityEntity(
                        text=m.group(),
                        entity_type=EntityType.ORGANIZATION,
                        confidence=0.50,
                        start_pos=m.start(),
                        end_pos=m.end(),
                        is_government=False,
                    )
                )

        return entities

    @staticmethod
    def _strip_accents(s: str) -> str:
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    def _is_government_entity(self, text: str) -> bool:
        """Check if an entity is likely a government organization."""
        t = self._strip_accents(text.lower())
        return bool(self.gov_keyword_re.search(t))

    @staticmethod
    def _span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Intersection-over-Union for character spans."""
        a0, a1 = a
        b0, b1 = b
        inter = max(0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        return (inter / union) if union > 0 else 0.0

    @staticmethod
    def _normalize_text_for_key(s: str) -> str:
        s = s.strip()
        s = unicodedata.normalize("NFKC", s)
        s = s.lower()
        s = "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"\s+", " ", s)

    def _dedupe_entities(
        self, entities: List[ResponsibilityEntity]
    ) -> List[ResponsibilityEntity]:
        """Deduplicate by normalized text and dominant span; keep highest-confidence instance."""
        bucket: Dict[str, ResponsibilityEntity] = {}
        for e in entities:
            key = f"{e.entity_type.value}:{self._normalize_text_for_key(e.text)}"
            if key not in bucket or e.confidence > bucket[key].confidence:
                bucket[key] = e
        return list(bucket.values())

    def _merge_overlapping_entities(
        self, entities: List[ResponsibilityEntity]
    ) -> List[ResponsibilityEntity]:
        """Merge overlapping entities, keeping the most responsible-leaning candidate with IoU-aware logic."""
        if not entities:
            return entities

        entities = sorted(entities, key=lambda x: (x.start_pos, -x.end_pos))
        merged: List[ResponsibilityEntity] = []
        current = entities[0]

        def priority_score(e: ResponsibilityEntity) -> Tuple[int, float]:
            # Prefer GOVERNMENT > POSITION > ORGANIZATION > PERSON, then confidence
            order = {
                EntityType.GOVERNMENT: 3,
                EntityType.POSITION: 2,
                EntityType.ORGANIZATION: 1,
                EntityType.PERSON: 0,
            }
            return (order.get(e.entity_type, 0), e.confidence)

        for nxt in entities[1:]:
            iou = self._span_iou(
                (current.start_pos, current.end_pos), (nxt.start_pos, nxt.end_pos)
            )
            if iou > 0.0:
                # Choose by priority; if tie, higher confidence
                cur_p, nxt_p = priority_score(current), priority_score(nxt)
                if nxt_p > cur_p:
                    current = nxt
                elif nxt_p == cur_p and nxt.confidence > current.confidence:
                    current = nxt
                else:
                    # keep current; ignore nxt
                    pass
            else:
                merged.append(current)
                current = nxt

        merged.append(current)
        return merged

    def _calculate_final_scores(
        self, text: str, entities: List[ResponsibilityEntity]
    ) -> List[ResponsibilityEntity]:
        """Calculate final confidence scores considering entity types, patterns, form, and contextual cues."""
        # Precompute context cue windows to cheaply boost nearby entities
        cue_spans = [m.span() for m in self.context_cues.finditer(text)]

        def near_cue(e: ResponsibilityEntity, window: int = 40) -> bool:
            for c0, c1 in cue_spans:
                if abs(e.start_pos - c1) <= window or abs(e.end_pos - c0) <= window:
                    return True
            return False

        for e in entities:
            base = e.confidence

            # Boost government entities
            if e.is_government:
                base *= 1.20

            # Boost official positions
            if e.entity_type == EntityType.POSITION:
                base *= 1.12

            # Slight penalty for generic non-gov organizations
            if e.entity_type == EntityType.ORGANIZATION and not e.is_government:
                base *= 0.90

            # Form-based calibration: title-case and token length (reduce false short hits)
            token_len = max(1, len(e.text.strip().split()))
            if token_len >= 3:
                base *= 1.05
            elif token_len == 1 and e.entity_type in (
                EntityType.ORGANIZATION,
                EntityType.GOVERNMENT,
            ):
                base *= 0.92

            # Title/Proper-casing heuristic (e.g., "Ministerio de Educación")
            if re.match(r"^[A-ZÁÉÍÓÚÑ][^\n]+", e.text.strip()):
                base *= 1.03

            # Contextual cue proximity
            if near_cue(e):
                base *= 1.08

            # Cap to [0.1, 1.0] to keep scores sane and comparable
            e.confidence = max(0.10, min(1.00, base))

        return entities


# Example usage and testing
if __name__ == "__main__":
    detector = ResponsibilityDetector()

    # Test samples
    test_texts = [
        "La Alcaldía Municipal coordinará con la Secretaría de Salud el programa de vacunación.",
        "Juan Pérez, director de la oficina regional, supervisará el proyecto.",
        "El Ministerio de Educación establecerá nuevas directrices para las instituciones educativas.",
        "La empresa XYZ trabajará con el equipo técnico en la implementación.",
        "Será responsabilidad de la Gobernación del Cauca y del Secretario de Gobierno ejecutar el plan.",
        "La Defensoría del Pueblo coordina con la Procuraduría y la Superintendencia de Salud.",
    ]

    for text in test_texts:
        print(f"\nTexto: {text}")
        result = detector.calculate_responsibility_score(text)
        print(f"Factibilidad: {result['factibility_score']:.3f}")
        print("Entidades encontradas:")
        for entity in result["entities"]:
            print(
                f"  - {entity.text} ({entity.entity_type.value}, conf: {entity.confidence:.3f})"
            )
