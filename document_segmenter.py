"""
Advanced Document Segmentation Module
=====================================
Implements dual-criteria segmentation targeting ~3 sentences or 700-900 characters.
Uses spaCy tokenizer for sentence boundary detection with intelligent fallback logic.
"""

import logging
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spacy_loader import SpacyModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentMetrics:
    """Metrics for document segment analysis."""

    char_count: int
    sentence_count: int
    word_count: int
    token_count: int
    semantic_coherence_score: float = 0.0
    segment_type: str = "unknown"  # sentence_based, character_based, hybrid


@dataclass
class SegmentationStats:
    """Statistics for segmentation quality analysis."""

    segments: List[SegmentMetrics] = field(default_factory=list)
    total_segments: int = 0
    segments_in_char_range: int = 0  # 700-900 chars
    segments_with_3_sentences: int = 0
    avg_char_length: float = 0.0
    avg_sentence_count: float = 0.0
    char_length_distribution: Dict[str, int] = field(default_factory=dict)
    sentence_count_distribution: Dict[int, int] = field(default_factory=dict)


class DocumentSegmenter:
    """
    Advanced document segmentation using dual criteria:
    - Primary: ~3 sentences per segment
    - Secondary: 700-900 character range
    - Fallback: Character-based splitting with semantic awareness
    """

    def __init__(
        self,
        target_char_min: int = 700,
        target_char_max: int = 900,
        target_sentences: int = 3,
        max_sentence_deviation: int = 1,
        min_segment_chars: int = 200,
        max_segment_chars: int = 1200,
        semantic_coherence_threshold: float = 0.6,
    ):
        """
        Initialize document segmenter with configurable parameters.

        Args:
            target_char_min: Minimum target character count
            target_char_max: Maximum target character count
            target_sentences: Target number of sentences per segment
            max_sentence_deviation: Maximum deviation from target sentences
            min_segment_chars: Absolute minimum segment size
            max_segment_chars: Absolute maximum segment size
            semantic_coherence_threshold: Threshold for semantic coherence
        """
        self.target_char_min = target_char_min
        self.target_char_max = target_char_max
        self.target_sentences = target_sentences
        self.max_sentence_deviation = max_sentence_deviation
        self.min_segment_chars = min_segment_chars
        self.max_segment_chars = max_segment_chars
        self.semantic_coherence_threshold = semantic_coherence_threshold

        # Initialize spaCy model with fallback
        self.spacy_loader = SpacyModelLoader()
        self.nlp = None
        self._initialize_spacy()

        # Metrics tracking
        self.segmentation_stats = SegmentationStats()

    def _initialize_spacy(self):
        """Initialize spaCy model with graceful degradation."""
        try:
            self.nlp = self.spacy_loader.load_model("es_core_news_sm")
            if self.nlp is None:
                logger.warning("spaCy Spanish model not available, using English model")
                self.nlp = self.spacy_loader.load_model("en_core_web_sm")

            if self.nlp is None:
                logger.warning(
                    "No spaCy models available, using rule-based segmentation"
                )

        except Exception as e:
            logger.error(f"Failed to initialize spaCy model: {e}")
            self.nlp = None

    def segment_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment document using dual-criteria approach.

        Args:
            text: Input text to segment

        Returns:
            List of segment dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []

        # Reset stats for new document
        self.segmentation_stats = SegmentationStats()

        try:
            # Primary approach: spaCy sentence segmentation
            if self.nlp is not None:
                segments = self._segment_with_spacy(text)
            else:
                # Fallback: Rule-based segmentation
                segments = self._segment_with_rules(text)

            # Apply post-processing to ensure quality
            segments = self._post_process_segments(segments)

            # Calculate final statistics
            self._calculate_segmentation_stats(segments)

            return segments

        except Exception as e:
            logger.error(f"Document segmentation failed: {e}")
            # Emergency fallback: simple character-based chunking
            return self._emergency_fallback_segmentation(text)

    def _segment_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Segment using spaCy sentence detection with dual criteria."""
        doc = self.nlp(text)
        sentences = [sent for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return self._segment_with_rules(text)

        segments = []
        current_segment_sents = []
        current_char_count = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_text = sent.text.strip()
            sent_char_count = len(sent_text)

            # Decision logic for dual criteria
            if not current_segment_sents:
                # Always start a new segment
                current_segment_sents.append(sent_text)
                current_char_count = sent_char_count
                i += 1
                continue

            # Check if adding this sentence meets our criteria
            projected_char_count = (
                current_char_count + sent_char_count + 1
            )  # +1 for space
            current_sent_count = len(current_segment_sents)

            # Primary criterion: sentence count (with flexibility)
            target_sentences_met = (
                current_sent_count
                >= self.target_sentences - self.max_sentence_deviation
                and current_sent_count
                <= self.target_sentences + self.max_sentence_deviation
            )

            # Secondary criterion: character range
            in_char_range = (
                self.target_char_min <= projected_char_count <= self.target_char_max
            )

            # Decision matrix for dual criteria
            should_finalize_segment = self._should_finalize_segment(
                current_sent_count,
                projected_char_count,
                sent_char_count,
                i,
                len(sentences),
            )

            if should_finalize_segment:
                # Finalize current segment
                segment_text = " ".join(current_segment_sents)
                segments.append(
                    self._create_segment_dict(
                        segment_text, current_segment_sents, "sentence_based"
                    )
                )

                # Start new segment
                current_segment_sents = [sent_text]
                current_char_count = sent_char_count
            else:
                # Add sentence to current segment
                current_segment_sents.append(sent_text)
                current_char_count = projected_char_count

            i += 1

        # Handle remaining sentences
        if current_segment_sents:
            segment_text = " ".join(current_segment_sents)
            segments.append(
                self._create_segment_dict(
                    segment_text, current_segment_sents, "sentence_based"
                )
            )

        return segments

    def _should_finalize_segment(
        self,
        current_sent_count: int,
        projected_char_count: int,
        next_sent_char_count: int,
        sent_index: int,
        total_sentences: int,
    ) -> bool:
        """
        Intelligent decision logic for segment finalization using dual criteria.
        """
        current_char_count = (
            projected_char_count - next_sent_char_count - 1
        )  # Approximate current count

        # Absolute limits
        if projected_char_count > self.max_segment_chars:
            return True

        if current_sent_count >= self.target_sentences + self.max_sentence_deviation:
            return True

        # If we have exactly target sentences and current segment is in character range, finalize
        if (
            current_sent_count == self.target_sentences
            and self.target_char_min <= current_char_count <= self.target_char_max
        ):
            return True

        # If we have minimum sentences and adding more would exceed target char range
        if (
            current_sent_count >= self.target_sentences - self.max_sentence_deviation
            and projected_char_count > self.target_char_max
        ):
            return True

        # If we're at minimum sentences and next sentence is very long
        if (
            current_sent_count >= self.target_sentences - self.max_sentence_deviation
            and next_sent_char_count > 400
        ):  # Very long sentence
            return True

        # If we're near the end and have reasonable content
        if (
            sent_index >= total_sentences - 2
            and current_sent_count >= 2
            and current_char_count >= self.min_segment_chars
        ):
            return True

        # If current segment meets char criteria and we have at least minimum sentences
        if (
            current_sent_count >= self.target_sentences - self.max_sentence_deviation
            and self.target_char_min <= current_char_count <= self.target_char_max
        ):
            return True

        return False

    def _segment_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """Fallback rule-based segmentation when spaCy is unavailable."""
        # Simple sentence detection using punctuation
        sentence_pattern = r"[.!?]+\s+"
        potential_sentences = re.split(sentence_pattern, text.strip())

        # Clean and filter sentences
        sentences = [
            s.strip() for s in potential_sentences if s.strip() and len(s.strip()) > 10
        ]

        if not sentences:
            # Character-based fallback
            return self._character_based_segmentation(text)

        segments = []
        current_segment_sents = []
        current_char_count = 0

        for i, sent in enumerate(sentences):
            sent_char_count = len(sent)

            if not current_segment_sents:
                current_segment_sents.append(sent)
                current_char_count = sent_char_count
                continue

            projected_char_count = current_char_count + sent_char_count + 1
            current_sent_count = len(current_segment_sents)

            should_finalize = self._should_finalize_segment(
                current_sent_count,
                projected_char_count,
                sent_char_count,
                i,
                len(sentences),
            )

            if should_finalize:
                segment_text = " ".join(current_segment_sents)
                segments.append(
                    self._create_segment_dict(
                        segment_text, current_segment_sents, "rule_based"
                    )
                )

                current_segment_sents = [sent]
                current_char_count = sent_char_count
            else:
                current_segment_sents.append(sent)
                current_char_count = projected_char_count

        # Handle remaining sentences
        if current_segment_sents:
            segment_text = " ".join(current_segment_sents)
            segments.append(
                self._create_segment_dict(
                    segment_text, current_segment_sents, "rule_based"
                )
            )

        return segments

    def _character_based_segmentation(self, text: str) -> List[Dict[str, Any]]:
        """Character-based segmentation with word boundary preservation."""
        segments = []
        words = text.split()

        current_segment_words = []
        current_char_count = 0
        target_chars = (
            self.target_char_min + self.target_char_max
        ) // 2  # Use midpoint

        for word in words:
            word_length = len(word)
            projected_length = (
                current_char_count + word_length + len(current_segment_words)
            )  # +spaces

            if (
                projected_length > target_chars
                and current_char_count >= self.min_segment_chars
                and current_segment_words
            ):
                # Finalize current segment
                segment_text = " ".join(current_segment_words)
                segments.append(
                    self._create_segment_dict(segment_text, [], "character_based")
                )

                # Start new segment
                current_segment_words = [word]
                current_char_count = word_length
            else:
                current_segment_words.append(word)
                current_char_count += word_length

        # Handle remaining words
        if current_segment_words:
            segment_text = " ".join(current_segment_words)
            segments.append(
                self._create_segment_dict(segment_text, [], "character_based")
            )

        return segments

    def _create_segment_dict(
        self, text: str, sentences: List[str], segment_type: str
    ) -> Dict[str, Any]:
        """Create segment dictionary with metadata."""
        # Calculate metrics
        char_count = len(text)
        sentence_count = (
            len(sentences) if sentences else self._estimate_sentence_count(text)
        )
        word_count = len(text.split())
        token_count = len(text.split())  # Simple approximation

        # Estimate semantic coherence (placeholder - could be enhanced with embeddings)
        coherence_score = self._estimate_semantic_coherence(text)

        metrics = SegmentMetrics(
            char_count=char_count,
            sentence_count=sentence_count,
            word_count=word_count,
            token_count=token_count,
            semantic_coherence_score=coherence_score,
            segment_type=segment_type,
        )

        return {
            "text": text,
            "sentences": sentences,
            "metrics": metrics,
            "meets_char_criteria": self.target_char_min
            <= char_count
            <= self.target_char_max,
            "meets_sentence_criteria": abs(sentence_count - self.target_sentences)
            <= self.max_sentence_deviation,
            "segment_type": segment_type,
        }

    def _estimate_sentence_count(self, text: str) -> int:
        """Estimate sentence count for character-based segments."""
        return len(re.findall(r"[.!?]+", text)) + 1

    def _estimate_semantic_coherence(self, text: str) -> float:
        """
        Estimate semantic coherence of a segment.
        Simple heuristic - could be enhanced with embeddings.
        """
        # Simple coherence indicators
        words = text.lower().split()
        if len(words) < 3:
            return 0.5

        # Repetition indicates coherence
        word_counts = Counter(words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_score = min(repeated_words / len(word_counts), 0.5)

        # Length consistency (sentences of similar length indicate coherence)
        if self.nlp:
            try:
                doc = self.nlp(text)
                sent_lengths = [
                    len(sent.text) for sent in doc.sents if sent.text.strip()
                ]
                if len(sent_lengths) > 1:
                    length_variance = statistics.variance(sent_lengths) / max(
                        statistics.mean(sent_lengths), 1
                    )
                    length_score = max(0, 1.0 - length_variance / 100)  # Normalize
                else:
                    length_score = 0.5
            except:
                length_score = 0.5
        else:
            length_score = 0.5

        return min((repetition_score + length_score) / 2, 1.0)

    def _post_process_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Post-process segments to handle edge cases and improve quality."""
        if not segments:
            return segments

        processed_segments = []

        for i, segment in enumerate(segments):
            metrics = segment["metrics"]

            # Handle segments that are too small (but be more lenient in tests)
            if metrics.char_count < self.min_segment_chars and len(segments) > 1:
                if processed_segments:
                    # Merge with previous segment, preserving original type if possible
                    prev_segment = processed_segments[-1]
                    merged_text = prev_segment["text"] + " " + segment["text"]
                    merged_sentences = prev_segment["sentences"] + segment["sentences"]
                    # Preserve the original segment type when merging from rule-based
                    original_type = prev_segment.get(
                        "segment_type", segment.get("segment_type", "merged")
                    )
                    processed_segments[-1] = self._create_segment_dict(
                        merged_text, merged_sentences, original_type
                    )
                else:
                    # Keep as is if it's the first segment
                    processed_segments.append(segment)

            # Handle segments that are too large
            elif metrics.char_count > self.max_segment_chars:
                # Split large segment
                sub_segments = self._split_large_segment(segment)
                processed_segments.extend(sub_segments)

            else:
                processed_segments.append(segment)

        return processed_segments

    def _split_large_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split segments that exceed maximum size."""
        text = segment["text"]
        sentences = segment["sentences"]

        if sentences and len(sentences) > 1:
            # Split by sentences
            mid_point = len(sentences) // 2
            first_half = sentences[:mid_point]
            second_half = sentences[mid_point:]

            return [
                self._create_segment_dict(
                    " ".join(first_half), first_half, "split_large"
                ),
                self._create_segment_dict(
                    " ".join(second_half), second_half, "split_large"
                ),
            ]
        else:
            # Character-based split
            return self._character_based_segmentation(text)

    def _calculate_segmentation_stats(self, segments: List[Dict[str, Any]]):
        """Calculate comprehensive segmentation statistics."""
        if not segments:
            return

        self.segmentation_stats.total_segments = len(segments)

        char_lengths = []
        sentence_counts = []

        for segment in segments:
            metrics = segment["metrics"]
            self.segmentation_stats.segments.append(metrics)

            char_lengths.append(metrics.char_count)
            sentence_counts.append(metrics.sentence_count)

            # Count segments meeting criteria
            if self.target_char_min <= metrics.char_count <= self.target_char_max:
                self.segmentation_stats.segments_in_char_range += 1

            if metrics.sentence_count == self.target_sentences:
                self.segmentation_stats.segments_with_3_sentences += 1

        # Calculate averages
        self.segmentation_stats.avg_char_length = statistics.mean(char_lengths)
        self.segmentation_stats.avg_sentence_count = statistics.mean(sentence_counts)

        # Calculate distributions
        self.segmentation_stats.char_length_distribution = (
            self._create_char_distribution(char_lengths)
        )
        self.segmentation_stats.sentence_count_distribution = Counter(sentence_counts)

    def _create_char_distribution(self, char_lengths: List[int]) -> Dict[str, int]:
        """Create character length distribution buckets."""
        distribution = defaultdict(int)

        for length in char_lengths:
            if length < 500:
                bucket = "< 500"
            elif length < 700:
                bucket = "500-699"
            elif length <= 900:
                bucket = "700-900 (target)"
            elif length <= 1200:
                bucket = "901-1200"
            else:
                bucket = "> 1200"

            distribution[bucket] += 1

        return dict(distribution)

    def _emergency_fallback_segmentation(self, text: str) -> List[Dict[str, Any]]:
        """Emergency fallback when all other methods fail."""
        target_size = (self.target_char_min + self.target_char_max) // 2
        segments = []

        for i in range(0, len(text), target_size):
            chunk = text[i : i + target_size].strip()
            if chunk:
                segments.append(
                    self._create_segment_dict(chunk, [], "emergency_fallback")
                )

        return segments

    def get_segmentation_report(self) -> Dict[str, Any]:
        """Generate comprehensive segmentation quality report."""
        stats = self.segmentation_stats

        if stats.total_segments == 0:
            return {"error": "No segments to analyze"}

        # Calculate success rates
        char_range_success_rate = (
            stats.segments_in_char_range / stats.total_segments
        ) * 100
        sentence_target_success_rate = (
            stats.segments_with_3_sentences / stats.total_segments
        ) * 100

        # Calculate quality metrics
        char_lengths = [seg.char_count for seg in stats.segments]
        char_std_dev = statistics.stdev(char_lengths) if len(char_lengths) > 1 else 0

        sentence_counts = [seg.sentence_count for seg in stats.segments]
        sentence_std_dev = (
            statistics.stdev(sentence_counts) if len(sentence_counts) > 1 else 0
        )

        report = {
            "summary": {
                "total_segments": stats.total_segments,
                "avg_char_length": round(stats.avg_char_length, 1),
                "avg_sentence_count": round(stats.avg_sentence_count, 1),
                "char_range_success_rate": round(char_range_success_rate, 1),
                "sentence_target_success_rate": round(sentence_target_success_rate, 1),
            },
            "character_analysis": {
                "target_range": f"{self.target_char_min}-{self.target_char_max}",
                "segments_in_target_range": stats.segments_in_char_range,
                "char_length_std_dev": round(char_std_dev, 1),
                "distribution": dict(stats.char_length_distribution),
            },
            "sentence_analysis": {
                "target_sentences": self.target_sentences,
                "segments_with_target_sentences": stats.segments_with_3_sentences,
                "sentence_count_std_dev": round(sentence_std_dev, 1),
                "distribution": dict(stats.sentence_count_distribution),
            },
            "quality_indicators": {
                "consistency_score": self._calculate_consistency_score(),
                "target_adherence_score": self._calculate_target_adherence_score(),
                "overall_quality_score": self._calculate_overall_quality_score(),
            },
        }

        return report

    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score based on segment size variation."""
        char_lengths = [seg.char_count for seg in self.segmentation_stats.segments]
        if len(char_lengths) < 2:
            return 1.0

        mean_length = statistics.mean(char_lengths)
        std_dev = statistics.stdev(char_lengths)
        cv = std_dev / mean_length if mean_length > 0 else 1

        # Lower coefficient of variation = higher consistency
        return max(0, 1.0 - cv)

    def _calculate_target_adherence_score(self) -> float:
        """Calculate how well segments adhere to dual criteria."""
        stats = self.segmentation_stats
        if stats.total_segments == 0:
            return 0.0

        char_score = stats.segments_in_char_range / stats.total_segments
        sentence_score = stats.segments_with_3_sentences / stats.total_segments

        # Weight character criteria slightly higher as it's the fallback
        return char_score * 0.6 + sentence_score * 0.4

    def _calculate_overall_quality_score(self) -> float:
        """Calculate overall segmentation quality score."""
        consistency = self._calculate_consistency_score()
        adherence = self._calculate_target_adherence_score()

        # Average coherence score
        coherence_scores = [
            seg.semantic_coherence_score for seg in self.segmentation_stats.segments
        ]
        avg_coherence = statistics.mean(coherence_scores) if coherence_scores else 0.5

        return consistency * 0.3 + adherence * 0.5 + avg_coherence * 0.2

    def log_segmentation_metrics(self):
        """Log detailed segmentation metrics for monitoring."""
        report = self.get_segmentation_report()

        logger.info("=== Document Segmentation Report ===")
        logger.info(f"Total segments: {report['summary']['total_segments']}")
        logger.info(f"Average character length: {report['summary']['avg_char_length']}")
        logger.info(
            f"Average sentence count: {report['summary']['avg_sentence_count']}"
        )
        logger.info(
            f"Character range success rate: {report['summary']['char_range_success_rate']}%"
        )
        logger.info(
            f"Sentence target success rate: {report['summary']['sentence_target_success_rate']}%"
        )
        logger.info(
            f"Overall quality score: {report['quality_indicators']['overall_quality_score']:.3f}"
        )

        logger.info("Character length distribution:")
        for bucket, count in report["character_analysis"]["distribution"].items():
            logger.info(f"  {bucket}: {count} segments")

        logger.info("Sentence count distribution:")
        for count, segments in report["sentence_analysis"]["distribution"].items():
            logger.info(f"  {count} sentences: {segments} segments")
