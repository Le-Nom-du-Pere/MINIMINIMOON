"""Tests for the modern SotaEmbedding backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, List
from unittest.mock import patch

import numpy as np

from embedding_model import (
    CalibrationCorpusStats,
    EmbeddingConfig,
    SotaEmbedding,
    get_default_embedding,
)


class _FakeSentenceTransformer:
    """Deterministic stub mimicking SentenceTransformer for tests."""

    def __init__(self, dimension: int = 2) -> None:
        self._dimension = dimension
        self.encode_invocations = 0
        self.request_history: List[List[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)

        self.request_history.append(batch)
        self.encode_invocations += 1

        # Produce repeatable orthogonal embeddings with shape (n, dimension)
        base = np.eye(self._dimension, dtype=np.float32)
        repeats = int(np.ceil(len(batch) / self._dimension))
        tiled = np.tile(base, (repeats, 1))
        return tiled[: len(batch)]


class _FakeIsotonicRegression:
    """Simple stub returning a predictable calibration curve."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature parity
        self.fit_args = None

    def fit_transform(self, scores: Iterable[float], labels: Iterable[int]) -> np.ndarray:
        scores = list(scores)
        self.fit_args = (scores, list(labels))
        # Return a smoothly increasing sequence with deterministic spread
        return np.linspace(0.2, 0.9, num=len(scores))


class TestSotaEmbedding(unittest.TestCase):
    """Exercise the modern embedding backend with deterministic stubs."""

    def setUp(self) -> None:  # noqa: D401 - unittest hook
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.calibration_path = Path(self.temp_dir.name) / "card.json"

        self.guard_patch = patch(
            "embedding_model.NoveltyGuard.check_dependencies", return_value=None
        )
        self.guard_patch.start()
        self.addCleanup(self.guard_patch.stop)

        def _build_fake_model(*args, **kwargs):
            model = _FakeSentenceTransformer()
            return model

        self.model_patch = patch(
            "embedding_model.SentenceTransformer", side_effect=_build_fake_model
        )
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

        self.config = EmbeddingConfig(
            model="test/model",
            precision="fp32",
            batch_size=8,
            normalize_l2=False,
            calibration_card=str(self.calibration_path),
            domain_hint_default="PDM",
            device="cpu",
        )

        self.backend = SotaEmbedding(self.config)

    def test_initialization_creates_calibration_card(self) -> None:
        """The backend seeds a default calibration card on startup."""

        card = self.backend.calibration_card
        self.assertIsNotNone(card)
        self.assertEqual(card.model_name, self.config.model)
        self.assertTrue(Path(self.config.calibration_card).exists())

        with open(self.config.calibration_card, "r", encoding="utf-8") as handle:
            card_payload = json.load(handle)

        self.assertEqual(card_payload["model_name"], self.config.model)
        self.assertIn("conformal_thresholds", card_payload)

    def test_embed_texts_applies_domain_smoothing_and_cache(self) -> None:
        """Domain priors influence embeddings and results are cached."""

        self.backend.calibration_card.domain_priors["PDM"] = 0.5

        first = self.backend.embed_texts(["uno", "dos"], domain_hint="PDM")
        expected = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32)
        np.testing.assert_allclose(first, expected)

        cached = self.backend.embed_texts(["uno", "dos"], domain_hint="PDM")
        np.testing.assert_allclose(cached, first)
        self.assertEqual(self.backend.model.encode_invocations, 1)

    def test_calibrate_updates_card_with_corpus_statistics(self) -> None:
        """Running calibration produces updated priors and thresholds."""

        corpus_stats = CalibrationCorpusStats(
            corpus_size=200,
            embedding_dim=2,
            similarity_mean=0.6,
            similarity_std=0.1,
            confidence_scores=list(np.linspace(0.1, 0.9, num=20)),
            gold_labels=[1] * 10 + [0] * 10,
            domain_distribution={"PDM": 80, "rural": 20},
        )

        with patch("embedding_model.IsotonicRegression", return_value=_FakeIsotonicRegression()):
            card = self.backend.calibrate(corpus_stats)

        self.assertEqual(card.embedding_dim, corpus_stats.embedding_dim)
        self.assertTrue(card.isotonic_calibrator["fitted"])
        self.assertAlmostEqual(card.normalization_params["mean"], 0.55, places=2)
        self.assertIn("alpha_0.1", card.conformal_thresholds)
        self.assertIn("rural", card.domain_priors)
        self.assertAlmostEqual(sum(card.domain_priors.values()), 1.0, places=2)

    def test_get_default_embedding_uses_configuration_factory(self) -> None:
        """The factory returns a SotaEmbedding wired with the supplied config."""

        config = self.config.model_copy(update={
            "calibration_card": str(Path(self.temp_dir.name) / "default_card.json")
        })

        with patch("embedding_model.load_embedding_config", return_value=config):
            backend = get_default_embedding()

        self.assertIsInstance(backend, SotaEmbedding)
        self.assertEqual(backend.config.model, config.model)
        self.assertTrue(Path(config.calibration_card).exists())

    def tearDown(self) -> None:  # noqa: D401 - unittest hook
        # Ensure patches from setUp are correctly removed even if assertions fail
        for patcher in [self.model_patch, self.guard_patch]:
            try:
                patcher.stop()
            except RuntimeError:
                # Already stopped by addCleanup
                pass


if __name__ == "__main__":
    unittest.main()
