# coding=utf-8
import logging
import os
import threading
import time
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

import spacy
import spacy.cli
from spacy.language import Language

from text_truncation_logger import log_debug_with_text, log_warning_with_text

logger = logging.getLogger(__name__)

_SPACY_SINGLETONS: dict[int, "SpacyModelLoader"] = {}
_SPACY_SINGLETON_LOCK = threading.RLock()


class SpacyModelLoader:
    """
    Robust spaCy model loader with automatic download, retry logic, and degraded mode fallback.
    """

    def __init__(
        self,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        *,
        max_cache_size: int = 4,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.degraded_mode = False
        self._max_cache_size = max(1, max_cache_size)
        self.loaded_models: "OrderedDict[str, Language]" = OrderedDict()
        self._lock = threading.RLock()
        return None

    def load_model(
        self, model_name: str, disable: Optional[list] = None
    ) -> Optional[Language]:
        """
        Load a spaCy model with automatic download and retry logic.

        Args:
            model_name: Name of the spaCy model to load
            disable: List of pipeline components to disable

        Returns:
            Loaded spaCy model or None if loading fails completely
        """
        with self._lock:
            if model_name in self.loaded_models:
                self.loaded_models.move_to_end(model_name)
                return self.loaded_models[model_name]

        # First attempt to load the model
        model = self._try_load_model(model_name, disable)
        if model is not None:
            with self._lock:
                self.loaded_models[model_name] = model
                self.loaded_models.move_to_end(model_name)
                self._prune_cache_if_needed()
            return model

        # Model not found, attempt automatic download
        logger.warning(
            f"spaCy model '{model_name}' not found. Attempting automatic download..."
        )

        if self._download_model_with_retry(model_name):
            # Try loading again after successful download
            model = self._try_load_model(model_name, disable)
            if model is not None:
                with self._lock:
                    self.loaded_models[model_name] = model
                    self.loaded_models.move_to_end(model_name)
                    self._prune_cache_if_needed()
                logger.info(
                    f"Successfully loaded spaCy model '{model_name}' after download"
                )
                return model

        # All attempts failed, enter degraded mode
        logger.error(
            f"Failed to load spaCy model '{model_name}'. Operating in degraded mode."
        )
        with self._lock:
            self.degraded_mode = True
        return None

    @staticmethod
    def _try_load_model(
        model_name: str, disable: Optional[list] = None
    ) -> Optional[Language]:
        """
        Attempt to load a spaCy model without downloading.

        Args:
            model_name: Name of the spaCy model to load
            disable: List of pipeline components to disable

        Returns:
            Loaded spaCy model or None if loading fails
        """
        try:
            return spacy.load(model_name, disable=disable or [])
        except Exception as e:  # Catch all exceptions to prevent SystemExit
            log_debug_with_text(
                logger, f"Failed to load model '{model_name}': {e}")
            return None

    def _download_model_with_retry(self, model_name: str) -> bool:
        """
        Download a spaCy model with retry logic.

        Args:
            model_name: Name of the spaCy model to download

        Returns:
            True if download successful, False otherwise
        """
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    f"Downloading spaCy model '{model_name}' (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                spacy.cli.download(model_name)
                logger.info(
                    f"Successfully downloaded spaCy model '{model_name}'")
                return True

            except Exception as e:
                logger.warning(
                    f"Download attempt {attempt + 1} failed for model '{model_name}': {e}"
                )

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"All download attempts failed for model '{model_name}'. "
                        f"Possible causes: offline environment, insufficient permissions, "
                        f"or model name not found."
                    )

        return False

    def _prune_cache_if_needed(self) -> None:
        """Ensure the in-memory cache respects the configured capacity."""
        while len(self.loaded_models) > self._max_cache_size:
            self.loaded_models.popitem(last=False)

    def is_degraded_mode(self) -> bool:
        """Check if the loader is operating in degraded mode."""
        with self._lock:
            return self.degraded_mode

    def get_loaded_models(self) -> dict:
        """Get dictionary of successfully loaded models."""
        with self._lock:
            return dict(self.loaded_models)


class SafeSpacyProcessor:
    """
    Example processor that gracefully handles missing spaCy models.
    """

    def __init__(
        self,
        preferred_model: str = "en_core_web_sm",
        *,
        loader: Optional[SpacyModelLoader] = None,
    ):
        self.loader = loader or get_spacy_model_loader()
        self.model = self.loader.load_model(preferred_model)
        self.preferred_model = preferred_model
        return None
    def process_text(self, text: str) -> dict:
        """
        Process text with available spaCy functionality or fallback methods.

        Args:
            text: Input text to process

        Returns:
            Dictionary with processing results
        """
        if self.model is not None:
            # Full functionality available
            doc = self.model(text)
            return {
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "sentences": [sent.text for sent in doc.sents],
                "processing_mode": "full",
            }
        else:
            # Degraded mode - basic text processing
            log_warning_with_text(
                logger,
                f"Processing text in degraded mode (no spaCy model available)",
                text,
            )
            return {
                "tokens": text.split(),  # Basic whitespace tokenization
                "lemmas": [],  # Not available in degraded mode
                "pos_tags": [],  # Not available in degraded mode
                "entities": [],  # Not available in degraded mode
                "sentences": text.split("."),  # Basic sentence splitting
                "processing_mode": "degraded",
            }

    def is_fully_functional(self) -> bool:
        """Check if processor has full spaCy functionality."""
        return self.model is not None and not self.loader.is_degraded_mode()


def get_spacy_model_loader() -> SpacyModelLoader:
    """Return a per-process singleton loader instance."""

    pid = os.getpid()
    with _SPACY_SINGLETON_LOCK:
        loader = _SPACY_SINGLETONS.get(pid)
        if loader is None:
            loader = SpacyModelLoader()
            _SPACY_SINGLETONS[pid] = loader
        return loader


def _reset_spacy_singleton_for_testing() -> None:
    """Clear singleton cache – intended for test suites only."""

    with _SPACY_SINGLETON_LOCK:
        _SPACY_SINGLETONS.pop(os.getpid(), None)


def setup_logging():
    """
    Setup logging with RotatingFileHandler and configurable log directory.
    Falls back to current working directory if LOG_DIR is not set or not writable.
    """
    # Get log directory from environment variable or fallback to current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())

    # Ensure log directory exists and is writable
    try:
        os.makedirs(log_dir, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(log_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (PermissionError, OSError, IOError) as e:
        # Fallback to current working directory
        warning_msg = f"Cannot write to LOG_DIR '{log_dir}': {e}. Falling back to current directory."
        log_dir = os.getcwd()
        # Log warning to console since file logging isn't set up yet
        print(f"WARNING: {warning_msg}")

        # Try current directory as final fallback
        try:
            test_file = os.path.join(log_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except (PermissionError, OSError, IOError) as fallback_e:
            print(
                f"WARNING: Cannot write to fallback directory '{log_dir}': {fallback_e}"
            )
            return None  # Skip file logging setup if no writable directory available

    log_file = os.path.join(log_dir, "spacy_loader.log")
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    try:
        # Setup RotatingFileHandler with appropriate parameters
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB per file
            backupCount=5,  # Keep 5 backup files
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Also add console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        logger.info(
            f"Logging configured successfully. Log files in: {log_dir}")

    except (PermissionError, OSError, IOError) as e:
        print(
            f"WARNING: Failed to setup file logging: {e}. Using console logging only."
        )
        # Fallback to console-only logging
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    return None


# Example usage and testing functions
def example_usage():
    """Demonstrate the robust spaCy model loading with enhanced logging."""
    setup_logging()

    # Test with a common model
    processor = SafeSpacyProcessor("en_core_web_sm")

    sample_text = "Apple is looking at buying U.K. startup for $1 billion. This is a test sentence."
    result = processor.process_text(sample_text)

    print(f"Processing mode: {result['processing_mode']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Entities: {result['entities']}")

    if not processor.is_fully_functional():
        logger.warning(
            "Application running in degraded mode due to spaCy model issues")
    
    return None


if __name__ == "__main__":
    example_usage()
