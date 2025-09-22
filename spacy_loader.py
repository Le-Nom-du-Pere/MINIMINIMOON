import logging
import time
from typing import Optional, Any
import spacy
import spacy.cli
from spacy.language import Language

logger = logging.getLogger(__name__)

class SpacyModelLoader:
    """
    Robust spaCy model loader with automatic download, retry logic, and degraded mode fallback.
    """
    
    def __init__(self, max_retries: int = 2, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.degraded_mode = False
        self.loaded_models = {}
    
    def load_model(self, model_name: str, disable: Optional[list] = None) -> Optional[Language]:
        """
        Load a spaCy model with automatic download and retry logic.
        
        Args:
            model_name: Name of the spaCy model to load
            disable: List of pipeline components to disable
            
        Returns:
            Loaded spaCy model or None if loading fails completely
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # First attempt to load the model
        model = self._try_load_model(model_name, disable)
        if model is not None:
            self.loaded_models[model_name] = model
            return model
        
        # Model not found, attempt automatic download
        logger.warning(f"spaCy model '{model_name}' not found. Attempting automatic download...")
        
        if self._download_model_with_retry(model_name):
            # Try loading again after successful download
            model = self._try_load_model(model_name, disable)
            if model is not None:
                self.loaded_models[model_name] = model
                logger.info(f"Successfully loaded spaCy model '{model_name}' after download")
                return model
        
        # All attempts failed, enter degraded mode
        logger.error(f"Failed to load spaCy model '{model_name}'. Operating in degraded mode.")
        self.degraded_mode = True
        return None
    
    def _try_load_model(self, model_name: str, disable: Optional[list] = None) -> Optional[Language]:
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
        except (IOError, OSError, ImportError, AttributeError) as e:
            logger.debug(f"Failed to load model '{model_name}': {e}")
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
                logger.info(f"Downloading spaCy model '{model_name}' (attempt {attempt + 1}/{self.max_retries + 1})")
                spacy.cli.download(model_name)
                logger.info(f"Successfully downloaded spaCy model '{model_name}'")
                return True
            
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed for model '{model_name}': {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All download attempts failed for model '{model_name}'. "
                               f"Possible causes: offline environment, insufficient permissions, "
                               f"or model name not found.")
        
        return False
    
    def is_degraded_mode(self) -> bool:
        """Check if the loader is operating in degraded mode."""
        return self.degraded_mode
    
    def get_loaded_models(self) -> dict:
        """Get dictionary of successfully loaded models."""
        return self.loaded_models.copy()


class SafeSpacyProcessor:
    """
    Example processor that gracefully handles missing spaCy models.
    """
    
    def __init__(self, preferred_model: str = "en_core_web_sm"):
        self.loader = SpacyModelLoader()
        self.model = self.loader.load_model(preferred_model)
        self.preferred_model = preferred_model
    
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
                'tokens': [token.text for token in doc],
                'lemmas': [token.lemma_ for token in doc],
                'pos_tags': [token.pos_ for token in doc],
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'sentences': [sent.text for sent in doc.sents],
                'processing_mode': 'full'
            }
        else:
            # Degraded mode - basic text processing
            logger.warning(f"Processing text in degraded mode (no spaCy model available)")
            return {
                'tokens': text.split(),  # Basic whitespace tokenization
                'lemmas': [],  # Not available in degraded mode
                'pos_tags': [],  # Not available in degraded mode
                'entities': [],  # Not available in degraded mode
                'sentences': text.split('.'),  # Basic sentence splitting
                'processing_mode': 'degraded'
            }
    
    def is_fully_functional(self) -> bool:
        """Check if processor has full spaCy functionality."""
        return self.model is not None and not self.loader.is_degraded_mode()


# Example usage and testing functions
def example_usage():
    """Demonstrate the robust spaCy model loading."""
    logging.basicConfig(level=logging.INFO)
    
    # Test with a common model
    processor = SafeSpacyProcessor("en_core_web_sm")
    
    sample_text = "Apple is looking at buying U.K. startup for $1 billion. This is a test sentence."
    result = processor.process_text(sample_text)
    
    print(f"Processing mode: {result['processing_mode']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Entities: {result['entities']}")
    
    if not processor.is_fully_functional():
        logger.warning("Application running in degraded mode due to spaCy model issues")


if __name__ == "__main__":
    example_usage()