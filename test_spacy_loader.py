import unittest
from unittest.mock import patch, MagicMock
import logging
from spacy_loader import SpacyModelLoader, SafeSpacyProcessor

class TestSpacyModelLoader(unittest.TestCase):
    
    def setUp(self):
        self.loader = SpacyModelLoader(max_retries=1, retry_delay=0.1)
        
    def test_successful_model_load(self):
        """Test successful model loading"""
        with patch('spacy.load') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            result = self.loader.load_model('en_core_web_sm')
            
            self.assertIsNotNone(result)
            self.assertEqual(result, mock_model)
            self.assertFalse(self.loader.is_degraded_mode())
    
    def test_model_load_with_download(self):
        """Test model loading with automatic download"""
        with patch('spacy.load') as mock_load, \
             patch('spacy.cli.download') as mock_download:
            
            # First call fails, second call succeeds after download
            mock_load.side_effect = [OSError("Model not found"), MagicMock()]
            
            result = self.loader.load_model('en_core_web_sm')
            
            mock_download.assert_called_once_with('en_core_web_sm')
            self.assertIsNotNone(result)
            self.assertFalse(self.loader.is_degraded_mode())
    
    def test_model_load_failure_degraded_mode(self):
        """Test graceful fallback to degraded mode"""
        with patch('spacy.load') as mock_load, \
             patch('spacy.cli.download') as mock_download:
            
            # Model load always fails
            mock_load.side_effect = OSError("Model not found")
            # Download also fails
            mock_download.side_effect = Exception("Download failed")
            
            result = self.loader.load_model('en_core_web_sm')
            
            self.assertIsNone(result)
            self.assertTrue(self.loader.is_degraded_mode())
    
    def test_cached_model_loading(self):
        """Test that models are cached after first load"""
        with patch('spacy.load') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            # Load same model twice
            result1 = self.loader.load_model('en_core_web_sm')
            result2 = self.loader.load_model('en_core_web_sm')
            
            # spacy.load should only be called once
            mock_load.assert_called_once()
            self.assertEqual(result1, result2)


class TestSafeSpacyProcessor(unittest.TestCase):
    
    def test_full_functionality_mode(self):
        """Test processor with full spaCy functionality"""
        with patch('spacy.load') as mock_load:
            # Create mock spaCy doc
            mock_doc = MagicMock()
            mock_token = MagicMock()
            mock_token.text = "test"
            mock_token.lemma_ = "test"
            mock_token.pos_ = "NOUN"
            mock_doc.__iter__ = MagicMock(return_value=iter([mock_token]))
            mock_doc.ents = []
            mock_doc.sents = [MagicMock(text="Test sentence.")]
            
            mock_model = MagicMock()
            mock_model.return_value = mock_doc
            mock_load.return_value = mock_model
            
            processor = SafeSpacyProcessor()
            result = processor.process_text("Test sentence.")
            
            self.assertEqual(result['processing_mode'], 'full')
            self.assertTrue(processor.is_fully_functional())
    
    def test_degraded_mode(self):
        """Test processor in degraded mode"""
        with patch('spacy.load') as mock_load, \
             patch('spacy.cli.download') as mock_download:
            
            # Model loading and download both fail
            mock_load.side_effect = OSError("Model not found")
            mock_download.side_effect = Exception("Download failed")
            
            processor = SafeSpacyProcessor()
            result = processor.process_text("Test sentence. Another sentence.")
            
            self.assertEqual(result['processing_mode'], 'degraded')
            self.assertEqual(result['tokens'], ['Test', 'sentence.', 'Another', 'sentence.'])
            self.assertEqual(result['sentences'], ['Test sentence', ' Another sentence', ''])
            self.assertFalse(processor.is_fully_functional())
    
    def test_no_system_exit_on_failure(self):
        """Test that SystemExit is never raised"""
        with patch('spacy.load') as mock_load, \
             patch('spacy.cli.download') as mock_download:
            
            # All operations fail
            mock_load.side_effect = Exception("Critical error")
            mock_download.side_effect = Exception("Download failed")
            
            # This should not raise SystemExit
            try:
                processor = SafeSpacyProcessor()
                result = processor.process_text("Test")
                # Should still get a result in degraded mode
                self.assertIsNotNone(result)
                self.assertEqual(result['processing_mode'], 'degraded')
            except SystemExit:
                self.fail("SystemExit should not be raised")


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    unittest.main()