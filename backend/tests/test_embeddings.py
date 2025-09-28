"""
Tests for the EmbeddingGenerator class.

This module contains comprehensive tests for embedding generation functionality,
including model loading, batch processing, normalization, and error handling.
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "src")))

from embeddings import EmbeddingGenerator


class TestEmbeddingGeneratorInitialization:
    """Test EmbeddingGenerator initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default embedding generator initialization."""
        with patch('embeddings.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            
            embedder = EmbeddingGenerator()
            
            assert embedder.model_name == "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
            assert embedder.device == "cpu"
            assert embedder.cache_dir is None
            assert embedder.model is None
            assert embedder.tokenizer is None
            assert embedder._model_loaded is False
    
    def test_custom_initialization(self):
        """Test embedding generator with custom parameters."""
        with patch('embeddings.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            
            embedder = EmbeddingGenerator(
                model_name="custom/model",
                device="cuda",
                cache_dir="./custom_cache"
            )
            
            assert embedder.model_name == "custom/model"
            assert embedder.device == "cuda"
            assert embedder.cache_dir == "./custom_cache"
    
    def test_auto_device_selection_cuda(self):
        """Test automatic CUDA device selection when available."""
        with patch('embeddings.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            
            embedder = EmbeddingGenerator()
            
            assert embedder.device == "cuda"
    
    def test_auto_device_selection_cpu(self):
        """Test automatic CPU device selection when CUDA unavailable."""
        with patch('embeddings.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            
            embedder = EmbeddingGenerator()
            
            assert embedder.device == "cpu"


class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('embeddings.TRANSFORMERS_AVAILABLE', True)
    @patch('embeddings.AutoModel')
    @patch('embeddings.AutoTokenizer')
    def test_successful_model_loading(self, mock_tokenizer, mock_model):
        """Test successful model loading."""
        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('builtins.print') as mock_print:
            embedder = EmbeddingGenerator()
            embedder.load_model()
            
            # Check that model and tokenizer were loaded
            mock_tokenizer.from_pretrained.assert_called_once_with(
                "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                trust_remote_code=True,
                cache_dir=None
            )
            
            mock_model.from_pretrained.assert_called_once_with(
                "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=None
            )
            
            assert embedder.tokenizer == mock_tokenizer_instance
            assert embedder.model == mock_model_instance
            assert embedder._model_loaded is True
            
            # Check print statements
            mock_print.assert_any_call("📥 Loading model: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
            mock_print.assert_any_call("✅ Model loaded successfully!")
    
    @patch('embeddings.TRANSFORMERS_AVAILABLE', True)
    @patch('embeddings.AutoModel')
    @patch('embeddings.AutoTokenizer')
    def test_model_loading_with_custom_cache(self, mock_tokenizer, mock_model):
        """Test model loading with custom cache directory."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        embedder = EmbeddingGenerator(cache_dir="./custom_cache")
        embedder.load_model()
        
        # Check that cache_dir was passed
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            cache_dir="./custom_cache"
        )
        
        mock_model.from_pretrained.assert_called_once_with(
            "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="./custom_cache"
        )
    
    @patch('embeddings.TRANSFORMERS_AVAILABLE', False)
    def test_model_loading_without_transformers(self):
        """Test model loading failure when transformers not available."""
        embedder = EmbeddingGenerator()
        
        with pytest.raises(ImportError, match="Transformers library not available"):
            embedder.load_model()
    
    @patch('embeddings.TRANSFORMERS_AVAILABLE', True)
    @patch('embeddings.AutoTokenizer')
    def test_model_loading_error_handling(self, mock_tokenizer):
        """Test error handling during model loading."""
        mock_tokenizer.from_pretrained.side_effect = Exception("Loading failed")
        
        embedder = EmbeddingGenerator()
        
        with pytest.raises(Exception, match="Loading failed"):
            embedder.load_model()
    
    @patch('embeddings.TRANSFORMERS_AVAILABLE', True)
    @patch('embeddings.AutoModel')
    @patch('embeddings.AutoTokenizer')
    def test_model_already_loaded(self, mock_tokenizer, mock_model):
        """Test that model loading is skipped if already loaded."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        embedder = EmbeddingGenerator()
        embedder.load_model()  # First load
        
        # Reset mocks
        mock_tokenizer.reset_mock()
        mock_model.reset_mock()
        
        with patch('builtins.print') as mock_print:
            embedder.load_model()  # Second load
            
            # Should not call from_pretrained again
            mock_tokenizer.from_pretrained.assert_not_called()
            mock_model.from_pretrained.assert_not_called()
            
            mock_print.assert_called_with("✅ Model already loaded")


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock embeddings
        self.mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.get_text_embeddings.return_value = self.mock_embeddings
        
        # Create mock tokenizer
        self.mock_tokenizer = Mock()
    
    def test_generate_embeddings_single_text(self):
        """Test generating embeddings for a single text."""
        with patch('embeddings.torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()
            
            embedder = EmbeddingGenerator()
            embedder.model = self.mock_model
            embedder.tokenizer = self.mock_tokenizer
            embedder._model_loaded = True
            
            result = embedder.generate_embeddings("test text")
            
            # Check that model was called with correct input
            self.mock_model.get_text_embeddings.assert_called_once_with(["test text"])
            
            # Check result
            np.testing.assert_array_equal(result, self.mock_embeddings)
    
    def test_generate_embeddings_list_of_texts(self):
        """Test generating embeddings for multiple texts."""
        with patch('embeddings.torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()
            
            embedder = EmbeddingGenerator()
            embedder.model = self.mock_model
            embedder.tokenizer = self.mock_tokenizer
            embedder._model_loaded = True
            
            texts = ["text 1", "text 2"]
            result = embedder.generate_embeddings(texts)
            
            self.mock_model.get_text_embeddings.assert_called_once_with(texts)
            np.testing.assert_array_equal(result, self.mock_embeddings)
    
    def test_generate_embeddings_empty_input(self):
        """Test generating embeddings for empty input."""
        embedder = EmbeddingGenerator()
        embedder._model_loaded = True
        
        result = embedder.generate_embeddings([])
        
        assert result.size == 0
        assert isinstance(result, np.ndarray)
    
    def test_generate_embeddings_empty_strings(self):
        """Test generating embeddings for empty strings."""
        embedder = EmbeddingGenerator()
        embedder._model_loaded = True
        
        with patch('builtins.print') as mock_print:
            result = embedder.generate_embeddings(["", "   ", "\n\t"])
            
            assert result.size == 0
            mock_print.assert_called_with("⚠️  No valid texts to process")
    
    def test_generate_embeddings_torch_tensor_conversion(self):
        """Test conversion of torch tensor to numpy array."""
        with patch('embeddings.torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()
            
            # Create a mock torch tensor
            mock_tensor = Mock()
            mock_tensor.cpu.return_value.numpy.return_value = self.mock_embeddings
            
            embedder = EmbeddingGenerator()
            embedder.model = Mock()
            embedder.model.get_text_embeddings.return_value = mock_tensor
            embedder.tokenizer = self.mock_tokenizer
            embedder._model_loaded = True
            
            result = embedder.generate_embeddings("test")
            
            # Check that tensor was converted to numpy
            mock_tensor.cpu.assert_called_once()
            mock_tensor.cpu.return_value.numpy.assert_called_once()
            np.testing.assert_array_equal(result, self.mock_embeddings)
    
    def test_generate_embeddings_auto_load_model(self):
        """Test that model is automatically loaded if not loaded."""
        with patch('embeddings.torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()
            
            embedder = EmbeddingGenerator()
            embedder._model_loaded = False
            
            # Mock the load_model method
            with patch.object(embedder, 'load_model') as mock_load:
                # Set up the mock to simulate successful loading
                def mock_load_side_effect():
                    embedder.model = self.mock_model
                    embedder.tokenizer = self.mock_tokenizer
                    embedder._model_loaded = True
                
                mock_load.side_effect = mock_load_side_effect
                
                embedder.generate_embeddings("test")
                
                mock_load.assert_called_once()
    
    def test_generate_embeddings_error_handling(self):
        """Test error handling during embedding generation."""
        with patch('embeddings.torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()
            
            embedder = EmbeddingGenerator()
            embedder.model = Mock()
            embedder.model.get_text_embeddings.side_effect = Exception("Generation failed")
            embedder._model_loaded = True
            
            # Should raise the exception from the model
            with pytest.raises(Exception):
                embedder.generate_embeddings("test")


class TestEmbeddingUtilities:
    """Test utility functions for embeddings."""
    
    def test_generate_single_embedding(self):
        """Test generating a single embedding."""
        embedder = EmbeddingGenerator()
        
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        with patch.object(embedder, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = mock_embeddings
            
            result = embedder.generate_single_embedding("test")
            
            mock_generate.assert_called_once_with(["test"])
            np.testing.assert_array_equal(result, mock_embeddings[0])
    
    def test_generate_single_embedding_empty_result(self):
        """Test generating single embedding with empty result."""
        embedder = EmbeddingGenerator()
        
        with patch.object(embedder, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = np.array([])
            
            result = embedder.generate_single_embedding("test")
            
            assert result.size == 0
    
    @patch('builtins.print')
    def test_generate_batch_embeddings(self, mock_print):
        """Test batch embedding generation."""
        embedder = EmbeddingGenerator()
        
        # Mock embeddings for each batch
        batch1_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        batch2_embeddings = np.array([[0.5, 0.6]])
        
        with patch.object(embedder, 'generate_embeddings') as mock_generate:
            mock_generate.side_effect = [batch1_embeddings, batch2_embeddings]
            
            texts = ["text1", "text2", "text3"]
            result = embedder.generate_batch_embeddings(texts, batch_size=2)
            
            # Should call generate_embeddings twice
            assert mock_generate.call_count == 2
            mock_generate.assert_any_call(["text1", "text2"])
            mock_generate.assert_any_call(["text3"])
            
            # Check result shape
            expected = np.vstack([batch1_embeddings, batch2_embeddings])
            np.testing.assert_array_equal(result, expected)
            
            # Check progress messages
            mock_print.assert_any_call("📦 Processing batch 1/2 (2 texts)")
            mock_print.assert_any_call("📦 Processing batch 2/2 (1 texts)")
            mock_print.assert_any_call("✅ Generated 3 total embeddings")
    
    def test_generate_batch_embeddings_empty_input(self):
        """Test batch embedding generation with empty input."""
        embedder = EmbeddingGenerator()
        
        result = embedder.generate_batch_embeddings([])
        
        assert result.size == 0
        assert isinstance(result, np.ndarray)
    
    @patch('builtins.print')
    def test_generate_batch_embeddings_no_progress(self, mock_print):
        """Test batch embedding generation without progress display."""
        embedder = EmbeddingGenerator()
        
        with patch.object(embedder, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = np.array([[0.1, 0.2]])
            
            embedder.generate_batch_embeddings(["text"], show_progress=False)
            
            # Should not show batch progress
            assert not any("📦 Processing batch" in str(call) for call in mock_print.call_args_list)


class TestEmbeddingNormalization:
    """Test embedding normalization functionality."""
    
    def test_normalize_embeddings_basic(self):
        """Test basic L2 normalization."""
        embedder = EmbeddingGenerator()
        
        # Create test embeddings
        embeddings = np.array([[3.0, 4.0], [1.0, 0.0]])
        
        with patch('builtins.print') as mock_print:
            result = embedder.normalize_embeddings(embeddings)
            
            # Check L2 normalization
            expected = np.array([[0.6, 0.8], [1.0, 0.0]])
            np.testing.assert_array_almost_equal(result, expected)
            
            mock_print.assert_called_with("✅ Normalized 2 embeddings")
    
    def test_normalize_embeddings_zero_vector(self):
        """Test normalization with zero vectors."""
        embedder = EmbeddingGenerator()
        
        # Include a zero vector
        embeddings = np.array([[0.0, 0.0], [3.0, 4.0]])
        
        result = embedder.normalize_embeddings(embeddings)
        
        # Zero vector should remain zero
        expected = np.array([[0.0, 0.0], [0.6, 0.8]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_normalize_embeddings_empty_input(self):
        """Test normalization with empty input."""
        embedder = EmbeddingGenerator()
        
        empty_embeddings = np.array([])
        result = embedder.normalize_embeddings(empty_embeddings)
        
        assert result.size == 0
        np.testing.assert_array_equal(result, empty_embeddings)


class TestModelInfo:
    """Test model information and utility functions."""
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        embedder = EmbeddingGenerator()
        
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        
        with patch.object(embedder, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = mock_embeddings
            embedder._model_loaded = True
            
            dimension = embedder.get_embedding_dimension()
            
            assert dimension == 3
            mock_generate.assert_called_once_with(["test"])
    
    def test_get_embedding_dimension_auto_load(self):
        """Test that get_embedding_dimension loads model if needed."""
        embedder = EmbeddingGenerator()
        embedder._model_loaded = False
        
        with patch.object(embedder, 'load_model') as mock_load:
            with patch.object(embedder, 'generate_embeddings') as mock_generate:
                mock_generate.return_value = np.array([[0.1, 0.2]])
                
                dimension = embedder.get_embedding_dimension()
                
                mock_load.assert_called_once()
                assert dimension == 2
    
    def test_get_model_info_loaded(self):
        """Test getting model info when model is loaded."""
        embedder = EmbeddingGenerator(
            model_name="test/model",
            device="cuda",
            cache_dir="./cache"
        )
        embedder._model_loaded = True
        
        with patch.object(embedder, 'get_embedding_dimension') as mock_dim:
            mock_dim.return_value = 768
            
            info = embedder.get_model_info()
            
            expected = {
                "model_name": "test/model",
                "device": "cuda",
                "loaded": True,
                "embedding_dim": 768,
                "cache_dir": "./cache"
            }
            
            assert info == expected
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model is not loaded."""
        embedder = EmbeddingGenerator()
        embedder._model_loaded = False
        
        info = embedder.get_model_info()
        
        assert info["loaded"] is False
        assert info["embedding_dim"] is None


class TestCacheManagement:
    """Test model cache management."""
    
    @patch('embeddings.torch')
    def test_clear_cache_with_model(self, mock_torch):
        """Test clearing cache when model is loaded."""
        mock_torch.cuda.is_available.return_value = True
        
        embedder = EmbeddingGenerator()
        embedder.model = Mock()
        embedder.tokenizer = Mock()
        embedder._model_loaded = True
        
        with patch('builtins.print') as mock_print:
            embedder.clear_cache()
            
            assert embedder.model is None
            assert embedder.tokenizer is None
            assert embedder._model_loaded is False
            
            # Should clear CUDA cache
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_print.assert_called_with("🗑️  Model cache cleared")
    
    @patch('embeddings.torch')
    def test_clear_cache_without_model(self, mock_torch):
        """Test clearing cache when no model is loaded."""
        mock_torch.cuda.is_available.return_value = False
        
        embedder = EmbeddingGenerator()
        embedder.model = None
        embedder.tokenizer = None
        embedder._model_loaded = False
        
        with patch('builtins.print') as mock_print:
            embedder.clear_cache()
            
            # Should not crash and should not call CUDA cache clear
            mock_torch.cuda.empty_cache.assert_not_called()
            mock_print.assert_called_with("🗑️  Model cache cleared")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_transformers_not_available_initialization(self):
        """Test initialization when transformers not available."""
        with patch('embeddings.TRANSFORMERS_AVAILABLE', False):
            with patch('builtins.print') as mock_print:
                embedder = EmbeddingGenerator()
                
                # Should still initialize but print warning during import
                assert embedder.model_name == "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        embedder = EmbeddingGenerator()
        embedder._model_loaded = True
        
        # Test with None
        result = embedder.generate_embeddings(None)
        assert result.size == 0
        
        # Test with non-string items in list
        with patch.object(embedder, 'model') as mock_model:
            mock_model.get_text_embeddings.return_value = np.array([[0.1, 0.2]])
            
            with patch('embeddings.torch.no_grad') as mock_no_grad:
                mock_no_grad.return_value.__enter__ = Mock()
                mock_no_grad.return_value.__exit__ = Mock()
                
                # Should filter out None values
                result = embedder.generate_embeddings([None, "valid text", ""])
                
                # Should only process valid text
                mock_model.get_text_embeddings.assert_called_with(["valid text"])


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
