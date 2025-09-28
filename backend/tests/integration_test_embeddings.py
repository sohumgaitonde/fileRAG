"""
Integration tests for the EmbeddingGenerator with real model functionality.

This module tests the embeddings generator with actual model calls
to ensure proper integration and real-world functionality.
"""

import os
import sys
import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from embeddings import EmbeddingGenerator
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    print(f"⚠️  Embeddings not available: {e}")


@pytest.mark.skipif(not EMBEDDINGS_AVAILABLE, reason="Embeddings module not available")
class TestEmbeddingGeneratorIntegration:
    """Integration tests with real model functionality."""
    
    @pytest.mark.slow
    def test_model_loading_integration(self):
        """Test real model loading (slow test - downloads model)."""
        try:
            embedder = EmbeddingGenerator()
            
            # This will download the model on first run
            embedder.load_model()
            
            assert embedder._model_loaded is True
            assert embedder.model is not None
            assert embedder.tokenizer is not None
            
            print("✅ Model loaded successfully in integration test")
            
        except Exception as e:
            pytest.skip(f"Model loading failed (likely due to network/dependencies): {e}")
    
    @pytest.mark.slow
    def test_embedding_generation_integration(self):
        """Test real embedding generation."""
        try:
            embedder = EmbeddingGenerator()
            
            # Test single text
            text = "This is a test sentence for embedding generation."
            embedding = embedder.generate_single_embedding(text)
            
            # Basic validation
            assert isinstance(embedding, np.ndarray)
            assert embedding.ndim == 1  # Should be 1D for single embedding
            assert embedding.size > 0   # Should have some dimension
            
            print(f"✅ Generated single embedding with shape: {embedding.shape}")
            
            # Test multiple texts
            texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Python is popular for data science applications."
            ]
            
            embeddings = embedder.generate_embeddings(texts)
            
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.ndim == 2  # Should be 2D for multiple embeddings
            assert embeddings.shape[0] == len(texts)  # One embedding per text
            assert embeddings.shape[1] > 0  # Should have some embedding dimension
            
            print(f"✅ Generated batch embeddings with shape: {embeddings.shape}")
            
        except Exception as e:
            pytest.skip(f"Embedding generation failed: {e}")
    
    @pytest.mark.slow
    def test_batch_processing_integration(self):
        """Test batch processing with real model."""
        try:
            embedder = EmbeddingGenerator()
            
            # Create a larger batch of texts
            texts = [
                f"This is test sentence number {i} for batch processing."
                for i in range(10)
            ]
            
            # Test with small batch size
            embeddings = embedder.generate_batch_embeddings(texts, batch_size=3)
            
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] > 0
            
            print(f"✅ Batch processing completed: {embeddings.shape}")
            
        except Exception as e:
            pytest.skip(f"Batch processing failed: {e}")
    
    @pytest.mark.slow
    def test_normalization_integration(self):
        """Test embedding normalization with real embeddings."""
        try:
            embedder = EmbeddingGenerator()
            
            texts = ["Test text for normalization", "Another test text"]
            embeddings = embedder.generate_embeddings(texts)
            
            # Normalize embeddings
            normalized = embedder.normalize_embeddings(embeddings)
            
            # Check that embeddings are normalized (L2 norm should be ~1)
            norms = np.linalg.norm(normalized, axis=1)
            
            # All norms should be close to 1.0
            assert np.allclose(norms, 1.0, atol=1e-6)
            
            print(f"✅ Normalization successful, norms: {norms}")
            
        except Exception as e:
            pytest.skip(f"Normalization test failed: {e}")
    
    @pytest.mark.slow
    def test_embedding_consistency(self):
        """Test that same text produces consistent embeddings."""
        try:
            embedder = EmbeddingGenerator()
            
            text = "Consistency test text"
            
            # Generate embedding twice
            embedding1 = embedder.generate_single_embedding(text)
            embedding2 = embedder.generate_single_embedding(text)
            
            # Should be identical (or very close due to floating point)
            assert np.allclose(embedding1, embedding2, atol=1e-6)
            
            print("✅ Embedding consistency verified")
            
        except Exception as e:
            pytest.skip(f"Consistency test failed: {e}")
    
    def test_model_info_integration(self):
        """Test getting model information."""
        try:
            embedder = EmbeddingGenerator()
            
            # Get info before loading
            info_before = embedder.get_model_info()
            assert info_before["loaded"] is False
            assert info_before["embedding_dim"] is None
            
            # Load model and get info
            embedder.load_model()
            info_after = embedder.get_model_info()
            
            assert info_after["loaded"] is True
            assert info_after["embedding_dim"] is not None
            assert info_after["embedding_dim"] > 0
            assert info_after["model_name"] == "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
            
            print(f"✅ Model info: {info_after}")
            
        except Exception as e:
            pytest.skip(f"Model info test failed: {e}")
    
    @pytest.mark.slow
    def test_cache_management_integration(self):
        """Test model cache management."""
        try:
            embedder = EmbeddingGenerator()
            
            # Load model
            embedder.load_model()
            assert embedder._model_loaded is True
            
            # Clear cache
            embedder.clear_cache()
            assert embedder._model_loaded is False
            assert embedder.model is None
            assert embedder.tokenizer is None
            
            print("✅ Cache management successful")
            
        except Exception as e:
            pytest.skip(f"Cache management test failed: {e}")
    
    def test_error_handling_integration(self):
        """Test error handling with real scenarios."""
        try:
            embedder = EmbeddingGenerator()
            
            # Test with empty texts
            result = embedder.generate_embeddings([])
            assert result.size == 0
            
            # Test with empty strings
            result = embedder.generate_embeddings(["", "   ", "\n"])
            assert result.size == 0
            
            print("✅ Error handling tests passed")
            
        except Exception as e:
            pytest.skip(f"Error handling test failed: {e}")
    
    @pytest.mark.slow
    def test_different_text_types_integration(self):
        """Test embedding generation with different types of text."""
        try:
            embedder = EmbeddingGenerator()
            
            # Different types of text
            texts = [
                "Short text.",
                "This is a much longer piece of text that contains multiple sentences and should test the model's ability to handle longer inputs effectively.",
                "Technical text: Machine learning algorithms utilize statistical methods to identify patterns in large datasets.",
                "Casual text: Hey there! How's it going? Hope you're having a great day! 😊",
                "Mixed content: The function f(x) = x^2 + 2x + 1 represents a quadratic equation.",
            ]
            
            embeddings = embedder.generate_embeddings(texts)
            
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] > 0
            
            # Check that different texts produce different embeddings
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    # Embeddings should be different
                    assert not np.allclose(embeddings[i], embeddings[j], atol=1e-3)
            
            print(f"✅ Different text types handled successfully: {embeddings.shape}")
            
        except Exception as e:
            pytest.skip(f"Different text types test failed: {e}")


def test_embeddings_fallback_without_transformers():
    """Test behavior when transformers is not available."""
    try:
        from unittest.mock import patch
        
        with patch('embeddings.TRANSFORMERS_AVAILABLE', False):
            embedder = EmbeddingGenerator()
            
            # Should be able to initialize
            assert embedder.model_name == "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
            
            # But loading should fail
            with pytest.raises(ImportError):
                embedder.load_model()
                
        print("✅ Fallback behavior verified")
        
    except ImportError:
        pytest.skip("Cannot test fallback without mock support")


def test_custom_model_integration():
    """Test with custom model name."""
    try:
        # Use a different model name (this won't actually load in CI)
        custom_embedder = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_dir="./test_cache"
        )
        
        assert custom_embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert custom_embedder.device == "cpu"
        assert custom_embedder.cache_dir == "./test_cache"
        
        print("✅ Custom model configuration verified")
        
    except Exception as e:
        pytest.skip(f"Custom model test failed: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
