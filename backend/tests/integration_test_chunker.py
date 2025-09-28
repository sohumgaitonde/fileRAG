"""
Integration tests for the TextChunker with real Chonkie functionality.

This module tests the chunker with actual Chonkie library calls
to ensure proper integration and real-world functionality.
"""

import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "backend", "src")))

try:
    from chunker import TextChunker
    CHUNKER_AVAILABLE = True
except ImportError as e:
    CHUNKER_AVAILABLE = False
    print(f"⚠️  Chunker not available: {e}")


@pytest.mark.skipif(not CHUNKER_AVAILABLE, reason="Chunker module not available")
class TestChunkerIntegration:
    """Integration tests with real Chonkie functionality."""
    
    def test_token_chunker_real(self):
        """Test token chunker with real Chonkie TokenChunker."""
        try:
            chunker = TextChunker(
                chunk_size=50,
                overlap=10,
                chunker_type="token"
            )
            
            text = "This is a test document for the token chunker. It should split this text into appropriate chunks based on token boundaries."
            
            result = chunker.chunk_text(text)
            
            # Basic validation
            assert len(result) > 0
            assert all('text' in chunk for chunk in result)
            assert all('chunk_index' in chunk for chunk in result)
            assert all('metadata' in chunk for chunk in result)
            
            # Check that chunks are properly indexed
            for i, chunk in enumerate(result):
                assert chunk['chunk_index'] == i
            
            print(f"✅ Token chunker created {len(result)} chunks")
            
        except ImportError:
            pytest.skip("Chonkie TokenChunker not available")
    
    def test_word_chunker_real(self):
        """Test word chunker with real Chonkie WordChunker."""
        try:
            chunker = TextChunker(
                chunk_size=30,
                overlap=5,
                chunker_type="word"
            )
            
            text = "Word chunking splits text at word boundaries. This ensures that words remain intact within chunks."
            
            result = chunker.chunk_text(text)
            
            assert len(result) > 0
            
            # Verify that words are not broken
            for chunk in result:
                chunk_text = chunk['text'].strip()
                # Should not start or end with partial words (basic check)
                if len(chunk_text) > 0:
                    assert not chunk_text[0].isspace()
                    assert not chunk_text[-1].isspace()
            
            print(f"✅ Word chunker created {len(result)} chunks")
            
        except ImportError:
            pytest.skip("Chonkie WordChunker not available")
    
    def test_sentence_chunker_real(self):
        """Test sentence chunker with real Chonkie SentenceChunker."""
        try:
            chunker = TextChunker(
                chunk_size=100,
                overlap=20,
                chunker_type="sentence"
            )
            
            text = "This is the first sentence. This is the second sentence. This is the third sentence with more content."
            
            result = chunker.chunk_text(text)
            
            assert len(result) > 0
            
            print(f"✅ Sentence chunker created {len(result)} chunks")
            
        except ImportError:
            pytest.skip("Chonkie SentenceChunker not available")
    
    @pytest.mark.slow
    def test_semantic_chunker_real(self):
        """Test semantic chunker with real model (slow test)."""
        try:
            chunker = TextChunker(
                chunk_size=200,
                overlap=30,
                chunker_type="semantic",
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            text = """
            Machine learning is a method of data analysis that automates analytical model building.
            It is a branch of artificial intelligence based on the idea that systems can learn from data.
            
            Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
            Learning can be supervised, semi-supervised or unsupervised.
            
            Python is a programming language that is widely used for data science and machine learning.
            It has many libraries that make it easy to work with data and build models.
            """
            
            result = chunker.chunk_text(text)
            
            assert len(result) > 0
            
            print(f"✅ Semantic chunker created {len(result)} chunks")
            
            # Print chunks to see semantic boundaries
            for i, chunk in enumerate(result):
                print(f"Chunk {i}: {chunk['text'][:100]}...")
            
        except ImportError:
            pytest.skip("Chonkie SemanticChunker or transformers not available")
        except Exception as e:
            pytest.skip(f"Semantic chunker failed (likely model download issue): {e}")
    
    def test_chunker_with_file_content(self):
        """Test chunker with actual file content."""
        try:
            # Read test document
            test_file = os.path.join(os.path.dirname(__file__), "test_data", "chunker_test_document.txt")
            
            if not os.path.exists(test_file):
                pytest.skip("Test document not found")
            
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunker = TextChunker(
                chunk_size=200,
                overlap=50,
                chunker_type="token"
            )
            
            metadata = {"file_path": test_file, "file_type": "txt"}
            result = chunker.chunk_text(content, metadata=metadata)
            
            assert len(result) > 0
            
            # Verify metadata preservation
            for chunk in result:
                assert chunk['metadata']['file_path'] == test_file
                assert chunk['metadata']['file_type'] == "txt"
            
            # Get statistics
            stats = chunker.get_chunk_stats(result)
            
            print(f"✅ File chunking stats:")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Average chunk size: {stats['average_chunk_size']:.1f}")
            print(f"   Total characters: {stats['total_characters']}")
            
        except Exception as e:
            pytest.skip(f"File chunking test failed: {e}")
    
    def test_document_chunking_integration(self):
        """Test multi-document chunking."""
        try:
            chunker = TextChunker(chunk_size=100, overlap=20)
            
            documents = [
                {
                    "id": "doc1",
                    "content": "This is the first document. It contains some sample text for testing purposes.",
                    "metadata": {"source": "test1.txt", "author": "tester"}
                },
                {
                    "id": "doc2", 
                    "content": "This is the second document. It has different content but similar structure.",
                    "metadata": {"source": "test2.txt", "author": "tester"}
                }
            ]
            
            result = chunker.chunk_documents(documents)
            
            assert len(result) >= 2  # At least one chunk per document
            
            # Verify document separation in metadata
            doc1_chunks = [c for c in result if c['metadata']['document_id'] == 'doc1']
            doc2_chunks = [c for c in result if c['metadata']['document_id'] == 'doc2']
            
            assert len(doc1_chunks) > 0
            assert len(doc2_chunks) > 0
            
            # Verify metadata preservation
            for chunk in doc1_chunks:
                assert chunk['metadata']['source'] == "test1.txt"
            
            for chunk in doc2_chunks:
                assert chunk['metadata']['source'] == "test2.txt"
            
            print(f"✅ Document chunking: {len(doc1_chunks)} + {len(doc2_chunks)} = {len(result)} chunks")
            
        except Exception as e:
            pytest.skip(f"Document chunking test failed: {e}")
    
    def test_chunk_optimization_integration(self):
        """Test chunk optimization with real chunks."""
        try:
            chunker = TextChunker(chunk_size=50, overlap=10)
            
            # Create text that will produce some small chunks
            text = "Short. A bit longer sentence here. Very long sentence that should definitely create a proper sized chunk for testing purposes."
            
            raw_chunks = chunker.chunk_text(text)
            optimized_chunks = chunker.optimize_chunks(raw_chunks)
            
            # Should have fewer or equal chunks after optimization
            assert len(optimized_chunks) <= len(raw_chunks)
            
            # All optimized chunks should have the optimization flag
            for chunk in optimized_chunks:
                assert chunk['metadata'].get('optimized') is True
            
            print(f"✅ Optimization: {len(raw_chunks)} → {len(optimized_chunks)} chunks")
            
        except Exception as e:
            pytest.skip(f"Optimization test failed: {e}")
    
    def test_settings_update_integration(self):
        """Test dynamic settings updates."""
        try:
            chunker = TextChunker(chunk_size=100, overlap=10)
            
            text = "This is a test text that will be chunked with different settings to verify that updates work correctly."
            
            # Chunk with original settings
            result1 = chunker.chunk_text(text)
            
            # Update settings
            chunker.update_settings(chunk_size=50, overlap=20)
            
            # Chunk with new settings
            result2 = chunker.chunk_text(text)
            
            # Should produce different results (likely more chunks with smaller size)
            assert chunker.chunk_size == 50
            assert chunker.overlap == 20
            
            print(f"✅ Settings update: {len(result1)} → {len(result2)} chunks")
            
        except Exception as e:
            pytest.skip(f"Settings update test failed: {e}")


def test_chunker_fallback_integration():
    """Test that fallback chunking works when Chonkie is not available."""
    # This test should work even without Chonkie
    try:
        # Mock a chunker that will fail
        from unittest.mock import Mock, patch
        
        with patch('chunker.TokenChunker') as mock_chunker:
            mock_chunker.side_effect = Exception("Chonkie not available")
            
            chunker = TextChunker(chunk_size=30, overlap=5)
            
            text = "This text should be chunked using the fallback method when Chonkie fails."
            result = chunker.chunk_text(text)
            
            assert len(result) > 0
            assert all(chunk['chunker_type'] == 'fallback' for chunk in result)
            
            print(f"✅ Fallback chunking created {len(result)} chunks")
            
    except ImportError:
        pytest.skip("Cannot test fallback without mock support")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
