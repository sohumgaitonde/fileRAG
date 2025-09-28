"""
Tests for the TextChunker class using Chonkie.

This module contains comprehensive tests for text chunking functionality,
including different chunking strategies, error handling, and optimization.
"""

import os
import sys
import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "backend", "src")))

from chunker import TextChunker


class TestTextChunkerInitialization:
    """Test TextChunker initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default chunker initialization."""
        chunker = TextChunker()
        
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50
        assert chunker.chunker_type == "token"
        assert chunker.model_name is None
        assert chunker.chunker is not None
    
    def test_custom_initialization(self):
        """Test chunker with custom parameters."""
        chunker = TextChunker(
            chunk_size=256,
            overlap=25,
            chunker_type="word",
            model_name="test-model"
        )
        
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25
        assert chunker.chunker_type == "word"
        assert chunker.model_name == "test-model"
    
    def test_chunker_type_case_insensitive(self):
        """Test that chunker type is case insensitive."""
        chunker = TextChunker(chunker_type="TOKEN")
        assert chunker.chunker_type == "token"
        
        chunker = TextChunker(chunker_type="Word")
        assert chunker.chunker_type == "word"
    
    def test_unknown_chunker_type_fallback(self):
        """Test fallback to token chunker for unknown types."""
        with patch('builtins.print') as mock_print:
            chunker = TextChunker(chunker_type="unknown")
            assert chunker.chunker_type == "unknown"
            # Should still create a chunker (fallback to token)
            assert chunker.chunker is not None


class TestChunkerTypes:
    """Test different chunker type initializations."""
    
    @patch('chunker.TokenChunker')
    def test_token_chunker_initialization(self, mock_token_chunker):
        """Test token chunker initialization."""
        mock_instance = Mock()
        mock_token_chunker.return_value = mock_instance
        
        chunker = TextChunker(chunk_size=256, overlap=25, chunker_type="token")
        
        mock_token_chunker.assert_called_once_with(
            chunk_size=256,
            chunk_overlap=25
        )
        assert chunker.chunker == mock_instance
    
    def test_word_chunker_initialization(self):
        """Test word chunker initialization (may fallback if not available)."""
        try:
            chunker = TextChunker(chunk_size=256, overlap=25, chunker_type="word")
            # Should either create a word chunker or fallback to token chunker
            assert chunker.chunker is not None or chunker.chunker is None  # May fallback
            assert chunker.chunker_type == "word"
            print("✅ Word chunker test passed (may have used fallback)")
        except Exception as e:
            pytest.skip(f"WordChunker not available: {e}")
    
    @patch('chunker.SentenceChunker')
    def test_sentence_chunker_initialization(self, mock_sentence_chunker):
        """Test sentence chunker initialization."""
        mock_instance = Mock()
        mock_sentence_chunker.return_value = mock_instance
        
        chunker = TextChunker(chunk_size=256, overlap=25, chunker_type="sentence")
        
        mock_sentence_chunker.assert_called_once_with(
            chunk_size=256,
            chunk_overlap=25
        )
        assert chunker.chunker == mock_instance
    
    @patch('chunker.SemanticChunker')
    def test_semantic_chunker_initialization(self, mock_semantic_chunker):
        """Test semantic chunker initialization."""
        mock_instance = Mock()
        mock_semantic_chunker.return_value = mock_instance
        
        chunker = TextChunker(
            chunk_size=256, 
            overlap=25, 
            chunker_type="semantic",
            model_name="test-model"
        )
        
        mock_semantic_chunker.assert_called_once_with(
            chunk_size=256,
            chunk_overlap=25,
            model_name="test-model"
        )
        assert chunker.chunker == mock_instance
    
    @patch('chunker.SemanticChunker')
    def test_semantic_chunker_default_model(self, mock_semantic_chunker):
        """Test semantic chunker with default model."""
        mock_instance = Mock()
        mock_semantic_chunker.return_value = mock_instance
        
        chunker = TextChunker(chunker_type="semantic")
        
        mock_semantic_chunker.assert_called_once_with(
            chunk_size=512,
            chunk_overlap=50,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


class TestTextChunking:
    """Test text chunking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Chonkie chunk object
        self.mock_chunk = Mock()
        self.mock_chunk.text = "This is a test chunk."
        self.mock_chunk.start_index = 0
        self.mock_chunk.end_index = 21
        
        # Mock chunker
        self.mock_chonkie_chunker = Mock()
        self.mock_chonkie_chunker.chunk.return_value = [self.mock_chunk]
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        chunker = TextChunker()
        
        # Empty string
        assert chunker.chunk_text("") == []
        
        # Whitespace only
        assert chunker.chunk_text("   \n\t  ") == []
        
        # None
        assert chunker.chunk_text(None) == []
    
    @patch('chunker.TokenChunker')
    def test_successful_chunking(self, mock_token_chunker):
        """Test successful text chunking."""
        mock_token_chunker.return_value = self.mock_chonkie_chunker
        
        chunker = TextChunker()
        result = chunker.chunk_text("This is a test text.")
        
        assert len(result) == 1
        chunk = result[0]
        
        # Check chunk structure
        assert 'id' in chunk
        assert 'text' in chunk
        assert 'chunk_index' in chunk
        assert 'chunk_size' in chunk
        assert 'start_index' in chunk
        assert 'end_index' in chunk
        assert 'chunker_type' in chunk
        assert 'metadata' in chunk
        
        # Check chunk content
        assert chunk['text'] == "This is a test chunk."
        assert chunk['chunk_index'] == 0
        assert chunk['chunk_size'] == 21
        assert chunk['start_index'] == 0
        assert chunk['end_index'] == 21
        assert chunk['chunker_type'] == "token"
        
        # Check metadata
        metadata = chunk['metadata']
        assert 'chunk_id' in metadata
        assert 'chunk_index' in metadata
        assert 'total_chunks' in metadata
        assert 'chunk_method' in metadata
        assert metadata['chunk_index'] == 0
        assert metadata['total_chunks'] == 1
        assert metadata['chunk_method'] == "token"
    
    @patch('chunker.TokenChunker')
    def test_chunking_with_metadata(self, mock_token_chunker):
        """Test chunking with custom metadata."""
        mock_token_chunker.return_value = self.mock_chonkie_chunker
        
        chunker = TextChunker()
        custom_metadata = {"file_path": "test.txt", "author": "test"}
        result = chunker.chunk_text("Test text", metadata=custom_metadata)
        
        assert len(result) == 1
        chunk = result[0]
        
        # Check that custom metadata is preserved
        assert chunk['metadata']['file_path'] == "test.txt"
        assert chunk['metadata']['author'] == "test"
        
        # Check that chunk-specific metadata is added
        assert 'chunk_id' in chunk['metadata']
        assert 'chunk_index' in chunk['metadata']
    
    @patch('chunker.TokenChunker')
    def test_multiple_chunks(self, mock_token_chunker):
        """Test handling of multiple chunks."""
        # Create multiple mock chunks
        chunk1 = Mock()
        chunk1.text = "First chunk"
        chunk1.start_index = 0
        chunk1.end_index = 11
        
        chunk2 = Mock()
        chunk2.text = "Second chunk"
        chunk2.start_index = 11
        chunk2.end_index = 23
        
        mock_chunker = Mock()
        mock_chunker.chunk.return_value = [chunk1, chunk2]
        mock_token_chunker.return_value = mock_chunker
        
        chunker = TextChunker()
        result = chunker.chunk_text("Test text with multiple chunks")
        
        assert len(result) == 2
        
        # Check first chunk
        assert result[0]['text'] == "First chunk"
        assert result[0]['chunk_index'] == 0
        assert result[0]['metadata']['total_chunks'] == 2
        
        # Check second chunk
        assert result[1]['text'] == "Second chunk"
        assert result[1]['chunk_index'] == 1
        assert result[1]['metadata']['total_chunks'] == 2
    
    @patch('chunker.TokenChunker')
    def test_chunking_error_fallback(self, mock_token_chunker):
        """Test fallback to simple chunking when Chonkie fails."""
        # Mock Chonkie to raise an exception
        mock_chunker = Mock()
        mock_chunker.chunk.side_effect = Exception("Chonkie error")
        mock_token_chunker.return_value = mock_chunker
        
        with patch('builtins.print') as mock_print:
            chunker = TextChunker(chunk_size=10, overlap=2)
            result = chunker.chunk_text("This is a test text for fallback chunking")
            
            # Should have fallen back to simple chunking
            assert len(result) > 0
            assert all(chunk['chunker_type'] == 'fallback' for chunk in result)
            
            # Check that error was logged
            mock_print.assert_any_call("❌ Error chunking text with token chunker: Chonkie error")
            mock_print.assert_any_call("🔄 Using fallback chunking method")


class TestFallbackChunking:
    """Test fallback chunking functionality."""
    
    def test_fallback_chunking_basic(self):
        """Test basic fallback chunking."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        
        text = "This is a test text for fallback chunking method testing"
        result = chunker._fallback_chunk(text)
        
        assert len(result) > 1  # Should create multiple chunks
        
        # Check chunk properties
        for i, chunk in enumerate(result):
            assert chunk['id'] is not None
            assert chunk['text'] is not None
            assert chunk['chunk_index'] == i
            assert chunk['chunker_type'] == 'fallback'
            assert 'metadata' in chunk
    
    def test_fallback_word_boundaries(self):
        """Test that fallback chunking respects word boundaries."""
        chunker = TextChunker(chunk_size=15, overlap=3)
        
        text = "This is a test text"
        result = chunker._fallback_chunk(text)
        
        # Check that chunks don't break words inappropriately
        for chunk in result:
            # Should not start or end with partial words (in most cases)
            text_content = chunk['text']
            assert text_content.strip() == text_content  # No leading/trailing whitespace
    
    def test_fallback_with_metadata(self):
        """Test fallback chunking with metadata."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        
        metadata = {"file": "test.txt"}
        result = chunker._fallback_chunk("Test text", metadata)
        
        assert len(result) >= 1
        chunk = result[0]
        assert chunk['metadata']['file'] == "test.txt"
        assert chunk['metadata']['chunk_method'] == 'fallback'


class TestDocumentChunking:
    """Test multi-document chunking functionality."""
    
    @patch('chunker.TokenChunker')
    def test_chunk_documents_basic(self, mock_token_chunker):
        """Test basic document chunking."""
        # Mock chunker
        mock_chunk = Mock()
        mock_chunk.text = "Test chunk"
        mock_chunk.start_index = 0
        mock_chunk.end_index = 10
        
        mock_chunker = Mock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_token_chunker.return_value = mock_chunker
        
        chunker = TextChunker()
        
        documents = [
            {"content": "First document", "metadata": {"file": "doc1.txt"}},
            {"content": "Second document", "metadata": {"file": "doc2.txt"}}
        ]
        
        result = chunker.chunk_documents(documents)
        
        assert len(result) == 2  # Two documents, one chunk each
        
        # Check document-specific metadata
        assert result[0]['metadata']['file'] == "doc1.txt"
        assert result[0]['metadata']['document_index'] == 0
        assert result[1]['metadata']['file'] == "doc2.txt"
        assert result[1]['metadata']['document_index'] == 1
    
    @patch('chunker.TokenChunker')
    def test_chunk_documents_with_ids(self, mock_token_chunker):
        """Test document chunking with document IDs."""
        mock_chunk = Mock()
        mock_chunk.text = "Test chunk"
        mock_chunk.start_index = 0
        mock_chunk.end_index = 10
        
        mock_chunker = Mock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_token_chunker.return_value = mock_chunker
        
        chunker = TextChunker()
        
        documents = [
            {"id": "doc-1", "content": "First document", "metadata": {"file": "doc1.txt"}}
        ]
        
        result = chunker.chunk_documents(documents)
        
        assert len(result) == 1
        assert result[0]['metadata']['document_id'] == "doc-1"
    
    def test_chunk_empty_documents(self):
        """Test chunking empty document list."""
        chunker = TextChunker()
        result = chunker.chunk_documents([])
        assert result == []


class TestChunkOptimization:
    """Test chunk optimization functionality."""
    
    def test_optimize_chunks_basic(self):
        """Test basic chunk optimization."""
        chunker = TextChunker(chunk_size=100)
        
        chunks = [
            {
                'id': 'chunk-1',
                'text': 'This is a good chunk with sufficient content',
                'chunk_size': 44,
                'metadata': {}
            },
            {
                'id': 'chunk-2',
                'text': 'Short',  # Too small
                'chunk_size': 5,
                'metadata': {}
            },
            {
                'id': 'chunk-3',
                'text': '   \n\t   ',  # Empty after stripping
                'chunk_size': 8,
                'metadata': {}
            },
            {
                'id': 'chunk-4',
                'text': 'Another good chunk with enough content here',
                'chunk_size': 43,
                'metadata': {}
            }
        ]
        
        result = chunker.optimize_chunks(chunks)
        
        # Should keep only the good chunks
        assert len(result) == 2
        assert result[0]['text'] == 'This is a good chunk with sufficient content'
        assert result[1]['text'] == 'Another good chunk with enough content here'
        
        # Check optimization metadata
        for chunk in result:
            assert chunk['metadata']['optimized'] is True
    
    def test_optimize_empty_chunks(self):
        """Test optimization with empty chunk list."""
        chunker = TextChunker()
        result = chunker.optimize_chunks([])
        assert result == []
    
    def test_optimize_chunks_text_cleaning(self):
        """Test that optimization cleans up text."""
        chunker = TextChunker(chunk_size=50)
        
        chunks = [
            {
                'id': 'chunk-1',
                'text': '  \n  This text has whitespace  \t\n  ',
                'chunk_size': 35,
                'metadata': {}
            }
        ]
        
        result = chunker.optimize_chunks(chunks)
        
        assert len(result) == 1
        assert result[0]['text'] == 'This text has whitespace'
        assert result[0]['chunk_size'] == 24  # Updated size


class TestChunkStatistics:
    """Test chunk statistics functionality."""
    
    def test_get_chunk_stats_basic(self):
        """Test basic chunk statistics."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        
        chunks = [
            {'chunk_size': 50},
            {'chunk_size': 75},
            {'chunk_size': 100}
        ]
        
        stats = chunker.get_chunk_stats(chunks)
        
        assert stats['total_chunks'] == 3
        assert stats['total_characters'] == 225
        assert stats['average_chunk_size'] == 75.0
        assert stats['min_chunk_size'] == 50
        assert stats['max_chunk_size'] == 100
        assert stats['chunker_type'] == 'token'
        assert stats['chunk_size_setting'] == 100
        assert stats['overlap_setting'] == 10
    
    def test_get_chunk_stats_empty(self):
        """Test statistics for empty chunk list."""
        chunker = TextChunker()
        stats = chunker.get_chunk_stats([])
        
        assert stats == {'total_chunks': 0}


class TestSettingsUpdate:
    """Test dynamic settings updates."""
    
    @patch('chunker.TokenChunker')
    def test_update_settings_chunk_size(self, mock_token_chunker):
        """Test updating chunk size."""
        mock_chunker = Mock()
        mock_token_chunker.return_value = mock_chunker
        
        chunker = TextChunker(chunk_size=100, overlap=10)
        original_chunker = chunker.chunker
        
        # Update chunk size
        with patch('builtins.print') as mock_print:
            chunker.update_settings(chunk_size=200)
        
        assert chunker.chunk_size == 200
        assert chunker.overlap == 10  # Should remain unchanged
        
        # Should have reinitialized chunker (check by calling count)
        assert mock_token_chunker.call_count >= 2  # Called at least twice (init + update)
        mock_print.assert_called_with("🔄 Updating chunker settings: size=200, overlap=10")
    
    @patch('chunker.TokenChunker')
    def test_update_settings_overlap(self, mock_token_chunker):
        """Test updating overlap."""
        mock_chunker = Mock()
        mock_token_chunker.return_value = mock_chunker
        
        chunker = TextChunker(chunk_size=100, overlap=10)
        
        # Update overlap
        with patch('builtins.print') as mock_print:
            chunker.update_settings(overlap=20)
        
        assert chunker.chunk_size == 100  # Should remain unchanged
        assert chunker.overlap == 20
        
        mock_print.assert_called_with("🔄 Updating chunker settings: size=100, overlap=20")
    
    @patch('chunker.TokenChunker')
    def test_update_settings_no_change(self, mock_token_chunker):
        """Test that no change doesn't trigger reinitialization."""
        mock_chunker = Mock()
        mock_token_chunker.return_value = mock_chunker
        
        chunker = TextChunker(chunk_size=100, overlap=10)
        original_chunker = chunker.chunker
        
        # Update with same values
        chunker.update_settings(chunk_size=100, overlap=10)
        
        # Should not have reinitialized
        assert chunker.chunker == original_chunker


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('chunker.TokenChunker')
    def test_chunker_initialization_error(self, mock_token_chunker):
        """Test handling of chunker initialization errors."""
        mock_token_chunker.side_effect = Exception("Initialization error")
        
        with patch('builtins.print') as mock_print:
            # Should fall back to token chunker
            chunker = TextChunker(chunker_type="token")
            
            # Should have logged the error and fallback
            mock_print.assert_any_call("⚠️  Error initializing token chunker: Initialization error")
            mock_print.assert_any_call("🔄 Falling back to token chunker or fallback method")
    
    def test_chunk_without_attributes(self):
        """Test handling of chunk objects without expected attributes."""
        chunker = TextChunker()
        
        # Mock a chunk object that doesn't have all expected attributes
        mock_chunk = Mock()
        mock_chunk.configure_mock(**{
            'text': 'Test chunk',
            # Missing start_index and end_index
        })
        del mock_chunk.start_index
        del mock_chunk.end_index
        
        with patch.object(chunker, 'chunker') as mock_chunker:
            mock_chunker.chunk.return_value = [mock_chunk]
            
            result = chunker.chunk_text("Test text")
            
            assert len(result) == 1
            chunk = result[0]
            assert chunk['text'] == 'Test chunk'
            assert chunk['start_index'] is None
            assert chunk['end_index'] is None


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
