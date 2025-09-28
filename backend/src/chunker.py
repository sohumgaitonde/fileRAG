"""
INDEXING: Text chunking using Chonkie.

This module handles:
- Text chunking with configurable strategies
- Chunk size optimization
- Metadata preservation
- Overlap management
"""

from typing import List, Dict, Optional, Union
import uuid
try:
    from chonkie import TokenChunker, SentenceChunker, SemanticChunker
    # WordChunker may not be available in all versions
    try:
        from chonkie import WordChunker
        WORD_CHUNKER_AVAILABLE = True
    except ImportError:
        WORD_CHUNKER_AVAILABLE = False
        print("⚠️  WordChunker not available in this version of Chonkie")
    
    CHONKIE_AVAILABLE = True
except ImportError:
    print("⚠️  Chonkie not available, using fallback chunking only")
    CHONKIE_AVAILABLE = False
    WORD_CHUNKER_AVAILABLE = False


class TextChunker:
    """Chunks text content using Chonkie for optimal processing."""
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        overlap: int = 50,
        chunker_type: str = "token",
        model_name: Optional[str] = None
    ):
        """
        Initialize the text chunker with Chonkie.
        
        Args:
            chunk_size: Maximum size of each chunk
            overlap: Number of characters/tokens to overlap between chunks
            chunker_type: Type of chunker ('token', 'word', 'sentence', 'semantic')
            model_name: Model name for semantic chunking (if using semantic chunker)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunker_type = chunker_type.lower()
        self.model_name = model_name
        
        # Initialize the appropriate Chonkie chunker
        self.chunker = self._initialize_chunker()
    
    def _initialize_chunker(self):
        """Initialize the appropriate Chonkie chunker based on type."""
        if not CHONKIE_AVAILABLE:
            print("⚠️  Chonkie not available, will use fallback chunking")
            return None
            
        try:
            if self.chunker_type == "token":
                return TokenChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.overlap
                )
            elif self.chunker_type == "word":
                if not WORD_CHUNKER_AVAILABLE:
                    print(f"⚠️  WordChunker not available, falling back to token chunker")
                    return TokenChunker(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.overlap
                    )
                return WordChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.overlap
                )
            elif self.chunker_type == "sentence":
                return SentenceChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.overlap
                )
            elif self.chunker_type == "semantic":
                if not self.model_name:
                    self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                return SemanticChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.overlap,
                    model_name=self.model_name
                )
            else:
                print(f"⚠️  Unknown chunker type '{self.chunker_type}', defaulting to token chunker")
                return TokenChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.overlap
                )
        except Exception as e:
            print(f"⚠️  Error initializing {self.chunker_type} chunker: {e}")
            print("🔄 Falling back to token chunker or fallback method")
            try:
                return TokenChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.overlap
                )
            except:
                return None
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text into smaller segments using Chonkie.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text, metadata, and chunk info
        """
        if not text or not text.strip():
            return []
        
        # If no Chonkie chunker available, use fallback
        if self.chunker is None:
            return self._fallback_chunk(text, metadata)
            
        try:
            # Use Chonkie to chunk the text
            chunks = self.chunker.chunk(text)
            
            # Convert Chonkie chunks to our format
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    'id': str(uuid.uuid4()),
                    'text': chunk.text if hasattr(chunk, 'text') else str(chunk),
                    'chunk_index': i,
                    'chunk_size': len(chunk.text) if hasattr(chunk, 'text') else len(str(chunk)),
                    'start_index': chunk.start_index if hasattr(chunk, 'start_index') else None,
                    'end_index': chunk.end_index if hasattr(chunk, 'end_index') else None,
                    'chunker_type': self.chunker_type,
                    'metadata': metadata.copy() if metadata else {}
                }
                
                # Add chunk-specific metadata
                chunk_dict['metadata'].update({
                    'chunk_id': chunk_dict['id'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_method': self.chunker_type
                })
                
                processed_chunks.append(chunk_dict)
            
            print(f"✅ Chunked text into {len(processed_chunks)} chunks using {self.chunker_type} chunker")
            return processed_chunks
            
        except Exception as e:
            print(f"❌ Error chunking text with {self.chunker_type} chunker: {e}")
            # Fallback to simple text splitting
            return self._fallback_chunk(text, metadata)
    
    def _fallback_chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Fallback chunking method using simple text splitting.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
            
        Returns:
            List of chunk dictionaries
        """
        print("🔄 Using fallback chunking method")
        
        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at word boundaries
            if end < text_length:
                # Look for the last space within the overlap range
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_dict = {
                    'id': str(uuid.uuid4()),
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'chunk_size': len(chunk_text),
                    'start_index': start,
                    'end_index': end,
                    'chunker_type': 'fallback',
                    'metadata': metadata.copy() if metadata else {}
                }
                
                chunk_dict['metadata'].update({
                    'chunk_id': chunk_dict['id'],
                    'chunk_index': chunk_index,
                    'chunk_method': 'fallback'
                })
                
                chunks.append(chunk_dict)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap, end)
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_index, doc in enumerate(documents):
            content = doc.get('content', '')
            doc_metadata = doc.get('metadata', {})
            
            # Add document-level metadata
            doc_metadata.update({
                'document_index': doc_index,
                'document_id': doc.get('id', str(uuid.uuid4()))
            })
            
            # Chunk the document
            doc_chunks = self.chunk_text(content, doc_metadata)
            all_chunks.extend(doc_chunks)
        
        print(f"✅ Processed {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks
    
    def optimize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Optimize chunks for better retrieval.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of optimized chunks
        """
        if not chunks:
            return chunks
        
        optimized_chunks = []
        
        for chunk in chunks:
            # Skip very small chunks (less than 10% of target size)
            min_size = max(10, self.chunk_size * 0.1)
            if chunk['chunk_size'] < min_size:
                print(f"⚠️  Skipping small chunk ({chunk['chunk_size']} chars)")
                continue
            
            # Clean up the text
            text = chunk['text'].strip()
            
            # Skip empty chunks
            if not text:
                continue
            
            # Update chunk with cleaned text
            optimized_chunk = chunk.copy()
            optimized_chunk['text'] = text
            optimized_chunk['chunk_size'] = len(text)
            
            # Add optimization metadata
            optimized_chunk['metadata']['optimized'] = True
            
            optimized_chunks.append(optimized_chunk)
        
        print(f"✅ Optimized {len(chunks)} chunks to {len(optimized_chunks)} chunks")
        return optimized_chunks
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {'total_chunks': 0}
        
        chunk_sizes = [chunk['chunk_size'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'chunker_type': self.chunker_type,
            'chunk_size_setting': self.chunk_size,
            'overlap_setting': self.overlap
        }
    
    def update_settings(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        """
        Update chunker settings and reinitialize if needed.
        
        Args:
            chunk_size: New chunk size
            overlap: New overlap size
        """
        settings_changed = False
        
        if chunk_size is not None and chunk_size != self.chunk_size:
            self.chunk_size = chunk_size
            settings_changed = True
        
        if overlap is not None and overlap != self.overlap:
            self.overlap = overlap
            settings_changed = True
        
        if settings_changed:
            print(f"🔄 Updating chunker settings: size={self.chunk_size}, overlap={self.overlap}")
            self.chunker = self._initialize_chunker()
