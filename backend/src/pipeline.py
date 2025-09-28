"""
INDEXING: Processing pipeline orchestration.

This module coordinates the entire indexing process:
- File discovery and parsing
- Text chunking and embedding generation  
- Database storage
- Error handling and retry logic
"""

from .crawler import FileCrawler
from .chunker import TextChunker
from .embeddings import EmbeddingGenerator


class IndexingPipeline:
    """Orchestrates the complete file indexing process."""
    
    def __init__(self):
        self.crawler = FileCrawler()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
    
    def process_directory(self, directory_path: str):
        """Process all files in a directory."""
        # TODO: Implement complete pipeline
        pass
    
    def process_file(self, file_path: str):
        """Process a single file through the pipeline."""
        # TODO: Implement single file processing
        pass
    
    def retry_failed(self):
        """Retry processing failed files."""
        # TODO: Implement retry logic
        pass
