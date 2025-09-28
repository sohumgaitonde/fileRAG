"""
INDEXING: Processing pipeline orchestration.

This module coordinates the entire indexing process:
- File discovery and parsing
- Text chunking and embedding generation  
- Database storage
- Error handling and retry logic
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .crawler import FileCrawler
from .chunker import TextChunker
from .embeddings import EmbeddingGenerator
from .db import VectorDatabase
from .parsers import PDFParser, DOCXParser, TXTParser, MDParser, ImageParser


@dataclass
class ProcessingResult:
    """Result of processing a file."""
    file_path: str
    success: bool
    chunks_created: int
    chunks_stored: int
    error_message: Optional[str] = None
    processing_time: float = 0.0


class IndexingPipeline:
    """Orchestrates the complete file indexing process."""
    
    def __init__(
        self, 
        db_path: str = "./chroma_db",
        collection_name: str = "fileRAG",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_file_size_mb: int = 50
    ):
        """
        Initialize the indexing pipeline.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            max_file_size_mb: Maximum file size to process
        """
        self.crawler = FileCrawler(max_file_size_mb=max_file_size_mb)
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = EmbeddingGenerator()
        self.database = VectorDatabase(db_path=db_path, collection_name=collection_name)
        
        # Initialize parsers
        self.parsers = {
            '.pdf': PDFParser(),
            '.docx': DOCXParser(),
            '.doc': DOCXParser(),  
            '.txt': TXTParser(),
            '.md': MDParser(),
            '.png': ImageParser(),
            '.jpg': ImageParser(),
            '.jpeg': ImageParser(),
            '.gif': ImageParser(),
            '.bmp': ImageParser(),
            '.tiff': ImageParser()
        }
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_chunks': 0,
            'total_processing_time': 0.0
        }
        
        # Failed files for retry
        self.failed_files: List[str] = []
    
    def initialize(self) -> bool:
        """Initialize all pipeline components."""
        print("🚀 Initializing indexing pipeline...")
        
        try:
            # Initialize database
            if not self.database.initialize():
                print("❌ Failed to initialize database")
                return False
            
            # Load embedding model
            print("📚 Loading embedding model...")
            self.embedder.load_model()
            
            print("✅ Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {str(e)}")
            return False
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[ProcessingResult]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            
        Returns:
            List of processing results
        """
        print(f"📁 Processing directory: {directory_path}")
        print("=" * 60)
        
        # Discover files
        try:
            files = self.crawler.crawl_directory(directory_path, recursive=recursive)
        except Exception as e:
            print(f"❌ Failed to crawl directory: {str(e)}")
            return []
        
        if not files:
            print("📭 No supported files found")
            return []
        
        print(f"📄 Found {len(files)} files to process")
        
        # Process each file
        results = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {Path(file_path).name}")
            result = self.process_file(file_path)
            results.append(result)
            
            # Update statistics
            self.stats['files_processed'] += 1
            if result.success:
                self.stats['files_successful'] += 1
                self.stats['total_chunks'] += result.chunks_stored
            else:
                self.stats['files_failed'] += 1
                self.failed_files.append(file_path)
            
            self.stats['total_processing_time'] += result.processing_time
        
        # Print summary
        self._print_summary()
        return results
    
    def process_file(self, file_path: str) -> ProcessingResult:
        """
        Process a single file through the pipeline.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            ProcessingResult with details about the processing
        """
        start_time = time.time()
        
        try:
            # Get file extension
            file_ext = Path(file_path).suffix.lower()
            
            # Get appropriate parser
            parser = self.parsers.get(file_ext)
            if not parser:
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    chunks_created=0,
                    chunks_stored=0,
                    error_message=f"Unsupported file type: {file_ext}",
                    processing_time=time.time() - start_time
                )
            
            # Parse file
            print(f"   📖 Parsing {file_ext} file...")
            parse_result = parser.parse(file_path)
            
            if not parse_result.get('content'):
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    chunks_created=0,
                    chunks_stored=0,
                    error_message="No content extracted from file",
                    processing_time=time.time() - start_time
                )
            
            # Print parsing results
            self._print_parse_results(file_ext, parse_result)
            
            # Chunk text
            print(f"   ✂️  Chunking text...")
            chunk_dicts = self.chunker.chunk_text(
                text=parse_result['content'],
                metadata=parse_result.get('metadata', {})
            )
            
            if not chunk_dicts:
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    chunks_created=0,
                    chunks_stored=0,
                    error_message="No chunks created from text",
                    processing_time=time.time() - start_time
                )
            
            # Extract text strings from chunk dictionaries for embedding generation
            chunks = [chunk_dict['text'] for chunk_dict in chunk_dicts]
            
            # Print chunk details
            self._print_chunk_details(chunk_dicts)
            
            # Generate embeddings
            print(f"   🧠 Generating embeddings for {len(chunks)} chunks...")
            embeddings_array = self.embedder.generate_embeddings(chunks)
            
            # Print embedding details
            self._print_embedding_details(embeddings_array)
            
            if len(embeddings_array) != len(chunks):
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    chunks_created=len(chunk_dicts),
                    chunks_stored=0,
                    error_message=f"Embedding count mismatch: {len(chunks)} chunks, {len(embeddings_array)} embeddings",
                    processing_time=time.time() - start_time
                )
            
            # Convert numpy array to List[List[float]] for database compatibility
            embeddings = embeddings_array.tolist()
            
            # Store in database
            print(f"   💾 Storing in database...")
            filename = Path(file_path).name
            metadata = parse_result.get('metadata', {})
            
            # Print metadata details
            self._print_metadata_details(metadata)
            
            chunk_ids = self.database.store(
                filename=filename,
                filepath=file_path,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            
            processing_time = time.time() - start_time
            print(f"   ✅ Successfully processed in {processing_time:.2f}s")
            
            return ProcessingResult(
                file_path=file_path,
                success=True,
                chunks_created=len(chunk_dicts),
                chunks_stored=len(chunk_ids),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing error: {str(e)}"
            print(f"   ❌ {error_msg}")
            
            return ProcessingResult(
                file_path=file_path,
                success=False,
                chunks_created=0,
                chunks_stored=0,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    def retry_failed(self) -> List[ProcessingResult]:
        """
        Retry processing failed files.
        
        Returns:
            List of processing results for retried files
        """
        if not self.failed_files:
            print("📭 No failed files to retry")
            return []
        
        print(f"🔄 Retrying {len(self.failed_files)} failed files...")
        
        # Clear failed files list
        files_to_retry = self.failed_files.copy()
        self.failed_files.clear()
        
        # Reset failure stats
        self.stats['files_failed'] = 0
        
        # Retry each file
        results = []
        for file_path in files_to_retry:
            print(f"\n🔄 Retrying: {Path(file_path).name}")
            result = self.process_file(file_path)
            results.append(result)
            
            if result.success:
                self.stats['files_successful'] += 1
                self.stats['total_chunks'] += result.chunks_stored
            else:
                self.stats['files_failed'] += 1
                self.failed_files.append(file_path)
        
        self._print_summary()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def get_failed_files(self) -> List[str]:
        """Get list of files that failed processing."""
        return self.failed_files.copy()
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"📄 Files processed: {self.stats['files_processed']}")
        print(f"✅ Files successful: {self.stats['files_successful']}")
        print(f"❌ Files failed: {self.stats['files_failed']}")
        print(f"🧩 Total chunks stored: {self.stats['total_chunks']}")
        print(f"⏱️  Total processing time: {self.stats['total_processing_time']:.2f}s")
        
        if self.stats['files_processed'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['files_processed']
            print(f"📈 Average time per file: {avg_time:.2f}s")
        
        if self.failed_files:
            print(f"\n⚠️  Failed files ({len(self.failed_files)}):")
            for file_path in self.failed_files[:5]:  # Show first 5
                print(f"   • {Path(file_path).name}")
            if len(self.failed_files) > 5:
                print(f"   ... and {len(self.failed_files) - 5} more")
        
        print("=" * 60)
    
    def _print_parse_results(self, file_ext: str, parse_result: Dict[str, Any]):
        """Print detailed parsing results."""
        content = parse_result.get('content', '')
        metadata = parse_result.get('metadata', {})
        
        print(f"   📄 Content extracted: {len(content)} characters")
        
        # Show content preview
        if content:
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   📝 Content preview: {repr(preview)}")
        
        # For images, show caption if available
        if file_ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            caption = (metadata.get('caption') or 
                      metadata.get('description') or 
                      metadata.get('semantic_description'))
            if caption:
                print(f"   🖼️  Image caption: {caption}")
            else:
                print(f"   🖼️  Image processed (no caption generated)")
    
    def _print_chunk_details(self, chunk_dicts: List[Dict]):
        """Print detailed chunk information."""
        print(f"   📊 Chunk details:")
        for i, chunk_dict in enumerate(chunk_dicts[:3]):  # Show first 3 chunks
            text = chunk_dict.get('text', '')
            chunk_size = len(text)
            preview = text[:100] + "..." if len(text) > 100 else text
            
            print(f"     Chunk {i+1}: {chunk_size} chars")
            print(f"     Preview: {repr(preview)}")
            
            # Show chunk metadata
            chunk_meta = chunk_dict.get('metadata', {})
            if chunk_meta:
                relevant_keys = ['chunk_index', 'chunk_method', 'start_index', 'end_index']
                meta_info = {k: v for k, v in chunk_meta.items() if k in relevant_keys}
                if meta_info:
                    print(f"     Metadata: {meta_info}")
        
        if len(chunk_dicts) > 3:
            print(f"     ... and {len(chunk_dicts) - 3} more chunks")
    
    def _print_embedding_details(self, embeddings_array):
        """Print embedding information."""
        if len(embeddings_array) > 0:
            shape = embeddings_array.shape
            print(f"   🔢 Embeddings shape: {shape}")
            
            # Show sample of first embedding
            if len(embeddings_array) > 0:
                first_embedding = embeddings_array[0]
                sample_size = min(10, len(first_embedding))
                sample = first_embedding[:sample_size]
                print(f"   🔍 First embedding sample ({sample_size} dims): {sample.tolist()}")
                
                # Show embedding statistics
                mean_val = float(embeddings_array.mean())
                std_val = float(embeddings_array.std())
                print(f"   📈 Embedding stats: mean={mean_val:.4f}, std={std_val:.4f}")
    
    def _print_metadata_details(self, metadata: Dict[str, Any]):
        """Print metadata information."""
        if not metadata:
            print(f"   📋 No metadata available")
            return
        
        print(f"   📋 Metadata ({len(metadata)} keys):")
        
        # Show important metadata keys
        important_keys = ['total_pages', 'file_size', 'author', 'title', 'created_date', 'caption', 'description']
        shown_keys = set()
        
        for key in important_keys:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, (list, dict)):
                    print(f"     {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"     {key}: {value}")
                shown_keys.add(key)
        
        # Show other keys (limited)
        other_keys = [k for k in metadata.keys() if k not in shown_keys]
        if other_keys:
            for key in other_keys[:3]:  # Show first 3 other keys
                value = metadata[key]
                if isinstance(value, (list, dict)):
                    print(f"     {key}: {type(value).__name__} with {len(value)} items")
                else:
                    value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"     {key}: {value_str}")
            
            if len(other_keys) > 3:
                print(f"     ... and {len(other_keys) - 3} more metadata keys")