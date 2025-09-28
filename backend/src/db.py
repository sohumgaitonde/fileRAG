"""
ChromaDB database operations for FileRAG.

This module provides the essential database operations:
- Initialize ChromaDB
- Store document chunks with embeddings
- Search for similar content
"""

import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorDatabase:
    """Manages ChromaDB operations for vector storage and retrieval."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "fileRAG"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "FileRAG document embeddings"}
                )
                print(f"Created new collection: {self.collection_name}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {str(e)}")
            return False
    
    def store(
        self, 
        filename: str, 
        filepath: str, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Store document chunks with embeddings in the database.
        
        Args:
            filename: Original filename
            filepath: Full path to the file
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            metadata: Additional metadata for the document
            
        Returns:
            List of unique IDs for the stored chunks
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize ChromaDB")
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Generate unique IDs for each chunk
        chunk_ids = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID: filename_chunk_index
            chunk_id = f"{filename}_{i}_{uuid.uuid4().hex[:8]}"
            chunk_ids.append(chunk_id)
            
            # Prepare metadata for this chunk
            chunk_metadata = {
                "filename": filename,
                "filepath": filepath,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_content": chunk,
                **metadata  # Include any additional metadata
            }
            metadatas.append(chunk_metadata)
        
        # Store in ChromaDB
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"Stored {len(chunks)} chunks for {filename}")
        return chunk_ids
    
    def search(
        self, 
        query_text: str, 
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search using text query (ChromaDB will handle embedding generation).
        
        Args:
            query_text: Text query
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary with search results
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize ChromaDB")
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "count": len(results["ids"][0]) if results["ids"] else 0
        }
