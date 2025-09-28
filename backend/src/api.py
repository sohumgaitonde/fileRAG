"""
INTERFACE: FastAPI application for the fileRAG system.

This module provides the REST API endpoints for:
- File indexing operations
- Search queries
- System status and health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import asyncio
import os
from pathlib import Path

from .pipeline import IndexingPipeline

app = FastAPI(
    title="fileRAG API",
    description="A simple file RAG system for searching through local files",
    version="0.1.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response data
class IndexRequest(BaseModel):
    directory_path: str
    recursive: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class SearchResult(BaseModel):
    filename: str
    content: str
    file_path: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class IndexResponse(BaseModel):
    job_id: str
    status: str
    message: str
    directory_path: str

class IndexStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Dict[str, Any]
    directory_path: str
    started_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

class StatsResponse(BaseModel):
    file_count: int
    db_size: str
    last_indexed: str
    status: str
    total_chunks: int

# Global pipeline instance
pipeline: Optional[IndexingPipeline] = None

# In-memory storage for indexing jobs
indexing_jobs = {}
search_results_cache = {}

def get_pipeline() -> IndexingPipeline:
    """Get or create the global pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = IndexingPipeline()
        if not pipeline.initialize():
            raise RuntimeError("Failed to initialize indexing pipeline")
    return pipeline

async def run_indexing_job(job_id: str, directory_path: str, recursive: bool, chunk_size: int, chunk_overlap: int):
    """Background task to run the indexing job."""
    try:
        # Update job status
        indexing_jobs[job_id]["status"] = "running"
        indexing_jobs[job_id]["progress"] = {"message": "Initializing pipeline...", "files_processed": 0, "files_total": 0}
        
        # Get pipeline instance
        job_pipeline = IndexingPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize pipeline
        if not job_pipeline.initialize():
            raise Exception("Failed to initialize pipeline")
        
        # Update progress
        indexing_jobs[job_id]["progress"]["message"] = "Processing files..."
        
        # Run the indexing
        results = job_pipeline.process_directory(directory_path, recursive=recursive)
        
        # Get final stats
        stats = job_pipeline.get_stats()
        failed_files = job_pipeline.get_failed_files()
        
        # Update job completion
        indexing_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "progress": {
                "message": "Indexing completed successfully",
                "files_processed": stats["files_processed"],
                "files_successful": stats["files_successful"],
                "files_failed": stats["files_failed"],
                "total_chunks": stats["total_chunks"],
                "processing_time": stats["total_processing_time"],
                "failed_files": failed_files[:10]  # Limit to first 10 failed files
            }
        })
        
    except Exception as e:
        # Update job with error
        indexing_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error_message": str(e),
            "progress": {
                "message": f"Indexing failed: {str(e)}",
                "files_processed": indexing_jobs[job_id]["progress"].get("files_processed", 0)
            }
        })

@app.get("/")
async def root():
    """Root endpoint - system status."""
    return {"message": "fileRAG API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Indexing endpoint
@app.post("/api/index", response_model=IndexResponse)
async def start_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """Start indexing files in a directory."""
    
    # Validate directory path
    if not os.path.exists(request.directory_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory_path}")
    
    if not os.path.isdir(request.directory_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.directory_path}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job info
    indexing_jobs[job_id] = {
        "status": "queued",
        "directory_path": request.directory_path,
        "started_at": datetime.now().isoformat(),
        "progress": {
            "message": "Job queued for processing",
            "files_processed": 0,
            "files_total": 0
        },
        "recursive": request.recursive,
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap
    }
    
    # Start background indexing task
    background_tasks.add_task(
        run_indexing_job,
        job_id,
        request.directory_path,
        request.recursive,
        request.chunk_size,
        request.chunk_overlap
    )
    
    return IndexResponse(
        job_id=job_id,
        status="queued",
        message=f"Indexing job queued for directory: {request.directory_path}",
        directory_path=request.directory_path
    )

# Search endpoint
@app.post("/api/search", response_model=List[SearchResult])
async def search_files(request: SearchRequest):
    """Search through indexed files."""
    
    try:
        # Get pipeline instance to access database
        current_pipeline = get_pipeline()
        
        # Perform vector search
        search_results = current_pipeline.database.search(
            query_text=request.query,
            n_results=request.limit
        )
        
        # Convert ChromaDB results to API response format
        results = []
        for i in range(len(search_results['ids'])):
            chunk_id = search_results['ids'][i]
            document_content = search_results['documents'][i]
            distance = search_results['distances'][i]
            metadata = search_results['metadatas'][i]
            
            # Extract metadata fields
            filename = metadata.get('filename', 'Unknown')
            file_path = metadata.get('filepath', '')
            chunk_index = metadata.get('chunk_index', 0)
            
            # Convert distance to similarity score (lower distance = higher similarity)
            # ChromaDB uses L2 distance, convert to 0-1 similarity score
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
            
            # Create search result
            result = SearchResult(
                filename=filename,
                content=document_content,
                file_path=file_path,
                score=round(similarity_score, 4),
                metadata={
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'distance': round(distance, 4),
                    'total_chunks': metadata.get('total_chunks', 1)
                }
            )
            results.append(result)
        
        # Cache results
        search_results_cache[request.query] = results
        
        print(f"🔍 Search completed: '{request.query}' -> {len(results)} results")
        return results
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Search failed: {str(e)}"
        )

# Status endpoint
@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        # Get pipeline instance to access database
        current_pipeline = get_pipeline()
        
        # Get collection info if available
        if current_pipeline.database.collection is not None:
            # Get collection count
            collection_count = current_pipeline.database.collection.count()
            
            # Calculate approximate database size
            db_path = Path(current_pipeline.database.db_path)
            db_size = "0 B"
            if db_path.exists():
                total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
                if total_size > 1024 * 1024:
                    db_size = f"{total_size / (1024 * 1024):.1f} MB"
                elif total_size > 1024:
                    db_size = f"{total_size / 1024:.1f} KB"
                else:
                    db_size = f"{total_size} B"
            
            # Get last indexed time from recent jobs
            last_indexed = "Never"
            if indexing_jobs:
                completed_jobs = [job for job in indexing_jobs.values() if job["status"] == "completed"]
                if completed_jobs:
                    latest_job = max(completed_jobs, key=lambda x: x.get("completed_at", ""))
                    last_indexed = latest_job.get("completed_at", "Unknown")
            
            # Estimate file count (collection count represents chunks, not files)
            # Rough estimate: average 3-5 chunks per file
            estimated_file_count = max(1, collection_count // 4)
            
            return StatsResponse(
                file_count=estimated_file_count,
                db_size=db_size,
                last_indexed=last_indexed,
                status="healthy",
                total_chunks=collection_count
            )
        else:
            return StatsResponse(
                file_count=0,
                db_size="0 B",
                last_indexed="Never",
                status="not_initialized",
                total_chunks=0
            )
            
    except Exception as e:
        return StatsResponse(
            file_count=0,
            db_size="Unknown",
            last_indexed="Unknown",
            status=f"error: {str(e)}",
            total_chunks=0
        )

# Indexing status endpoint
@app.get("/api/index/status/{job_id}", response_model=IndexStatusResponse)
async def get_indexing_status(job_id: str):
    """Get indexing job status."""
    if job_id not in indexing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = indexing_jobs[job_id]
    
    return IndexStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        directory_path=job["directory_path"],
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        error_message=job.get("error_message")
    )

# List all indexing jobs endpoint
@app.get("/api/index/jobs")
async def list_indexing_jobs():
    """List all indexing jobs."""
    jobs = []
    for job_id, job_data in indexing_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": job_data["status"],
            "directory_path": job_data["directory_path"],
            "started_at": job_data["started_at"],
            "completed_at": job_data.get("completed_at"),
            "progress": job_data.get("progress", {}),
            "error_message": job_data.get("error_message")
        })
    
    # Sort by started_at descending (newest first)
    jobs.sort(key=lambda x: x["started_at"], reverse=True)
    return {"jobs": jobs}
