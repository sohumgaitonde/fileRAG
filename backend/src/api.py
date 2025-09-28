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
import logging
from datetime import datetime
from multi_query_search import MultiQuerySearch

# Set up logging
logger = logging.getLogger(__name__)

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
    result_limit: int = 20  # New parameter for multi-query system

class SearchResult(BaseModel):
    filename: str
    content: str
    file_path: str
    score: float
    weighted_score: Optional[float] = None
    query_importance: Optional[float] = None
    found_by_queries: Optional[List[str]] = None
    total_matches: Optional[int] = None
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


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_variations: List[str]
    performance_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    total_time: float

# In-memory storage for demo (replace with real database later)
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

# Initialize multi-query search system
multi_query_search = MultiQuerySearch()

# Search endpoint
@app.post("/api/search", response_model=SearchResponse)
async def search_files(request: SearchRequest):
    """Search through indexed files using multi-query system."""
    try:
        # Use multi-query search system
        search_results = multi_query_search.search_multiple_queries(
            user_query=request.query,
            n_results_per_query=5,  # 5 results per query variation
            global_limit=request.result_limit
        )
        
        # Convert results to SearchResult format
        formatted_results = []
        for result in search_results["results"]:
            formatted_results.append(SearchResult(
                filename=result.get("metadata", {}).get("filename", "Unknown"),
                content=result.get("document", ""),
                file_path=result.get("metadata", {}).get("filepath", ""),
                score=result.get("base_score", 0.0),
                weighted_score=result.get("weighted_score", 0.0),
                query_importance=result.get("query_importance", 1.0),
                found_by_queries=result.get("found_by_queries", []),
                total_matches=result.get("total_matches", 1)
            ))
        
        # Return enhanced search response
        return SearchResponse(
            results=formatted_results,
            query_variations=search_results["query_variations"],
            performance_metrics=search_results["performance_metrics"],
            quality_metrics=search_results["performance_metrics"].get("quality_metrics", {}),
            total_time=search_results["performance_metrics"]["total_time"]
        )
        
    except Exception as e:
        # Fallback to single query if multi-query fails
        logger.error(f"Multi-query search failed: {e}")
        
        # Simple fallback search (you could implement a single-query fallback here)
        fallback_results = [
            SearchResult(
                filename="fallback_result.txt",
                content=f"Fallback search result for '{request.query}'. Multi-query system temporarily unavailable.",
                file_path="/fallback/result.txt",
                score=0.5
            )
        ]
        
        return SearchResponse(
            results=fallback_results,
            query_variations=[request.query],
            performance_metrics={"error": str(e), "fallback_used": True},
            quality_metrics={"fallback": True},
            total_time=0.0

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
