"""
INTERFACE: FastAPI application for the fileRAG system.

This module provides the REST API endpoints for:
- File indexing operations
- Search queries
- System status and health checks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import logging
from datetime import datetime
from multi_query_search import MultiQuerySearch

# Set up logging
logger = logging.getLogger(__name__)

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

class IndexResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatsResponse(BaseModel):
    file_count: int
    db_size: str
    last_indexed: str
    status: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_variations: List[str]
    performance_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    total_time: float

# In-memory storage for demo (replace with real database later)
indexing_jobs = {}
search_results_cache = {}

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
async def start_indexing(request: IndexRequest):
    """Start indexing files in a directory."""
    job_id = str(uuid.uuid4())
    
    # Store job info (in real implementation, this would be in a database)
    indexing_jobs[job_id] = {
        "status": "started",
        "directory_path": request.directory_path,
        "started_at": datetime.now().isoformat(),
        "progress": 0,
        "total_files": 0
    }
    
    return IndexResponse(
        job_id=job_id,
        status="started",
        message=f"Indexing started for directory: {request.directory_path}"
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
    return StatsResponse(
        file_count=42,
        db_size="15.7 MB",
        last_indexed="2024-01-15 14:30:00",
        status="healthy"
    )

# Indexing status endpoint
@app.get("/api/index/status/{job_id}")
async def get_indexing_status(job_id: str):
    """Get indexing job status."""
    if job_id not in indexing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = indexing_jobs[job_id]
    
    # Simulate progress for demo
    if job["status"] == "started":
        job["progress"] = min(job["progress"] + 25, 100)
        if job["progress"] >= 100:
            job["status"] = "completed"
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "directory_path": job["directory_path"],
        "started_at": job["started_at"]
    }
