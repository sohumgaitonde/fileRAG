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
from typing import List, Dict, Any
import uuid
from datetime import datetime

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

class SearchResult(BaseModel):
    filename: str
    content: str
    file_path: str
    score: float

class IndexResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatsResponse(BaseModel):
    file_count: int
    db_size: str
    last_indexed: str
    status: str

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

# Search endpoint
@app.post("/api/search", response_model=List[SearchResult])
async def search_files(request: SearchRequest):
    """Search through indexed files."""
    
    # Dummy search results for demo
    dummy_results = [
        SearchResult(
            filename="sample_document.pdf",
            content=f"Found content related to '{request.query}' in this document. This is a sample search result showing how the search functionality works.",
            file_path="/Users/example/documents/sample_document.pdf",
            score=0.95
        ),
        SearchResult(
            filename="research_paper.docx",
            content=f"Another document containing information about '{request.query}'. This demonstrates multiple search results.",
            file_path="/Users/example/documents/research_paper.docx",
            score=0.87
        ),
        SearchResult(
            filename="notes.txt",
            content=f"Quick notes and thoughts about '{request.query}'. This shows how different file types are handled.",
            file_path="/Users/example/notes/notes.txt",
            score=0.72
        )
    ]
    
    # Cache results for demo
    search_results_cache[request.query] = dummy_results
    
    return dummy_results[:request.limit]

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
