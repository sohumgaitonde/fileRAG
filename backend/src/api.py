"""
INTERFACE: FastAPI application for the fileRAG system.

This module provides the REST API endpoints for:
- File indexing operations
- Search queries
- System status and health checks
"""

from fastapi import FastAPI

app = FastAPI(
    title="fileRAG API",
    description="A simple file RAG system for searching through local files",
    version="0.1.0"
)


@app.get("/")
async def root():
    """Root endpoint - system status."""
    return {"message": "fileRAG API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# TODO: Add indexing endpoints
# TODO: Add search endpoints
# TODO: Add file management endpoints
