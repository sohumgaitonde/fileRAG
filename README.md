# fileRAG

A simple file RAG (Retrieval-Augmented Generation) system for effectively searching through local files.

## Architecture

The system is organized into three main sections:

### 🖥️ Interface
- **API Layer**: FastAPI-based REST API for interacting with the system

### 📚 Indexing
- **File Crawling**: Discover and monitor files in specified directories
- **Content Parsing**: Extract text from various file formats (PDF, DOCX, TXT, MD, Images)
- **Text Chunking**: Split content into manageable chunks using Chonkie
- **Embeddings**: Generate vector embeddings using Qwen 2B model
- **Pipeline**: Orchestrate the indexing process

### 🔍 Querying
- **Query Generation**: Generate search queries using SLM via Ollama
- **Vector Database**: Store and retrieve embeddings using ChromaDB locally

## Tech Stack

- **Vector Database**: ChromaDB (local)
- **Chunking**: Chonkie
- **Embeddings**: Qwen 2B
- **Query Generation**: SLM via Ollama
- **API**: FastAPI
- **File Parsing**: PyPDF2, python-docx, Pillow

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Start the API server
uvicorn src.api:app --reload
```

## Project Structure

```
fileRAG/
├── src/
│   ├── api.py              # Interface: FastAPI application
│   ├── crawler.py          # Indexing: File discovery
│   ├── chunker.py          # Indexing: Text chunking
│   ├── embeddings.py       # Indexing: Vector embeddings
│   ├── pipeline.py         # Indexing: Processing pipeline
│   ├── query_generator.py  # Querying: Query generation
│   ├── db.py              # Querying: Database operations
│   └── parsers/           # Indexing: File format parsers
│       ├── pdf_parser.py
│       ├── docx_parser.py
│       ├── txt_parser.py
│       ├── md_parser.py
│       └── image_parser.py
```