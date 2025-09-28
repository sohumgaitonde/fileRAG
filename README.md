# fileRAG - AI-Powered File Search System

A desktop application for semantic file search using AI-powered embeddings and vector similarity search.

## 🏗️ Architecture

This repository contains both the backend API and frontend desktop application:

- **Backend**: FastAPI-based Python service for file indexing and search
- **Frontend**: Electron desktop app with Perplexity-style interface

## 📁 Project Structure

```
fileRAG/
├── backend/          # Python FastAPI backend
│   ├── src/         # Source code
│   ├── tests/       # Test files
│   └── requirements.txt
├── frontend/        # Electron desktop app
│   ├── main.js      # Electron main process
│   ├── index.html   # UI
│   ├── renderer.js  # Frontend logic
│   └── package.json
└── README.md
```

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn src.api:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - REST API framework
- **ChromaDB** - Vector database
- **Transformers** - AI embeddings (Qwen 2B)
- **Ollama** - Query generation
- **PyPDF2, python-docx** - File parsing

### Frontend
- **Electron** - Desktop app framework
- **HTML/CSS/JavaScript** - UI
- **Axios** - HTTP client

## 📋 Features

- **File Indexing**: Support for PDF, DOCX, TXT, MD, Images
- **Semantic Search**: AI-powered search using embeddings
- **Desktop Interface**: Clean, Perplexity-style UI
- **Real-time Updates**: Live indexing progress
- **Local Storage**: ChromaDB for vector storage

## 🔧 Development

### Backend Development
```bash
cd backend
make dev-install  # Install dev dependencies
make test        # Run tests
make lint        # Code linting
```

### Frontend Development
```bash
cd frontend
npm run dev      # Development mode
npm run build    # Build for production
```

## 📝 API Endpoints

- `GET /` - Health check
- `GET /health` - System status
- `POST /api/index` - Start file indexing
- `POST /api/search` - Search files
- `GET /api/stats` - System statistics

## 🎯 Usage

1. **Start Backend**: Run the FastAPI server
2. **Start Frontend**: Launch the Electron app
3. **Index Files**: Use the Setup tab to select directories
4. **Search**: Use the main search bar to find files

## 📄 License

MIT License - see LICENSE file for details.
