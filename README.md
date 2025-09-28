# fileRAG

A simple file RAG (Retrieval-Augmented Generation) system for effectively searching through local files.

## ✨ Features

- 🔍 **Smart File Discovery**: Automatically crawls directories for supported file types
- 📄 **Multi-Format Support**: PDF, DOCX, TXT, Markdown, and Images (with OCR)
- 🧠 **Advanced Chunking**: Intelligent text splitting using Chonkie
- 🎯 **Vector Search**: Fast similarity search with ChromaDB
- 🤖 **AI-Powered Queries**: Query optimization using SLM via Ollama
- ⚡ **FastAPI Interface**: Modern REST API for easy integration

## 🏗️ Architecture

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

## 🛠️ Tech Stack

- **Vector Database**: ChromaDB (local)
- **Chunking**: Chonkie
- **Embeddings**: Qwen 2B
- **Query Generation**: SLM via Ollama
- **API**: FastAPI
- **File Parsing**: PyPDF2, python-docx, Pillow

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### 🔧 Installation

#### Automated Setup 
```bash
# Clone the repository
git clone <your-repo-url>
cd fileRAG

# Run automated setup
./setup_env.sh

# Activate the environment
source venv/bin/activate
```

### 📦 Dependencies

The installation includes:

**Core Dependencies:**
- `chromadb>=0.4.0` - Vector database
- `chonkie>=0.1.0` - Text chunking
- `fastapi>=0.104.0` - API framework
- `ollama>=0.2.0` - SLM integration
- `transformers>=4.35.0` + `torch>=2.0.0` - Qwen 2B embeddings

**File Processing:**
- `PyPDF2>=3.0.0` - PDF parsing
- `python-docx>=0.8.11` - DOCX parsing
- `Pillow>=10.0.0` - Image processing

**Development Tools:**
- `pytest>=7.0.0` + `pytest-cov>=4.0.0` - Testing
- `black`, `isort`, `flake8` - Code formatting

## 🎯 Usage

### Starting the API Server
```bash
# Activate environment
source venv/bin/activate

# Start the server
make run
# or
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Testing the System
```bash
# Run all tests
make test

# Test the file crawler specifically
make test-crawler

# Run tests with coverage
make test-coverage
```

### Development Workflow
```bash
# Activate environment (do this each session)
source venv/bin/activate

# Format code
make format

# Check code quality
make lint

# Run tests
make test

# Start development server
make run

# Deactivate when done
deactivate
```

## 📁 Project Structure

```
fileRAG/
├── 📄 Project Configuration
│   ├── pyproject.toml          # Project metadata and dependencies
│   ├── requirements.txt        # Python dependencies
│   ├── Makefile               # Development commands
│   └── setup_env.sh           # Automated environment setup
│
├── 💻 Source Code
│   └── src/
│       ├── 🖥️ Interface Layer
│       │   └── api.py              # FastAPI application
│       │
│       ├── 📚 Indexing Layer
│       │   ├── crawler.py          # File discovery and crawling
│       │   ├── chunker.py          # Text chunking with Chonkie
│       │   ├── embeddings.py       # Vector embeddings (Qwen 2B)
│       │   ├── pipeline.py         # Processing orchestration
│       │   └── parsers/            # File format parsers
│       │       ├── pdf_parser.py   # PDF document parsing
│       │       ├── docx_parser.py  # Word document parsing
│       │       ├── txt_parser.py   # Plain text parsing
│       │       ├── md_parser.py    # Markdown parsing
│       │       └── image_parser.py # Image OCR parsing
│       │
│       └── 🔍 Querying Layer
│           ├── query_generator.py  # SLM query generation (Ollama)
│           └── db.py              # ChromaDB operations
│
└── 🧪 Testing
    ├── tests/
    │   ├── test_crawler.py        # Crawler functionality tests
    │   └── test_data/            # Sample files for testing
    └── testing_main.py           # End-to-end testing script
```

## 🎮 Available Commands

All commands are available through the Makefile:

```bash
# Environment Setup
make setup          # Create venv and install dependencies
make venv           # Create virtual environment only
make install        # Install dependencies
make dev-install    # Install with development dependencies

# Development
make clean          # Remove Python cache files
make format         # Format code with black and isort
make lint           # Run code quality checks

# Testing
make test           # Run all tests
make test-crawler   # Run crawler tests specifically
make test-coverage  # Run tests with coverage report

# Running
make run            # Start the API server
make reset-db       # Reset the ChromaDB database

# Help
make help           # Show all available commands
```

## 🔧 Configuration

The system uses environment variables for configuration. Key settings include:

- **File Processing**: Maximum file size, supported extensions
- **Chunking**: Chunk size and overlap settings
- **Database**: ChromaDB storage path
- **Models**: Embedding and query generation model settings
- **API**: Host and port configuration

## 🧪 Testing

The project includes comprehensive tests:

### Running Tests
```bash
# All tests
make test

# Specific test files
pytest tests/test_crawler.py -v

# With coverage
make test-coverage
```

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Test Data**: Sample files for realistic testing scenarios

## 🤝 Development

### Code Quality
The project enforces code quality through:
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting and style checks
- **Type hints**: For better code documentation

### Contributing Workflow
1. Activate the virtual environment: `source venv/bin/activate`
2. Make your changes
3. Format code: `make format`
4. Run tests: `make test`
5. Check linting: `make lint`
6. Submit your changes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Virtual Environment Issues:**
```bash
# If venv activation fails
python3 -m venv venv --clear
source venv/bin/activate
```

**Dependency Issues:**
```bash
# Upgrade pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Permission Issues:**
```bash
# Make setup script executable
chmod +x setup_env.sh
```

### Getting Help

1. Check the test files for usage examples
2. Run `make help` for available commands
3. Check the logs for detailed error messages
4. Ensure all prerequisites are installed
