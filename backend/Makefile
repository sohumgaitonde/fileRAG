.PHONY: setup install dev-install clean test lint format run venv

# Environment setup
setup: venv install

venv:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Development
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

test:
	pytest tests/ -v

test-crawler:
	pytest tests/test_crawler.py -v

test-chunker:
	pytest tests/test_chunker.py -v

test-chunker-integration:
	pytest tests/integration_test_chunker.py -v -s

test-coverage:
	pytest tests/ --cov=src --cov-report=html

lint:
	flake8 src/
	black --check src/
	isort --check-only src/

format:
	black src/
	isort src/

# Run the application
run:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Database operations
reset-db:
	rm -rf ./chroma_db

# Help
help:
	@echo "Available commands:"
	@echo "  setup       - Create venv and install dependencies"
	@echo "  install     - Install the package"
	@echo "  dev-install - Install with development dependencies"
	@echo "  clean       - Remove Python cache files"
	@echo "  test        - Run all tests"
	@echo "  test-crawler - Run crawler tests only"
	@echo "  test-chunker - Run chunker unit tests"
	@echo "  test-chunker-integration - Run chunker integration tests"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  run         - Start the API server"
	@echo "  reset-db    - Reset the ChromaDB database"