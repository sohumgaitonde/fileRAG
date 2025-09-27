.PHONY: install dev-install clean test lint format run

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# Development
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

test:
	pytest tests/

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
	@echo "  install     - Install the package"
	@echo "  dev-install - Install with development dependencies"
	@echo "  clean       - Remove Python cache files"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  run         - Start the API server"
	@echo "  reset-db    - Reset the ChromaDB database"