#!/bin/bash

# fileRAG Environment Setup Script
# This script creates a virtual environment and installs all dependencies

echo "🚀 Setting up fileRAG development environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install in development mode
echo "🔧 Installing fileRAG in development mode..."
pip install -e .

# Install additional dev dependencies for testing
echo "🧪 Installing testing dependencies..."
pip install pytest-cov

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "   deactivate"
echo ""
echo "To run tests:"
echo "   make test"
echo ""
echo "To start the API server:"
echo "   make run"
