#!/usr/bin/env python3
"""
Simple test runner for the FileRAG parser test suite.

Usage:
    python run_tests.py
"""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "tests"))

from test_parsers import run_tests

if __name__ == "__main__":
    run_tests()
