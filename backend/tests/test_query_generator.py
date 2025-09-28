#!/usr/bin/env python3
"""
Test script for QueryGenerator Phase 1 implementation.
This tests the SLM integration and query generation functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_generator import QueryGenerator

def main():
    print("🚀 Testing QueryGenerator Phase 1 Implementation")
    print("=" * 50)
    
    # Initialize the query generator
    print("1. Initializing QueryGenerator...")
    qg = QueryGenerator()
    
    # Test query generation
    test_queries = [
        "machine learning algorithms",
        "python web development",
        "database optimization techniques"
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing with query: '{test_query}'")
        print("-" * 40)
        
        try:
            variations = qg.generate_query_variations(test_query)
            print(f"✅ Generated {len(variations)} variations:")
            for j, variation in enumerate(variations, 1):
                print(f"   {j}. {variation}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Phase 1 testing complete!")

if __name__ == "__main__":
    main()
