#!/usr/bin/env python3
"""
Test OpenAI integration for query generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_generator import QueryGenerator

def main():
    print("🚀 Testing OpenAI QueryGenerator")
    print("=" * 50)
    
    # Initialize with OpenAI
    qg = QueryGenerator(use_fallback=True)
    
    # Test queries
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
    print("🎉 OpenAI testing complete!")

if __name__ == "__main__":
    main()
