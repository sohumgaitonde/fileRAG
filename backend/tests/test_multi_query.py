#!/usr/bin/env python3
"""
Test script for Multi-Query Search functionality.
This tests the multi-query search with database integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_query_search import MultiQuerySearch

def main():
    print("🚀 Testing Multi-Query Search (Phase 2.2)")
    print("=" * 60)
    
    # Initialize multi-query search
    print("1. Initializing Multi-Query Search...")
    try:
        search = MultiQuerySearch()
        print("✅ Multi-Query Search initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "python web development", 
        "database optimization techniques"
    ]
    
    print(f"\n2. Testing with {len(test_queries)} different queries...")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Perform multi-query search
            results = search.search_multiple_queries(
                user_query=query,
                n_results_per_query=5,  # 5 results per query variation
                global_limit=10         # 10 total results
            )
            
            # Display results
            print(f"✅ Generated {len(results['query_variations'])} query variations")
            print(f"📊 Found {len(results['results'])} unique results")
            print(f"⏱️  Total time: {results['performance_metrics']['total_time']:.2f}s")
            print(f"📈 Avg query time: {results['performance_metrics']['avg_query_time']:.2f}s")
            
            # Show query variations
            print(f"\n📝 Query variations:")
            for j, variation in enumerate(results['query_variations'], 1):
                print(f"   {j}. {variation}")
            
            # Show top results
            print(f"\n🏆 Top results:")
            for j, result in enumerate(results['results'][:3], 1):
                doc = result.get('document', 'No content')
                score = 1.0 - result.get('distance', 1.0)
                filename = result.get('metadata', {}).get('filename', 'Unknown')
                print(f"   {j}. Score: {score:.3f} | {filename}")
                print(f"      Content: {doc[:80]}...")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Phase 2.2 testing complete!")
    print("\n💡 Next steps:")
    print("   - Add sample data to database for real testing")
    print("   - Integrate with API endpoints")
    print("   - Test with frontend integration")

if __name__ == "__main__":
    main()
