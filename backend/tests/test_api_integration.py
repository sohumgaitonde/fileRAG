#!/usr/bin/env python3
"""
Test script for API integration with multi-query search system.
This tests the enhanced API endpoints with Phase 3 and 4 features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import requests
import json
import time

def test_api_integration():
    """Test the enhanced API with multi-query search."""
    print("🚀 Testing API Integration (Phase 3 & 4)")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "python web development",
        "database optimization"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Test enhanced search endpoint
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/search",
                json={
                    "query": query,
                    "limit": 5,
                    "result_limit": 10
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Search successful in {end_time - start_time:.2f}s")
                print(f"📊 Found {len(data['results'])} results")
                print(f"🔄 Generated {len(data['query_variations'])} query variations")
                print(f"⏱️  Total time: {data['total_time']:.2f}s")
                
                # Display quality metrics
                if 'quality_metrics' in data:
                    quality = data['quality_metrics']
                    print(f"📈 Quality metrics:")
                    print(f"   - Average score: {quality.get('avg_score', 0):.3f}")
                    print(f"   - Diversity score: {quality.get('diversity_score', 0):.3f}")
                    print(f"   - Coverage score: {quality.get('coverage_score', 0):.3f}")
                    print(f"   - Unique documents: {quality.get('unique_documents', 0)}")
                
                # Display query variations
                print(f"\n📝 Query variations:")
                for j, variation in enumerate(data['query_variations'], 1):
                    print(f"   {j}. {variation}")
                
                # Display top results with enhanced metadata
                print(f"\n🏆 Top results:")
                for j, result in enumerate(data['results'][:3], 1):
                    print(f"   {j}. {result['filename']}")
                    print(f"      Score: {result['score']:.3f}")
                    if result.get('weighted_score'):
                        print(f"      Weighted Score: {result['weighted_score']:.3f}")
                    if result.get('query_importance'):
                        print(f"      Query Importance: {result['query_importance']:.3f}")
                    if result.get('found_by_queries'):
                        print(f"      Found by: {len(result['found_by_queries'])} queries")
                    print(f"      Content: {result['content'][:80]}...")
                
            else:
                print(f"❌ Search failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed: Backend server not running")
            print("   Start the backend with: make run")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 API integration testing complete!")
    print("\n💡 Next steps:")
    print("   - Test with frontend integration")
    print("   - Add error handling scenarios")
    print("   - Performance optimization")

if __name__ == "__main__":
    test_api_integration()
