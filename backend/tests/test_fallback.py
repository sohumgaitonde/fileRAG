#!/usr/bin/env python3
"""
Test the fallback functionality by simulating OpenAI failures.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_generator import QueryGenerator

def test_fallback_scenarios():
    """Test different fallback scenarios."""
    print("🧪 Testing Fallback Scenarios")
    print("=" * 50)
    
    # Test 1: No API key (should trigger fallback)
    print("\n1. Testing with no API key (should use fallback):")
    print("-" * 50)
    
    # Temporarily remove API key
    original_key = os.environ.get('OPENAI_API_KEY')
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    qg_no_key = QueryGenerator(use_fallback=True)
    variations = qg_no_key.generate_query_variations("machine learning")
    print(f"✅ Generated {len(variations)} variations:")
    for i, variation in enumerate(variations, 1):
        print(f"   {i}. {variation}")
    
    # Restore API key
    if original_key:
        os.environ['OPENAI_API_KEY'] = original_key
    
    # Test 2: With API key but fallback disabled
    print("\n2. Testing with API key but fallback disabled:")
    print("-" * 50)
    
    qg_no_fallback = QueryGenerator(use_fallback=False)
    variations = qg_no_fallback.generate_query_variations("machine learning")
    print(f"✅ Generated {len(variations)} variations:")
    for i, variation in enumerate(variations, 1):
        print(f"   {i}. {variation}")
    
    # Test 3: With API key and fallback enabled (normal case)
    print("\n3. Testing with API key and fallback enabled (normal case):")
    print("-" * 50)
    
    qg_normal = QueryGenerator(use_fallback=True)
    variations = qg_normal.generate_query_variations("machine learning")
    print(f"✅ Generated {len(variations)} variations:")
    for i, variation in enumerate(variations, 1):
        print(f"   {i}. {variation}")
    
    print("\n" + "=" * 50)
    print("🎉 Fallback testing complete!")

if __name__ == "__main__":
    test_fallback_scenarios()
