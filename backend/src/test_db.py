"""
Local test for the VectorDatabase class.

This test demonstrates:
1. Initialize the database
2. Store sample document chunks
3. Search for similar content
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db import VectorDatabase


def test_database():
    """Test the VectorDatabase functionality."""
    
    print("🧪 Testing VectorDatabase...")
    
    # 1. Initialize database
    print("\n1️⃣ Initializing database...")
    db = VectorDatabase(
        db_path="./test_chroma_db", 
        collection_name="test_collection"
    )
    
    if not db.initialize():
        print("❌ Failed to initialize database")
        return False
    
    print("✅ Database initialized successfully!")
    
    # 2. Prepare sample data
    print("\n2️⃣ Preparing sample data...")
    
    # Sample document chunks
    filename = "test_document.txt"
    filepath = "/path/to/test_document.txt"
    chunks = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret and analyze visual information."
    ]
    
    # Sample embeddings (384-dimensional vectors to match ChromaDB's default model)
    # These are realistic-looking embeddings for testing
    embeddings = [
        [0.1] * 384,  # All 0.1s for first chunk
        [0.2] * 384,  # All 0.2s for second chunk  
        [0.3] * 384,  # All 0.3s for third chunk
        [0.4] * 384   # All 0.4s for fourth chunk
    ]
    
    # Sample metadata
    metadata = {
        "file_type": "txt",
        "author": "Test Author",
        "topic": "artificial_intelligence",
        "created_date": "2024-01-01"
    }
    
    print(f"   📝 {len(chunks)} chunks prepared")
    print(f"   🧠 {len(embeddings)} embeddings prepared")
    print(f"   📋 Metadata: {metadata}")
    
    # 3. Store data
    print("\n3️⃣ Storing data in database...")
    
    try:
        chunk_ids = db.store(
            filename=filename,
            filepath=filepath,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )
        
        print(f"✅ Successfully stored {len(chunk_ids)} chunks")
        print(f"   Chunk IDs: {chunk_ids[:2]}...")  # Show first 2 IDs
        
    except Exception as e:
        print(f"❌ Failed to store data: {str(e)}")
        return False
    
    # 4. Test search functionality
    print("\n4️⃣ Testing search functionality...")
    
    # Test 1: Basic search
    print("\n   🔍 Test 1: Basic search")
    try:
        results = db.search(
            query_text="machine learning algorithms",
            n_results=3
        )
        
        print(f"   ✅ Found {results['count']} results")
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"      {i+1}. {doc[:50]}... (from {metadata['filename']})")
            
    except Exception as e:
        print(f"   ❌ Search failed: {str(e)}")
        return False
    
    # Test 2: Search with metadata filter
    print("\n   🔍 Test 2: Search with filter")
    try:
        filtered_results = db.search(
            query_text="neural networks",
            n_results=2,
            where={"topic": "artificial_intelligence"}
        )
        
        print(f"   ✅ Found {filtered_results['count']} filtered results")
        for i, doc in enumerate(filtered_results['documents']):
            print(f"      {i+1}. {doc[:50]}...")
            
    except Exception as e:
        print(f"   ❌ Filtered search failed: {str(e)}")
        return False
    
    # Test 3: Search with file type filter
    print("\n   🔍 Test 3: Search by file type")
    try:
        type_results = db.search(
            query_text="computer vision",
            n_results=2,
            where={"file_type": "txt"}
        )
        
        print(f"   ✅ Found {type_results['count']} results for txt files")
        for i, doc in enumerate(type_results['documents']):
            print(f"      {i+1}. {doc[:50]}...")
            
    except Exception as e:
        print(f"   ❌ File type search failed: {str(e)}")
        return False
    
    print("\n🎉 All tests passed successfully!")
    print("\n📊 Test Summary:")
    print(f"   ✅ Database initialization: PASSED")
    print(f"   ✅ Data storage: PASSED")
    print(f"   ✅ Basic search: PASSED")
    print(f"   ✅ Filtered search: PASSED")
    print(f"   ✅ File type search: PASSED")
    
    return True


def cleanup_test_db():
    """Clean up test database files."""
    import shutil
    import os
    
    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        print(f"🧹 Cleaned up test database: {test_db_path}")


def main():
    """Main test function."""
    print("🚀 Starting VectorDatabase Local Test")
    print("=" * 50)
    
    try:
        success = test_database()
        
        if success:
            print("\n✅ All tests completed successfully!")
            
            # Ask if user wants to keep test database
            keep_db = input("\n🤔 Keep test database for inspection? (y/N): ").lower()
            if keep_db != 'y':
                cleanup_test_db()
            else:
                print(f"📁 Test database kept at: ./test_chroma_db")
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
        cleanup_test_db()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        cleanup_test_db()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
