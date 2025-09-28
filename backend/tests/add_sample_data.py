#!/usr/bin/env python3
"""
Add sample data to the database for testing multi-query search.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import VectorDatabase
import numpy as np

def add_sample_data():
    """Add sample documents to the database."""
    print("📚 Adding sample data to database...")
    
    # Initialize database
    db = VectorDatabase()
    
    # Sample documents
    sample_docs = [
        {
            "filename": "machine_learning_guide.txt",
            "filepath": "/sample/machine_learning_guide.txt",
            "content": """
            Machine Learning Algorithms Guide
            
            This comprehensive guide covers the most important machine learning algorithms:
            
            1. Supervised Learning Algorithms:
               - Linear Regression: Used for predicting continuous values
               - Logistic Regression: Used for classification problems
               - Decision Trees: Tree-based models for both regression and classification
               - Random Forest: Ensemble method using multiple decision trees
               - Support Vector Machines (SVM): Effective for high-dimensional data
               - Naive Bayes: Probabilistic classifier based on Bayes' theorem
            
            2. Unsupervised Learning Algorithms:
               - K-Means Clustering: Groups data into k clusters
               - Hierarchical Clustering: Creates tree of clusters
               - Principal Component Analysis (PCA): Dimensionality reduction
               - DBSCAN: Density-based clustering
            
            3. Deep Learning Algorithms:
               - Neural Networks: Multi-layer perceptrons
               - Convolutional Neural Networks (CNN): For image processing
               - Recurrent Neural Networks (RNN): For sequential data
               - Long Short-Term Memory (LSTM): Advanced RNN for long sequences
               - Transformer: Attention-based models for NLP
            
            Best practices for choosing algorithms:
            - Start with simple algorithms (linear regression, decision trees)
            - Use ensemble methods for better performance
            - Consider deep learning for complex patterns
            - Always validate with cross-validation
            """,
            "metadata": {"category": "machine_learning", "type": "guide"}
        },
        {
            "filename": "python_web_development.txt", 
            "filepath": "/sample/python_web_development.txt",
            "content": """
            Python Web Development Best Practices
            
            Python is one of the most popular languages for web development due to its simplicity and powerful frameworks.
            
            Popular Python Web Frameworks:
            1. Django: Full-featured framework with built-in admin, ORM, and security features
            2. Flask: Lightweight and flexible microframework
            3. FastAPI: Modern framework for building APIs with automatic documentation
            4. Pyramid: Flexible framework that scales from simple to complex applications
            
            Key Python Web Development Concepts:
            - WSGI (Web Server Gateway Interface): Standard interface between web servers and Python applications
            - Virtual Environments: Isolate project dependencies using venv or conda
            - Package Management: Use pip for installing packages, requirements.txt for dependencies
            - Database Integration: SQLAlchemy ORM, Django ORM, or raw SQL
            - Testing: pytest, unittest for writing and running tests
            - Deployment: Docker containers, cloud platforms (AWS, Heroku, DigitalOcean)
            
            Best Practices:
            - Follow PEP 8 style guidelines
            - Use type hints for better code documentation
            - Implement proper error handling and logging
            - Write comprehensive tests
            - Use environment variables for configuration
            - Implement security best practices (CSRF protection, input validation)
            """,
            "metadata": {"category": "web_development", "type": "best_practices"}
        },
        {
            "filename": "database_optimization.txt",
            "filepath": "/sample/database_optimization.txt", 
            "content": """
            Database Optimization Techniques
            
            Database optimization is crucial for maintaining high performance in production systems.
            
            Query Optimization Techniques:
            1. Indexing Strategies:
               - Primary indexes on frequently queried columns
               - Composite indexes for multi-column queries
               - Partial indexes for filtered queries
               - Covering indexes to avoid table lookups
            
            2. Query Performance:
               - Use EXPLAIN to analyze query execution plans
               - Avoid SELECT * and specify only needed columns
               - Use LIMIT to restrict result sets
               - Optimize JOIN operations and avoid N+1 queries
               - Use prepared statements for repeated queries
            
            3. Database Design:
               - Normalize data to reduce redundancy
               - Denormalize strategically for read performance
               - Use appropriate data types and sizes
               - Implement proper foreign key constraints
               - Consider partitioning for large tables
            
            4. Caching Strategies:
               - Application-level caching (Redis, Memcached)
               - Database query result caching
               - CDN caching for static content
               - Browser caching with proper headers
            
            5. Monitoring and Profiling:
               - Monitor slow query logs
               - Use database profiling tools
               - Track performance metrics over time
               - Set up alerts for performance degradation
            
            Common Performance Issues:
            - Missing or inefficient indexes
            - Poorly written queries with unnecessary complexity
            - Lack of connection pooling
            - Inadequate hardware resources
            - Poor database schema design
            """,
            "metadata": {"category": "database", "type": "optimization"}
        }
    ]
    
    # Process each document
    for doc in sample_docs:
        print(f"📄 Processing: {doc['filename']}")
        
        # Split content into chunks (simple chunking for demo)
        content = doc['content'].strip()
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        
        # Generate dummy embeddings for each chunk (ChromaDB will handle real embeddings)
        embeddings_list = []
        for chunk in chunks:
            # Create dummy embedding (ChromaDB will replace with real embeddings)
            dummy_embedding = np.random.rand(384).tolist()  # 384 is a common embedding dimension
            embeddings_list.append(dummy_embedding)
        
        # Store in database
        try:
            chunk_ids = db.store(
                filename=doc['filename'],
                filepath=doc['filepath'],
                chunks=chunks,
                embeddings=embeddings_list,
                metadata=doc['metadata']
            )
            print(f"✅ Stored {len(chunks)} chunks for {doc['filename']}")
        except Exception as e:
            print(f"❌ Failed to store {doc['filename']}: {e}")
    
    print("\n🎉 Sample data added successfully!")
    print("Now you can test multi-query search with real data!")

if __name__ == "__main__":
    add_sample_data()
