"""
Multi-Query Search functionality for FileRAG.

This module handles:
- Searching multiple query variations
- Collecting and ranking results globally
- Performance monitoring and logging
"""

import time
from typing import List, Dict, Any, Optional
import logging
from .db import VectorDatabase
from .query_generator import QueryGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiQuerySearch:
    """Handles multi-query search with global result ranking."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "fileRAG"):
        self.db = VectorDatabase(db_path, collection_name)
        self.query_generator = QueryGenerator(use_fallback=True)
        self._initialize_db()
    
    def _initialize_db(self) -> bool:
        """Initialize the database connection."""
        try:
            success = self.db.initialize()
            if success:
                logger.info("Database initialized successfully")
            else:
                logger.error("Failed to initialize database")
            return success
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def search_multiple_queries(
        self, 
        user_query: str, 
        n_results_per_query: int = 10,
        global_limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search using multiple query variations and return globally ranked results.
        
        Args:
            user_query: Original user query
            n_results_per_query: Number of results per query variation
            global_limit: Maximum total results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate query variations
            logger.info(f"Generating query variations for: '{user_query}'")
            query_variations = self.query_generator.generate_query_variations(user_query)
            logger.info(f"Generated {len(query_variations)} query variations")
            
            # Step 2: Execute all queries
            all_results = []
            query_timings = []
            
            for i, query in enumerate(query_variations, 1):
                query_start = time.time()
                
                try:
                    logger.info(f"Executing query {i}/{len(query_variations)}: '{query}'")
                    results = self.db.search(
                        query_text=query,
                        n_results=n_results_per_query
                    )
                    
                    # Add query metadata to results
                    for j, doc in enumerate(results.get("documents", [])):
                        result_item = {
                            "document": doc,
                            "metadata": results.get("metadatas", [])[j] if j < len(results.get("metadatas", [])) else {},
                            "distance": results.get("distances", [])[j] if j < len(results.get("distances", [])) else 1.0,
                            "query": query,
                            "query_index": i,
                            "result_index": j
                        }
                        all_results.append(result_item)
                    
                    query_time = time.time() - query_start
                    query_timings.append({
                        "query": query,
                        "time": query_time,
                        "results_count": results.get("count", 0)
                    })
                    
                    logger.info(f"Query {i} completed in {query_time:.2f}s with {results.get('count', 0)} results")
                    
                except Exception as e:
                    logger.error(f"Query {i} failed: {e}")
                    query_timings.append({
                        "query": query,
                        "time": 0,
                        "results_count": 0,
                        "error": str(e)
                    })
            
            # Step 3: Global ranking and deduplication
            logger.info("Ranking and deduplicating results...")
            ranked_results = self._rank_and_deduplicate_results(all_results, global_limit)
            
            # Step 4: Calculate performance metrics and quality scores
            total_time = time.time() - start_time
            quality_metrics = self._calculate_quality_metrics(ranked_results)
            
            performance_metrics = {
                "total_time": total_time,
                "query_timings": query_timings,
                "total_queries": len(query_variations),
                "total_results_before_dedup": len(all_results),
                "final_results_count": len(ranked_results),
                "avg_query_time": sum(t["time"] for t in query_timings) / len(query_timings) if query_timings else 0,
                "quality_metrics": quality_metrics
            }
            
            logger.info(f"Multi-query search completed in {total_time:.2f}s")
            logger.info(f"Found {len(ranked_results)} unique results from {len(all_results)} total results")
            
            return {
                "results": ranked_results,
                "query_variations": query_variations,
                "performance_metrics": performance_metrics,
                "user_query": user_query
            }
            
        except Exception as e:
            logger.error(f"Multi-query search failed: {e}")
            return {
                "results": [],
                "query_variations": [],
                "performance_metrics": {"error": str(e)},
                "user_query": user_query
            }
    
    def _rank_and_deduplicate_results(self, all_results: List[Dict], limit: int) -> List[Dict]:
        """
        Rank and deduplicate results globally by weighted similarity score.
        
        Args:
            all_results: List of all search results
            limit: Maximum number of results to return
            
        Returns:
            List of ranked and deduplicated results with enhanced metadata
        """
        # Group results by document ID (filename + chunk_index)
        doc_groups = {}
        
        for result in all_results:
            metadata = result.get("metadata", {})
            doc_id = f"{metadata.get('filename', 'unknown')}_{metadata.get('chunk_index', 0)}"
            
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # For each document, keep the best result and calculate weighted score
        best_results = []
        
        for doc_id, results in doc_groups.items():
            # Sort by distance (lower is better)
            best_result = min(results, key=lambda x: x.get("distance", 1.0))
            
            # Calculate weighted score with query importance
            base_score = 1.0 - best_result.get("distance", 1.0)
            query_importance = self._calculate_query_importance(best_result.get("query", ""))
            weighted_score = base_score * query_importance
            
            # Add enhanced metadata
            best_result["weighted_score"] = weighted_score
            best_result["base_score"] = base_score
            best_result["query_importance"] = query_importance
            best_result["found_by_queries"] = [r.get("query", "") for r in results]
            best_result["total_matches"] = len(results)
            
            best_results.append(best_result)
        
        # Sort all results by weighted score
        best_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        # Return top results
        return best_results[:limit]
    
    def _calculate_query_importance(self, query: str) -> float:
        """
        Calculate query importance weight based on query characteristics.
        
        Args:
            query: The query string
            
        Returns:
            Importance weight (0.5 to 1.5)
        """
        # Base importance
        importance = 1.0
        
        # Boost for specific query types
        if any(word in query.lower() for word in ["best", "top", "popular", "effective"]):
            importance += 0.2  # Boost for quality-focused queries
        
        if any(word in query.lower() for word in ["how", "what", "why", "when", "where"]):
            importance += 0.1  # Boost for question queries
        
        if any(word in query.lower() for word in ["tutorial", "guide", "learn", "beginner"]):
            importance += 0.15  # Boost for educational queries
        
        # Penalty for very long queries (might be too specific)
        if len(query.split()) > 8:
            importance -= 0.1
        
        # Ensure importance is within reasonable bounds
        return max(0.5, min(1.5, importance))
    
    def _calculate_quality_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate quality metrics for the search results.
        
        Args:
            results: List of ranked search results
            
        Returns:
            Dictionary with quality metrics
        """
        if not results:
            return {
                "avg_score": 0.0,
                "score_distribution": "no_results",
                "diversity_score": 0.0,
                "coverage_score": 0.0
            }
        
        # Calculate average weighted score
        scores = [r.get("weighted_score", 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate score distribution
        high_quality = len([s for s in scores if s > 0.7])
        medium_quality = len([s for s in scores if 0.4 <= s <= 0.7])
        low_quality = len([s for s in scores if s < 0.4])
        
        # Calculate diversity (unique documents)
        unique_docs = len(set(r.get("metadata", {}).get("filename", "") for r in results))
        diversity_score = unique_docs / len(results) if results else 0.0
        
        # Calculate coverage (how many queries found results)
        all_queries = set()
        for result in results:
            all_queries.update(result.get("found_by_queries", []))
        coverage_score = len(all_queries) / 6.0  # 6 is the expected number of queries
        
        return {
            "avg_score": avg_score,
            "score_distribution": {
                "high_quality": high_quality,
                "medium_quality": medium_quality,
                "low_quality": low_quality
            },
            "diversity_score": diversity_score,
            "coverage_score": coverage_score,
            "unique_documents": unique_docs,
            "total_queries_used": len(all_queries)
        }
    
    def test_multi_query_search(self, test_queries: List[str] = None) -> None:
        """Test the multi-query search functionality."""
        if test_queries is None:
            test_queries = [
                "machine learning algorithms",
                "python web development",
                "database optimization techniques"
            ]
        
        logger.info("🧪 Testing Multi-Query Search")
        logger.info("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{i}. Testing query: '{query}'")
            logger.info("-" * 40)
            
            try:
                results = self.search_multiple_queries(query, n_results_per_query=5, global_limit=10)
                
                logger.info(f"✅ Found {len(results['results'])} results")
                logger.info(f"⏱️  Total time: {results['performance_metrics']['total_time']:.2f}s")
                logger.info(f"📊 Avg query time: {results['performance_metrics']['avg_query_time']:.2f}s")
                
                # Show top 3 results
                for j, result in enumerate(results['results'][:3], 1):
                    doc = result.get('document', 'No content')
                    score = 1.0 - result.get('distance', 1.0)
                    logger.info(f"   {j}. Score: {score:.3f} | {doc[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Test failed: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 Multi-query search testing complete!")


def main():
    """Test the multi-query search functionality."""
    search = MultiQuerySearch()
    search.test_multi_query_search()


if __name__ == "__main__":
    main()
