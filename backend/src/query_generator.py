"""
QUERYING: Query generation using SLM via Ollama.

This module handles:
- Natural language query processing
- Query expansion and refinement
- Context-aware query generation
- Integration with Ollama SLM models
"""

import requests
import json
import time
import os
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAISLMProvider:
    """OpenAI-based SLM provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = 30
        
        if not self.api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
    
    def generate(self, prompt: str, model: str = None) -> str:
        """Generate response using OpenAI API."""
        model = model or "gpt-3.5-turbo"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.2,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.0
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")


class QueryGenerator:
    """Generates and refines search queries using OpenAI with local fallback."""
    
    def __init__(self, api_key: str = None, model_name: str = None, use_fallback: bool = True):
        self.provider = OpenAISLMProvider(api_key)
        self.model_name = model_name
        self.timeout = 30
        self.use_fallback = use_fallback
    
    def _test_connection(self) -> bool:
        """Test connection to OpenAI."""
        try:
            # Test with a simple prompt
            test_response = self.provider.generate("Hello", self.model_name)
            if test_response:
                logger.info("Connected to OpenAI")
                return True
            else:
                logger.warning("OpenAI returned empty response")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            return False
    
    def _create_query_variation_prompt(self, user_query: str) -> str:
        """Create a prompt for generating 6 very similar query variations."""
        prompt = f"""Generate 6 very similar search queries for: "{user_query}"

Make them almost identical - use the same words with minor changes. Include the original query as one of the 6.

1. [variation]
2. [variation]
3. [variation]
4. [variation]
5. [variation]
6. [variation]"""
        return prompt
    
    def generate_query_variations(self, user_query: str) -> List[str]:
        """Generate 6 different query variations using OpenAI with local fallback."""
        try:
            start_time = time.time()
            
            # Try OpenAI first
            if self.provider.api_key:
                prompt = self._create_query_variation_prompt(user_query)
                response = self.provider.generate(prompt, self.model_name)
                queries = self._parse_query_variations(response)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Generated {len(queries)} query variations in {elapsed_time:.2f}s using OpenAI")
                return queries
            else:
                raise Exception("No OpenAI API key provided")
                
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")
            
            if self.use_fallback:
                logger.info("Falling back to fast local generation...")
                return self._generate_fast_local_variations(user_query)
            else:
                # Return original query 6 times
                return [user_query] * 6
    
    def _generate_fast_local_variations(self, user_query: str) -> List[str]:
        """Generate fast local variations using rule-based approach."""
        base_query = user_query.strip()
        
        # Create diverse variations using simple rules
        variations = [
            base_query,  # Original
            f"what is {base_query}",  # Question format
            f"how to {base_query}",  # How-to format
            f"best {base_query}",  # Best practices
            f"{base_query} examples",  # Examples
            f"{base_query} tutorial"  # Tutorial format
        ]
        
        # Add some keyword variations for diversity
        keyword_variations = [
            f"{base_query} guide",
            f"{base_query} techniques", 
            f"{base_query} methods",
            f"{base_query} strategies",
            f"{base_query} tips",
            f"{base_query} tools"
        ]
        
        # Mix and match to get 6 unique variations
        all_variations = variations + keyword_variations
        unique_variations = list(dict.fromkeys(all_variations))  # Remove duplicates
        
        return unique_variations[:6]
    
    
    def _parse_query_variations(self, response: str) -> List[str]:
        """Parse the SLM response to extract 6 query variations."""
        queries = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered lines (1., 2., 3., etc.)
            if line and (line[0].isdigit() and '.' in line):
                # Extract the query after the number
                query = line.split('.', 1)[1].strip()
                if query:
                    queries.append(query)
        
        # If we didn't get exactly 6, pad or truncate
        if len(queries) < 6:
            # Pad with the first query repeated
            while len(queries) < 6:
                queries.append(queries[0] if queries else "search query")
        elif len(queries) > 6:
            # Truncate to 6
            queries = queries[:6]
        
        return queries
    
