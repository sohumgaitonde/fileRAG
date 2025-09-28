"""
INDEXING: Vector embeddings generation using Qwen 2B VL.

This module handles:
- Text embedding generation using Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
- Batch processing for efficiency
- Embedding normalization
- Model management with caching
"""

from typing import List, Optional, Union
import numpy as np
import torch
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False


class EmbeddingGenerator:
    """Generates vector embeddings using Qwen 2B VL model."""
    
    def __init__(
        self, 
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        print(f"🧠 Initializing EmbeddingGenerator with {model_name}")
        print(f"🖥️  Using device: {self.device}")
    
    def load_model(self):
        """Load the Qwen 2B VL embedding model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers torch")
        
        if self._model_loaded:
            print("✅ Model already loaded")
            return
        
        try:
            print(f"📥 Loading model: {self.model_name}")
            print("⚠️  This may take a while on first run (downloading ~4GB model)...")
            
            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # Load model
            print("🤖 Loading model...")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=self.cache_dir
            )
            
            self._model_loaded = True
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        if not self._model_loaded:
            self.load_model()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            print(f"🔢 Generating embeddings for {len(texts)} text(s)...")
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                print("⚠️  No valid texts to process")
                return np.array([])
            
            # Generate embeddings using the model's custom method
            with torch.no_grad():
                embeddings = self.model.get_text_embeddings(valid_texts)
            
            # Convert to numpy array
            if hasattr(embeddings, 'cpu') and hasattr(embeddings, 'numpy'):
                # It's a torch tensor
                embeddings = embeddings.cpu().numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            print(f"✅ Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            1D numpy array embedding
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings in batches for efficiency.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress
            
        Returns:
            numpy array of all embeddings
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            if show_progress:
                print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            batch_embeddings = self.generate_embeddings(batch)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        result = np.vstack(all_embeddings) if all_embeddings else np.array([])
        print(f"✅ Generated {len(result)} total embeddings")
        return result
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for better similarity search.
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            L2-normalized embeddings
        """
        if embeddings.size == 0:
            return embeddings
        
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms
        
        print(f"✅ Normalized {len(normalized)} embeddings")
        return normalized
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.
        
        Returns:
            Embedding dimension
        """
        if not self._model_loaded:
            self.load_model()
        
        # Generate a test embedding to get dimension
        test_embedding = self.generate_embeddings(["test"])
        return test_embedding.shape[1] if len(test_embedding) > 0 else 0
    
    def clear_cache(self):
        """Clear model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._model_loaded = False
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("🗑️  Model cache cleared")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model_loaded,
            "embedding_dim": self.get_embedding_dimension() if self._model_loaded else None,
            "cache_dir": self.cache_dir
        }
