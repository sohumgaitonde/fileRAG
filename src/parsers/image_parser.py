"""
Image file parser using BLIP for semantic image understanding.

Extracts semantic descriptions from images using BLIP (Bootstrapping Language-Image Pre-training)
for better semantic search capabilities in RAG systems.
"""

import os
from typing import Dict, Optional
from PIL import Image


class ImageParser:
    """Parser for image files with BLIP-based semantic understanding."""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the BLIP model to avoid loading on import."""
        if not self._model_loaded:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                print("Loading BLIP model for image captioning...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self._model_loaded = True
                print("BLIP model loaded successfully!")
                
            except ImportError as e:
                raise ImportError(
                    "BLIP dependencies not installed. Please install with: "
                    "pip install transformers torch"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load BLIP model: {str(e)}") from e
    
    def parse(self, file_path: str) -> dict:
        """Parse image file and extract semantic description."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load model if not already loaded
            self._load_model()
            
            # Load and process image
            image = Image.open(file_path)
            
            # Get image metadata
            image_metadata = self._get_image_metadata(image, file_path)
            
            # Generate semantic description
            description = self._generate_description(image)
            
            # Combine description with metadata
            content = f"Image Description: {description}"
            
            return {
                "content": content,
                "metadata": {
                    **image_metadata,
                    "semantic_description": description,
                    "model_used": "BLIP-image-captioning-base"
                }
            }
            
        except Exception as e:
            return {
                "content": "",
                "metadata": {
                    "file_path": file_path,
                    "file_type": "image",
                    "error": str(e),
                    "width": 0,
                    "height": 0,
                    "format": "",
                    "model_used": "none"
                }
            }
    
    def _get_image_metadata(self, image: Image.Image, file_path: str) -> Dict:
        """Extract basic image metadata."""
        file_stats = os.stat(file_path)
        
        return {
            "file_path": file_path,
            "file_type": "image",
            "width": image.width,
            "height": image.height,
            "format": image.format or "unknown",
            "mode": image.mode,
            "file_size": file_stats.st_size,
            "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        }
    
    def _generate_description(self, image: Image.Image) -> str:
        """Generate semantic description using BLIP model."""
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Generate description
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
            
            # Decode the generated text
            description = self.processor.decode(out[0], skip_special_tokens=True)
            
            return description.strip()
            
        except Exception as e:
            return f"Error generating description: {str(e)}"
    
