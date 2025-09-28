"""
Image file parser using BLIP for semantic image understanding and OCR for text extraction.

Extracts semantic descriptions from images using BLIP (Bootstrapping Language-Image Pre-training)
and performs OCR to extract any text content from images for comprehensive RAG systems.
"""

import os
from typing import Dict, Optional, List, Tuple
from PIL import Image


class ImageParser:
    """Parser for image files with BLIP-based semantic understanding and OCR text extraction."""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.ocr_reader = None
        self._model_loaded = False
        self._ocr_loaded = False
    
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
    
    def _load_ocr(self):
        """Lazy load the OCR model to avoid loading on import."""
        if not self._ocr_loaded:
            try:
                import easyocr
                
                print("Loading EasyOCR model for text extraction...")
                # Initialize EasyOCR with English language support
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                self._ocr_loaded = True
                print("EasyOCR model loaded successfully!")
                
            except ImportError as e:
                raise ImportError(
                    "EasyOCR dependencies not installed. Please install with: "
                    "pip install easyocr"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load EasyOCR model: {str(e)}") from e
    
    def parse(self, file_path: str) -> dict:
        """Parse image file and extract both semantic description and OCR text."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load models if not already loaded
            self._load_model()
            self._load_ocr()
            
            # Load and process image
            image = Image.open(file_path)
            
            # Get image metadata
            image_metadata = self._get_image_metadata(image, file_path)
            
            # Generate semantic description using BLIP
            description = self._generate_description(image)
            
            # Extract text using OCR
            ocr_text, ocr_confidence = self._extract_text(image)
            
            # Combine description and OCR text
            content_parts = []
            if description:
                content_parts.append(f"Image Description: {description}")
            if ocr_text:
                content_parts.append(f"Extracted Text: {ocr_text}")
            
            content = "\n\n".join(content_parts) if content_parts else "No content extracted from image"
            
            return {
                "content": content,
                "metadata": {
                    **image_metadata,
                    "semantic_description": description,
                    "ocr_text": ocr_text,
                    "ocr_confidence": ocr_confidence,
                    "models_used": "BLIP-image-captioning-base, EasyOCR"
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
                    "models_used": "none"
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
    
    def _extract_text(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text from image using OCR."""
        try:
            # Convert PIL image to numpy array for EasyOCR
            import numpy as np
            image_array = np.array(image)
            
            # Perform OCR
            results = self.ocr_reader.readtext(image_array)
            
            # Extract text and calculate average confidence
            extracted_texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter out low-confidence detections
                    extracted_texts.append(text.strip())
                    confidences.append(confidence)
            
            # Combine all text
            combined_text = " ".join(extracted_texts)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return combined_text, avg_confidence
            
        except Exception as e:
            return f"Error extracting text: {str(e)}", 0.0
    