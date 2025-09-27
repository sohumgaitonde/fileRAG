"""
PDF file parser using PyPDF2.

Extracts text content from PDF files while preserving structure and metadata.
"""

import os
from typing import Dict, List, Optional
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


class PDFParser:
    """Parser for PDF files."""
    
    def __init__(self):
        pass
    
    def parse(self, file_path: str) -> dict:
        """Parse PDF file and extract text content."""
        try:
            # Open and read the PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Extract text from all pages
                content = self._extract_text(pdf_reader)
                
                # Extract metadata
                metadata = self.extract_metadata(file_path)
                metadata.update({
                    "file_path": file_path,
                    "file_type": "pdf",
                    "pages": len(pdf_reader.pages),
                    "content_length": len(content),
                    "is_encrypted": pdf_reader.is_encrypted
                })
                
                return {
                    "content": content,
                    "metadata": metadata
                }
                
        except PdfReadError as e:
            return {
                "content": "",
                "metadata": {
                    "file_path": file_path,
                    "file_type": "pdf",
                    "error": f"PDF read error: {str(e)}",
                    "pages": 0,
                    "title": "",
                    "author": ""
                }
            }
        except Exception as e:
            return {
                "content": "",
                "metadata": {
                    "file_path": file_path,
                    "file_type": "pdf",
                    "error": str(e),
                    "pages": 0,
                    "title": "",
                    "author": ""
                }
            }
    
    def _extract_text(self, pdf_reader: PdfReader) -> str:
        """Extract text content from all pages of the PDF."""
        text_parts = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    # Add page separator for better readability
                    text_parts.append(f"[Page {page_num}]\n{page_text.strip()}")
            except Exception as e:
                # If a page fails, add a note and continue
                text_parts.append(f"[Page {page_num} - Error extracting text: {str(e)}]")
        
        return "\n\n".join(text_parts)
    
    def extract_metadata(self, file_path: str) -> dict:
        """Extract metadata from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Get file stats
                file_stats = os.stat(file_path)
                
                # Extract PDF metadata
                metadata = pdf_reader.metadata
                
                # Extract additional information
                page_info = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        page_info.append({
                            "page_number": page_num,
                            "text_length": len(page_text),
                            "has_text": bool(page_text.strip())
                        })
                    except Exception:
                        page_info.append({
                            "page_number": page_num,
                            "text_length": 0,
                            "has_text": False
                        })
                
                return {
                    "title": metadata.get("/Title", "") if metadata else "",
                    "author": metadata.get("/Author", "") if metadata else "",
                    "subject": metadata.get("/Subject", "") if metadata else "",
                    "creator": metadata.get("/Creator", "") if metadata else "",
                    "producer": metadata.get("/Producer", "") if metadata else "",
                    "creation_date": str(metadata.get("/CreationDate", "")) if metadata else "",
                    "modification_date": str(metadata.get("/ModDate", "")) if metadata else "",
                    "keywords": metadata.get("/Keywords", "") if metadata else "",
                    "file_size": file_stats.st_size,
                    "page_info": page_info,
                    "total_pages": len(pdf_reader.pages),
                    "is_encrypted": pdf_reader.is_encrypted
                }
                
        except Exception as e:
            return {
                "title": "",
                "author": "",
                "subject": "",
                "creator": "",
                "producer": "",
                "creation_date": "",
                "modification_date": "",
                "keywords": "",
                "file_size": 0,
                "page_info": [],
                "total_pages": 0,
                "is_encrypted": False,
                "error": str(e)
            }
    
    def extract_text_by_page(self, file_path: str) -> List[Dict[str, str]]:
        """Extract text from each page separately."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                pages = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        pages.append({
                            "page_number": page_num,
                            "content": page_text,
                            "text_length": len(page_text),
                            "has_text": bool(page_text.strip())
                        })
                    except Exception as e:
                        pages.append({
                            "page_number": page_num,
                            "content": "",
                            "text_length": 0,
                            "has_text": False,
                            "error": str(e)
                        })
                
                return pages
                
        except Exception as e:
            return [{
                "page_number": 0,
                "content": "",
                "text_length": 0,
                "has_text": False,
                "error": str(e)
            }]
    
    def get_page_count(self, file_path: str) -> int:
        """Get the number of pages in the PDF."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                return len(pdf_reader.pages)
        except Exception:
            return 0
    
    def is_encrypted(self, file_path: str) -> bool:
        """Check if the PDF is encrypted."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                return pdf_reader.is_encrypted
        except Exception:
            return False
