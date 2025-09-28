"""
Plain text file parser.

Handles various text file formats with encoding detection and content extraction.
"""

import os
import chardet
from typing import Dict, List, Optional


class TXTParser:
    """Parser for plain text files."""
    
    def __init__(self):
        # Common encodings to try in order of preference
        self.common_encodings = [
            'utf-8',
            'utf-8-sig', 
            'latin-1',
            'cp1252',
            'iso-8859-1',
            'ascii'
        ]
    
    def parse(self, file_path: str) -> dict:
        """Parse text file and extract content."""
        try:
            # Detect encoding
            encoding = self.detect_encoding(file_path)
            
            # Read file with detected encoding
            content = self._read_file_with_encoding(file_path, encoding)
            
            # Get file statistics
            file_stats = os.stat(file_path)
            
            # Analyze content
            content_analysis = self._analyze_content(content)
            
            metadata = {
                "file_path": file_path,
                "file_type": "txt",
                "encoding": encoding,
                "file_size": file_stats.st_size,
                "content_length": len(content),
                **content_analysis
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "content": "",
                "metadata": {
                    "file_path": file_path,
                    "file_type": "txt",
                    "error": str(e),
                    "encoding": "unknown",
                    "lines": 0,
                    "size": 0
                }
            }
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet and fallback methods."""
        try:
            # First, try chardet for automatic detection
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                
                if result['encoding'] and result['confidence'] > 0.7:
                    return result['encoding']
            
            # If chardet fails or confidence is low, try common encodings
            return self._try_common_encodings(file_path)
            
        except Exception:
            # Fallback to utf-8
            return 'utf-8'
    
    def _try_common_encodings(self, file_path: str) -> str:
        """Try common encodings to find one that works."""
        for encoding in self.common_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    file.read(1000)  # Try to read first 1000 characters
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If all fail, return utf-8 as last resort
        return 'utf-8'
    
    def _read_file_with_encoding(self, file_path: str, encoding: str) -> str:
        """Read file with specified encoding, with fallback handling."""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except (UnicodeDecodeError, UnicodeError):
            # If the detected encoding fails, try with error handling
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                    return file.read()
            except Exception:
                # Last resort: read as binary and decode with errors='replace'
                with open(file_path, 'rb') as file:
                    raw_data = file.read()
                    return raw_data.decode('utf-8', errors='replace')
    
    def _analyze_content(self, content: str) -> Dict:
        """Analyze text content for various characteristics."""
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Count different types of characters
        char_counts = {
            'total_chars': len(content),
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'whitespace_lines': len(lines) - len(non_empty_lines),
            'words': len(content.split()) if content.strip() else 0,
            'paragraphs': len([p for p in content.split('\n\n') if p.strip()])
        }
        
        # Detect line endings
        line_endings = self._detect_line_endings(content)
        
        # Detect file type hints
        file_type_hints = self._detect_file_type_hints(content)
        
        return {
            **char_counts,
            "line_endings": line_endings,
            "file_type_hints": file_type_hints,
            "has_bom": content.startswith('\ufeff'),
            "is_binary": self._is_likely_binary(content)
        }
    
    def _detect_line_endings(self, content: str) -> str:
        """Detect the type of line endings used in the file."""
        if '\r\n' in content:
            return 'CRLF (Windows)'
        elif '\r' in content:
            return 'CR (Mac)'
        elif '\n' in content:
            return 'LF (Unix/Linux)'
        else:
            return 'Unknown'
    
    def _detect_file_type_hints(self, content: str) -> List[str]:
        """Detect hints about what type of text content this might be."""
        hints = []
        content_lower = content.lower()
        
        # Simple content type detection for .txt files
        if any(pattern in content_lower for pattern in ['error', 'warning', 'info', 'debug']):
            hints.append('Log file')
        
        if any(pattern in content_lower for pattern in ['config', 'setting', 'option']):
            hints.append('Configuration file')
        
        if any(pattern in content_lower for pattern in ['readme', 'documentation', 'instructions']):
            hints.append('Documentation')
        
        if any(pattern in content_lower for pattern in ['note', 'todo', 'task']):
            hints.append('Notes/Todo list')
        
        return hints
    
    def _is_likely_binary(self, content: str) -> bool:
        """Check if content is likely binary data."""
        # Check for null bytes and other binary indicators
        if '\x00' in content:
            return True
        
        # Check ratio of printable vs non-printable characters
        printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
        total_chars = len(content)
        
        if total_chars > 0:
            printable_ratio = printable_chars / total_chars
            return printable_ratio < 0.7  # Less than 70% printable characters
        
        return False
    
    def get_encoding_info(self, file_path: str) -> Dict:
        """Get detailed encoding information for a file."""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)
                result = chardet.detect(raw_data)
                
                return {
                    "detected_encoding": result.get('encoding', 'unknown'),
                    "confidence": result.get('confidence', 0.0),
                    "language": result.get('language', 'unknown'),
                    "file_size": len(raw_data)
                }
        except Exception as e:
            return {
                "detected_encoding": "unknown",
                "confidence": 0.0,
                "language": "unknown",
                "error": str(e)
            }