"""
INDEXING: File crawler for discovering and monitoring files.

This module handles:
- Directory traversal and file discovery
- File filtering by supported types
- Basic file metadata extraction
"""

import os
from pathlib import Path
from typing import List, Dict


class FileCrawler:
    """Crawls directories to discover files for indexing."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf',
        '.docx', '.doc',
        '.txt',
        '.md',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'
    }
    
    def __init__(self, max_file_size_mb: int = 50):
        """
        Initialize crawler with optional file size limit.
        
        Args:
            max_file_size_mb: Maximum file size in MB to process (default: 50MB)
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def crawl_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """
        Crawl a directory and return list of supported files.
        
        Args:
            directory_path: Path to directory to crawl
            recursive: Whether to crawl subdirectories (default: True)
            
        Returns:
            List of file paths that match supported extensions
        """
        directory_path = os.path.abspath(directory_path)
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        print(f"🔍 Crawling directory: {directory_path}")
        
        all_files = []
        
        if recursive:
            # Recursively walk through all subdirectories
            for root, dirs, files in os.walk(directory_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if not file.startswith('.'):  # Skip hidden files
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
        else:
            # Only process files in the current directory
            try:
                for item in os.listdir(directory_path):
                    if not item.startswith('.'):  # Skip hidden files
                        item_path = os.path.join(directory_path, item)
                        if os.path.isfile(item_path):
                            all_files.append(item_path)
            except PermissionError:
                print(f"⚠️  Permission denied accessing: {directory_path}")
                return []
        
        # Filter files by supported types and other criteria
        filtered_files = self.filter_files(all_files)
        
        print(f"📁 Found {len(all_files)} total files, {len(filtered_files)} supported files")
        
        return filtered_files
    
    def filter_files(self, files: List[str]) -> List[str]:
        """
        Filter files by supported types and other criteria.
        
        Args:
            files: List of file paths to filter
            
        Returns:
            List of filtered file paths
        """
        filtered_files = []
        
        for file_path in files:
            try:
                # Check if file exists and is accessible
                if not os.path.isfile(file_path):
                    continue
                
                # Get file extension
                file_ext = Path(file_path).suffix.lower()
                
                # Check if extension is supported
                if file_ext not in self.SUPPORTED_EXTENSIONS:
                    continue
                
                # Check file size
                file_size = os.path.getsize(file_path)
                if file_size > self.max_file_size_bytes:
                    print(f"⚠️  Skipping large file ({file_size / 1024 / 1024:.1f}MB): {file_path}")
                    continue
                
                # Check if file is readable
                try:
                    with open(file_path, 'rb') as f:
                        f.read(1)  # Try to read one byte
                    filtered_files.append(file_path)
                except (PermissionError, OSError):
                    print(f"⚠️  Cannot read file: {file_path}")
                    continue
                    
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
                continue
        
        return filtered_files
    
    def get_file_info(self, file_path: str) -> Dict:
        """
        Get basic file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata
        """
        try:
            stat_info = os.stat(file_path)
            path_obj = Path(file_path)
            
            return {
                'file_path': file_path,
                'file_name': path_obj.name,
                'file_extension': path_obj.suffix.lower(),
                'file_size': stat_info.st_size,
                'modified_time': stat_info.st_mtime,
                'created_time': stat_info.st_ctime,
                'is_readable': os.access(file_path, os.R_OK)
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e)
            }
    
    def get_supported_extensions(self) -> set:
        """Return set of supported file extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()
    
    def watch_directory(self, directory_path: str):
        """Monitor directory for file changes."""
        # TODO: Implement file watching (for future use)
        pass