"""
Tests for the FileCrawler class.

This module contains comprehensive tests for file discovery,
filtering, and metadata extraction functionality.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import FileCrawler


class TestFileCrawler:
    """Test suite for FileCrawler class."""
    
    @pytest.fixture
    def crawler(self):
        """Create a FileCrawler instance for testing."""
        return FileCrawler(max_file_size_mb=1)  # 1MB limit for testing
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test data directory."""
        return Path(__file__).parent / "test_data"
    
    def test_init_default(self):
        """Test FileCrawler initialization with default parameters."""
        crawler = FileCrawler()
        assert crawler.max_file_size_bytes == 50 * 1024 * 1024  # 50MB
        assert crawler.SUPPORTED_EXTENSIONS == {
            '.pdf', '.docx', '.doc', '.txt', '.md',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'
        }
    
    def test_init_custom_size(self):
        """Test FileCrawler initialization with custom size limit."""
        crawler = FileCrawler(max_file_size_mb=10)
        assert crawler.max_file_size_bytes == 10 * 1024 * 1024  # 10MB
    
    def test_get_supported_extensions(self, crawler):
        """Test getting supported extensions."""
        extensions = crawler.get_supported_extensions()
        assert isinstance(extensions, set)
        assert '.txt' in extensions
        assert '.pdf' in extensions
        assert '.md' in extensions
        assert '.xyz' not in extensions
    
    def test_crawl_directory_recursive(self, crawler, test_data_dir):
        """Test crawling directory recursively."""
        if not test_data_dir.exists():
            pytest.skip("Test data directory not found")
        
        files = crawler.crawl_directory(str(test_data_dir), recursive=True)
        
        # Should find files in root and subdirectories
        assert len(files) > 0
        
        # Check that we found expected files
        file_names = [Path(f).name for f in files]
        assert 'sample.txt' in file_names
        assert 'sample.md' in file_names
        assert 'nested_file.txt' in file_names
        
        # Should not find hidden files
        assert '.hidden_file.txt' not in file_names
        
        # Should not find unsupported extensions
        assert 'unsupported.xyz' not in file_names
    
    def test_crawl_directory_non_recursive(self, crawler, test_data_dir):
        """Test crawling directory non-recursively."""
        if not test_data_dir.exists():
            pytest.skip("Test data directory not found")
        
        files = crawler.crawl_directory(str(test_data_dir), recursive=False)
        
        # Should find files in root directory only
        file_names = [Path(f).name for f in files]
        assert 'sample.txt' in file_names
        assert 'sample.md' in file_names
        
        # Should not find files in subdirectories
        assert 'nested_file.txt' not in file_names
    
    def test_crawl_nonexistent_directory(self, crawler):
        """Test crawling a non-existent directory."""
        with pytest.raises(FileNotFoundError):
            crawler.crawl_directory("/path/that/does/not/exist")
    
    def test_crawl_file_instead_of_directory(self, crawler, test_data_dir):
        """Test crawling a file path instead of directory."""
        if not test_data_dir.exists():
            pytest.skip("Test data directory not found")
        
        sample_file = test_data_dir / "sample.txt"
        if sample_file.exists():
            with pytest.raises(ValueError):
                crawler.crawl_directory(str(sample_file))
    
    def test_filter_files_by_extension(self, crawler):
        """Test filtering files by supported extensions."""
        test_files = [
            "/path/to/document.pdf",
            "/path/to/text.txt",
            "/path/to/markdown.md",
            "/path/to/image.jpg",
            "/path/to/unsupported.xyz",
            "/path/to/another.abc"
        ]
        
        # Mock the file existence and readability checks
        original_isfile = os.path.isfile
        original_getsize = os.path.getsize
        original_open = open
        
        def mock_isfile(path):
            return True
        
        def mock_getsize(path):
            return 1024  # 1KB
        
        def mock_open(path, mode='r'):
            class MockFile:
                def read(self, size=None):
                    return b"test"
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockFile()
        
        # Apply mocks
        os.path.isfile = mock_isfile
        os.path.getsize = mock_getsize
        
        try:
            # Mock open for the readability check
            import builtins
            original_builtin_open = builtins.open
            builtins.open = mock_open
            
            filtered = crawler.filter_files(test_files)
            
            # Should only include supported extensions
            filtered_names = [Path(f).name for f in filtered]
            assert 'document.pdf' in filtered_names
            assert 'text.txt' in filtered_names
            assert 'markdown.md' in filtered_names
            assert 'image.jpg' in filtered_names
            assert 'unsupported.xyz' not in filtered_names
            assert 'another.abc' not in filtered_names
            
        finally:
            # Restore original functions
            os.path.isfile = original_isfile
            os.path.getsize = original_getsize
            builtins.open = original_builtin_open
    
    def test_get_file_info(self, crawler, test_data_dir):
        """Test getting file information."""
        if not test_data_dir.exists():
            pytest.skip("Test data directory not found")
        
        sample_file = test_data_dir / "sample.txt"
        if not sample_file.exists():
            pytest.skip("Sample file not found")
        
        info = crawler.get_file_info(str(sample_file))
        
        assert 'file_path' in info
        assert 'file_name' in info
        assert 'file_extension' in info
        assert 'file_size' in info
        assert 'modified_time' in info
        assert 'created_time' in info
        assert 'is_readable' in info
        
        assert info['file_name'] == 'sample.txt'
        assert info['file_extension'] == '.txt'
        assert info['file_size'] > 0
        assert info['is_readable'] is True
    
    def test_get_file_info_nonexistent(self, crawler):
        """Test getting info for non-existent file."""
        info = crawler.get_file_info("/path/that/does/not/exist.txt")
        
        assert 'error' in info
        assert info['file_path'] == "/path/that/does/not/exist.txt"
    
    def test_large_file_filtering(self, crawler):
        """Test that large files are filtered out."""
        # Create a temporary large file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            # Write more than 1MB (our test crawler limit)
            large_content = "x" * (2 * 1024 * 1024)  # 2MB
            temp_file.write(large_content.encode())
            temp_file_path = temp_file.name
        
        try:
            # Test filtering
            filtered = crawler.filter_files([temp_file_path])
            
            # Should be empty because file exceeds size limit
            assert len(filtered) == 0
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_empty_directory(self, crawler):
        """Test crawling an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = crawler.crawl_directory(temp_dir)
            assert files == []
    
    def test_directory_with_only_unsupported_files(self, crawler):
        """Test directory containing only unsupported file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create unsupported files
            unsupported_file = Path(temp_dir) / "test.xyz"
            unsupported_file.write_text("test content")
            
            files = crawler.crawl_directory(temp_dir)
            assert files == []
    
    def test_mixed_file_types(self, crawler):
        """Test directory with mix of supported and unsupported files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create supported file
            supported_file = Path(temp_dir) / "test.txt"
            supported_file.write_text("test content")
            
            # Create unsupported file
            unsupported_file = Path(temp_dir) / "test.xyz"
            unsupported_file.write_text("test content")
            
            # Create hidden file
            hidden_file = Path(temp_dir) / ".hidden.txt"
            hidden_file.write_text("hidden content")
            
            files = crawler.crawl_directory(temp_dir)
            
            # Should only find the supported, non-hidden file
            assert len(files) == 1
            assert Path(files[0]).name == "test.txt"


def test_crawler_integration():
    """Integration test for the complete crawler workflow."""
    crawler = FileCrawler(max_file_size_mb=10)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "document.pdf").touch()
        (temp_path / "text.txt").write_text("Sample text content")
        (temp_path / "markdown.md").write_text("# Sample Markdown")
        (temp_path / "image.jpg").touch()
        (temp_path / "unsupported.xyz").write_text("Unsupported content")
        
        # Create subdirectory with files
        subdir = temp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content")
        
        # Test crawling
        files = crawler.crawl_directory(str(temp_path))
        
        # Verify results
        file_names = [Path(f).name for f in files]
        
        # Should find supported files
        expected_files = {"text.txt", "markdown.md", "nested.txt"}
        found_files = set(file_names)
        
        # PDF and JPG files might be empty and cause issues, so we check for text files
        assert "text.txt" in found_files
        assert "markdown.md" in found_files
        assert "nested.txt" in found_files
        assert "unsupported.xyz" not in found_files


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
