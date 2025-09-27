"""
Test suite for all parsers in the fileRAG system.

This test suite verifies that each parser correctly extracts content and metadata
from various file types.
"""

import os
import sys
import unittest
from pathlib import Path

# Add src to path so we can import our parsers
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsers import PDFParser, DOCXParser, TXTParser, MDParser, ImageParser


class TestTxtParser(unittest.TestCase):
    """Test the TXT parser with various text files."""
    
    def setUp(self):
        self.parser = TXTParser()
        self.sample_file = Path(__file__).parent / "sample_files" / "sample.txt"
    
    def test_parse_txt_file(self):
        """Test parsing a basic text file."""
        result = self.parser.parse(str(self.sample_file))
        
        # Check basic structure
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        
        # Check content
        self.assertIsInstance(result["content"], str)
        self.assertGreater(len(result["content"]), 0)
        
        # Check metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["file_type"], "txt")
        self.assertEqual(metadata["file_path"], str(self.sample_file))
        self.assertIn("encoding", metadata)
        self.assertGreater(metadata["total_lines"], 0)
        self.assertGreater(metadata["words"], 0)
    
    def test_encoding_detection(self):
        """Test encoding detection functionality."""
        encoding = self.parser.detect_encoding(str(self.sample_file))
        self.assertIsInstance(encoding, str)
        self.assertGreater(len(encoding), 0)
    
    def test_encoding_info(self):
        """Test detailed encoding information."""
        info = self.parser.get_encoding_info(str(self.sample_file))
        self.assertIn("detected_encoding", info)
        self.assertIn("confidence", info)
        self.assertIsInstance(info["confidence"], float)


class TestMdParser(unittest.TestCase):
    """Test the Markdown parser."""
    
    def setUp(self):
        self.parser = MDParser()
        self.sample_file = Path(__file__).parent / "sample_files" / "sample.md"
    
    def test_parse_md_file(self):
        """Test parsing a markdown file."""
        result = self.parser.parse(str(self.sample_file))
        
        # Check basic structure
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        
        # Check content
        self.assertIsInstance(result["content"], str)
        self.assertGreater(len(result["content"]), 0)
        
        # Check metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["file_type"], "md")
        self.assertEqual(metadata["file_path"], str(self.sample_file))
        
        # Check structure extraction
        self.assertIn("headings", metadata)
        self.assertIn("links", metadata)
        self.assertIn("images", metadata)
        self.assertIn("code_blocks", metadata)
        self.assertIn("tables", metadata)
        
        # Verify extracted elements
        self.assertGreater(len(metadata["headings"]), 0)
        self.assertGreater(len(metadata["links"]), 0)
        self.assertGreater(len(metadata["images"]), 0)
        self.assertGreater(len(metadata["code_blocks"]), 0)
        self.assertGreater(len(metadata["tables"]), 0)
    
    def test_structure_extraction(self):
        """Test markdown structure extraction."""
        with open(self.sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        structure = self.parser.extract_structure(content)
        
        # Check headings
        headings = structure["headings"]
        self.assertGreater(len(headings), 0)
        self.assertIn("level", headings[0])
        self.assertIn("text", headings[0])
        self.assertIn("anchor", headings[0])
        
        # Check links
        links = structure["links"]
        self.assertGreater(len(links), 0)
        self.assertIn("text", links[0])
        self.assertIn("url", links[0])
        
        # Check images
        images = structure["images"]
        self.assertGreater(len(images), 0)
        self.assertIn("alt", images[0])
        self.assertIn("url", images[0])
        
        # Check code blocks
        code_blocks = structure["code_blocks"]
        self.assertGreater(len(code_blocks), 0)
        self.assertIn("language", code_blocks[0])
        self.assertIn("code", code_blocks[0])
        
        # Check tables
        tables = structure["tables"]
        self.assertGreater(len(tables), 0)
        self.assertIn("headers", tables[0])
        self.assertIn("rows", tables[0])


class TestPdfParser(unittest.TestCase):
    """Test the PDF parser."""
    
    def setUp(self):
        self.parser = PDFParser()
        # Note: We'll create a simple PDF for testing if needed
    
    def test_parser_initialization(self):
        """Test that the parser initializes correctly."""
        self.assertIsNotNone(self.parser)
    
    def test_page_count_method(self):
        """Test the get_page_count method with non-existent file."""
        # Test with non-existent file
        count = self.parser.get_page_count("non_existent.pdf")
        self.assertEqual(count, 0)
    
    def test_encryption_detection(self):
        """Test the is_encrypted method with non-existent file."""
        # Test with non-existent file
        is_encrypted = self.parser.is_encrypted("non_existent.pdf")
        self.assertFalse(is_encrypted)


class TestDocxParser(unittest.TestCase):
    """Test the DOCX parser."""
    
    def setUp(self):
        self.parser = DOCXParser()
    
    def test_parser_initialization(self):
        """Test that the parser initializes correctly."""
        self.assertIsNotNone(self.parser)
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        result = self.parser.parse("non_existent.docx")
        
        # Should return error in metadata
        self.assertIn("error", result["metadata"])
        self.assertEqual(result["content"], "")


class TestImageParser(unittest.TestCase):
    """Test the Image parser."""
    
    def setUp(self):
        self.parser = ImageParser()
    
    def test_parser_initialization(self):
        """Test that the parser initializes correctly."""
        self.assertIsNotNone(self.parser)
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        result = self.parser.parse("non_existent.jpg")
        
        # Should return error in metadata
        self.assertIn("error", result["metadata"])
        self.assertEqual(result["content"], "")


class TestParserIntegration(unittest.TestCase):
    """Integration tests for all parsers."""
    
    def test_all_parsers_importable(self):
        """Test that all parsers can be imported."""
        parsers = [PDFParser, DOCXParser, TXTParser, MDParser, ImageParser]
        
        for parser_class in parsers:
            with self.subTest(parser=parser_class.__name__):
                parser = parser_class()
                self.assertIsNotNone(parser)
    
    def test_parser_interface_consistency(self):
        """Test that all parsers have the same interface."""
        parsers = [
            PDFParser(),
            DOCXParser(),
            TXTParser(),
            MDParser(),
            ImageParser()
        ]
        
        for parser in parsers:
            with self.subTest(parser=parser.__class__.__name__):
                # All parsers should have a parse method
                self.assertTrue(hasattr(parser, 'parse'))
                self.assertTrue(callable(getattr(parser, 'parse')))


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTxtParser,
        TestMdParser,
        TestPdfParser,
        TestDocxParser,
        TestImageParser,
        TestParserIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running FileRAG Parser Test Suite")
    print("=" * 50)
    
    result = run_tests()
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    sys.exit(0 if result.wasSuccessful() else 1)
