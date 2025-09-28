"""
Markdown file parser.

Extracts text content from Markdown files while preserving structure and metadata.
"""

import os
import re
from typing import Dict, List, Tuple
import markdown
from markdown.extensions import meta


class MDParser:
    """Parser for Markdown files."""
    
    def __init__(self):
        self.md = markdown.Markdown(extensions=['meta', 'tables', 'codehilite', 'fenced_code'])
    
    def parse(self, file_path: str) -> dict:
        """Parse Markdown file and extract content."""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract structure information
            structure = self.extract_structure(content)
            
            # Convert markdown to HTML for better text extraction
            html_content = self.md.convert(content)
            
            # Extract plain text from HTML (remove HTML tags)
            plain_text = self._html_to_text(html_content)
            
            # Get file metadata
            file_stats = os.stat(file_path)
            
            # Get frontmatter metadata if present
            frontmatter_meta = getattr(self.md, 'Meta', {})
            
            metadata = {
                "file_path": file_path,
                "file_type": "md",
                "file_size": file_stats.st_size,
                "content_length": len(plain_text),
                "html_length": len(html_content),
                **structure,
                **self._process_frontmatter(frontmatter_meta)
            }
            
            return {
                "content": plain_text,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "content": "",
                "metadata": {
                    "file_path": file_path,
                    "file_type": "md",
                    "error": str(e),
                    "headings": [],
                    "links": [],
                    "images": []
                }
            }
    
    def extract_structure(self, content: str) -> dict:
        """Extract Markdown structure (headings, links, etc.)."""
        headings = self._extract_headings(content)
        links = self._extract_links(content)
        images = self._extract_images(content)
        code_blocks = self._extract_code_blocks(content)
        tables = self._extract_tables(content)
        
        return {
            "headings": headings,
            "links": links,
            "images": images,
            "code_blocks": code_blocks,
            "tables": tables,
            "heading_count": len(headings),
            "link_count": len(links),
            "image_count": len(images),
            "code_block_count": len(code_blocks),
            "table_count": len(tables)
        }
    
    def _extract_headings(self, content: str) -> List[Dict[str, str]]:
        """Extract all headings with their levels and text."""
        headings = []
        # Match headings with # syntax
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        
        for line in content.split('\n'):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append({
                    "level": level,
                    "text": text,
                    "anchor": self._create_anchor(text)
                })
        
        return headings
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract all links from markdown content."""
        links = []
        # Match markdown links [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        for match in re.finditer(link_pattern, content):
            links.append({
                "text": match.group(1),
                "url": match.group(2)
            })
        
        return links
    
    def _extract_images(self, content: str) -> List[Dict[str, str]]:
        """Extract all images from markdown content."""
        images = []
        # Match markdown images ![alt](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for match in re.finditer(image_pattern, content):
            images.append({
                "alt": match.group(1),
                "url": match.group(2)
            })
        
        return images
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        # Match fenced code blocks
        code_pattern = r'```(\w*)\n(.*?)\n```'
        
        for match in re.finditer(code_pattern, content, re.DOTALL):
            code_blocks.append({
                "language": match.group(1) or "text",
                "code": match.group(2)
            })
        
        return code_blocks
    
    def _extract_tables(self, content: str) -> List[Dict[str, any]]:
        """Extract table information from markdown content."""
        tables = []
        lines = content.split('\n')
        in_table = False
        current_table = []
        
        for line in lines:
            if '|' in line and line.strip():
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line.strip())
            elif in_table:
                # End of table
                if current_table:
                    tables.append(self._parse_table(current_table))
                current_table = []
                in_table = False
        
        # Handle table at end of file
        if in_table and current_table:
            tables.append(self._parse_table(current_table))
        
        return tables
    
    def _parse_table(self, table_lines: List[str]) -> Dict[str, any]:
        """Parse a markdown table into structured data."""
        if len(table_lines) < 2:
            return {"headers": [], "rows": [], "column_count": 0}
        
        # Parse headers (first line)
        headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
        
        # Skip separator line (second line)
        rows = []
        for line in table_lines[2:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells:  # Skip empty rows
                rows.append(cells)
        
        return {
            "headers": headers,
            "rows": rows,
            "column_count": len(headers),
            "row_count": len(rows)
        }
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to plain text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        # Decode HTML entities
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _create_anchor(self, text: str) -> str:
        """Create an anchor link from heading text."""
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        anchor = re.sub(r'[^\w\s-]', '', text.lower())
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')
    
    def _process_frontmatter(self, frontmatter_meta: Dict) -> Dict:
        """Process frontmatter metadata."""
        processed = {}
        for key, value in frontmatter_meta.items():
            if isinstance(value, list) and len(value) == 1:
                processed[key] = value[0]
            else:
                processed[key] = value
        return processed