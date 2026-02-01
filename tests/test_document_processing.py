"""
Tests for document processing module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.document_processing.parsers import (
    TextParser,
    HTMLParser,
    DocumentParser
)
from src.document_processing.chunking import (
    FixedSizeChunking,
    RecursiveChunking,
    SentenceChunking,
    TextChunk
)


class TestTextParser:
    """Tests for TextParser."""
    
    def test_parse_text_file(self):
        """Test parsing a text file."""
        content = "Hello, this is a test document.\nWith multiple lines."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            
            parser = TextParser()
            result = parser.parse(f.name)
            
            assert content in result
            os.unlink(f.name)
    
    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            f.flush()
            
            parser = TextParser()
            result = parser.parse(f.name)
            
            assert result == ""
            os.unlink(f.name)
    
    def test_parse_unicode_content(self):
        """Test parsing file with unicode content."""
        content = "Hello 世界! Émojis: 🚀 📚"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            f.flush()
            
            parser = TextParser()
            result = parser.parse(f.name)
            
            assert "世界" in result
            assert "🚀" in result
            os.unlink(f.name)


class TestHTMLParser:
    """Tests for HTMLParser."""
    
    def test_parse_basic_html(self):
        """Test parsing basic HTML."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Hello World</h1>
                <p>This is a paragraph.</p>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            f.flush()
            
            parser = HTMLParser()
            result = parser.parse(f.name)
            
            assert "Hello World" in result
            assert "This is a paragraph" in result
            assert "<h1>" not in result  # Tags should be stripped
            os.unlink(f.name)
    
    def test_parse_html_strips_scripts(self):
        """Test that scripts are removed from HTML."""
        html_content = """
        <html>
            <body>
                <p>Content</p>
                <script>alert('malicious');</script>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            f.flush()
            
            parser = HTMLParser()
            result = parser.parse(f.name)
            
            assert "Content" in result
            assert "alert" not in result
            os.unlink(f.name)


class TestFixedSizeChunking:
    """Tests for FixedSizeChunking strategy."""
    
    def test_basic_chunking(self):
        """Test basic fixed-size chunking."""
        text = "This is a test. " * 100  # ~1600 chars
        chunker = FixedSizeChunking(chunk_size=200, chunk_overlap=20)
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        # Check chunks are roughly the right size
        assert all(len(c.content) <= 220 for c in chunks)
    
    def test_small_text(self):
        """Test chunking text smaller than chunk size."""
        text = "Small text."
        chunker = FixedSizeChunking(chunk_size=200, chunk_overlap=20)
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
    
    def test_chunk_metadata(self):
        """Test that chunks have proper metadata."""
        text = "Test content " * 50
        chunker = FixedSizeChunking(chunk_size=100, chunk_overlap=10)
        
        chunks = chunker.chunk(text)
        
        for i, chunk in enumerate(chunks):
            assert 'chunk_index' in chunk.metadata
            assert chunk.metadata['chunk_index'] == i
    
    def test_overlap(self):
        """Test that chunks have overlap."""
        text = "ABCDEFGHIJ" * 50
        chunker = FixedSizeChunking(chunk_size=50, chunk_overlap=10)
        
        chunks = chunker.chunk(text)
        
        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            # The end of chunk 0 should appear at start of chunk 1
            overlap_content = chunks[0].content[-10:]
            assert overlap_content in chunks[1].content


class TestRecursiveChunking:
    """Tests for RecursiveChunking strategy."""
    
    def test_paragraph_splitting(self):
        """Test splitting on paragraph boundaries."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = RecursiveChunking(chunk_size=50, chunk_overlap=0)
        
        chunks = chunker.chunk(text)
        
        # Should split on paragraph boundaries when possible
        assert len(chunks) >= 1
    
    def test_sentence_splitting(self):
        """Test splitting on sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = RecursiveChunking(chunk_size=40, chunk_overlap=0)
        
        chunks = chunker.chunk(text)
        
        # Sentences should not be split in the middle
        for chunk in chunks:
            # Each chunk should end with punctuation or be the last chunk
            content = chunk.content.strip()
            if content:
                assert content[-1] in '.!?' or chunk == chunks[-1]


class TestSentenceChunking:
    """Tests for SentenceChunking strategy."""
    
    def test_sentence_grouping(self):
        """Test grouping sentences."""
        text = "First. Second. Third. Fourth. Fifth."
        chunker = SentenceChunking(sentences_per_chunk=2, overlap_sentences=0)
        
        chunks = chunker.chunk(text)
        
        # Should have 3 chunks: [First, Second], [Third, Fourth], [Fifth]
        assert len(chunks) == 3
    
    def test_sentence_overlap(self):
        """Test sentence overlap."""
        text = "One. Two. Three. Four."
        chunker = SentenceChunking(sentences_per_chunk=2, overlap_sentences=1)
        
        chunks = chunker.chunk(text)
        
        # With overlap, "Two" should appear in both first and second chunk
        if len(chunks) > 1:
            assert "Two" in chunks[0].content
            assert "Two" in chunks[1].content
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SentenceChunking(sentences_per_chunk=3, overlap_sentences=1)
        
        chunks = chunker.chunk("")
        
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0].content == "")


class TestChunkMetadata:
    """Tests for chunk metadata handling."""
    
    def test_metadata_preservation(self):
        """Test that source metadata is preserved in chunks."""
        text = "Test content for chunking."
        source_metadata = {"source": "test.txt", "author": "Test Author"}
        
        chunker = FixedSizeChunking(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(text, metadata=source_metadata)
        
        for chunk in chunks:
            assert chunk.metadata.get('source') == "test.txt"
            assert chunk.metadata.get('author') == "Test Author"
    
    def test_chunk_index_metadata(self):
        """Test that chunk index is added to metadata."""
        text = "Test " * 100
        chunker = FixedSizeChunking(chunk_size=50, chunk_overlap=5)
        
        chunks = chunker.chunk(text)
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_index'] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
