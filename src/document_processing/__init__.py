"""
Document Processing Module
Handles document parsing and chunking for the RAG pipeline.
"""

from .parsers import (
    DocumentParser,
    PDFParser,
    DocxParser,
    HTMLParser,
    TextParser,
    AIParser,
    parse_document,
    parse_directory
)
from .chunking import (
    ChunkingStrategy,
    FixedSizeChunker,
    SemanticChunker,
    RecursiveChunker,
    SentenceChunker,
    chunk_documents
)

__all__ = [
    # Parsers
    "DocumentParser",
    "PDFParser",
    "DocxParser",
    "HTMLParser",
    "TextParser",
    "AIParser",
    "parse_document",
    "parse_directory",
    # Chunking
    "ChunkingStrategy",
    "FixedSizeChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "chunk_documents",
]
