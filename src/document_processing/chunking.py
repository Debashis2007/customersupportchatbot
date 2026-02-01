"""
Document Chunking Strategies
Implements various chunking strategies for RAG systems:
- Fixed-size chunking
- Semantic chunking
- Recursive chunking
- Sentence-based chunking
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    start_index: int = 0
    end_index: int = 0
    
    def __len__(self):
        return len(self.content)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-Size Chunking Strategy
    
    Splits documents into chunks of fixed character/token size with overlap.
    Simple but effective for uniform content.
    
    Pros:
    - Predictable chunk sizes
    - Easy to implement
    - Good for dense, uniform text
    
    Cons:
    - May split sentences/paragraphs mid-way
    - Doesn't respect document structure
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        separator: str = " "
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separator = separator
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to find a good break point
            if end < len(text):
                # Look for separator near the end
                break_point = text.rfind(self.separator, start + self.chunk_size // 2, end)
                if break_point != -1:
                    end = break_point + 1
            else:
                end = len(text)
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk = Chunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "fixed_size",
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    },
                    chunk_id=f"{metadata.get('source', 'doc')}_{chunk_index}",
                    start_index=start,
                    end_index=end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break
        
        logger.debug(f"Created {len(chunks)} fixed-size chunks")
        return chunks


class SentenceChunker(ChunkingStrategy):
    """
    Sentence-Based Chunking Strategy
    
    Chunks text by grouping complete sentences together.
    Respects sentence boundaries for better semantic coherence.
    
    Pros:
    - Maintains sentence integrity
    - Better for QA systems
    - Natural breakpoints
    
    Cons:
    - Variable chunk sizes
    - May create very small or large chunks
    """
    
    def __init__(
        self,
        sentences_per_chunk: int = 5,
        sentence_overlap: int = 1,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Sentence splitting pattern
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+'
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use regex for sentence splitting
        sentences = self.sentence_pattern.split(text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into sentence-based chunks."""
        if not text:
            return []
        
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [Chunk(
                content=text,
                metadata={**metadata, "chunking_strategy": "sentence"},
                chunk_id=f"{metadata.get('source', 'doc')}_0"
            )]
        
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(sentences):
            # Group sentences
            end_idx = min(i + self.sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            chunk_content = " ".join(chunk_sentences)
            
            # Handle minimum size - add more sentences if needed
            while len(chunk_content) < self.min_chunk_size and end_idx < len(sentences):
                end_idx += 1
                chunk_sentences = sentences[i:end_idx]
                chunk_content = " ".join(chunk_sentences)
            
            # Handle maximum size - split if needed
            if len(chunk_content) > self.max_chunk_size:
                # Use fixed-size chunker as fallback
                fallback_chunker = FixedSizeChunker(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=200
                )
                sub_chunks = fallback_chunker.chunk(chunk_content, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["chunking_strategy"] = "sentence_fallback"
                    sub_chunk.chunk_id = f"{metadata.get('source', 'doc')}_{chunk_index}"
                    chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                chunk = Chunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "sentence",
                        "num_sentences": len(chunk_sentences),
                    },
                    chunk_id=f"{metadata.get('source', 'doc')}_{chunk_index}"
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next group with overlap
            i = end_idx - self.sentence_overlap
            if i <= 0 or i >= len(sentences) - self.sentence_overlap:
                i = end_idx
        
        logger.debug(f"Created {len(chunks)} sentence-based chunks")
        return chunks


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive Character Text Splitting
    
    Recursively splits text using a hierarchy of separators.
    Tries to keep semantically related content together.
    
    Separator hierarchy (default):
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Spaces
    5. Characters
    
    Pros:
    - Respects document structure
    - Adapts to content
    - Good balance of size and coherence
    
    Cons:
    - More complex implementation
    - May still split mid-sentence in edge cases
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",     # Paragraphs
            "\n",       # Lines
            ". ",       # Sentences
            "! ",       # Exclamations
            "? ",       # Questions
            "; ",       # Clauses
            ", ",       # Phrases
            " ",        # Words
            ""          # Characters
        ]
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level splitting
            return list(text)
        
        splits = text.split(separator)
        
        # Rejoin separator with content (except for whitespace separators)
        if separator.strip():
            splits = [s + separator if i < len(splits) - 1 else s 
                     for i, s in enumerate(splits)]
        
        # Check if any split is still too large
        final_splits = []
        for split in splits:
            if len(split) <= self.chunk_size:
                if split.strip():
                    final_splits.append(split)
            else:
                # Recursively split with next separator
                sub_splits = self._split_text(split, remaining_separators)
                final_splits.extend(sub_splits)
        
        return final_splits
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits into chunks respecting chunk_size."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            if current_length + split_length <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_length
            else:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [split]
                current_length = split_length
        
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text using recursive strategy."""
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Split text recursively
        splits = self._split_text(text, self.separators)
        
        # Merge splits into appropriately sized chunks
        merged = self._merge_splits(splits)
        
        # Create Chunk objects with overlap
        chunks = []
        for i, chunk_content in enumerate(merged):
            chunk_content = chunk_content.strip()
            if chunk_content:
                chunk = Chunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "chunking_strategy": "recursive",
                    },
                    chunk_id=f"{metadata.get('source', 'doc')}_{i}"
                )
                chunks.append(chunk)
        
        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        logger.debug(f"Created {len(chunks)} recursive chunks")
        return chunks
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add overlap between consecutive chunks."""
        for i in range(1, len(chunks)):
            prev_content = chunks[i-1].content
            if len(prev_content) > self.chunk_overlap:
                overlap_text = prev_content[-self.chunk_overlap:]
                # Find word boundary
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx+1:]
                chunks[i].content = overlap_text + " " + chunks[i].content
                chunks[i].metadata["has_overlap"] = True
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic Chunking Strategy
    
    Uses embeddings to identify semantic boundaries.
    Chunks text where there are significant semantic shifts.
    
    Pros:
    - Maintains semantic coherence
    - Adapts to content meaning
    - Ideal for diverse content types
    
    Cons:
    - Requires embedding model
    - Computationally expensive
    - Variable chunk sizes
    """
    
    def __init__(
        self,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.8,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000,
        buffer_size: int = 3
    ):
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_size = buffer_size  # Number of sentences to consider for similarity
    
    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text based on semantic boundaries."""
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Fallback to recursive chunking if no embedding function
        if not self.embedding_function:
            logger.warning("No embedding function provided, falling back to recursive chunking")
            fallback = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
            return fallback.chunk(text, metadata)
        
        sentences = self._get_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(
                content=text,
                metadata={**metadata, "chunking_strategy": "semantic"},
                chunk_id=f"{metadata.get('source', 'doc')}_0"
            )]
        
        # Get embeddings for sentences (using buffer for smoothing)
        sentence_embeddings = []
        for i in range(len(sentences)):
            # Combine buffer of sentences for embedding
            start = max(0, i - self.buffer_size // 2)
            end = min(len(sentences), i + self.buffer_size // 2 + 1)
            buffer_text = " ".join(sentences[start:end])
            embedding = self.embedding_function(buffer_text)
            sentence_embeddings.append(embedding)
        
        # Find semantic breakpoints
        breakpoints = [0]
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(
                sentence_embeddings[i-1],
                sentence_embeddings[i]
            )
            if similarity < self.similarity_threshold:
                breakpoints.append(i)
        breakpoints.append(len(sentences))
        
        # Create chunks from breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = " ".join(chunk_sentences)
            
            # Handle size constraints
            if len(chunk_content) < self.min_chunk_size and i < len(breakpoints) - 2:
                continue  # Will be merged with next chunk
            
            if len(chunk_content) > self.max_chunk_size:
                # Split oversized chunks
                sub_chunker = RecursiveChunker(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=100
                )
                sub_chunks = sub_chunker.chunk(chunk_content, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["chunking_strategy"] = "semantic_recursive"
                    chunks.append(sub_chunk)
            else:
                chunk = Chunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunking_strategy": "semantic",
                        "num_sentences": len(chunk_sentences),
                    },
                    chunk_id=f"{metadata.get('source', 'doc')}_{len(chunks)}"
                )
                chunks.append(chunk)
        
        logger.debug(f"Created {len(chunks)} semantic chunks")
        return chunks


class MarkdownChunker(ChunkingStrategy):
    """
    Markdown-Aware Chunking Strategy
    
    Chunks markdown documents respecting headers and structure.
    Ideal for documentation and structured content.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        headers_to_split_on: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on or ["#", "##", "###"]
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split markdown text by headers."""
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Pattern to match markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        # Find all headers
        matches = list(header_pattern.finditer(text))
        
        if not matches:
            # No headers, use recursive chunker
            fallback = RecursiveChunker(self.chunk_size, self.chunk_overlap)
            return fallback.chunk(text, metadata)
        
        chunks = []
        current_headers = {}
        
        for i, match in enumerate(matches):
            header_level = len(match.group(1))
            header_text = match.group(2).strip()
            start_pos = match.start()
            
            # Get end position (start of next section or end of text)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            
            # Update header hierarchy
            current_headers[header_level] = header_text
            # Clear lower level headers
            for level in list(current_headers.keys()):
                if level > header_level:
                    del current_headers[level]
            
            # Create chunk
            if len(section_content) <= self.chunk_size:
                chunk = Chunk(
                    content=section_content,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunking_strategy": "markdown",
                        "header": header_text,
                        "header_level": header_level,
                        "header_hierarchy": dict(current_headers),
                    },
                    chunk_id=f"{metadata.get('source', 'doc')}_{len(chunks)}"
                )
                chunks.append(chunk)
            else:
                # Split large sections
                sub_chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                sub_chunks = sub_chunker.chunk(section_content, {
                    **metadata,
                    "header": header_text,
                    "header_level": header_level,
                })
                chunks.extend(sub_chunks)
        
        logger.debug(f"Created {len(chunks)} markdown chunks")
        return chunks


def get_chunker(strategy: str = "recursive", **kwargs) -> ChunkingStrategy:
    """
    Factory function to get a chunking strategy.
    
    Args:
        strategy: Name of the strategy (fixed, sentence, recursive, semantic, markdown)
        **kwargs: Additional arguments for the chunker
        
    Returns:
        ChunkingStrategy instance
    """
    chunkers = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "markdown": MarkdownChunker,
    }
    
    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Available: {list(chunkers.keys())}")
    
    return chunkers[strategy](**kwargs)


def chunk_documents(
    documents: List[Any],  # List of Document objects
    strategy: str = "recursive",
    **kwargs
) -> List[Chunk]:
    """
    Chunk a list of documents.
    
    Args:
        documents: List of Document objects to chunk
        strategy: Chunking strategy to use
        **kwargs: Additional arguments for the chunker
        
    Returns:
        List of Chunk objects
    """
    chunker = get_chunker(strategy, **kwargs)
    all_chunks = []
    
    for doc in documents:
        chunks = chunker.chunk(doc.content, doc.metadata)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
    return all_chunks
