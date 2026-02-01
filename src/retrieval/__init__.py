"""
Retrieval Module
Handles search and reranking for the RAG pipeline.
"""

from .search import (
    SearchEngine,
    VectorSearchEngine,
    HybridSearchEngine,
    search_documents
)
from .reranking import (
    Reranker,
    CrossEncoderReranker,
    LLMReranker,
    CohereReranker,
    rerank_results
)

__all__ = [
    # Search
    "SearchEngine",
    "VectorSearchEngine",
    "HybridSearchEngine",
    "search_documents",
    # Reranking
    "Reranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "CohereReranker",
    "rerank_results",
]
