"""
Indexing Module
Handles embeddings and vector store operations for the RAG pipeline.
"""

from .embeddings import (
    EmbeddingModel,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    get_embedding_model
)
from .vector_stores import (
    VectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    get_vector_store,
    build_index
)

__all__ = [
    # Embeddings
    "EmbeddingModel",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "get_embedding_model",
    # Vector Stores
    "VectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "get_vector_store",
    "build_index",
]
