"""
Vector Stores
Implements vector storage and retrieval using ChromaDB and FAISS.
Supports both keyword-based and vector-based indexing.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np

from .embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a search result from the vector store."""
    
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        score: float,
        doc_id: str = ""
    ):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.doc_id = doc_id
    
    def __repr__(self):
        return f"SearchResult(score={self.score:.4f}, content={self.content[:50]}...)"


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        pass


class ChromaVectorStore(VectorStore):
    """
    ChromaDB Vector Store
    
    Features:
    - Persistent storage
    - Metadata filtering
    - Built-in embedding support
    - Easy to use and deploy
    
    Best for:
    - Small to medium datasets
    - Development and prototyping
    - Applications needing metadata filtering
    """
    
    def __init__(
        self,
        collection_name: str = "customer_support",
        persist_directory: str = "./data/chroma_db",
        embedding_model: Optional[EmbeddingModel] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            if embedding_model:
                # Use custom embedding function
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            logger.info(f"Initialized ChromaDB collection: {collection_name}")
            
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to ChromaDB."""
        if not documents:
            return
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Generate embeddings if model provided
        embeddings = None
        if self.embedding_model:
            embeddings = self.embedding_model.embed_texts(documents)
        
        # Clean metadata - ChromaDB requires specific types
        if metadatas:
            clean_metadatas = []
            for meta in metadatas:
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif isinstance(v, (list, dict)):
                        clean_meta[k] = json.dumps(v)
                    else:
                        clean_meta[k] = str(v)
                clean_metadatas.append(clean_meta)
            metadatas = clean_metadatas
        
        # Add to collection
        if embeddings:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents in ChromaDB."""
        # Get query embedding
        query_embedding = None
        if self.embedding_model:
            query_embedding = self.embedding_model.embed_query(query)
        
        # Perform search
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
        
        # Convert to SearchResult objects
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity score (cosine distance to similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Convert distance to similarity
                
                search_results.append(SearchResult(
                    content=doc,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=score,
                    doc_id=results["ids"][0][i] if results["ids"] else ""
                ))
        
        return search_results
    
    def search_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance (MMR) search.
        Balances relevance with diversity in results.
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of candidates to fetch
            lambda_mult: Balance between relevance (1) and diversity (0)
            filter: Metadata filter
        """
        # Get more candidates than needed
        candidates = self.search(query, k=fetch_k, filter=filter)
        
        if not candidates or not self.embedding_model:
            return candidates[:k]
        
        # Get query embedding
        query_embedding = np.array(self.embedding_model.embed_query(query))
        
        # Get embeddings for all candidates
        candidate_embeddings = np.array(
            self.embedding_model.embed_texts([c.content for c in candidates])
        )
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        for _ in range(min(k, len(candidates))):
            if not remaining_indices:
                break
            
            if not selected_indices:
                # First selection: most similar to query
                similarities = np.dot(candidate_embeddings[remaining_indices], query_embedding)
                best_idx = remaining_indices[np.argmax(similarities)]
            else:
                # Subsequent selections: balance relevance and diversity
                scores = []
                for idx in remaining_indices:
                    # Relevance to query
                    relevance = np.dot(candidate_embeddings[idx], query_embedding)
                    
                    # Maximum similarity to already selected
                    max_sim = max(
                        np.dot(candidate_embeddings[idx], candidate_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    )
                    
                    # MMR score
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                    scores.append((idx, mmr_score))
                
                best_idx = max(scores, key=lambda x: x[1])[0]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [candidates[i] for i in selected_indices]
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from ChromaDB")
    
    def count(self) -> int:
        """Return the number of documents."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared ChromaDB collection: {self.collection_name}")


class FAISSVectorStore(VectorStore):
    """
    FAISS Vector Store
    
    Features:
    - High-performance similarity search
    - Supports billions of vectors
    - Various index types (Flat, IVF, HNSW)
    - GPU acceleration support
    
    Best for:
    - Large-scale applications
    - Production deployments
    - High-throughput requirements
    
    Index Types:
    - Flat: Exact search, best quality, O(n) search
    - IVF: Approximate, good balance, O(sqrt(n)) search
    - HNSW: Fast approximate, O(log n) search
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index_path: str = "./data/faiss_index",
        index_type: str = "flat",  # flat, ivf, hnsw
        nlist: int = 100,  # For IVF index
        use_gpu: bool = False
    ):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.index_type = index_type
        self.dimension = embedding_model.dimension
        
        # Storage for documents and metadata
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        try:
            import faiss
            self.faiss = faiss
            
            # Create index based on type
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            elif index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 = M parameter
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            # GPU support
            if use_gpu and hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            # Try to load existing index
            self._load_index()
            
            logger.info(f"Initialized FAISS index: {index_type}, dimension={self.dimension}")
            
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu or faiss-gpu")
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity with inner product."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-9)
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to FAISS index."""
        if not documents:
            return
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
        
        # Default metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Get embeddings
        embeddings = np.array(
            self.embedding_model.embed_texts(documents),
            dtype=np.float32
        )
        
        # Normalize for cosine similarity
        embeddings = self._normalize(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Persist
        self._save_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []
        
        # Get query embedding
        query_embedding = np.array(
            [self.embedding_model.embed_query(query)],
            dtype=np.float32
        )
        query_embedding = self._normalize(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k * 2, self.index.ntotal))
        
        # Filter and convert to SearchResult
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            
            # Apply metadata filter if provided
            if filter:
                metadata = self.metadatas[idx]
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue
            
            results.append(SearchResult(
                content=self.documents[idx],
                metadata=self.metadatas[idx],
                score=float(score),
                doc_id=self.ids[idx]
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID (requires index rebuild)."""
        # Find indices to keep
        indices_to_remove = set()
        for i, doc_id in enumerate(self.ids):
            if doc_id in ids:
                indices_to_remove.add(i)
        
        if not indices_to_remove:
            return
        
        # Filter out removed documents
        new_documents = []
        new_metadatas = []
        new_ids = []
        
        for i in range(len(self.documents)):
            if i not in indices_to_remove:
                new_documents.append(self.documents[i])
                new_metadatas.append(self.metadatas[i])
                new_ids.append(self.ids[i])
        
        # Rebuild index
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.index.reset()
        
        if new_documents:
            self.add_documents(new_documents, new_metadatas, new_ids)
        
        logger.info(f"Deleted {len(ids)} documents from FAISS index")
    
    def count(self) -> int:
        """Return the number of documents."""
        return self.index.ntotal
    
    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(self.index_path, "index.faiss")
        self.faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata_file = os.path.join(self.index_path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump({
                "documents": self.documents,
                "metadatas": self.metadatas,
                "ids": self.ids
            }, f)
    
    def _load_index(self) -> None:
        """Load index and metadata from disk."""
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.json")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            # Load FAISS index
            self.index = self.faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, "r") as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]
                self.ids = data["ids"]
            
            logger.info(f"Loaded FAISS index with {self.count()} documents")


class HybridVectorStore(VectorStore):
    """
    Hybrid Vector Store
    
    Combines keyword search (BM25) with vector search for better retrieval.
    Uses rank fusion to combine results from both methods.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        alpha: float = 0.5  # Weight for vector search (1-alpha for keyword)
    ):
        self.vector_store = vector_store
        self.alpha = alpha
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        # BM25 index
        self.bm25 = None
        self._tokenized_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        return re.findall(r'\w+', text.lower())
    
    def _build_bm25(self) -> None:
        """Build BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
            self._tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(self._tokenized_docs)
        except ImportError:
            logger.warning("rank_bm25 not installed. Keyword search disabled.")
            self.bm25 = None
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to both vector and keyword indices."""
        # Add to vector store
        self.vector_store.add_documents(documents, metadatas, ids)
        
        # Store for keyword search
        self.documents.extend(documents)
        self.metadatas.extend(metadatas or [{} for _ in documents])
        self.ids.extend(ids or [f"doc_{len(self.ids) + i}" for i in range(len(documents))])
        
        # Rebuild BM25 index
        self._build_bm25()
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Hybrid search combining vector and keyword search."""
        # Vector search
        vector_results = self.vector_store.search(query, k=k * 2, filter=filter)
        
        # Keyword search
        keyword_results = []
        if self.bm25:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k keyword results
            top_indices = np.argsort(bm25_scores)[::-1][:k * 2]
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    keyword_results.append(SearchResult(
                        content=self.documents[idx],
                        metadata=self.metadatas[idx],
                        score=float(bm25_scores[idx]),
                        doc_id=self.ids[idx]
                    ))
        
        # Reciprocal Rank Fusion
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, SearchResult] = {}
        
        # Score from vector search
        for rank, result in enumerate(vector_results):
            score = self.alpha * (1 / (rank + 60))  # RRF formula
            fused_scores[result.doc_id] = fused_scores.get(result.doc_id, 0) + score
            doc_map[result.doc_id] = result
        
        # Score from keyword search
        for rank, result in enumerate(keyword_results):
            score = (1 - self.alpha) * (1 / (rank + 60))
            fused_scores[result.doc_id] = fused_scores.get(result.doc_id, 0) + score
            if result.doc_id not in doc_map:
                doc_map[result.doc_id] = result
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # Return top-k results
        results = []
        for doc_id in sorted_ids[:k]:
            result = doc_map[doc_id]
            result.score = fused_scores[doc_id]
            results.append(result)
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents from both indices."""
        self.vector_store.delete(ids)
        
        # Remove from keyword index
        indices_to_remove = set()
        for i, doc_id in enumerate(self.ids):
            if doc_id in ids:
                indices_to_remove.add(i)
        
        self.documents = [d for i, d in enumerate(self.documents) if i not in indices_to_remove]
        self.metadatas = [m for i, m in enumerate(self.metadatas) if i not in indices_to_remove]
        self.ids = [id_ for i, id_ in enumerate(self.ids) if i not in indices_to_remove]
        
        self._build_bm25()
    
    def count(self) -> int:
        return self.vector_store.count()


def get_vector_store(
    store_type: str = "chroma",
    embedding_model: Optional[EmbeddingModel] = None,
    **kwargs
) -> VectorStore:
    """
    Factory function to get a vector store.
    
    Args:
        store_type: Type of vector store (chroma, faiss, hybrid)
        embedding_model: Embedding model to use
        **kwargs: Additional arguments for the store
        
    Returns:
        VectorStore instance
    """
    if embedding_model is None:
        embedding_model = get_embedding_model("sentence_transformer")
    
    if store_type == "chroma":
        return ChromaVectorStore(embedding_model=embedding_model, **kwargs)
    elif store_type == "faiss":
        return FAISSVectorStore(embedding_model=embedding_model, **kwargs)
    elif store_type == "hybrid":
        base_store = get_vector_store(
            kwargs.pop("base_store", "chroma"),
            embedding_model,
            **kwargs
        )
        return HybridVectorStore(base_store, **kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def build_index(
    documents_path: str,
    store_type: str = "chroma",
    embedding_provider: str = "sentence_transformer",
    chunking_strategy: str = "recursive",
    **kwargs
) -> VectorStore:
    """
    Build a vector index from documents.
    
    Args:
        documents_path: Path to documents directory
        store_type: Type of vector store
        embedding_provider: Embedding model provider
        chunking_strategy: Document chunking strategy
        **kwargs: Additional arguments
        
    Returns:
        Populated VectorStore instance
    """
    from ..document_processing import parse_directory, chunk_documents
    
    # Parse documents
    logger.info(f"Parsing documents from: {documents_path}")
    documents = parse_directory(documents_path)
    
    # Chunk documents
    logger.info(f"Chunking {len(documents)} documents with strategy: {chunking_strategy}")
    chunks = chunk_documents(documents, strategy=chunking_strategy)
    
    # Create embedding model
    embedding_model = get_embedding_model(embedding_provider)
    
    # Create vector store
    vector_store = get_vector_store(store_type, embedding_model)
    
    # Add chunks to store
    logger.info(f"Adding {len(chunks)} chunks to vector store")
    vector_store.add_documents(
        documents=[chunk.content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.chunk_id for chunk in chunks]
    )
    
    logger.info(f"Index built with {vector_store.count()} documents")
    return vector_store


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Build index from sample documents
    import sys
    if len(sys.argv) > 1:
        docs_path = sys.argv[1]
        store = build_index(docs_path)
        print(f"Built index with {store.count()} documents")
