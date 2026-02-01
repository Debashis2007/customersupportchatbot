"""
Embedding Models
Implements various embedding models for converting text to vectors.
Supports OpenAI embeddings and local Sentence Transformers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (may use different model/processing than documents)."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbeddings(EmbeddingModel):
    """
    OpenAI Embeddings
    
    Uses OpenAI's text-embedding models for high-quality embeddings.
    
    Models:
    - text-embedding-3-small: 1536 dimensions, cost-effective
    - text-embedding-3-large: 3072 dimensions, highest quality
    - text-embedding-ada-002: 1536 dimensions, legacy
    
    Features:
    - High-quality semantic embeddings
    - Support for multiple languages
    - Configurable dimensions (for v3 models)
    """
    
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        batch_size: int = 100
    ):
        self.model = model
        self.batch_size = batch_size
        self._dimensions = dimensions or self.MODEL_DIMENSIONS.get(model, 1536)
        
        try:
            from openai import OpenAI
            import os
            
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            logger.info(f"Initialized OpenAI embeddings with model: {model}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    @property
    def dimension(self) -> int:
        return self._dimensions
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with batching."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Clean texts - remove empty strings and limit length
            batch = [t[:8191] if t else " " for t in batch]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self._dimensions if self.model.startswith("text-embedding-3") else None
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error getting embeddings: {e}")
                raise
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (same as regular embedding for OpenAI)."""
        return self.embed_text(query)


class SentenceTransformerEmbeddings(EmbeddingModel):
    """
    Sentence Transformer Embeddings
    
    Uses local Sentence Transformer models for embeddings.
    Free, fast, and can run entirely offline.
    
    Popular Models:
    - all-MiniLM-L6-v2: Fast, good quality, 384 dimensions
    - all-mpnet-base-v2: High quality, 768 dimensions
    - multi-qa-mpnet-base-dot-v1: Optimized for QA, 768 dimensions
    - paraphrase-multilingual-MiniLM-L12-v2: Multilingual, 384 dimensions
    
    Features:
    - No API costs
    - Fast local inference
    - GPU acceleration support
    - Many specialized models available
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Auto-detect device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            self.device = device
            self.model = SentenceTransformer(model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Initialized SentenceTransformer: {model_name} on {device}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (same as regular embedding)."""
        return self.embed_text(query)


class HuggingFaceEmbeddings(EmbeddingModel):
    """
    HuggingFace Embeddings
    
    Uses HuggingFace's inference API or local models.
    Good for accessing a wide variety of models.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        api_key: Optional[str] = None,
        use_api: bool = False
    ):
        self.model_name = model_name
        self.use_api = use_api
        
        if use_api:
            import os
            self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            self._dimension = 384  # Default, may vary by model
        else:
            # Use local model via sentence-transformers
            self._local_model = SentenceTransformerEmbeddings(model_name)
            self._dimension = self._local_model.dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        if self.use_api:
            return self._embed_via_api([text])[0]
        return self._local_model.embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.use_api:
            return self._embed_via_api(texts)
        return self._local_model.embed_texts(texts)
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)
    
    def _embed_via_api(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings via HuggingFace API."""
        import requests
        
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.post(api_url, headers=headers, json={"inputs": texts})
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        
        return response.json()


class CohereEmbeddings(EmbeddingModel):
    """
    Cohere Embeddings
    
    Uses Cohere's embedding models with specialized types
    for search queries vs documents.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0"
    ):
        self.model = model
        
        try:
            import cohere
            import os
            
            self.client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))
            self._dimension = 1024  # Default for v3 models
            
        except ImportError:
            raise ImportError("cohere package not installed. Run: pip install cohere")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts as documents (for indexing)."""
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"
        )
        return [list(e) for e in response.embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query (with search_query input type)."""
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query"
        )
        return list(response.embeddings[0])


def get_embedding_model(
    provider: str = "openai",
    **kwargs
) -> EmbeddingModel:
    """
    Factory function to get an embedding model.
    
    Args:
        provider: Embedding provider (openai, sentence_transformer, huggingface, cohere)
        **kwargs: Additional arguments for the model
        
    Returns:
        EmbeddingModel instance
    """
    providers = {
        "openai": OpenAIEmbeddings,
        "sentence_transformer": SentenceTransformerEmbeddings,
        "huggingface": HuggingFaceEmbeddings,
        "cohere": CohereEmbeddings,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown embedding provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider](**kwargs)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.linalg.norm(vec1 - vec2))
