"""
Search Methods
Implements exact and approximate nearest neighbor search methods.
Supports various search strategies for optimal retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..indexing.vector_stores import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    top_k: int = 5
    similarity_threshold: float = 0.0
    search_type: str = "similarity"  # similarity, mmr, hybrid
    mmr_lambda: float = 0.5
    rerank: bool = False
    filter: Optional[Dict[str, Any]] = None


class SearchEngine(ABC):
    """Abstract base class for search engines."""
    
    @abstractmethod
    def search(
        self,
        query: str,
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """Execute a search query."""
        pass
    
    @abstractmethod
    def multi_query_search(
        self,
        queries: List[str],
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """Execute multiple queries and combine results."""
        pass


class VectorSearchEngine(SearchEngine):
    """
    Vector-based Search Engine
    
    Supports multiple search strategies:
    - Exact Nearest Neighbor (Flat index)
    - Approximate Nearest Neighbor (IVF, HNSW)
    - MMR (Maximal Marginal Relevance) for diversity
    
    Search Methods:
    
    1. Exact NN (Brute Force):
       - Compares query to all vectors
       - O(n) complexity
       - Best quality, slowest
       
    2. Approximate NN:
       - IVF (Inverted File Index): Clusters vectors, searches subset
       - HNSW (Hierarchical Navigable Small World): Graph-based
       - Faster but may miss some results
       
    3. MMR (Maximal Marginal Relevance):
       - Balances relevance with diversity
       - Reduces redundancy in results
       - Good for comprehensive answers
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        default_config: Optional[SearchConfig] = None
    ):
        self.vector_store = vector_store
        self.default_config = default_config or SearchConfig()
    
    def search(
        self,
        query: str,
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        Execute a search query.
        
        Args:
            query: Search query string
            config: Search configuration
            
        Returns:
            List of SearchResult objects
        """
        config = config or self.default_config
        
        # Choose search method
        if config.search_type == "mmr" and hasattr(self.vector_store, "search_mmr"):
            results = self.vector_store.search_mmr(
                query=query,
                k=config.top_k,
                lambda_mult=config.mmr_lambda,
                filter=config.filter
            )
        else:
            results = self.vector_store.search(
                query=query,
                k=config.top_k,
                filter=config.filter
            )
        
        # Apply similarity threshold
        if config.similarity_threshold > 0:
            results = [r for r in results if r.score >= config.similarity_threshold]
        
        logger.debug(f"Search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    def multi_query_search(
        self,
        queries: List[str],
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        Execute multiple queries and combine results using rank fusion.
        
        Useful for:
        - Query expansion (multiple phrasings)
        - Handling ambiguous queries
        - Comprehensive retrieval
        """
        config = config or self.default_config
        
        # Collect results from all queries
        all_results: Dict[str, Tuple[SearchResult, float]] = {}
        
        for i, query in enumerate(queries):
            results = self.search(query, config)
            
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                
                # Reciprocal Rank Fusion score
                rrf_score = 1 / (rank + 60)
                
                if doc_id in all_results:
                    # Combine scores
                    existing_result, existing_score = all_results[doc_id]
                    all_results[doc_id] = (existing_result, existing_score + rrf_score)
                else:
                    all_results[doc_id] = (result, rrf_score)
        
        # Sort by combined score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Update scores and return top-k
        final_results = []
        for result, score in sorted_results[:config.top_k]:
            result.score = score
            final_results.append(result)
        
        return final_results
    
    def semantic_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Semantic similarity search.
        
        Uses cosine similarity between query and document embeddings.
        """
        config = SearchConfig(
            top_k=k,
            similarity_threshold=threshold,
            search_type="similarity"
        )
        return self.search(query, config)
    
    def diverse_search(
        self,
        query: str,
        k: int = 5,
        diversity: float = 0.5
    ) -> List[SearchResult]:
        """
        Diversity-aware search using MMR.
        
        Args:
            query: Search query
            k: Number of results
            diversity: 0 = pure relevance, 1 = pure diversity
        """
        config = SearchConfig(
            top_k=k,
            search_type="mmr",
            mmr_lambda=1 - diversity  # Convert to lambda
        )
        return self.search(query, config)


class HybridSearchEngine(SearchEngine):
    """
    Hybrid Search Engine
    
    Combines multiple search strategies:
    - Dense vector search (semantic)
    - Sparse vector search (keyword/BM25)
    - Optional metadata filtering
    
    Fusion methods:
    - Reciprocal Rank Fusion (RRF)
    - Linear combination
    - Learned fusion
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_weight: float = 0.3,
        default_config: Optional[SearchConfig] = None
    ):
        self.vector_store = vector_store
        self.keyword_weight = keyword_weight
        self.default_config = default_config or SearchConfig()
        
        # Initialize keyword index
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._bm25 = None
    
    def _init_keyword_index(self, documents: List[str], doc_ids: List[str]) -> None:
        """Initialize BM25 keyword index."""
        try:
            from rank_bm25 import BM25Okapi
            import re
            
            self._documents = documents
            self._doc_ids = doc_ids
            
            # Tokenize documents
            tokenized = [re.findall(r'\w+', doc.lower()) for doc in documents]
            self._bm25 = BM25Okapi(tokenized)
            
        except ImportError:
            logger.warning("rank_bm25 not installed. Keyword search disabled.")
    
    def search(
        self,
        query: str,
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        Execute hybrid search combining vector and keyword search.
        """
        config = config or self.default_config
        
        # Vector search
        vector_results = self.vector_store.search(
            query=query,
            k=config.top_k * 2,
            filter=config.filter
        )
        
        # Keyword search (if available)
        keyword_scores = {}
        if self._bm25 and self._documents:
            import re
            query_tokens = re.findall(r'\w+', query.lower())
            scores = self._bm25.get_scores(query_tokens)
            
            # Normalize scores
            max_score = max(scores) if max(scores) > 0 else 1
            for i, (doc_id, score) in enumerate(zip(self._doc_ids, scores)):
                keyword_scores[doc_id] = score / max_score
        
        # Combine using Reciprocal Rank Fusion
        combined_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        
        # Vector scores
        for rank, result in enumerate(vector_results):
            rrf_score = (1 - self.keyword_weight) * (1 / (rank + 60))
            combined_scores[result.doc_id] = rrf_score
            result_map[result.doc_id] = result
        
        # Keyword scores
        if keyword_scores:
            sorted_keyword = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_keyword[:config.top_k * 2]):
                rrf_score = self.keyword_weight * (1 / (rank + 60))
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
        
        # Sort and return
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids[:config.top_k]:
            if doc_id in result_map:
                result = result_map[doc_id]
                result.score = combined_scores[doc_id]
                results.append(result)
        
        return results
    
    def multi_query_search(
        self,
        queries: List[str],
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """Execute multiple queries and combine results."""
        config = config or self.default_config
        
        all_results: Dict[str, Tuple[SearchResult, float]] = {}
        
        for query in queries:
            results = self.search(query, config)
            
            for rank, result in enumerate(results):
                rrf_score = 1 / (rank + 60)
                
                if result.doc_id in all_results:
                    existing_result, existing_score = all_results[result.doc_id]
                    all_results[result.doc_id] = (existing_result, existing_score + rrf_score)
                else:
                    all_results[result.doc_id] = (result, rrf_score)
        
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        
        return [r for r, _ in sorted_results[:config.top_k]]


class QueryExpander:
    """
    Query Expansion
    
    Generates multiple query variations to improve retrieval.
    
    Techniques:
    - Synonym expansion
    - LLM-based query generation
    - Historical query mining
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query variations.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations including original
        """
        queries = [query]
        
        if self.llm_client:
            expansion_prompt = f"""Generate {num_variations} alternative phrasings of this search query.
Each variation should capture the same intent but use different words or structure.

Original query: {query}

Return only the alternative queries, one per line, without numbering or explanations."""
            
            try:
                response = self.llm_client.generate(expansion_prompt)
                variations = [v.strip() for v in response.strip().split("\n") if v.strip()]
                queries.extend(variations[:num_variations])
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
        
        return queries
    
    def hypothetical_document_embedding(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embedding)
        
        Generates a hypothetical answer to the query, then searches
        using that answer's embedding. Can improve retrieval for
        complex queries.
        """
        if not self.llm_client:
            return query
        
        hyde_prompt = f"""Write a brief passage that would be a good answer to this question:

Question: {query}

Write 2-3 sentences that directly answer the question with specific details."""
        
        try:
            response = self.llm_client.generate(hyde_prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query


def search_documents(
    vector_store: VectorStore,
    query: str,
    k: int = 5,
    search_type: str = "similarity",
    threshold: float = 0.0,
    filter: Optional[Dict[str, Any]] = None
) -> List[SearchResult]:
    """
    Convenience function for searching documents.
    
    Args:
        vector_store: Vector store to search
        query: Search query
        k: Number of results
        search_type: Type of search (similarity, mmr, hybrid)
        threshold: Minimum similarity threshold
        filter: Metadata filter
        
    Returns:
        List of SearchResult objects
    """
    config = SearchConfig(
        top_k=k,
        similarity_threshold=threshold,
        search_type=search_type,
        filter=filter
    )
    
    engine = VectorSearchEngine(vector_store)
    return engine.search(query, config)
