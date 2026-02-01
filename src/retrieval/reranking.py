"""
Reranking Strategies
Implements various reranking methods to improve retrieval quality.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import logging

from ..indexing.vector_stores import SearchResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Rerank search results."""
        pass


class CrossEncoderReranker(Reranker):
    """
    Cross-Encoder Reranker
    
    Uses a cross-encoder model to compute relevance scores.
    More accurate than bi-encoders but slower (can't pre-compute).
    
    Process:
    1. Concatenate query and document
    2. Pass through transformer
    3. Get relevance score
    
    Good for:
    - High-precision requirements
    - Reranking top candidates
    - When latency allows
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            self.device = device
            self.model = CrossEncoder(model_name, device=device)
            
            logger.info(f"Initialized CrossEncoder reranker: {model_name} on {device}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked search results
        """
        if not results:
            return results
        
        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine with results
        scored_results = list(zip(results, scores))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return
        reranked = []
        for result, score in scored_results:
            result.score = float(score)
            result.metadata["reranked"] = True
            result.metadata["reranker"] = "cross_encoder"
            reranked.append(result)
        
        if top_k:
            reranked = reranked[:top_k]
        
        logger.debug(f"Reranked {len(results)} results")
        return reranked


class LLMReranker(Reranker):
    """
    LLM-based Reranker
    
    Uses a language model to judge relevance.
    Most flexible but slowest approach.
    
    Methods:
    - Pointwise: Score each document independently
    - Pairwise: Compare documents against each other
    - Listwise: Rank all documents at once
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank results using LLM.
        """
        if not results:
            return results
        
        # Use pointwise scoring
        scored_results = []
        
        for result in results:
            score = self._score_document(query, result.content)
            scored_results.append((result, score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return
        reranked = []
        for result, score in scored_results:
            result.score = score
            result.metadata["reranked"] = True
            result.metadata["reranker"] = "llm"
            reranked.append(result)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def _score_document(self, query: str, document: str) -> float:
        """Score a single document's relevance to the query."""
        prompt = f"""Rate the relevance of the following document to the query on a scale of 0-10.
Only respond with a single number.

Query: {query}

Document: {document[:2000]}

Relevance score (0-10):"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=5)
            score = float(response.strip().split()[0])
            return min(max(score, 0), 10) / 10  # Normalize to 0-1
        except:
            return 0.5  # Default score on error
    
    def rerank_listwise(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank using listwise approach - rank all at once.
        More efficient for larger result sets.
        """
        if not results or len(results) <= 1:
            return results
        
        # Prepare document list
        doc_list = "\n".join([
            f"[{i+1}] {r.content[:500]}" for i, r in enumerate(results)
        ])
        
        prompt = f"""Given the query and list of documents, rank them by relevance.
Return only the document numbers in order of relevance, from most to least relevant.
Format: 1, 3, 2, 5, 4

Query: {query}

Documents:
{doc_list}

Ranking (comma-separated numbers):"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=50)
            
            # Parse ranking
            ranking = []
            for num in response.strip().split(","):
                try:
                    idx = int(num.strip()) - 1
                    if 0 <= idx < len(results) and idx not in ranking:
                        ranking.append(idx)
                except:
                    continue
            
            # Add any missing indices
            for i in range(len(results)):
                if i not in ranking:
                    ranking.append(i)
            
            # Reorder results
            reranked = [results[i] for i in ranking]
            
            # Update scores
            for i, result in enumerate(reranked):
                result.score = 1 - (i / len(reranked))
                result.metadata["reranked"] = True
                result.metadata["reranker"] = "llm_listwise"
            
            if top_k:
                reranked = reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            logger.warning(f"Listwise reranking failed: {e}")
            return results


class CohereReranker(Reranker):
    """
    Cohere Reranker API
    
    Uses Cohere's dedicated reranking model.
    High quality, production-ready.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v2.0"):
        self.model = model
        
        try:
            import cohere
            import os
            
            self.client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))
            
        except ImportError:
            raise ImportError("cohere not installed. Run: pip install cohere")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank using Cohere's reranking model.
        """
        if not results:
            return results
        
        documents = [r.content for r in results]
        
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k or len(results)
        )
        
        # Map results
        reranked = []
        for item in response.results:
            result = results[item.index]
            result.score = item.relevance_score
            result.metadata["reranked"] = True
            result.metadata["reranker"] = "cohere"
            reranked.append(result)
        
        return reranked


class EnsembleReranker(Reranker):
    """
    Ensemble Reranker
    
    Combines multiple rerankers for robust performance.
    Uses weighted score aggregation.
    """
    
    def __init__(
        self,
        rerankers: List[Tuple[Reranker, float]]  # List of (reranker, weight)
    ):
        self.rerankers = rerankers
        
        # Normalize weights
        total_weight = sum(w for _, w in rerankers)
        self.rerankers = [(r, w/total_weight) for r, w in rerankers]
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank using ensemble of rerankers.
        """
        if not results:
            return results
        
        # Collect scores from all rerankers
        aggregated_scores = {r.doc_id: 0.0 for r in results}
        result_map = {r.doc_id: r for r in results}
        
        for reranker, weight in self.rerankers:
            try:
                reranked = reranker.rerank(query, results.copy())
                
                for rank, result in enumerate(reranked):
                    # RRF-style scoring
                    rrf_score = weight * (1 / (rank + 60))
                    aggregated_scores[result.doc_id] += rrf_score
                    
            except Exception as e:
                logger.warning(f"Reranker failed: {e}")
        
        # Sort by aggregated score
        sorted_ids = sorted(
            aggregated_scores.keys(),
            key=lambda x: aggregated_scores[x],
            reverse=True
        )
        
        # Return reranked results
        reranked = []
        for doc_id in sorted_ids:
            result = result_map[doc_id]
            result.score = aggregated_scores[doc_id]
            result.metadata["reranked"] = True
            result.metadata["reranker"] = "ensemble"
            reranked.append(result)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked


def rerank_results(
    query: str,
    results: List[SearchResult],
    method: str = "cross_encoder",
    top_k: Optional[int] = None,
    **kwargs
) -> List[SearchResult]:
    """
    Convenience function for reranking results.
    
    Args:
        query: Search query
        results: Search results to rerank
        method: Reranking method (cross_encoder, llm, cohere)
        top_k: Number of results to return
        **kwargs: Additional arguments for the reranker
        
    Returns:
        Reranked search results
    """
    if method == "cross_encoder":
        reranker = CrossEncoderReranker(**kwargs)
    elif method == "cohere":
        reranker = CohereReranker(**kwargs)
    elif method == "llm":
        if "llm_client" not in kwargs:
            raise ValueError("llm_client required for LLM reranker")
        reranker = LLMReranker(kwargs["llm_client"])
    else:
        raise ValueError(f"Unknown reranking method: {method}")
    
    return reranker.rerank(query, results, top_k)
