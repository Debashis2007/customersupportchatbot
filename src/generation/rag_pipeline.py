"""
RAG Pipeline
Complete Retrieval-Augmented Generation pipeline implementation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import time

from .llm import LLMClient, get_llm_client
from ..indexing.vector_stores import VectorStore, SearchResult
from ..retrieval.search import VectorSearchEngine, SearchConfig
from ..retrieval.reranking import Reranker, CrossEncoderReranker
from ..prompts.templates import RAGPromptTemplate, CustomerSupportPrompt
from ..prompts.strategies import apply_prompt_strategy

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.5
    search_type: str = "mmr"  # similarity, mmr
    rerank: bool = False
    rerank_top_k: int = 3
    
    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.7
    
    # Prompt settings
    prompt_strategy: str = "default"
    include_sources: bool = True
    max_context_tokens: int = 4000
    
    # Pipeline settings
    stream: bool = False
    debug: bool = False


@dataclass
class RAGResponse:
    """Represents a RAG response."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class RAGPipeline:
    """
    Complete RAG Pipeline
    
    Orchestrates the full retrieval-augmented generation process:
    1. Query processing
    2. Document retrieval
    3. Optional reranking
    4. Context construction
    5. Prompt engineering
    6. LLM generation
    7. Response formatting
    
    Architecture:
    
    ┌────────────┐
    │   Query    │
    └─────┬──────┘
          │
          ▼
    ┌────────────────┐
    │ Query Processor│ ← Query expansion, intent detection
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │   Retriever    │ ← Vector store search
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │   Reranker     │ ← Optional: Cross-encoder, LLM
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │Context Builder │ ← Format documents, add metadata
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │Prompt Engineer │ ← Apply strategies (CoT, few-shot, etc.)
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │   LLM Client   │ ← Generate response
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │ Post-Processor │ ← Format, validate, add sources
    └───────┬────────┘
            │
            ▼
    ┌────────────┐
    │  Response  │
    └────────────┘
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        config: Optional[RAGConfig] = None,
        reranker: Optional[Reranker] = None,
        prompt_template: Optional[RAGPromptTemplate] = None
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config or RAGConfig()
        self.reranker = reranker
        self.prompt_template = prompt_template or RAGPromptTemplate()
        
        # Initialize search engine
        self.search_engine = VectorSearchEngine(
            vector_store,
            SearchConfig(
                top_k=self.config.top_k,
                similarity_threshold=self.config.similarity_threshold,
                search_type=self.config.search_type
            )
        )
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("RAG Pipeline initialized")
    
    def query(
        self,
        question: str,
        filter: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RAGResponse:
        """
        Execute a RAG query.
        
        Args:
            question: User's question
            filter: Optional metadata filter for retrieval
            conversation_history: Optional conversation history
            
        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()
        metadata = {"query_time": 0, "retrieval_time": 0, "generation_time": 0}
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        search_results = self.search_engine.search(
            question,
            SearchConfig(
                top_k=self.config.top_k * 2 if self.config.rerank else self.config.top_k,
                similarity_threshold=self.config.similarity_threshold,
                search_type=self.config.search_type,
                filter=filter
            )
        )
        metadata["retrieval_time"] = time.time() - retrieval_start
        
        if self.config.debug:
            logger.debug(f"Retrieved {len(search_results)} documents")
        
        # Step 2: Rerank if enabled
        if self.config.rerank and self.reranker and search_results:
            search_results = self.reranker.rerank(
                question,
                search_results,
                top_k=self.config.rerank_top_k
            )
            if self.config.debug:
                logger.debug(f"Reranked to {len(search_results)} documents")
        
        # Step 3: Build context from retrieved documents
        context = self._build_context(search_results)
        
        # Step 4: Format conversation history
        history_str = ""
        if conversation_history:
            history_str = self._format_history(conversation_history)
        
        # Step 5: Construct prompt using template and strategy
        prompt = self._construct_prompt(question, context, history_str, search_results)
        
        # Step 6: Generate response
        generation_start = time.time()
        
        if self.config.stream:
            # For streaming, we'd return a generator
            answer = self._generate_streaming(prompt)
        else:
            answer = self.llm_client.generate(
                prompt,
                system_prompt=CustomerSupportPrompt.get_system_prompt(),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
        
        metadata["generation_time"] = time.time() - generation_start
        
        # Step 7: Extract and format sources
        sources = self._extract_sources(search_results)
        
        # Calculate confidence based on retrieval scores
        confidence = self._calculate_confidence(search_results)
        
        metadata["query_time"] = time.time() - start_time
        metadata["num_sources"] = len(sources)
        metadata["context_length"] = len(context)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            context=context,
            confidence=confidence,
            metadata=metadata
        )
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results."""
        documents = [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score
            }
            for r in results
        ]
        
        return RAGPromptTemplate.format_context(
            documents,
            max_tokens=self.config.max_context_tokens,
            include_metadata=True
        )
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history."""
        lines = []
        for msg in history[-5:]:  # Keep last 5 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
    
    def _construct_prompt(
        self,
        question: str,
        context: str,
        history: str,
        search_results: List[SearchResult]
    ) -> str:
        """Construct the final prompt."""
        # Apply prompt strategy
        if self.config.prompt_strategy != "default":
            context_docs = [
                {"content": r.content, "score": r.score}
                for r in search_results
            ]
            question = apply_prompt_strategy(
                question,
                self.config.prompt_strategy,
                context_docs=context_docs
            )
        
        # Use prompt template
        return self.prompt_template.format(
            question=question,
            context=context,
            history=history if history else None
        )
    
    def _generate_streaming(self, prompt: str):
        """Generate streaming response."""
        full_response = []
        for chunk in self.llm_client.stream(
            prompt,
            system_prompt=CustomerSupportPrompt.get_system_prompt(),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        ):
            full_response.append(chunk)
            yield chunk
        
        return "".join(full_response)
    
    def _extract_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract source information from results."""
        sources = []
        seen_sources = set()
        
        for result in results:
            source = result.metadata.get("source", "Unknown")
            
            # Deduplicate by source
            if source in seen_sources:
                continue
            seen_sources.add(source)
            
            sources.append({
                "source": source,
                "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "relevance_score": round(result.score, 3),
                "metadata": {
                    k: v for k, v in result.metadata.items()
                    if k not in ["source", "content"]
                }
            })
        
        return sources
    
    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not results:
            return 0.0
        
        # Average of top scores, weighted by position
        weighted_sum = 0
        weight_total = 0
        
        for i, result in enumerate(results[:3]):  # Top 3 results
            weight = 1 / (i + 1)  # 1, 0.5, 0.33
            weighted_sum += result.score * weight
            weight_total += weight
        
        avg_score = weighted_sum / weight_total if weight_total > 0 else 0
        
        # Normalize to 0-1 range
        return min(max(avg_score, 0), 1)
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_similar_questions(self, question: str, k: int = 3) -> List[str]:
        """Find similar questions in the knowledge base."""
        results = self.search_engine.search(
            f"question: {question}",
            SearchConfig(top_k=k, search_type="similarity")
        )
        
        return [r.metadata.get("question", r.content[:100]) for r in results]


class CustomerSupportRAG(RAGPipeline):
    """
    Specialized RAG Pipeline for Customer Support
    
    Adds customer support-specific features:
    - Intent classification
    - Sentiment analysis
    - Escalation detection
    - Response personalization
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        config: Optional[RAGConfig] = None,
        **kwargs
    ):
        super().__init__(vector_store, llm_client, config, **kwargs)
        
        # Customer context
        self.customer_context: Dict[str, Any] = {}
    
    def set_customer_context(self, context: Dict[str, Any]) -> None:
        """Set customer context for personalization."""
        self.customer_context = context
    
    def query(
        self,
        question: str,
        filter: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RAGResponse:
        """Execute customer support query with additional analysis."""
        # Analyze intent and sentiment
        intent = self._classify_intent(question)
        sentiment = self._analyze_sentiment(question)
        
        # Check if escalation needed
        should_escalate = self._check_escalation(question, sentiment)
        
        # Get RAG response
        response = super().query(question, filter, conversation_history)
        
        # Add customer support metadata
        response.metadata["intent"] = intent
        response.metadata["sentiment"] = sentiment
        response.metadata["should_escalate"] = should_escalate
        
        # Personalize if customer context available
        if self.customer_context:
            response.answer = self._personalize_response(
                response.answer,
                self.customer_context
            )
        
        return response
    
    def _classify_intent(self, message: str) -> str:
        """Classify the intent of the customer message."""
        prompt = CustomerSupportPrompt.classify_intent(message)
        
        try:
            intent = self.llm_client.generate(
                prompt,
                max_tokens=20,
                temperature=0
            ).strip().upper()
            
            valid_intents = ["QUESTION", "ISSUE", "REQUEST", "FEEDBACK", "COMPLAINT", "OTHER"]
            return intent if intent in valid_intents else "OTHER"
        except:
            return "OTHER"
    
    def _analyze_sentiment(self, message: str) -> str:
        """Analyze the sentiment of the message."""
        prompt = CustomerSupportPrompt.analyze_sentiment(message)
        
        try:
            sentiment = self.llm_client.generate(
                prompt,
                max_tokens=20,
                temperature=0
            ).strip().upper()
            
            valid_sentiments = ["VERY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "VERY_POSITIVE"]
            return sentiment if sentiment in valid_sentiments else "NEUTRAL"
        except:
            return "NEUTRAL"
    
    def _check_escalation(self, message: str, sentiment: str) -> bool:
        """Check if the conversation should be escalated."""
        # Automatic escalation triggers
        escalation_keywords = ["speak to human", "talk to agent", "manager", "supervisor", "lawsuit", "lawyer"]
        
        message_lower = message.lower()
        for keyword in escalation_keywords:
            if keyword in message_lower:
                return True
        
        # Escalate on very negative sentiment
        if sentiment == "VERY_NEGATIVE":
            return True
        
        return False
    
    def _personalize_response(self, response: str, context: Dict[str, Any]) -> str:
        """Personalize the response based on customer context."""
        # Simple personalization - add customer name if available
        if "name" in context:
            name = context["name"]
            if not response.startswith(f"Hi {name}") and not response.startswith(f"Hello {name}"):
                response = f"Hi {name},\n\n{response}"
        
        return response


def create_rag_pipeline(
    vector_store: VectorStore,
    llm_provider: str = "openai",
    enable_reranking: bool = False,
    customer_support: bool = True,
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        vector_store: Vector store for document retrieval
        llm_provider: LLM provider (openai, anthropic, huggingface)
        enable_reranking: Whether to enable cross-encoder reranking
        customer_support: Whether to use customer support specialized pipeline
        **kwargs: Additional configuration
        
    Returns:
        Configured RAG pipeline
    """
    # Create LLM client
    llm_client = get_llm_client(llm_provider)
    
    # Create reranker if enabled
    reranker = None
    if enable_reranking:
        try:
            reranker = CrossEncoderReranker()
        except ImportError:
            logger.warning("Reranking disabled: sentence-transformers not installed")
    
    # Create config
    config = RAGConfig(**{k: v for k, v in kwargs.items() if hasattr(RAGConfig, k)})
    
    # Create appropriate pipeline
    if customer_support:
        return CustomerSupportRAG(
            vector_store=vector_store,
            llm_client=llm_client,
            config=config,
            reranker=reranker
        )
    else:
        return RAGPipeline(
            vector_store=vector_store,
            llm_client=llm_client,
            config=config,
            reranker=reranker
        )
