"""
Chat Handler
Manages conversations and provides the main chatbot interface.
"""

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
import json

from ..generation.rag_pipeline import RAGPipeline, CustomerSupportRAG, RAGConfig, RAGResponse
from ..generation.llm import LLMClient, get_llm_client
from ..indexing.vector_stores import VectorStore, get_vector_store
from ..indexing.embeddings import get_embedding_model, EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Conversation:
    """Represents a conversation session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history for RAG context."""
        messages = self.messages[-last_n * 2:] if last_n else self.messages
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class ConversationManager:
    """
    Manages multiple conversation sessions.
    Handles persistence and retrieval of conversations.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.conversations: Dict[str, Conversation] = {}
        self.storage_path = storage_path
        
        if storage_path:
            self._load_conversations()
    
    def create_conversation(self, **metadata) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(metadata=metadata)
        self.conversations[conversation.id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        return [
            {
                "id": conv.id,
                "created_at": conv.created_at.isoformat(),
                "message_count": len(conv.messages),
                "last_message": conv.messages[-1].content[:100] if conv.messages else ""
            }
            for conv in self.conversations.values()
        ]
    
    def save_conversations(self) -> None:
        """Save conversations to storage."""
        if not self.storage_path:
            return
        
        data = {
            conv_id: conv.to_dict()
            for conv_id, conv in self.conversations.items()
        }
        
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_conversations(self) -> None:
        """Load conversations from storage."""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            for conv_id, conv_data in data.items():
                conversation = Conversation(
                    id=conv_data["id"],
                    created_at=datetime.fromisoformat(conv_data["created_at"]),
                    updated_at=datetime.fromisoformat(conv_data["updated_at"]),
                    metadata=conv_data.get("metadata", {})
                )
                
                for msg_data in conv_data["messages"]:
                    conversation.messages.append(Message(
                        role=msg_data["role"],
                        content=msg_data["content"],
                        timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                        metadata=msg_data.get("metadata", {})
                    ))
                
                self.conversations[conv_id] = conversation
                
        except FileNotFoundError:
            pass


class CustomerSupportChatbot:
    """
    Main Customer Support Chatbot Interface
    
    Provides a high-level API for interacting with the RAG-powered
    customer support system.
    
    Features:
    - Multi-turn conversations
    - Intent detection and routing
    - Sentiment-aware responses
    - Source citation
    - Conversation persistence
    - Streaming responses
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[LLMClient] = None,
        config: Optional[RAGConfig] = None,
        knowledge_base_path: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the chatbot.
        
        Args:
            vector_store: Pre-initialized vector store
            llm_client: Pre-initialized LLM client
            config: RAG configuration
            knowledge_base_path: Path to knowledge base documents
            embedding_model: Pre-initialized embedding model
        """
        # Initialize components
        self.config = config or RAGConfig()
        
        # Use provided embedding model or create default
        self.embedding_model = embedding_model
        
        # Initialize vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = get_vector_store(
                "chroma",
                embedding_model=self.embedding_model
            )
        
        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            try:
                self.llm_client = get_llm_client("openai")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                logger.info("Attempting to use local model...")
                self.llm_client = get_llm_client("huggingface", use_api=False)
        
        # Initialize RAG pipeline
        self.rag_pipeline = CustomerSupportRAG(
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            config=self.config
        )
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
        
        # Current conversation
        self.current_conversation: Optional[Conversation] = None
        
        # Load knowledge base if provided
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
        
        logger.info("CustomerSupportChatbot initialized")
    
    def load_knowledge_base(self, path: str) -> int:
        """
        Load documents into the knowledge base.
        
        Args:
            path: Path to documents directory
            
        Returns:
            Number of documents loaded
        """
        from ..document_processing import parse_directory, chunk_documents
        
        logger.info(f"Loading knowledge base from: {path}")
        
        # Parse documents
        documents = parse_directory(path)
        
        # Chunk documents
        chunks = chunk_documents(documents, strategy="recursive")
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks]
        )
        
        logger.info(f"Loaded {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)
    
    def start_conversation(self, **metadata) -> str:
        """
        Start a new conversation.
        
        Returns:
            Conversation ID
        """
        self.current_conversation = self.conversation_manager.create_conversation(**metadata)
        logger.info(f"Started conversation: {self.current_conversation.id}")
        return self.current_conversation.id
    
    def set_conversation(self, conversation_id: str) -> bool:
        """
        Set the current conversation.
        
        Args:
            conversation_id: ID of conversation to use
            
        Returns:
            True if conversation found
        """
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation:
            self.current_conversation = conversation
            return True
        return False
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message and get a response.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
            stream: Whether to stream the response
            **kwargs: Additional arguments (e.g., customer context)
            
        Returns:
            Response dictionary with answer, sources, and metadata
        """
        # Get or create conversation
        if conversation_id:
            self.set_conversation(conversation_id)
        
        if not self.current_conversation:
            self.start_conversation()
        
        # Add user message to conversation
        self.current_conversation.add_message("user", message)
        
        # Set customer context if provided
        if "customer_context" in kwargs:
            self.rag_pipeline.set_customer_context(kwargs["customer_context"])
        
        # Get conversation history
        history = self.current_conversation.get_history(last_n=5)
        
        # Query RAG pipeline
        try:
            response: RAGResponse = self.rag_pipeline.query(
                question=message,
                conversation_history=history[:-1]  # Exclude current message
            )
            
            # Add assistant response to conversation
            self.current_conversation.add_message(
                "assistant",
                response.answer,
                metadata={
                    "sources": response.sources,
                    "confidence": response.confidence,
                    **response.metadata
                }
            )
            
            result = {
                "answer": response.answer,
                "sources": response.sources,
                "confidence": response.confidence,
                "conversation_id": self.current_conversation.id,
                "metadata": response.metadata
            }
            
            # Add intent and sentiment if available
            if "intent" in response.metadata:
                result["intent"] = response.metadata["intent"]
            if "sentiment" in response.metadata:
                result["sentiment"] = response.metadata["sentiment"]
            if response.metadata.get("should_escalate"):
                result["escalation_recommended"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            
            error_response = {
                "answer": "I apologize, but I'm having trouble processing your request. "
                         "Please try again or contact our support team for assistance.",
                "error": str(e),
                "conversation_id": self.current_conversation.id
            }
            
            self.current_conversation.add_message(
                "assistant",
                error_response["answer"],
                metadata={"error": str(e)}
            )
            
            return error_response
    
    def stream_chat(
        self,
        message: str,
        conversation_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream a chat response.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
            
        Yields:
            Response chunks
        """
        if conversation_id:
            self.set_conversation(conversation_id)
        
        if not self.current_conversation:
            self.start_conversation()
        
        self.current_conversation.add_message("user", message)
        
        # Get context from RAG
        history = self.current_conversation.get_history(last_n=5)
        
        # Get search results
        from ..retrieval.search import VectorSearchEngine, SearchConfig
        search_engine = VectorSearchEngine(self.vector_store)
        search_results = search_engine.search(
            message,
            SearchConfig(top_k=self.config.top_k)
        )
        
        # Build context
        from ..prompts.templates import RAGPromptTemplate, CustomerSupportPrompt
        context = RAGPromptTemplate.format_context([
            {"content": r.content, "metadata": r.metadata}
            for r in search_results
        ])
        
        # Build prompt
        prompt_template = RAGPromptTemplate()
        prompt = prompt_template.format(
            question=message,
            context=context,
            history="\n".join([f"{m['role']}: {m['content']}" for m in history[:-1]])
        )
        
        # Stream response
        full_response = []
        for chunk in self.llm_client.stream(
            prompt,
            system_prompt=CustomerSupportPrompt.get_system_prompt(),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        ):
            full_response.append(chunk)
            yield chunk
        
        # Save full response
        self.current_conversation.add_message(
            "assistant",
            "".join(full_response)
        )
    
    def get_suggested_responses(self, message: str, n: int = 3) -> List[str]:
        """
        Get suggested responses for a message.
        
        Args:
            message: User message
            n: Number of suggestions
            
        Returns:
            List of suggested responses
        """
        prompt = f"""Given this customer message, suggest {n} different helpful responses.

Customer message: {message}

Provide {n} concise, professional response options:"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=300, temperature=0.7)
            
            # Parse suggestions (assuming numbered list)
            suggestions = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove number/bullet
                    suggestion = line.lstrip("0123456789.-) ").strip()
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions[:n]
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    def get_conversation_summary(self) -> Optional[str]:
        """Get a summary of the current conversation."""
        if not self.current_conversation or len(self.current_conversation.messages) < 2:
            return None
        
        history = "\n".join([
            f"{m.role}: {m.content}"
            for m in self.current_conversation.messages
        ])
        
        prompt = f"""Summarize this customer support conversation in 2-3 sentences:

{history}

Summary:"""
        
        try:
            return self.llm_client.generate(prompt, max_tokens=150, temperature=0.3)
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
    
    def clear_conversation(self) -> None:
        """Clear the current conversation."""
        if self.current_conversation:
            self.current_conversation.messages = []
        self.rag_pipeline.clear_history()
    
    def end_conversation(self) -> Optional[Dict[str, Any]]:
        """
        End the current conversation and return summary.
        
        Returns:
            Conversation summary or None
        """
        if not self.current_conversation:
            return None
        
        summary = self.get_conversation_summary()
        
        result = {
            "conversation_id": self.current_conversation.id,
            "message_count": len(self.current_conversation.messages),
            "duration": (
                datetime.now() - self.current_conversation.created_at
            ).total_seconds(),
            "summary": summary
        }
        
        self.current_conversation = None
        return result


def create_chatbot(
    knowledge_base_path: Optional[str] = None,
    llm_provider: str = "openai",
    embedding_provider: str = "sentence_transformer",
    **config_kwargs
) -> CustomerSupportChatbot:
    """
    Factory function to create a chatbot instance.
    
    Args:
        knowledge_base_path: Path to knowledge base documents
        llm_provider: LLM provider to use
        embedding_provider: Embedding model provider
        **config_kwargs: Additional RAG configuration
        
    Returns:
        Configured CustomerSupportChatbot instance
    """
    config = RAGConfig(**config_kwargs)
    
    embedding_model = get_embedding_model(embedding_provider)
    vector_store = get_vector_store("chroma", embedding_model=embedding_model)
    
    try:
        llm_client = get_llm_client(llm_provider)
    except Exception:
        logger.warning(f"Failed to initialize {llm_provider}, using default")
        llm_client = None
    
    return CustomerSupportChatbot(
        vector_store=vector_store,
        llm_client=llm_client,
        config=config,
        knowledge_base_path=knowledge_base_path
    )
