"""
Customer Support Chatbot - Main Application Entry Point

This module provides the main entry point for running the customer support chatbot.
It supports both interactive CLI mode and programmatic usage.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import config
from src.chatbot.chat_handler import CustomerSupportChatbot, ConversationManager
from src.generation.rag_pipeline import RAGPipeline
from src.generation.llm import OpenAIClient, AnthropicClient
from src.indexing.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from src.indexing.vector_stores import ChromaVectorStore, FAISSVectorStore
from src.retrieval.search import VectorSearchEngine, HybridSearchEngine
from src.retrieval.reranking import CrossEncoderReranker, LLMReranker
from src.document_processing.parsers import DocumentParser, TextParser, PDFParser, DocxParser, HTMLParser
from src.document_processing.chunking import (
    FixedSizeChunker,
    SemanticChunker,
    RecursiveChunker,
    SentenceChunker
)


class ChatbotApplication:
    """Main application class for the Customer Support Chatbot."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        embedding_provider: str = "openai",
        vector_store_type: str = "chroma",
        search_type: str = "hybrid",
        use_reranking: bool = True
    ):
        """
        Initialize the chatbot application.
        
        Args:
            llm_provider: LLM provider ("openai" or "anthropic")
            embedding_provider: Embedding provider ("openai" or "sentence_transformers")
            vector_store_type: Vector store type ("chroma" or "faiss")
            search_type: Search type ("vector" or "hybrid")
            use_reranking: Whether to use reranking
        """
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.vector_store_type = vector_store_type
        self.search_type = search_type
        self.use_reranking = use_reranking
        
        self.chatbot: Optional[CustomerSupportChatbot] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.conversation_manager = ConversationManager()
        
    def _create_llm_client(self):
        """Create the LLM client based on configuration."""
        if self.llm_provider == "anthropic":
            return AnthropicClient(
                api_key=config.anthropic.api_key,
                model=config.anthropic.model
            )
        elif self.llm_provider == "ollama":
            from src.generation.llm import OllamaClient
            return OllamaClient(model="llama3.2:1b")
        else:
            return OpenAIClient(
                api_key=config.openai.api_key,
                model=config.openai.model
            )
    
    def _create_embedding_model(self):
        """Create the embedding model based on configuration."""
        if self.embedding_provider == "sentence_transformers":
            return SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            return OpenAIEmbeddings(
                api_key=config.openai.api_key,
                model=config.openai.embedding_model
            )
    
    def _create_vector_store(self, embedding_model):
        """Create the vector store based on configuration."""
        persist_dir = Path(config.vector_store.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        if self.vector_store_type == "faiss":
            return FAISSVectorStore(
                embedding_model=embedding_model,
                dimension=embedding_model.dimension
            )
        else:
            return ChromaVectorStore(
                embedding_model=embedding_model,
                collection_name="customer_support",
                persist_directory=str(persist_dir)
            )
    
    def _create_search_engine(self, vector_store):
        """Create the search engine based on configuration."""
        if self.search_type == "hybrid":
            return HybridSearchEngine(
                vector_store=vector_store,
                keyword_weight=0.3
            )
        else:
            return VectorSearchEngine(vector_store=vector_store)
    
    def _create_reranker(self, llm_client):
        """Create the reranker if enabled."""
        if not self.use_reranking:
            return None
        
        # Use LLM-based reranking for better quality
        return LLMReranker(llm_client=llm_client)
    
    def _get_document_parser(self, file_path: Path) -> DocumentParser:
        """Get the appropriate parser for a file."""
        suffix = file_path.suffix.lower()
        parsers = {
            '.pdf': PDFParser(),
            '.docx': DocxParser(),
            '.html': HTMLParser(),
            '.htm': HTMLParser(),
            '.txt': TextParser(),
            '.md': TextParser(),  # Markdown files handled as text
        }
        return parsers.get(suffix, TextParser())
    
    async def initialize(self):
        """Initialize all components of the chatbot."""
        print("🚀 Initializing Customer Support Chatbot...")
        
        # Create components
        print("  📦 Setting up LLM client...")
        llm_client = self._create_llm_client()
        
        print("  🔤 Setting up embedding model...")
        embedding_model = self._create_embedding_model()
        
        print("  💾 Setting up vector store...")
        vector_store = self._create_vector_store(embedding_model)
        
        print("  🔍 Setting up search engine...")
        search_engine = self._create_search_engine(vector_store)
        
        print("  📊 Setting up reranker...")
        reranker = self._create_reranker(llm_client) if self.use_reranking else None
        
        print("  ⚙️ Creating RAG pipeline...")
        self.rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_client=llm_client,
            reranker=reranker
        )
        
        print("  🤖 Initializing chatbot...")
        self.chatbot = CustomerSupportChatbot(
            vector_store=vector_store,
            llm_client=llm_client,
            embedding_model=embedding_model
        )
        
        print("✅ Chatbot initialized successfully!\n")
        
    async def ingest_knowledge_base(self, directory: str = "data/knowledge_base"):
        """
        Ingest documents from the knowledge base directory.
        
        Args:
            directory: Path to the knowledge base directory
        """
        kb_path = Path(directory)
        if not kb_path.exists():
            print(f"⚠️ Knowledge base directory not found: {directory}")
            return
        
        print(f"📚 Ingesting knowledge base from {directory}...")
        
        documents = []
        for file_path in kb_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.pdf', '.docx', '.html']:
                try:
                    parser = self._get_document_parser(file_path)
                    parsed_doc = parser.parse(str(file_path))
                    # Extract content string from Document object
                    content = parsed_doc.content if hasattr(parsed_doc, 'content') else str(parsed_doc)
                    documents.append({
                        "content": content,
                        "metadata": {
                            "source": str(file_path),
                            "filename": file_path.name,
                            "type": file_path.suffix
                        }
                    })
                    print(f"  ✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"  ✗ Error loading {file_path.name}: {e}")
        
        if documents:
            print(f"\n📄 Processing {len(documents)} documents...")
            # Extract texts and metadata for vector store
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            # Ingest directly to vector store
            self.chatbot.vector_store.add_documents(texts, metadatas=metadatas)
            print(f"✅ Knowledge base ingestion complete!\n")
        else:
            print("⚠️ No documents found to ingest.\n")
    
    async def chat(self, message: str, user_id: str = "default") -> str:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: User message
            user_id: User identifier for conversation tracking
            
        Returns:
            Chatbot response
        """
        if not self.chatbot:
            raise RuntimeError("Chatbot not initialized. Call initialize() first.")
        
        response = self.chatbot.chat(message, conversation_id=user_id)
        return response.get("answer", str(response))
    
    async def chat_stream(self, message: str, user_id: str = "default"):
        """
        Send a message and stream the response.
        
        Args:
            message: User message
            user_id: User identifier
            
        Yields:
            Response chunks
        """
        if not self.chatbot:
            raise RuntimeError("Chatbot not initialized. Call initialize() first.")
        
        async for chunk in self.chatbot.handle_message_stream(message, user_id):
            yield chunk
    
    async def run_interactive(self):
        """Run the chatbot in interactive CLI mode."""
        print("=" * 60)
        print("🤖 TechCorp Customer Support Chatbot")
        print("=" * 60)
        print("\nHello! I'm your TechCorp support assistant.")
        print("I can help you with:")
        print("  • Product information (TechCloud, SecureShield, DataSync)")
        print("  • Technical troubleshooting")
        print("  • Billing and account questions")
        print("  • Policies and procedures")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to start a new conversation.")
        print("-" * 60 + "\n")
        
        user_id = "interactive_user"
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using TechCorp Support. Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_manager.clear_history(user_id)
                    print("\n🔄 Conversation cleared. Starting fresh!\n")
                    continue
                
                print("\n🤔 Thinking...\n")
                
                # Get response
                response = await self.chat(user_input, user_id)
                
                print(f"Assistant: {response}\n")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                continue


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TechCorp Customer Support Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run interactive chatbot
  python main.py --llm ollama             # Use free local Ollama LLM
  python main.py --llm anthropic          # Use Anthropic Claude
  python main.py --no-reranking           # Disable reranking
  python main.py --ingest-only            # Only ingest documents
  python main.py --query "How do I reset my password?"
        """
    )
    
    parser.add_argument(
        "--llm",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai, use 'ollama' for free local)"
    )
    
    parser.add_argument(
        "--embeddings",
        choices=["openai", "sentence_transformers"],
        default="openai",
        help="Embedding provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--vector-store",
        choices=["chroma", "faiss"],
        default="chroma",
        help="Vector store to use (default: chroma)"
    )
    
    parser.add_argument(
        "--search",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Search type to use (default: hybrid)"
    )
    
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable result reranking"
    )
    
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest documents, don't start chat"
    )
    
    parser.add_argument(
        "--kb-path",
        default="data/knowledge_base",
        help="Path to knowledge base directory (default: data/knowledge_base)"
    )
    
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single query mode - ask a question and exit"
    )
    
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip knowledge base ingestion"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = ChatbotApplication(
        llm_provider=args.llm,
        embedding_provider=args.embeddings,
        vector_store_type=args.vector_store,
        search_type=args.search,
        use_reranking=not args.no_reranking
    )
    
    # Initialize
    await app.initialize()
    
    # Ingest knowledge base
    if not args.skip_ingest:
        await app.ingest_knowledge_base(args.kb_path)
    
    # Check if ingest-only mode
    if args.ingest_only:
        print("✅ Document ingestion complete. Exiting.")
        return
    
    # Check if single query mode
    if args.query:
        print(f"\n📝 Query: {args.query}\n")
        response = await app.chat(args.query)
        print(f"🤖 Response:\n{response}\n")
        return
    
    # Run interactive mode
    await app.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
