#!/usr/bin/env python3
"""
Customer Support Chatbot - Web UI
A beautiful Gradio-based web interface for the RAG chatbot.
"""

import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.generation.llm import OllamaClient
from src.indexing.embeddings import SentenceTransformerEmbeddings
from src.indexing.vector_stores import ChromaVectorStore
from src.retrieval.search import VectorSearchEngine, SearchConfig
from src.generation.rag_pipeline import RAGPipeline, RAGConfig
from src.document_processing.parsers import TextParser

# Global components
chatbot_instance = None


def initialize_chatbot():
    """Initialize the chatbot components."""
    global chatbot_instance
    
    if chatbot_instance is not None:
        return chatbot_instance
    
    print("🚀 Initializing Customer Support Chatbot...")
    
    # Create LLM client (Ollama - free and local)
    print("  📦 Setting up LLM client (Ollama)...")
    llm_client = OllamaClient(model="llama3.2:1b")
    
    # Create embedding model (Sentence Transformers - free and local)
    print("  🔤 Setting up embedding model...")
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    print("  💾 Setting up vector store...")
    persist_dir = Path("./data/chroma_db")
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    vector_store = ChromaVectorStore(
        embedding_model=embedding_model,
        collection_name="customer_support_ui",
        persist_directory=str(persist_dir)
    )
    
    # Load knowledge base
    print("  📚 Loading knowledge base...")
    kb_path = Path("./data/knowledge_base")
    if kb_path.exists():
        parser = TextParser()
        documents = []
        for file_path in kb_path.glob("*.md"):
            try:
                parsed = parser.parse(str(file_path))
                content = parsed.content if hasattr(parsed, 'content') else str(parsed)
                documents.append({
                    "content": content,
                    "metadata": {"source": file_path.name}
                })
                print(f"    ✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"    ✗ Error: {file_path.name}: {e}")
        
        if documents:
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            vector_store.add_documents(texts, metadatas=metadatas)
            print(f"  ✅ Indexed {len(documents)} documents")
    
    # Create RAG pipeline
    print("  ⚙️ Creating RAG pipeline...")
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=llm_client,
        config=RAGConfig(
            top_k=3,
            temperature=0.7,
            max_tokens=512
        )
    )
    
    chatbot_instance = {
        "llm_client": llm_client,
        "vector_store": vector_store,
        "rag_pipeline": rag_pipeline
    }
    
    print("✅ Chatbot initialized successfully!\n")
    return chatbot_instance


def chat(message: str, history: list) -> str:
    """Process a chat message and return a response."""
    if not message.strip():
        return ""
    
    try:
        bot = initialize_chatbot()
        
        # Convert history to conversation format
        conversation_history = []
        for human, assistant in history:
            conversation_history.append({"role": "user", "content": human})
            if assistant:
                conversation_history.append({"role": "assistant", "content": assistant})
        
        # Query RAG pipeline
        response = bot["rag_pipeline"].query(
            question=message,
            conversation_history=conversation_history
        )
        
        return response.answer
        
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."


def create_ui():
    """Create the Gradio UI."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    footer {
        display: none !important;
    }
    """
    
    with gr.Blocks(
        title="TechCorp Customer Support",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        )
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🤖 TechCorp Customer Support Chatbot
        
        Welcome! I'm your AI-powered support assistant. I can help you with:
        - 📦 **Product Information** - TechCloud, SecureShield, DataSync
        - 🔧 **Technical Troubleshooting** - Setup, configuration, issues
        - 💳 **Billing & Accounts** - Payments, subscriptions, invoices
        - 📋 **Policies** - Returns, warranties, privacy
        
        *Powered by RAG (Retrieval Augmented Generation) with local LLM*
        """)
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Chat",
            height=450,
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=support"),
            bubble_full_width=False
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your question here... (e.g., 'What is your return policy?')",
                show_label=False,
                scale=9,
                container=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Example questions
        gr.Markdown("### 💡 Quick Questions")
        with gr.Row():
            gr.Button("What is your return policy?", size="sm").click(
                lambda: "What is your return policy?", outputs=msg
            )
            gr.Button("How do I reset my password?", size="sm").click(
                lambda: "How do I reset my password?", outputs=msg
            )
            gr.Button("What products do you offer?", size="sm").click(
                lambda: "What products do you offer?", outputs=msg
            )
            gr.Button("How do I contact support?", size="sm").click(
                lambda: "How do I contact support?", outputs=msg
            )
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        <small>
        🔒 Running locally with Ollama | 🆓 No API costs | 📚 RAG-powered responses
        </small>
        </center>
        """)
        
        # Event handlers
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            
            bot_message = chat(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        # Submit on button click
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        # Submit on Enter
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        # Clear chat
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    return demo


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║       TechCorp Customer Support Chatbot - Web UI             ║
╠══════════════════════════════════════════════════════════════╣
║  🌐 Starting web interface...                                ║
║  📍 Open http://localhost:7860 in your browser               ║
║  🛑 Press Ctrl+C to stop                                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize chatbot on startup
    initialize_chatbot()
    
    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
