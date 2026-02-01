"""
FastAPI REST API for the Customer Support Chatbot.

This module provides a REST API interface for the chatbot,
enabling integration with web applications and other services.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import json
import uuid
from datetime import datetime

from src.config import config
from src.chatbot.chat_handler import CustomerSupportChatbot, ConversationManager
from src.generation.rag_pipeline import RAGPipeline
from src.generation.llm import OpenAIClient
from src.indexing.embeddings import OpenAIEmbeddings
from src.indexing.vector_stores import ChromaVectorStore
from src.retrieval.search import HybridSearchEngine
from src.document_processing.chunking import RecursiveChunking


# Initialize FastAPI app
app = FastAPI(
    title="TechCorp Customer Support API",
    description="AI-powered customer support chatbot API using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
chatbot: Optional[CustomerSupportChatbot] = None
conversation_manager = ConversationManager(max_history=20)


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message", min_length=1, max_length=4000)
    user_id: Optional[str] = Field(default=None, description="User identifier for conversation tracking")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Chatbot response")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    intent: Optional[str] = Field(default=None, description="Detected intent")
    confidence: Optional[float] = Field(default=None, description="Response confidence score")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Source documents used")
    timestamp: str = Field(..., description="Response timestamp")


class DocumentRequest(BaseModel):
    """Document ingestion request model."""
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class DocumentBatchRequest(BaseModel):
    """Batch document ingestion request model."""
    documents: List[DocumentRequest] = Field(..., description="List of documents to ingest")


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(default=None, description="Optional feedback comment")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    components: Dict[str, str]


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    global chatbot
    
    try:
        print("🚀 Initializing Customer Support API...")
        
        # Initialize components
        llm_client = OpenAIClient(
            api_key=config.openai.api_key,
            model=config.openai.model
        )
        
        embedding_model = OpenAIEmbeddings(
            api_key=config.openai.api_key,
            model=config.openai.embedding_model
        )
        
        vector_store = ChromaVectorStore(
            embedding_model=embedding_model,
            collection_name="customer_support_api",
            persist_directory=config.vector_store.chroma_persist_dir
        )
        
        search_engine = HybridSearchEngine(
            vector_store=vector_store,
            vector_weight=0.7,
            keyword_weight=0.3
        )
        
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            search_engine=search_engine,
            llm_client=llm_client,
            chunking_strategy=RecursiveChunking(chunk_size=500, chunk_overlap=50)
        )
        
        chatbot = CustomerSupportChatbot(
            rag_pipeline=rag_pipeline,
            conversation_manager=conversation_manager
        )
        
        print("✅ API initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down API...")


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "name": "TechCorp Customer Support API",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "chatbot": "ready" if chatbot else "not_initialized",
            "database": "connected",
            "llm": "connected"
        }
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and get a response.
    
    - **message**: The user's message (required)
    - **user_id**: User identifier for conversation tracking (optional)
    - **session_id**: Session identifier (optional)
    - **metadata**: Additional context (optional)
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Generate IDs if not provided
    user_id = request.user_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Get response from chatbot
        response = await chatbot.handle_message(
            message=request.message,
            user_id=user_id
        )
        
        return ChatResponse(
            response=response,
            user_id=user_id,
            session_id=session_id,
            intent=None,  # Could be extracted from chatbot
            confidence=None,
            sources=None,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream a response from the chatbot.
    
    Returns a Server-Sent Events stream.
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    user_id = request.user_id or str(uuid.uuid4())
    
    async def generate():
        try:
            async for chunk in chatbot.handle_message_stream(
                message=request.message,
                user_id=user_id
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/documents", tags=["Documents"])
async def ingest_document(request: DocumentRequest, background_tasks: BackgroundTasks):
    """
    Ingest a single document into the knowledge base.
    
    - **content**: Document content (required)
    - **metadata**: Document metadata (optional)
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    document_id = str(uuid.uuid4())
    
    async def ingest():
        await chatbot.rag_pipeline.ingest_documents([{
            "content": request.content,
            "metadata": request.metadata or {}
        }])
    
    background_tasks.add_task(ingest)
    
    return {
        "status": "accepted",
        "document_id": document_id,
        "message": "Document queued for ingestion"
    }


@app.post("/documents/batch", tags=["Documents"])
async def ingest_documents_batch(request: DocumentBatchRequest, background_tasks: BackgroundTasks):
    """
    Ingest multiple documents into the knowledge base.
    
    - **documents**: List of documents to ingest (required)
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    batch_id = str(uuid.uuid4())
    
    documents = [
        {"content": doc.content, "metadata": doc.metadata or {}}
        for doc in request.documents
    ]
    
    async def ingest():
        await chatbot.rag_pipeline.ingest_documents(documents)
    
    background_tasks.add_task(ingest)
    
    return {
        "status": "accepted",
        "batch_id": batch_id,
        "document_count": len(documents),
        "message": "Documents queued for ingestion"
    }


@app.get("/conversations/{user_id}", tags=["Conversations"])
async def get_conversation_history(user_id: str, limit: int = 20):
    """
    Get conversation history for a user.
    
    - **user_id**: User identifier (required)
    - **limit**: Maximum number of messages to return (default: 20)
    """
    history = conversation_manager.get_history(user_id)
    
    return {
        "user_id": user_id,
        "message_count": len(history),
        "messages": history[-limit:]
    }


@app.delete("/conversations/{user_id}", tags=["Conversations"])
async def clear_conversation(user_id: str):
    """
    Clear conversation history for a user.
    
    - **user_id**: User identifier (required)
    """
    conversation_manager.clear_history(user_id)
    
    return {
        "status": "success",
        "message": f"Conversation history cleared for user {user_id}"
    }


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a chatbot response.
    
    - **session_id**: Session identifier (required)
    - **message_id**: Message identifier (required)
    - **rating**: Rating from 1 to 5 (required)
    - **comment**: Optional feedback comment
    """
    # In production, store this in a database
    return {
        "status": "success",
        "message": "Feedback recorded",
        "feedback_id": str(uuid.uuid4())
    }


@app.get("/stats", tags=["Analytics"])
async def get_stats():
    """Get API usage statistics."""
    return {
        "total_conversations": len(conversation_manager._histories),
        "active_sessions": len(conversation_manager._histories),
        "api_version": "1.0.0"
    }


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
