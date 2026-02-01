"""
Example: Document Ingestion and Indexing

This script demonstrates how to:
1. Parse documents from various formats
2. Chunk documents using different strategies
3. Create embeddings
4. Store vectors in a vector database
"""

import asyncio
from pathlib import Path

from src.document_processing.parsers import (
    TextParser,
    PDFParser,
    DocxParser,
    HTMLParser,
    AIDocumentParser
)
from src.document_processing.chunking import (
    FixedSizeChunking,
    SemanticChunking,
    RecursiveChunking,
    SentenceChunking
)
from src.indexing.embeddings import (
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings
)
from src.indexing.vector_stores import (
    ChromaVectorStore,
    FAISSVectorStore
)
from src.config import config


async def demo_document_parsing():
    """Demonstrate document parsing."""
    print("\n" + "=" * 60)
    print("📄 DOCUMENT PARSING DEMO")
    print("=" * 60)
    
    # Sample text for demonstration
    sample_text = """
    # TechCloud Documentation
    
    TechCloud is our flagship cloud platform offering scalable 
    infrastructure solutions. It supports multiple deployment 
    options including public, private, and hybrid cloud.
    
    ## Features
    - Auto-scaling based on demand
    - Load balancing across regions
    - 99.99% uptime SLA
    - 24/7 monitoring and support
    
    ## Getting Started
    1. Sign up for an account
    2. Choose your plan
    3. Deploy your first application
    """
    
    # Parse text
    parser = TextParser()
    # Note: In real usage, you'd parse from a file:
    # content = parser.parse("path/to/document.txt")
    
    print("\n✅ Text parsed successfully!")
    print(f"   Sample content (first 100 chars): {sample_text[:100]}...")
    
    return sample_text


async def demo_chunking_strategies(text: str):
    """Demonstrate different chunking strategies."""
    print("\n" + "=" * 60)
    print("✂️ CHUNKING STRATEGIES DEMO")
    print("=" * 60)
    
    strategies = [
        ("Fixed Size", FixedSizeChunking(chunk_size=200, chunk_overlap=20)),
        ("Recursive", RecursiveChunking(chunk_size=200, chunk_overlap=20)),
        ("Sentence", SentenceChunking(sentences_per_chunk=3, overlap_sentences=1)),
    ]
    
    for name, strategy in strategies:
        chunks = strategy.chunk(text)
        print(f"\n📦 {name} Chunking:")
        print(f"   Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            preview = chunk.content[:80].replace('\n', ' ')
            print(f"   Chunk {i+1}: {preview}...")
    
    return strategies[1][1].chunk(text)  # Return recursive chunks


async def demo_embeddings(chunks):
    """Demonstrate embedding generation."""
    print("\n" + "=" * 60)
    print("🔤 EMBEDDINGS DEMO")
    print("=" * 60)
    
    # Using Sentence Transformers (doesn't require API key)
    print("\n📊 Using Sentence Transformers (local model)...")
    
    try:
        embedding_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        texts = [chunk.content for chunk in chunks[:3]]
        embeddings = await embedding_model.embed(texts)
        
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   First embedding (first 5 values): {embeddings[0][:5]}")
        
        return embedding_model
        
    except Exception as e:
        print(f"   ⚠️ Sentence Transformers not available: {e}")
        print("   Trying OpenAI embeddings...")
        
        if config.openai.api_key:
            embedding_model = OpenAIEmbeddings(
                api_key=config.openai.api_key
            )
            texts = [chunk.content for chunk in chunks[:3]]
            embeddings = await embedding_model.embed(texts)
            
            print(f"   Generated {len(embeddings)} embeddings")
            print(f"   Embedding dimension: {len(embeddings[0])}")
            
            return embedding_model
        else:
            print("   ❌ No embedding model available. Set OPENAI_API_KEY.")
            return None


async def demo_vector_store(embedding_model, chunks):
    """Demonstrate vector store operations."""
    print("\n" + "=" * 60)
    print("💾 VECTOR STORE DEMO")
    print("=" * 60)
    
    if not embedding_model:
        print("   ⚠️ Skipping - no embedding model available")
        return
    
    # Create ChromaDB vector store
    print("\n📚 Creating ChromaDB vector store...")
    
    vector_store = ChromaVectorStore(
        embedding_model=embedding_model,
        collection_name="demo_collection",
        persist_directory="./demo_data/chroma"
    )
    
    # Add documents
    documents = [
        {
            "content": chunk.content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    ids = await vector_store.add_documents(documents)
    print(f"   Added {len(ids)} documents to vector store")
    
    # Search
    query = "What are the features of TechCloud?"
    print(f"\n🔍 Searching for: '{query}'")
    
    results = await vector_store.search(query, top_k=3)
    
    print(f"   Found {len(results)} results:")
    for i, result in enumerate(results):
        preview = result['content'][:60].replace('\n', ' ')
        print(f"   {i+1}. Score: {result['score']:.4f} - {preview}...")
    
    return vector_store


async def main():
    """Run all demos."""
    print("\n" + "🚀" * 30)
    print("   DOCUMENT INGESTION & INDEXING EXAMPLES")
    print("🚀" * 30)
    
    # Demo 1: Document Parsing
    text = await demo_document_parsing()
    
    # Demo 2: Chunking
    chunks = await demo_chunking_strategies(text)
    
    # Demo 3: Embeddings
    embedding_model = await demo_embeddings(chunks)
    
    # Demo 4: Vector Store
    await demo_vector_store(embedding_model, chunks)
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
