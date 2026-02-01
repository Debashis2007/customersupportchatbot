"""
Example: Retrieval and Search

This script demonstrates how to:
1. Set up vector and hybrid search
2. Perform similarity search
3. Use reranking strategies
4. Compare different retrieval approaches
"""

import asyncio
from typing import List, Dict

from src.indexing.embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings
from src.indexing.vector_stores import ChromaVectorStore, FAISSVectorStore
from src.retrieval.search import VectorSearchEngine, HybridSearchEngine
from src.retrieval.reranking import (
    CrossEncoderReranker,
    ReciprocalRankFusion,
    Reranker
)
from src.config import config


# Sample knowledge base for demonstration
SAMPLE_DOCUMENTS = [
    {
        "content": "To reset your TechCloud password, go to the login page and click 'Forgot Password'. Enter your email address and we'll send you a reset link. The link expires in 24 hours.",
        "metadata": {"topic": "account", "category": "password_reset"}
    },
    {
        "content": "TechCloud pricing starts at $29/month for the Basic plan. The Professional plan is $99/month and includes advanced features like auto-scaling and priority support.",
        "metadata": {"topic": "billing", "category": "pricing"}
    },
    {
        "content": "If you're experiencing slow performance on TechCloud, first check your resource usage in the dashboard. You may need to upgrade your plan or optimize your application.",
        "metadata": {"topic": "troubleshooting", "category": "performance"}
    },
    {
        "content": "TechCloud supports deployment in multiple regions including US-East, US-West, EU-West, and Asia-Pacific. Choose the region closest to your users for best performance.",
        "metadata": {"topic": "features", "category": "regions"}
    },
    {
        "content": "To change your billing information, log into your account, go to Settings > Billing, and update your payment method. Changes take effect on your next billing cycle.",
        "metadata": {"topic": "billing", "category": "payment"}
    },
    {
        "content": "TechCloud's auto-scaling feature automatically adjusts your resources based on traffic. Enable it in your project settings under 'Scaling Configuration'.",
        "metadata": {"topic": "features", "category": "scaling"}
    },
    {
        "content": "For security, we recommend enabling two-factor authentication (2FA) on your TechCloud account. Go to Settings > Security > Enable 2FA.",
        "metadata": {"topic": "security", "category": "authentication"}
    },
    {
        "content": "If you need to delete your TechCloud account, contact our support team. Please note that all data will be permanently removed after a 30-day grace period.",
        "metadata": {"topic": "account", "category": "deletion"}
    },
]


async def setup_vector_store() -> ChromaVectorStore:
    """Set up a vector store with sample documents."""
    print("\n📚 Setting up vector store with sample documents...")
    
    # Try sentence transformers first (no API key needed)
    try:
        embedding_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        if config.openai.api_key:
            embedding_model = OpenAIEmbeddings(api_key=config.openai.api_key)
        else:
            raise RuntimeError("No embedding model available")
    
    vector_store = ChromaVectorStore(
        embedding_model=embedding_model,
        collection_name="retrieval_demo",
        persist_directory="./demo_data/retrieval"
    )
    
    # Add documents
    await vector_store.add_documents(SAMPLE_DOCUMENTS)
    print(f"   Added {len(SAMPLE_DOCUMENTS)} documents")
    
    return vector_store


async def demo_vector_search(vector_store: ChromaVectorStore):
    """Demonstrate basic vector search."""
    print("\n" + "=" * 60)
    print("🔍 VECTOR SEARCH DEMO")
    print("=" * 60)
    
    search_engine = VectorSearchEngine(vector_store=vector_store)
    
    queries = [
        "How do I reset my password?",
        "What is the pricing for TechCloud?",
        "My app is running slow",
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        results = await search_engine.search(query, top_k=3)
        
        print("   Top 3 results:")
        for i, result in enumerate(results):
            score = result.get('score', 0)
            preview = result['content'][:60].replace('\n', ' ')
            category = result.get('metadata', {}).get('category', 'unknown')
            print(f"   {i+1}. [{category}] Score: {score:.4f} - {preview}...")


async def demo_hybrid_search(vector_store: ChromaVectorStore):
    """Demonstrate hybrid search (vector + keyword)."""
    print("\n" + "=" * 60)
    print("🔀 HYBRID SEARCH DEMO")
    print("=" * 60)
    
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    
    queries = [
        "password reset email",
        "billing payment update",
        "auto-scaling configuration",
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        results = await hybrid_engine.search(query, top_k=3)
        
        print("   Top 3 hybrid results:")
        for i, result in enumerate(results):
            score = result.get('score', 0)
            preview = result['content'][:60].replace('\n', ' ')
            print(f"   {i+1}. Score: {score:.4f} - {preview}...")


async def demo_search_comparison(vector_store: ChromaVectorStore):
    """Compare vector vs hybrid search results."""
    print("\n" + "=" * 60)
    print("⚖️ SEARCH COMPARISON DEMO")
    print("=" * 60)
    
    vector_engine = VectorSearchEngine(vector_store=vector_store)
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        vector_weight=0.6,
        keyword_weight=0.4
    )
    
    query = "enable two-factor security authentication"
    print(f"\n📝 Query: '{query}'")
    
    # Vector search
    print("\n   📊 Vector Search Results:")
    vector_results = await vector_engine.search(query, top_k=3)
    for i, result in enumerate(vector_results):
        category = result.get('metadata', {}).get('category', 'unknown')
        print(f"   {i+1}. [{category}] {result['content'][:50]}...")
    
    # Hybrid search
    print("\n   📊 Hybrid Search Results:")
    hybrid_results = await hybrid_engine.search(query, top_k=3)
    for i, result in enumerate(hybrid_results):
        category = result.get('metadata', {}).get('category', 'unknown')
        print(f"   {i+1}. [{category}] {result['content'][:50]}...")


async def demo_reranking(vector_store: ChromaVectorStore):
    """Demonstrate reranking strategies."""
    print("\n" + "=" * 60)
    print("📊 RERANKING DEMO")
    print("=" * 60)
    
    search_engine = VectorSearchEngine(vector_store=vector_store)
    query = "How much does TechCloud cost per month?"
    
    # Initial search
    print(f"\n📝 Query: '{query}'")
    initial_results = await search_engine.search(query, top_k=5)
    
    print("\n   Initial ranking:")
    for i, result in enumerate(initial_results):
        preview = result['content'][:50].replace('\n', ' ')
        print(f"   {i+1}. {preview}...")
    
    # Try cross-encoder reranking if available
    try:
        print("\n   Attempting cross-encoder reranking...")
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked = await reranker.rerank(query, initial_results, top_k=3)
        
        print("   After cross-encoder reranking:")
        for i, result in enumerate(reranked):
            preview = result['content'][:50].replace('\n', ' ')
            print(f"   {i+1}. {preview}...")
    except Exception as e:
        print(f"   ⚠️ Cross-encoder not available: {e}")


async def demo_metadata_filtering(vector_store: ChromaVectorStore):
    """Demonstrate search with metadata filtering."""
    print("\n" + "=" * 60)
    print("🏷️ METADATA FILTERING DEMO")
    print("=" * 60)
    
    search_engine = VectorSearchEngine(vector_store=vector_store)
    
    query = "How do I change settings?"
    
    print(f"\n📝 Query: '{query}'")
    
    # Search without filter
    print("\n   Without filter:")
    results = await search_engine.search(query, top_k=3)
    for i, result in enumerate(results):
        topic = result.get('metadata', {}).get('topic', 'unknown')
        preview = result['content'][:40].replace('\n', ' ')
        print(f"   {i+1}. [{topic}] {preview}...")
    
    # Search with topic filter (if supported)
    print("\n   With filter (topic='billing'):")
    # Note: Filter implementation depends on vector store
    filtered_results = await search_engine.search(
        query, 
        top_k=3,
        filters={"topic": "billing"}
    )
    for i, result in enumerate(filtered_results):
        topic = result.get('metadata', {}).get('topic', 'unknown')
        preview = result['content'][:40].replace('\n', ' ')
        print(f"   {i+1}. [{topic}] {preview}...")


async def main():
    """Run all retrieval demos."""
    print("\n" + "🔍" * 30)
    print("   RETRIEVAL AND SEARCH EXAMPLES")
    print("🔍" * 30)
    
    # Setup
    vector_store = await setup_vector_store()
    
    # Demo 1: Basic Vector Search
    await demo_vector_search(vector_store)
    
    # Demo 2: Hybrid Search
    await demo_hybrid_search(vector_store)
    
    # Demo 3: Search Comparison
    await demo_search_comparison(vector_store)
    
    # Demo 4: Reranking
    await demo_reranking(vector_store)
    
    # Demo 5: Metadata Filtering
    await demo_metadata_filtering(vector_store)
    
    print("\n" + "=" * 60)
    print("✅ All retrieval demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
