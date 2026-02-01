"""
Tests configuration - pytest fixtures and settings.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = Mock()
    client.generate = AsyncMock(return_value="Mock response")
    client.generate_stream = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    model = Mock()
    model.dimension = 384
    model.embed = AsyncMock(return_value=[[0.1] * 384])
    return model


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    store = Mock()
    store.add_documents = AsyncMock(return_value=["id1", "id2"])
    store.search = AsyncMock(return_value=[
        {"content": "Result 1", "score": 0.9, "metadata": {}},
        {"content": "Result 2", "score": 0.8, "metadata": {}},
    ])
    store.delete_documents = AsyncMock()
    return store


@pytest.fixture
def mock_search_engine(mock_vector_store):
    """Create a mock search engine for testing."""
    engine = Mock()
    engine.search = AsyncMock(return_value=[
        {"content": "Search result 1", "score": 0.95, "metadata": {"source": "doc1"}},
        {"content": "Search result 2", "score": 0.85, "metadata": {"source": "doc2"}},
    ])
    return engine


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "TechCloud is our flagship cloud platform offering scalable infrastructure.",
            "metadata": {"source": "products.md", "topic": "product"}
        },
        {
            "content": "To reset your password, go to Settings > Security > Reset Password.",
            "metadata": {"source": "faq.md", "topic": "account"}
        },
        {
            "content": "The Pro plan costs $99/month and includes priority support.",
            "metadata": {"source": "pricing.md", "topic": "billing"}
        },
        {
            "content": "If you experience slow performance, try clearing your cache.",
            "metadata": {"source": "troubleshooting.md", "topic": "technical"}
        },
    ]


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    from src.document_processing.chunking import TextChunk
    
    return [
        TextChunk(
            content="This is the first chunk of text.",
            metadata={"chunk_index": 0, "source": "test.txt"}
        ),
        TextChunk(
            content="This is the second chunk of text.",
            metadata={"chunk_index": 1, "source": "test.txt"}
        ),
        TextChunk(
            content="This is the third chunk of text.",
            metadata={"chunk_index": 2, "source": "test.txt"}
        ),
    ]


@pytest.fixture
def sample_conversation():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello, I need help with my account."},
        {"role": "assistant", "content": "Hello! I'd be happy to help. What issue are you experiencing?"},
        {"role": "user", "content": "I can't log in."},
        {"role": "assistant", "content": "I'm sorry to hear that. Have you tried resetting your password?"},
    ]


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: marks tests that require API keys"
    )
