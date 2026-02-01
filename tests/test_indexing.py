"""
Tests for indexing module (embeddings and vector stores).
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
from typing import List

from src.indexing.embeddings import (
    EmbeddingModel,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings
)
from src.indexing.vector_stores import (
    VectorStore,
    ChromaVectorStore,
    FAISSVectorStore
)


class TestEmbeddingModels:
    """Tests for embedding models."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        with patch('src.indexing.embeddings.openai') as mock:
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1] * 1536),
                MagicMock(embedding=[0.2] * 1536),
            ]
            mock.OpenAI.return_value.embeddings.create.return_value = mock_response
            yield mock
    
    @pytest.mark.asyncio
    async def test_openai_embeddings_single_text(self, mock_openai_client):
        """Test embedding a single text."""
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result = await embedder.embed(["Hello world"])
        
        assert len(result) == 1
        assert len(result[0]) == 1536
    
    @pytest.mark.asyncio
    async def test_openai_embeddings_batch(self, mock_openai_client):
        """Test embedding multiple texts."""
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result = await embedder.embed(["Text 1", "Text 2"])
        
        assert len(result) == 2
        assert all(len(emb) == 1536 for emb in result)
    
    def test_openai_embeddings_dimension(self, mock_openai_client):
        """Test embedding dimension property."""
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        assert embedder.dimension == 1536
    
    @pytest.mark.asyncio
    async def test_openai_embeddings_empty_input(self, mock_openai_client):
        """Test embedding empty list."""
        embedder = OpenAIEmbeddings(api_key="test-key")
        
        result = await embedder.embed([])
        
        assert result == []


class TestSentenceTransformerEmbeddings:
    """Tests for SentenceTransformer embeddings."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        with patch('src.indexing.embeddings.SentenceTransformer') as mock:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock.return_value = mock_model
            yield mock
    
    @pytest.mark.asyncio
    async def test_sentence_transformer_embed(self, mock_sentence_transformer):
        """Test SentenceTransformer embedding."""
        embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        result = await embedder.embed(["Hello", "World"])
        
        assert len(result) == 2
        assert len(result[0]) == 384
    
    def test_sentence_transformer_dimension(self, mock_sentence_transformer):
        """Test SentenceTransformer dimension."""
        embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        assert embedder.dimension == 384


class TestChromaVectorStore:
    """Tests for ChromaDB vector store."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        model = Mock(spec=EmbeddingModel)
        model.dimension = 384
        model.embed = AsyncMock(return_value=[[0.1] * 384])
        return model
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock Chroma client."""
        with patch('src.indexing.vector_stores.chromadb') as mock:
            mock_collection = MagicMock()
            mock_collection.add = MagicMock()
            mock_collection.query = MagicMock(return_value={
                'ids': [['doc1', 'doc2']],
                'documents': [['Content 1', 'Content 2']],
                'metadatas': [[{'source': 'test1'}, {'source': 'test2'}]],
                'distances': [[0.1, 0.2]]
            })
            mock_collection.delete = MagicMock()
            mock_collection.count = MagicMock(return_value=10)
            
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock.Client.return_value = mock_client
            mock.PersistentClient.return_value = mock_client
            
            yield mock
    
    @pytest.mark.asyncio
    async def test_add_documents(self, mock_embedding_model, mock_chroma_client):
        """Test adding documents to ChromaDB."""
        store = ChromaVectorStore(
            embedding_model=mock_embedding_model,
            collection_name="test",
        )
        
        documents = [
            {"content": "Test content", "metadata": {"source": "test.txt"}}
        ]
        
        ids = await store.add_documents(documents)
        
        assert len(ids) == 1
    
    @pytest.mark.asyncio
    async def test_search(self, mock_embedding_model, mock_chroma_client):
        """Test searching in ChromaDB."""
        store = ChromaVectorStore(
            embedding_model=mock_embedding_model,
            collection_name="test",
        )
        
        results = await store.search("test query", top_k=3)
        
        assert len(results) > 0
        assert 'content' in results[0]
        assert 'score' in results[0]
    
    @pytest.mark.asyncio
    async def test_delete_documents(self, mock_embedding_model, mock_chroma_client):
        """Test deleting documents from ChromaDB."""
        store = ChromaVectorStore(
            embedding_model=mock_embedding_model,
            collection_name="test",
        )
        
        await store.delete_documents(["doc1", "doc2"])
        
        # Verify delete was called
        store._collection.delete.assert_called_once()


class TestFAISSVectorStore:
    """Tests for FAISS vector store."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        model = Mock(spec=EmbeddingModel)
        model.dimension = 384
        model.embed = AsyncMock(return_value=[[0.1] * 384])
        return model
    
    @pytest.fixture
    def mock_faiss(self):
        """Create a mock FAISS."""
        with patch('src.indexing.vector_stores.faiss') as mock:
            mock_index = MagicMock()
            mock_index.add = MagicMock()
            mock_index.search = MagicMock(return_value=(
                np.array([[0.1, 0.2, 0.3]]),  # distances
                np.array([[0, 1, 2]])  # indices
            ))
            mock_index.ntotal = 10
            
            mock.IndexFlatL2.return_value = mock_index
            mock.IndexIVFFlat.return_value = mock_index
            
            yield mock
    
    @pytest.mark.asyncio
    async def test_add_documents(self, mock_embedding_model, mock_faiss):
        """Test adding documents to FAISS."""
        store = FAISSVectorStore(
            embedding_model=mock_embedding_model,
            dimension=384
        )
        
        documents = [
            {"content": "Test content", "metadata": {"source": "test.txt"}}
        ]
        
        ids = await store.add_documents(documents)
        
        assert len(ids) == 1
    
    @pytest.mark.asyncio
    async def test_search(self, mock_embedding_model, mock_faiss):
        """Test searching in FAISS."""
        store = FAISSVectorStore(
            embedding_model=mock_embedding_model,
            dimension=384
        )
        
        # Add some documents first
        await store.add_documents([
            {"content": "Content 1", "metadata": {}},
            {"content": "Content 2", "metadata": {}},
            {"content": "Content 3", "metadata": {}},
        ])
        
        results = await store.search("test query", top_k=3)
        
        # Results depend on mock setup
        assert isinstance(results, list)


class TestVectorStoreAbstraction:
    """Tests for vector store abstraction."""
    
    def test_vector_store_is_abstract(self):
        """Test that VectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorStore()
    
    def test_embedding_model_is_abstract(self):
        """Test that EmbeddingModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingModel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
