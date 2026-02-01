# Customer Support Chatbot with RAGs and Prompt Engineering

A comprehensive customer support chatbot implementation demonstrating various adaptation techniques including RAG (Retrieval-Augmented Generation), prompt engineering, and evaluation methods.

## 🏗️ Project Structure

```
CustomerSupportChatbot/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── document_processing/      # Document parsing and chunking
│   │   ├── __init__.py
│   │   ├── parsers.py           # Rule-based and AI-based parsers
│   │   └── chunking.py          # Various chunking strategies
│   ├── indexing/                 # Indexing and vector stores
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Embedding models
│   │   └── vector_stores.py     # ChromaDB, FAISS implementations
│   ├── retrieval/               # Retrieval and search
│   │   ├── __init__.py
│   │   ├── search.py            # Exact and approximate NN search
│   │   └── reranking.py         # Result reranking strategies
│   ├── prompts/                 # Prompt engineering
│   │   ├── __init__.py
│   │   ├── templates.py         # Prompt templates
│   │   └── strategies.py        # Few-shot, CoT, role-based prompting
│   ├── generation/              # Response generation
│   │   ├── __init__.py
│   │   ├── llm.py              # LLM integrations (OpenAI, Anthropic)
│   │   └── rag_pipeline.py     # Complete RAG pipeline
│   ├── evaluation/              # RAG evaluation
│   │   ├── __init__.py
│   │   └── metrics.py          # Context relevance, faithfulness, etc.
│   └── chatbot/                 # Chatbot interface
│       ├── __init__.py
│       └── chat_handler.py     # Conversation management
├── data/
│   └── knowledge_base/          # Sample knowledge base documents
│       ├── faq.md               # Frequently asked questions
│       ├── product_docs.md      # Product documentation
│       ├── troubleshooting.md   # Troubleshooting guides
│       └── policies.md          # Company policies
├── examples/                    # Example scripts
│   ├── 01_document_ingestion.py # Document parsing and indexing
│   ├── 02_retrieval_search.py   # Search and retrieval examples
│   ├── 03_prompt_engineering.py # Prompt strategy examples
│   └── 04_evaluation.py         # RAG evaluation examples
├── tests/                       # Unit tests
│   ├── conftest.py             # Test configuration
│   ├── test_document_processing.py
│   ├── test_indexing.py
│   └── test_prompts.py
├── main.py                      # CLI entry point
├── api.py                       # FastAPI REST API
├── requirements.txt
├── .env.example
└── README.md
```

## 🎯 Features

### Adaptation Techniques Overview

#### 1. Fine-tuning Approaches (Conceptual)
- **Full Fine-tuning**: Training all model parameters
- **PEFT (Parameter-efficient fine-tuning)**: Training subset of parameters
- **LoRA (Low-Rank Adaptation)**: Adding trainable rank decomposition matrices
- **Adapters**: Adding small trainable modules between layers

#### 2. Prompt Engineering
- **Zero-shot prompting**: Direct task description without examples
- **Few-shot prompting**: Including examples in the prompt
- **Chain-of-thought (CoT)**: Step-by-step reasoning
- **Role-specific prompting**: Defining AI persona and behavior
- **User-context prompting**: Personalizing based on user information

#### 3. RAG (Retrieval-Augmented Generation)
- **Retrieval**:
  - Document parsing (PDF, DOCX, HTML, TXT)
  - Chunking strategies (fixed-size, semantic, recursive)
  - Vector embeddings with multiple models
  
- **Indexing**:
  - Keyword-based indexing
  - Full-text search
  - Vector-based (ChromaDB, FAISS)
  - Hybrid search approaches
  
- **Generation**:
  - Exact and approximate nearest neighbor search
  - MMR (Maximal Marginal Relevance) for diversity
  - Context-aware prompt construction

### Evaluation Metrics
- **Context Relevance**: How relevant retrieved documents are
- **Faithfulness**: Whether responses are grounded in context
- **Answer Correctness**: Quality of generated answers
- **Answer Relevance**: How well answers address queries

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
cd /path/to/CustomerSupportChatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

#### Option 1: Interactive CLI
```bash
python main.py
```

With options:
```bash
# Use Anthropic Claude instead of OpenAI
python main.py --llm anthropic

# Use local Sentence Transformers for embeddings
python main.py --embeddings sentence_transformers

# Single query mode
python main.py --query "How do I reset my password?"

# Skip knowledge base ingestion (use existing)
python main.py --skip-ingest
```

#### Option 2: REST API (FastAPI)
```bash
# Start the API server
uvicorn api:app --reload

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

API Usage:
```bash
# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What products do you offer?"}'

# Ingest a document
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "New product documentation...", "metadata": {"source": "manual"}}'
```

#### Option 3: Python Script
```python
import asyncio
from main import ChatbotApplication

async def main():
    app = ChatbotApplication(
        llm_provider="openai",
        embedding_provider="openai",
        vector_store_type="chroma",
        search_type="hybrid",
        use_reranking=True
    )
    
    await app.initialize()
    await app.ingest_knowledge_base("data/knowledge_base")
    
    response = await app.chat("How do I reset my password?")
    print(response)

asyncio.run(main())
```

### Running Examples

```bash
# Document ingestion and indexing
python examples/01_document_ingestion.py

# Retrieval and search
python examples/02_retrieval_search.py

# Prompt engineering strategies
python examples/03_prompt_engineering.py

# RAG evaluation
python examples/04_evaluation.py
```

## 📚 Key Concepts

### RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING                              │
│  • Query understanding                                          │
│  • Query expansion                                              │
│  • Intent classification                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Keyword   │  │   Vector    │  │   Hybrid    │             │
│  │   Search    │  │   Search    │  │   Search    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RERANKING                                    │
│  • Cross-encoder reranking                                      │
│  • LLM-based reranking                                          │
│  • Reciprocal Rank Fusion                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                PROMPT CONSTRUCTION                               │
│  • System prompt with role                                      │
│  • Retrieved context injection                                  │
│  • Few-shot examples                                            │
│  • User query formatting                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION (LLM)                              │
│  • OpenAI GPT-4 / GPT-3.5                                       │
│  • Anthropic Claude                                             │
│  • Response synthesis                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING                                │
│  • Response formatting                                          │
│  • Source attribution                                           │
│  • Conversation history                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Prompt Engineering Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Zero-Shot | Direct task instruction | Simple, well-defined tasks |
| Few-Shot | Include examples | Complex format requirements |
| Chain-of-Thought | Step-by-step reasoning | Complex reasoning tasks |
| Role-Context | Persona + user context | Personalized interactions |

### Evaluation Metrics

| Metric | Description | Score Range |
|--------|-------------|-------------|
| Context Relevance | Retrieved docs relevance | 0.0 - 1.0 |
| Faithfulness | Answer grounded in context | 0.0 - 1.0 |
| Answer Correctness | Semantic similarity to truth | 0.0 - 1.0 |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_prompts.py -v

# Run with coverage
pytest tests/ --cov=src
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines.
