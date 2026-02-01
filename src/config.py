"""
Configuration management for the Customer Support Chatbot.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"))


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    store_type: str = Field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "chroma"))
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )
    faiss_index_path: str = Field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    )


class DocumentConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096")))


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k_results: int = Field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5")))
    similarity_threshold: float = Field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    )
    search_type: str = Field(default_factory=lambda: os.getenv("SEARCH_TYPE", "mmr"))


class AppConfig(BaseModel):
    """Application configuration."""
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))


class Config(BaseModel):
    """Main configuration class aggregating all config sections."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    app: AppConfig = Field(default_factory=AppConfig)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def validate_config() -> dict:
    """Validate configuration and return status."""
    issues = []
    
    if not config.openai.api_key:
        issues.append("OPENAI_API_KEY is not set")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }
