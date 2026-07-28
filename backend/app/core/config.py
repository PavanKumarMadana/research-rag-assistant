"""
Application Configuration Module.

This module provides centralized configuration management using Pydantic Settings.
All environment variables are loaded and validated here.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "AI Research & Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Enterprise AI Research Assistant with RAG capabilities"
    DEBUG: bool = Field(default=False, validation_alias="APP_DEBUG")
    ENVIRONMENT: str = Field(default="development")

    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # Database
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./data/research_assistant.db"
    )
    DATABASE_ECHO: bool = Field(default=False)

    # Vector Store
    VECTOR_DB_PATH: str = Field(default="./data/chromadb")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # LLM Configuration
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    LLM_PROVIDER: str = Field(default="gemini")
    LLM_MODEL: str = Field(default="gemini-2.0-flash")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    TEMPERATURE: float = Field(default=0.3)
    MAX_TOKENS: int = Field(default=2048)

    # TensorFlow Model
    MODEL_PATH: str = Field(default="./models/document_classifier")
    NUM_CLASSES: int = Field(default=8)
    MAX_SEQUENCE_LENGTH: int = Field(default=512)

    # File Upload
    UPLOAD_DIR: str = Field(default="./data/uploads")
    MAX_UPLOAD_SIZE: int = Field(default=50 * 1024 * 1024)  # 50MB
    ALLOWED_EXTENSIONS: list[str] = Field(default=[".pdf"])

    # CORS
    CORS_ORIGINS: list[str] = Field(default=["*"])

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="./data/app.log")

    # Analytics
    ANALYTICS_ENABLED: bool = Field(default=True)

    # Search
    TOP_K_RESULTS: int = Field(default=5)
    SIMILARITY_THRESHOLD: float = Field(default=0.25)
    MMR_LAMBDA: float = Field(default=0.7)
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = Field(default=6)


settings = Settings()

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")), exist_ok=True)
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
