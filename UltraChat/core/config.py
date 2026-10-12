import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # App
    APP_NAME: str = "LLM Intelligent Router Thesis API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM Provider API Keys
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    # Qdrant Semantic Cache (Local disk storage by default)
    QDRANT_STORAGE_PATH: str = str(BASE_DIR / "qdrant_storage")
    QDRANT_LOCATION: Optional[str] = None
    QDRANT_HOST: Optional[str] = None
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "llm_prompt_cache"
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.92

    # Evaluator Deep Learning Models
    EVALUATOR_WEIGHTS_DIR: str = str(BASE_DIR / "evaluator_weights")
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    HIDDEN_DIM: int = 128

    # Routing and Evaluator Policies
    DEFAULT_ROUTING_STRATEGY: str = "balanced"  # balanced, best_quality, cost_optimized, instruction_focused
    FALLBACK_MODEL: str = "gemini-3.5-flash-lite"
    VALIDATOR_MODEL: str = "gemini-3.5-flash-lite"
    MAX_EVALUATOR_RETRIES: int = 2

    @field_validator("EVALUATOR_WEIGHTS_DIR", mode="after")
    @classmethod
    def resolve_evaluator_weights_dir(cls, v: str) -> str:
        path = Path(v)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        return str(path)

    @field_validator("QDRANT_STORAGE_PATH", mode="after")
    @classmethod
    def resolve_qdrant_storage_path(cls, v: str) -> str:
        path = Path(v)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        return str(path)

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
