"""
app/config.py

All configuration for the intent service.
Reads from .env via python-dotenv (loaded in main.py before this is imported).
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Database ---
    db_host:     str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port:     int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name:     str = field(default_factory=lambda: os.getenv("DB_NAME", "ragdb"))
    db_user:     str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "postgres"))
    db_pool_min: int = 1
    db_pool_max: int = 10

    # --- Embedding service (called via HTTP) ---
    embedding_service_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    # Timeout in seconds for HTTP calls to the embedding service.
    # Ingestion of many utterances can be slow; keep this generous.
    embedding_timeout: float = field(
        default_factory=lambda: float(os.getenv("EMBEDDING_TIMEOUT", "60.0"))
    )

    # --- Retrieval ---
    top_k:                int   = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")))
    ambiguity_margin:     float = field(default_factory=lambda: float(os.getenv("AMBIGUITY_MARGIN", "0.03")))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
