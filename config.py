"""
config.py

Single source of truth for DB connection and embedding model settings.
No chunking params here -- intent utterances are atomic phrases, there's
nothing to chunk.

Reads from environment variables (with .env auto-loaded via python-dotenv
if present), so this stays in sync with docker-compose.yml's pgvector
service automatically -- both read the same .env file. If you override a
value, change .env, not the defaults below.
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()  # no-op if .env doesn't exist or python-dotenv isn't installed
except ImportError:
    pass


@dataclass
class Config:
    # --- Database (matches docker-compose.yml / .env) ---
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "ragdb"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "postgres"))

    # --- Embedding model ---
    # model_path: local folder with the full sentence-transformers file set
    # (see embedder.py's docstring for exactly what's required). No Hugging
    # Face access from this deployment -- the model is downloaded elsewhere
    # and copied in.
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "./models/bge-large-en-v1.5"))
    # model_name: human-readable label stored in the DB for provenance only --
    # NOT used to load anything. Keep it accurate.
    model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024   # hard requirement -- enforced in embedder.py and db_writer.py
    device: str = field(default_factory=lambda: os.getenv("EMBED_DEVICE", "cpu"))
    batch_size: int = 32

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )