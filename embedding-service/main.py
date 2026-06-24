"""
main.py  --  Embedding Service

Startup:
  1. Read MODEL_PATH, EMBED_DEVICE, EMBEDDING_DIM from .env
  2. Load BGEEmbedder ONCE -- model stays resident for the lifetime of the
     process, reused by every /embed request.

This service has zero knowledge of intents, utterances, or the database.
It exists purely to convert text → vectors and can be reused by any other
service that needs embeddings.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.embedder import BGEEmbedder
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "./models/bge-large-en-v1.5")
    model_label = os.getenv("MODEL_NAME", "BAAI/bge-large-en-v1.5")
    device = os.getenv("EMBED_DEVICE", "cpu")
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

    print(f"[startup] loading model from {model_path} on {device}")
    app.state.embedder = BGEEmbedder(
        model_path=model_path,
        model_label=model_label,
        device=device,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
    )
    print(f"[startup] model ready ({embedding_dim}-dim)")
    yield
    print("[shutdown] embedding service stopped")


app = FastAPI(
    title="Embedding Service",
    description="Text → 1024-dim vector conversion. Reusable by any downstream service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
