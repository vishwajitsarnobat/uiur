"""
main.py  --  Intent Service

Startup:
  1. Load config from .env
  2. Initialise DB connection pool
  3. Create EmbeddingClient (async httpx, kept alive for the process lifetime)
  4. Register routes

Shutdown:
  Close httpx client and DB pool cleanly.

Endpoints:
  Company-facing:
    POST /ingest                              -- ingest intents + utterances
    GET  /intents                             -- list all active intents
    GET  /intents/{intent_code}/utterances    -- list utterances for an intent

  User-facing:
    POST /intent_identifiers                  -- classify a user query

  Infrastructure:
    GET  /health                              -- DB + embedding service probe

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()

from app.config import Config
from app.database import connection as db_pool
from app.embedding_client import EmbeddingClient
from app.routes.catalog import router as catalog_router
from app.routes.user import router as user_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config()
    app.state.cfg = cfg

    # DB pool
    print(f"[startup] connecting to DB {cfg.db_host}:{cfg.db_port}/{cfg.db_name}")
    db_pool.init_pool(cfg.dsn, minconn=cfg.db_pool_min, maxconn=cfg.db_pool_max)
    print("[startup] DB pool ready")

    # Embedding client -- created once, stays alive
    print(f"[startup] embedding service → {cfg.embedding_service_url}")
    app.state.embedding_client = EmbeddingClient(
        base_url=cfg.embedding_service_url,
        timeout=cfg.embedding_timeout,
    )
    print("[startup] embedding client ready")

    yield

    # Shutdown
    await app.state.embedding_client.close()
    db_pool.close_pool()
    print("[shutdown] intent service stopped")


app = FastAPI(
    title="Intent Service",
    description=(
        "Intent ingestion and classification for banking search. "
        "Company-facing: /ingest, /intents. "
        "User-facing: /intent_identifiers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(catalog_router)
app.include_router(user_router)


@app.get("/health", tags=["infrastructure"])
async def health(request: Request):
    """Probes both the DB pool and the embedding service."""
    db_ok = False
    try:
        conn = db_pool.get_conn()
        db_pool.release_conn(conn)
        db_ok = True
    except Exception as exc:
        print(f"[health] DB check failed: {exc}")

    embed_status = await request.app.state.embedding_client.health()

    overall = "ok" if db_ok and embed_status.get("reachable") else "degraded"
    return {
        "status":            overall,
        "db":                db_ok,
        "embedding_service": embed_status,
    }
