"""
app/routes.py

Two endpoints, nothing else:
  POST /embed   -- texts → embeddings
  GET  /health  -- liveness probe

The embedder instance lives on app.state (loaded once at startup in main.py).
Route handlers are thin: validate → call embedder → serialise.
"""

from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Texts to embed.")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, request: Request):
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts list must not be empty")

    embedder = request.app.state.embedder
    try:
        vecs = embedder.embed(req.texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    return EmbedResponse(
        embeddings=vecs.tolist(),
        model=embedder.model_label,
        dim=embedder.expected_dim,
        count=len(req.texts),
    )


@router.get("/health")
async def health(request: Request):
    embedder = request.app.state.embedder
    return {
        "status": "ok",
        "model": embedder.model_label,
        "dim": embedder.expected_dim,
        "device": str(embedder.model.device),
    }
