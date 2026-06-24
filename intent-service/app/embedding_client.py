"""
app/embedding_client.py

Async HTTP client that calls the embedding service's POST /embed endpoint.

Created once at startup (main.py lifespan), stored on app.state, passed
to ingestion and retrieval modules. Never instantiate this per-request.

Returns (embeddings: np.ndarray, model_label: str) so the caller can store
the model label alongside the vectors for provenance.
"""

from typing import List, Tuple

import httpx
import numpy as np


class EmbeddingClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
        )

    async def embed(self, texts: List[str]) -> Tuple[np.ndarray, str]:
        """
        POST /embed to the embedding service.
        Returns:
          embeddings -- float32 array of shape (len(texts), dim)
          model_label -- the model name string from the service response
        Raises RuntimeError with an actionable message if the service is down.
        """
        try:
            resp = await self._client.post("/embed", json={"texts": texts})
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach embedding service at {self.base_url}. "
                "Is it running? Check EMBEDDING_SERVICE_URL in .env"
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Embedding service returned {exc.response.status_code}: "
                f"{exc.response.text}"
            )

        data = resp.json()
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        model_label = data["model"]
        return embeddings, model_label

    async def health(self) -> dict:
        """Lightweight probe -- used by /health on the intent service."""
        try:
            resp = await self._client.get("/health", timeout=5.0)
            resp.raise_for_status()
            return {"reachable": True, **resp.json()}
        except Exception as exc:
            return {"reachable": False, "error": str(exc)}

    async def close(self) -> None:
        await self._client.aclose()
