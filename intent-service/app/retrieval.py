"""
app/retrieval.py

Retrieval workflow (user-facing):

  raw query
      ↓
  normalize
      ↓
  call embedding service  (HTTP POST /embed)
      ↓
  cosine similarity search in pgvector
      ↓
  collapse to best-per-intent
      ↓
  confidence + ambiguity check
      ↓
  ClassificationResult

ClassificationResult and IntentMatch live here because they are the
retrieval module's output contract -- routes serialise them into JSON.
"""

from dataclasses import dataclass
from typing import List, Optional

from app.embedding_client import EmbeddingClient
from app.normalizer import normalize
import app.database.repository as repo


@dataclass
class IntentMatch:
    intent_code:      str
    display_name:     str
    similarity:       float
    matched_utterance: str


@dataclass
class ClassificationResult:
    query_raw:        str
    query_normalized: str
    top:              Optional[IntentMatch]
    candidates:       List[IntentMatch]   # best-per-intent, best first
    confident:        bool                # top sim ≥ confidence_threshold
    ambiguous:        bool                # top two too close (only when confident)


async def classify(
    query: str,
    client: EmbeddingClient,
    top_k: int = 5,
    confidence_threshold: float = 0.75,
    ambiguity_margin: float = 0.03,
) -> ClassificationResult:
    """
    Main entry point for the retrieval workflow.
    Called by app/routes/user.py.
    """
    norm_query = normalize(query)

    # Single-text embed; returns (1, dim) array
    embeddings, _ = await client.embed([norm_query])
    query_vec = embeddings[0]

    # Over-fetch so we have headroom to collapse per-intent before top_k cut
    rows = repo.search_utterances(query_vec, limit=top_k * 3)

    # Collapse: keep the highest-scoring utterance per distinct intent so
    # the top-k candidates list represents distinct intents, not paraphrases.
    best: dict[str, IntentMatch] = {}
    for intent_code, display_name, utt_raw, sim in rows:
        sim = float(sim)
        if intent_code not in best or sim > best[intent_code].similarity:
            best[intent_code] = IntentMatch(
                intent_code=intent_code,
                display_name=display_name,
                similarity=sim,
                matched_utterance=utt_raw,
            )

    candidates = sorted(best.values(), key=lambda m: m.similarity, reverse=True)[:top_k]

    if not candidates:
        return ClassificationResult(query, norm_query, None, [], False, False)

    top = candidates[0]
    confident = top.similarity >= confidence_threshold
    close_second = (
        len(candidates) > 1
        and (top.similarity - candidates[1].similarity) < ambiguity_margin
    )

    return ClassificationResult(
        query_raw=query,
        query_normalized=norm_query,
        top=top,
        candidates=candidates,
        confident=confident,
        # Flag ambiguity only when we would otherwise auto-route -- low-
        # confidence results are already handled by the confident=False path.
        ambiguous=confident and close_second,
    )
