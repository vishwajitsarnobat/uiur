"""
app/ingestion.py

Ingestion workflow (company-facing):

  JSON payload
      ↓
  validate structure + utterance quality
      ↓
  normalize utterance text
      ↓
  call embedding service  (HTTP POST /embed)
      ↓
  store intents + utterance embeddings in pgvector

Called directly by app/routes/catalog.py. No HTTP between here and the DB.
"""

from typing import Any, Dict, List

from app.embedding_client import EmbeddingClient
from app.normalizer import normalize
import app.database.repository as repo


# ---------------------------------------------------------------------------
# Input validation (no external dependency -- basic checks only)
# ---------------------------------------------------------------------------

def _validate_payload(payload: Dict) -> List[str]:
    """
    Returns a list of validation warnings. Does not raise -- the caller
    decides whether to abort or proceed with warnings.
    """
    warnings: List[str] = []
    for intent in payload.get("intents", []):
        code = intent.get("code", "")
        for utt in intent.get("utterances", []):
            norm = normalize(utt)
            if len(norm.split()) < 2:
                warnings.append(
                    f"[{code}] '{utt}': very short utterance -- may not embed distinctively"
                )
    return warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def ingest(payload: Dict[str, Any], client: EmbeddingClient, embedding_dim: int) -> Dict:
    """
    Full ingestion pipeline. Called by the /ingest route handler.

    payload format:
    {
      "intents": [
        {
          "code": "check_balance",
          "display_name": "Check Balance",
          "description": "...",          # optional
          "utterances": ["what is my balance", ...]
        },
        ...
      ]
    }

    Returns a summary dict returned directly as the API response.
    """
    intents_raw: List[Dict] = payload.get("intents", [])
    if not intents_raw:
        return {"intents_processed": 0, "utterances_processed": 0,
                "inserted": 0, "skipped": 0, "warnings": []}

    # --- validation ---
    warnings = _validate_payload(payload)

    # --- flatten to parallel lists ---
    intents_meta: List[Dict[str, str]] = []
    codes:         List[str] = []
    raw_texts:     List[str] = []
    norm_texts:    List[str] = []

    for intent in intents_raw:
        code = intent["code"]
        intents_meta.append({
            "code":         code,
            "display_name": intent.get("display_name", code),
            "description":  intent.get("description", ""),
        })
        for utt in intent.get("utterances", []):
            codes.append(code)
            raw_texts.append(utt)
            norm_texts.append(normalize(utt))

    n_utterances = len(norm_texts)

    # --- embed via HTTP (embedding service handles internal batching) ---
    embeddings, model_label = await client.embed(norm_texts)

    # --- store ---
    repo.upsert_intents(intents_meta)
    inserted = repo.write_utterances(
        codes=codes,
        raw_texts=raw_texts,
        normalized_texts=norm_texts,
        embeddings=embeddings,
        embedding_model=model_label,
        embedding_dim=embedding_dim,
    )
    skipped = n_utterances - inserted

    return {
        "intents_processed":    len(intents_meta),
        "utterances_processed": n_utterances,
        "inserted":             inserted,
        "skipped":              skipped,
        "warnings":             warnings,
    }
