"""
app/routes/catalog.py

Company-facing endpoints -- transport only.

  POST /ingest
      Body: JSON intent taxonomy (intents + utterances)
      Calls: ingestion.ingest()

  GET /intents
      Lists all active intents with utterance counts.

  GET /intents/{intent_code}/utterances
      Lists all utterances stored for a specific intent.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app import ingestion
import app.database.repository as repo

router = APIRouter(tags=["catalog"])


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------

class IntentPayload(BaseModel):
    code:         str            = Field(..., description="Unique intent code, e.g. 'check_balance'")
    display_name: str            = Field(..., description="Human-readable intent name")
    description:  Optional[str]  = Field(default="")
    utterances:   List[str]      = Field(..., min_length=1, description="Example phrases for this intent")


class IngestRequest(BaseModel):
    intents: List[IntentPayload] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    intents_processed:    int
    utterances_processed: int
    inserted:             int
    skipped:              int
    warnings:             List[str]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request):
    """
    Ingest intents and their example utterances. Safe to call multiple times:
    existing utterances (matched by content hash) are skipped, new ones added.
    Intent metadata (display_name, description) is updated if already present.
    """
    client = request.app.state.embedding_client
    cfg    = request.app.state.cfg

    try:
        result = await ingestion.ingest(
            payload=req.model_dump(),
            client=client,
            embedding_dim=cfg.embedding_dim,
        )
    except RuntimeError as exc:
        # Catches embedding service connectivity errors with an actionable message
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ---------------------------------------------------------------------------
# /intents
# ---------------------------------------------------------------------------

class IntentSummary(BaseModel):
    id:               int
    intent_code:      str
    display_name:     str
    description:      Optional[str]
    is_active:        bool
    utterance_count:  int
    created_at:       Any     # datetime serialised as string by FastAPI


class IntentsListResponse(BaseModel):
    count:   int
    intents: List[IntentSummary]


@router.get("/intents", response_model=IntentsListResponse)
async def list_intents():
    """List all active intents with their utterance counts."""
    intents = repo.list_intents()
    return IntentsListResponse(count=len(intents), intents=intents)


# ---------------------------------------------------------------------------
# /intents/{intent_code}/utterances
# ---------------------------------------------------------------------------

class UtteranceSummary(BaseModel):
    id:                   int
    utterance_raw:        str
    utterance_normalized: str
    created_at:           Any


class UtterancesListResponse(BaseModel):
    intent_code: str
    count:       int
    utterances:  List[UtteranceSummary]


@router.get("/intents/{intent_code}/utterances", response_model=UtterancesListResponse)
async def list_utterances(intent_code: str):
    """List all utterances stored for a specific intent."""
    utterances = repo.list_utterances(intent_code)
    if utterances is None:
        raise HTTPException(
            status_code=404,
            detail=f"Intent '{intent_code}' not found or not active."
        )
    return UtterancesListResponse(
        intent_code=intent_code,
        count=len(utterances),
        utterances=utterances,
    )
