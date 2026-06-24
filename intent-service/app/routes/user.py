"""
app/routes/user.py

User-facing endpoint -- transport only.

  POST /intent_identifiers
      Body: raw user query from the search bar
      Calls: retrieval.classify()
      Returns: top intent match with confidence and ambiguity signals
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app import retrieval

router = APIRouter(tags=["user"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class IntentIdentifierRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Raw user query from the search bar.")
    # Optional per-request overrides; if omitted the service defaults apply.
    top_k:                Optional[int]   = Field(default=None, ge=1, le=20)
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ambiguity_margin:     Optional[float] = Field(default=None, ge=0.0, le=1.0)


class IntentMatchResponse(BaseModel):
    intent_code:       str
    display_name:      str
    similarity:        float
    matched_utterance: str


class IntentIdentifierResponse(BaseModel):
    query_raw:        str
    query_normalized: str
    top:              Optional[IntentMatchResponse]
    candidates:       List[IntentMatchResponse]
    confident:        bool
    ambiguous:        bool


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/intent_identifiers", response_model=IntentIdentifierResponse)
async def identify_intent(req: IntentIdentifierRequest, request: Request):
    """
    Classify a user query against the stored intent taxonomy.

    Response fields:
      top       -- best matching intent (None if DB is empty)
      candidates -- top-k distinct intents considered, best first
      confident  -- True if top similarity ≥ confidence_threshold
      ambiguous  -- True if top two candidates are too close to auto-route
                    (only relevant when confident=True)

    Routing logic for the caller:
      confident=True,  ambiguous=False → auto-route to top.intent_code
      confident=True,  ambiguous=True  → show top 2 options, ask user to confirm
      confident=False                  → show a clarification prompt or fallback
    """
    client = request.app.state.embedding_client
    cfg    = request.app.state.cfg

    try:
        result = await retrieval.classify(
            query=req.query,
            client=client,
            top_k=req.top_k or cfg.top_k,
            confidence_threshold=req.confidence_threshold or cfg.confidence_threshold,
            ambiguity_margin=req.ambiguity_margin or cfg.ambiguity_margin,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IntentIdentifierResponse(
        query_raw=result.query_raw,
        query_normalized=result.query_normalized,
        top=IntentMatchResponse(**vars(result.top)) if result.top else None,
        candidates=[IntentMatchResponse(**vars(c)) for c in result.candidates],
        confident=result.confident,
        ambiguous=result.ambiguous,
    )
