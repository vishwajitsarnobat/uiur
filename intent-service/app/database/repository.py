"""
app/database/repository.py

All pgvector read/write operations. No business logic -- that lives in
ingestion.py and retrieval.py. Callers never manage connections directly;
every function borrows from the pool and returns it in a finally block.

Flat intent structure: no parent_intent_code, no subintents.
"""

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from psycopg2.extras import execute_values, RealDictCursor

from app.database.connection import get_conn, release_conn


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

def upsert_intents(
    intents: List[Dict[str, str]]
) -> None:
    """
    intents: list of dicts with keys: code, display_name, description.
    Updates display_name and description if the code already exists.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for intent in intents:
                cur.execute(
                    """
                    INSERT INTO intents (intent_code, display_name, description)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (intent_code) DO UPDATE
                        SET display_name = EXCLUDED.display_name,
                            description  = EXCLUDED.description
                    """,
                    (intent["code"], intent["display_name"], intent.get("description", "")),
                )
        conn.commit()
    finally:
        release_conn(conn)


def list_intents() -> List[Dict[str, Any]]:
    """
    Returns all active intents with utterance counts.
    Used by GET /intents.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT  i.id,
                        i.intent_code,
                        i.display_name,
                        i.description,
                        i.is_active,
                        i.created_at,
                        COUNT(u.id) AS utterance_count
                FROM    intents i
                LEFT JOIN intent_utterances u ON u.intent_id = i.id
                WHERE   i.is_active = TRUE
                GROUP BY i.id
                ORDER BY i.intent_code
                """
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_conn(conn)


def list_utterances(intent_code: str) -> Optional[List[Dict[str, Any]]]:
    """
    Returns all utterances for a given intent_code, or None if the intent
    doesn't exist (so the route can return 404 rather than an empty list).
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify intent exists first
            cur.execute(
                "SELECT id FROM intents WHERE intent_code = %s AND is_active = TRUE",
                (intent_code,),
            )
            if cur.fetchone() is None:
                return None

            cur.execute(
                """
                SELECT  u.id,
                        u.utterance_raw,
                        u.utterance_normalized,
                        u.created_at
                FROM    intent_utterances u
                JOIN    intents i ON i.id = u.intent_id
                WHERE   i.intent_code = %s
                ORDER BY u.id
                """,
                (intent_code,),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_conn(conn)


def get_intent_code_to_id() -> Dict[str, int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT intent_code, id FROM intents")
            return dict(cur.fetchall())
    finally:
        release_conn(conn)


# ---------------------------------------------------------------------------
# Utterances
# ---------------------------------------------------------------------------

def write_utterances(
    codes: Sequence[str],
    raw_texts: Sequence[str],
    normalized_texts: Sequence[str],
    embeddings: np.ndarray,
    embedding_model: str,
    embedding_dim: int = 1024,
) -> int:
    """
    Batch-insert utterance embeddings. Idempotent via ON CONFLICT on
    utterance_hash (sha256 of intent_code + normalized_text).
    Returns the number of newly inserted rows.
    """
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Got {embeddings.shape[1]}-dim embeddings, "
            f"schema requires {embedding_dim}."
        )

    code_to_id = get_intent_code_to_id()
    rows = []
    for code, raw, norm, emb in zip(codes, raw_texts, normalized_texts, embeddings):
        intent_id = code_to_id.get(code)
        if intent_id is None:
            raise ValueError(
                f"Intent code '{code}' not found in intents table. "
                "upsert_intents() must be called before write_utterances()."
            )
        utterance_hash = hashlib.sha256(f"{code}::{norm}".encode()).hexdigest()
        rows.append((intent_id, raw, norm, utterance_hash, emb, embedding_model))

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            inserted = execute_values(
                cur,
                """
                INSERT INTO intent_utterances
                    (intent_id, utterance_raw, utterance_normalized,
                     utterance_hash, embedding, embedding_model)
                VALUES %s
                ON CONFLICT (utterance_hash) DO NOTHING
                RETURNING id
                """,
                rows,
                fetch=True,
            )
        conn.commit()
        return len(inserted)
    finally:
        release_conn(conn)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_utterances(
    query_vec: np.ndarray, limit: int
) -> List[Tuple[str, str, float]]:
    """
    Cosine similarity search.
    Returns list of (intent_code, display_name, utterance_raw, similarity).
    <=> is pgvector's cosine-distance operator; ORDER BY ASC = most similar first.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT  i.intent_code,
                        i.display_name,
                        u.utterance_raw,
                        1 - (u.embedding <=> %s) AS similarity
                FROM    intent_utterances u
                JOIN    intents i ON i.id = u.intent_id
                WHERE   i.is_active = TRUE
                ORDER BY u.embedding <=> %s
                LIMIT   %s
                """,
                (query_vec, query_vec, limit),
            )
            return cur.fetchall()
    finally:
        release_conn(conn)
