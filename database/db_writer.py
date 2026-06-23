"""
db_writer.py

Writes the intent taxonomy (intents table) and intent utterances +
embeddings (intent_utterances table) to pgvector.

- Intents are upserted by intent_code -- re-running ingestion after editing
  display names/descriptions updates them in place.
- Utterances are idempotent on utterance_hash -- re-running ingestion over
  an unchanged YAML file inserts zero new rows.
- Embedding dimension is re-validated here, even though embedder.py already
  enforces it -- fail fast with a clear message rather than a cryptic
  psycopg2 type error from the VECTOR(1024) column constraint.
"""

import hashlib
from typing import List, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


class IntentWriter:
    def __init__(self, dsn: str, embedding_model: str, embedding_dim: int = 1024):
        self.conn = psycopg2.connect(dsn)
        register_vector(self.conn)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    def upsert_intents(self, intents_meta: List[Tuple[str, str, str, str]]):
        """intents_meta: list of (code, parent_code, display_name, description).
        Inserted in two passes so the parent_intent_code FK never points at a
        not-yet-inserted row."""
        with self.conn.cursor() as cur:
            for code, parent_code, display_name, description in intents_meta:
                if parent_code is None:
                    cur.execute(
                        """
                        INSERT INTO intents (intent_code, parent_intent_code, display_name, description)
                        VALUES (%s, NULL, %s, %s)
                        ON CONFLICT (intent_code) DO UPDATE
                            SET display_name = EXCLUDED.display_name,
                                description = EXCLUDED.description
                        """,
                        (code, display_name, description),
                    )
            for code, parent_code, display_name, description in intents_meta:
                if parent_code is not None:
                    cur.execute(
                        """
                        INSERT INTO intents (intent_code, parent_intent_code, display_name, description)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (intent_code) DO UPDATE
                            SET parent_intent_code = EXCLUDED.parent_intent_code,
                                display_name = EXCLUDED.display_name,
                                description = EXCLUDED.description
                        """,
                        (code, parent_code, display_name, description),
                    )
        self.conn.commit()

    def _intent_code_to_id(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT intent_code, id FROM intents")
            return dict(cur.fetchall())

    def write_utterances(
        self,
        codes: Sequence[str],
        raw_texts: Sequence[str],
        normalized_texts: Sequence[str],
        embeddings: np.ndarray,
    ) -> int:
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Got {embeddings.shape[1]}-dim embeddings, schema requires "
                f"exactly {self.embedding_dim}. Refusing to write."
            )

        code_to_id = self._intent_code_to_id()
        rows = []
        for code, raw, norm, emb in zip(codes, raw_texts, normalized_texts, embeddings):
            intent_id = code_to_id.get(code)
            if intent_id is None:
                raise ValueError(
                    f"Intent code '{code}' not found in intents table -- "
                    f"call upsert_intents() before write_utterances()"
                )
            utterance_hash = hashlib.sha256(f"{code}::{norm}".encode()).hexdigest()
            rows.append((intent_id, raw, norm, utterance_hash, emb, self.embedding_model))

        with self.conn.cursor() as cur:
            inserted = execute_values(
                cur,
                """
                INSERT INTO intent_utterances
                    (intent_id, utterance_raw, utterance_normalized, utterance_hash,
                     embedding, embedding_model)
                VALUES %s
                ON CONFLICT (utterance_hash) DO NOTHING
                RETURNING id
                """,
                rows,
                fetch=True,
            )
        self.conn.commit()
        return len(inserted)

    def close(self):
        self.conn.close()
