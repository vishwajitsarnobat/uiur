-- db_schema.sql
-- Run once: psql -d ragdb -f db_schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Self-referencing taxonomy: top-level intents have parent_intent_code = NULL,
-- subintents point back to their parent's intent_code. Supports arbitrary
-- depth even though banking intents are realistically 2 levels.
CREATE TABLE IF NOT EXISTS intents (
    id                  SERIAL PRIMARY KEY,
    intent_code         TEXT NOT NULL UNIQUE,
    parent_intent_code  TEXT REFERENCES intents (intent_code),
    display_name        TEXT NOT NULL,
    description         TEXT,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS intent_utterances (
    id                    BIGSERIAL PRIMARY KEY,
    intent_id             INT NOT NULL REFERENCES intents (id) ON DELETE CASCADE,
    utterance_raw         TEXT NOT NULL,
    utterance_normalized  TEXT NOT NULL,
    utterance_hash        TEXT NOT NULL UNIQUE,   -- idempotent re-ingestion
    embedding             VECTOR(1024) NOT NULL,  -- hard requirement, see config.py
    embedding_model       TEXT NOT NULL,
    created_at            TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_utterances_intent_id ON intent_utterances (intent_id);

-- Deliberately NO HNSW/IVFFlat index here. A predefined intent taxonomy is
-- realistically tens to low hundreds of utterances -- exact cosine scan over
-- that is sub-millisecond and gives guaranteed-correct nearest neighbors,
-- which matters more for correctness testing than ANN's latency win. Add an
-- index later only if this taxonomy grows into the tens of thousands.
