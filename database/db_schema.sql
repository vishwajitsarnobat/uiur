-- database/db_schema.sql
--
-- Applied automatically by Docker on first volume init via
-- /docker-entrypoint-initdb.d. For an existing container:
--   ./db.sh apply-schema

CREATE EXTENSION IF NOT EXISTS vector;

-- Flat intent structure: no parent_intent_code, no subintents.
-- The intent taxonomy is managed entirely via the intent-service API.
CREATE TABLE IF NOT EXISTS intents (
    id           SERIAL      PRIMARY KEY,
    intent_code  TEXT        NOT NULL UNIQUE,
    display_name TEXT        NOT NULL,
    description  TEXT        DEFAULT '',
    is_active    BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS intent_utterances (
    id                    BIGSERIAL   PRIMARY KEY,
    intent_id             INT         NOT NULL REFERENCES intents (id) ON DELETE CASCADE,
    utterance_raw         TEXT        NOT NULL,
    utterance_normalized  TEXT        NOT NULL,
    -- sha256(intent_code + "::" + normalized_text) -- drives idempotent re-ingestion
    utterance_hash        TEXT        NOT NULL UNIQUE,
    embedding             VECTOR(1024) NOT NULL,
    embedding_model       TEXT        NOT NULL,
    created_at            TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_utterances_intent_id
    ON intent_utterances (intent_id);

-- No HNSW/IVFFlat: a predefined banking taxonomy is realistically tens to
-- low hundreds of utterances. Exact cosine scan at that scale is
-- sub-millisecond and gives guaranteed-correct nearest neighbors.
-- Add an approximate index only if the taxonomy grows into the tens of thousands.
