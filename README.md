# Intent Matching Pipeline (BGE-large, 1024-dim + pgvector + spaCy)

For a banking search bar: predefined intents/subintents are embedded once
from a YAML taxonomy; live user queries are normalized and embedded the same
way at request time, then matched via cosine similarity.

## Setup

**1. Start Postgres/pgvector via Docker** (schema auto-applies on first run):

```bash
cp .env.example .env        # already filled with working local-dev defaults
sudo docker-compose up -d --wait # or: ./db.sh up
```

This pulls `pgvector/pgvector:pg16` (Postgres with the pgvector extension
preinstalled), creates the `ragdb_pgvector` container, and runs
`db_schema.sql` automatically via Docker's `/docker-entrypoint-initdb.d`
mount the first time the data volume is created. No local Postgres install,
no manual `psql -f db_schema.sql` step.

Quick checks:

```bash
sudo docker-compose ps                 # should show pgvector as healthy
sudo ./db.sh apply-schema
sudo ./db.sh shell                     # drops you into psql inside the container
  \dt                              #   -> should list intents, intent_utterances
  \q
```

If you edit `db_schema.sql` *after* the container has already initialized,
that mount won't re-run automatically (it only fires against an empty
volume) -- use `./db.sh apply-schema` to re-apply it to the running
container, or `./db.sh reset` to wipe the volume and start clean.

**2. Python environment** (runs on your host, connects to the Dockerized DB
over `localhost:5432`):

```bash
uv add -r requirements.txt
uv run python -m spacy download en_core_web_sm
```

`config.py` reads DB credentials from the same `.env` file docker-compose
used, via `python-dotenv` -- there's nothing to configure twice.

## Manually downloading the model (no Hugging Face access)

`embedder.py` loads from a local folder (`MODEL_PATH` in `.env`), never from
the Hugging Face hub directly -- `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are
set and `local_files_only=True` is passed, so it won't even attempt a
network call.

**On a machine that does have access**, get the complete file set -- not
just the weights:

```bash
uv add huggingface_hub
hf download BAAI/bge-large-en-v1.5 --local-dir ./bge-large-en-v1.5
```

(or `git clone https://huggingface.co/BAAI/bge-large-en-v1.5` with git-lfs
installed). Either way you need the **entire** folder, ~1.3GB, including:

```
config.json
config_sentence_transformers.json
modules.json
sentence_bert_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.txt
pytorch_model.bin   (or model.safetensors)
1_Pooling/config.json
```

`1_Pooling/config.json` is the one most likely to get missed if someone
grabs "just the model file" by hand -- it's what tells sentence-transformers
to use BGE's CLS-token pooling instead of the mean pooling most
sentence-transformer models default to. Missing it doesn't always error;
it can silently produce different (wrong) embeddings instead, which is a
much worse failure mode than a crash.

**Transfer the whole folder** (USB / internal file share / however your
org moves files across the air gap) to wherever the pipeline runs, then
point `MODEL_PATH` in `.env` at it:

```
MODEL_PATH=/path/to/bge-large-en-v1.5
```

`embedder.py` will refuse to start with a clear `FileNotFoundError` if
`MODEL_PATH` isn't a real directory, and will refuse with a clear
`ValueError` if whatever's loaded doesn't produce 1024-dim output --
both fail fast, before touching the database, rather than failing deep
inside a Postgres error.

## Windows / restricted-environment notes

- **`db.sh` needs a POSIX shell** (Git Bash, or WSL) -- it won't run in
  plain PowerShell or cmd.exe. It's a convenience wrapper, not a
  requirement: every command it runs is a plain `docker compose ...`
  invocation, which works identically typed directly into PowerShell. If
  you don't have Git Bash/WSL, just run the `docker compose` commands
  shown in `db.sh` directly instead.
- **Docker Desktop with the WSL2 backend** handles the relative bind mount
  (`./db_schema.sql:/docker-entrypoint-initdb.d/...`) correctly as long as
  your project folder is somewhere Docker Desktop's file sharing can see --
  either inside the WSL filesystem, or on a Windows drive you've enabled
  under Settings > Resources > File sharing.
- **If you can't pull arbitrary images** (common on a locked-down company
  PC), see `DOCKER_IMAGE_REQUEST.md` for exactly what to request and why --
  it's written to be handed directly to whoever provisions the container
  for you.

## Run order

```bash
# 1. Ingest the taxonomy (edit intents.yaml first to match your real intents)
uv run ingest_intents.py intents.yaml

# 2. Visual sanity check -- writes pca_intents_leaf.png / umap_intents_leaf.png
uv run verify_embeddings.py --method both --by leaf
uv run verify_embeddings.py --method both --by parent   # rolled up to top-level intent

# 3. Functional correctness: self-retrieval, cross-intent ambiguity, labeled eval
#    (edit labeled_queries in test_intent_matching.py to match your taxonomy)
uv run test_intent_matching.py

# 4. Try the live classifier directly
uv run query_intent.py "send 500 to john" "whats my balance" "block my card"
```

## Module map

| File                       | Responsibility                                                  |
|-----------------------------|-------------------------------------------------------------------|
| `docker-compose.yml`         | Postgres + pgvector container, auto-applies `db_schema.sql`      |
| `DOCKER_IMAGE_REQUEST.md`    | What to hand IT/infra to get the image whitelisted/provisioned   |
| `.env` / `.env.example`      | Single source of truth for DB credentials -- read by both Docker and Python|
| `db.sh`                       | Convenience wrapper: up / down / reset / shell / logs / apply-schema|
| `config.py`                 | DB / model settings, loaded from `.env`                          |
| `db_schema.sql`              | `intents` + `intent_utterances` tables, no ANN index (see below)|
| `intents.yaml`               | The actual intent/subintent taxonomy -- edit this                |
| `normalizer.py`              | Shared lowercase/normalize fn, used at BOTH ingestion and query  |
| `quality_checks.py`          | spaCy authoring-quality checks on the taxonomy (not on embed text)|
| `embedder.py`                | BGE wrapper, symmetric s2s embedding, hard 1024-dim enforcement  |
| `db_writer.py`               | Idempotent batch insert of intents + utterance embeddings        |
| `ingest_intents.py`          | Orchestrates: load YAML -> validate -> normalize -> embed -> write|
| `verify_embeddings.py`       | PCA/UMAP plots colored by intent                                  |
| `test_intent_matching.py`    | Self-retrieval, cross-intent ambiguity, labeled accuracy eval     |
| `query_intent.py`            | Runtime classifier: confidence threshold + ambiguity margin       |

## Design decisions specific to this use case (vs. a general document-RAG pipeline)

- **No chunking.** Utterances are atomic phrases; chunking/sentence-splitting
  modules from a document pipeline don't apply and aren't here.
- **spaCy validates authoring quality, not embedding input.** It runs on the
  YAML source to catch too-short/low-content utterances and duplicates
  (including duplicates that conflict across two different intents). It does
  NOT lemmatize or strip stopwords from the text that gets embedded --
  stopwords like "to"/"from" can be the entire intent signal, and lemmatized
  text is out-of-distribution for a model trained on fluent language.
- **One normalize() function, used identically at ingestion and query time**
  (`normalizer.py`). This is what makes "lowercase everything" actually hold
  in production instead of drifting between the two code paths.
- **No query/passage asymmetry.** Intent matching is query-vs-query
  (symmetric s2s), not query-vs-long-passage. `embedder.py` exposes a single
  `embed_text()` used on both sides -- do not add BGE's retrieval query
  prefix back in here, it's the wrong convention for this task.
- **1024-dim is enforced, not just configured.** `BGEEmbedder` raises at
  construction if the model isn't genuinely 1024-dim; `IntentWriter`
  re-checks before insert.
- **No HNSW/IVFFlat index.** At realistic taxonomy size (tens to low
  hundreds of utterances), exact cosine scan is fast and exact. Add an index
  only if this grows into the tens of thousands of utterances.
- **Confidence threshold + ambiguity margin in `query_intent.py`.** A wrong
  intent guess in a banking app has real consequences, so the classifier
  distinguishes "confident single match," "low confidence," and "ambiguous
  (top two candidates too close)" rather than always committing to a guess.
  Both thresholds are starting points -- tune them against
  `test_intent_matching.py`'s labeled eval set once you have real query logs.
- **Postgres runs in Docker, nothing else does.** The Python scripts still
  run directly on your host (or wherever you actually run ingestion/queries
  from) and connect to the container over `localhost:${DB_PORT}`. Only the
  database itself is containerized -- that's what "we're only allowed to use
  Docker for that" means in practice here. If you later need the Python side
  containerized too (e.g. to deploy `query_intent.py` as a service), that's
  a separate Dockerfile/service, not a change to this one.
- **One `.env` file, read by both docker-compose and `config.py`.** This is
  the same principle as `normalizer.py` being shared between ingestion and
  query time: one place defines truth, both consumers read it, so they can't
  silently drift apart. Don't hardcode a different value in `docker-compose.yml`
  or `config.py` -- change `.env`.