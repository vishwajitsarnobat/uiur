"""
embedder.py

BGE wrapper for intent matching. Unlike a general document-RAG setup, here
both stored utterances AND live queries are short phrases being compared to
each other -- this is a symmetric (sentence-to-sentence) similarity task,
not asymmetric query-to-long-passage retrieval. BGE's documented guidance
for s2s tasks (STS, paraphrase, classification) is that the query
instruction prefix is not required on either side, as long as you're
consistent about it.

So: a single embed_text() is used for BOTH ingestion and live queries.
Do not reintroduce an asymmetric query/passage split here -- it would only
help if stored utterances were long passages, which they aren't.

LOADING FROM A LOCAL FOLDER (no Hugging Face access)
-----------------------------------------------------
model_path points at a local directory containing the FULL sentence-transformers
file set for BAAI/bge-large-en-v1.5 -- not just the raw weights. BGE uses
CLS-token pooling, not the mean pooling most sentence-transformer models
default to; that's configured in 1_Pooling/config.json, modules.json, and
sentence_bert_config.json. If those are missing, loading either fails
outright or silently falls back to the wrong pooling strategy -- same
dimensions, wrong embeddings, no error. Required files in model_path:

  config.json, config_sentence_transformers.json, modules.json,
  sentence_bert_config.json, special_tokens_map.json, tokenizer.json,
  tokenizer_config.json, vocab.txt, 1_Pooling/config.json,
  and pytorch_model.bin OR model.safetensors (~1.3GB)

model_label is separate from model_path: it's a human-readable string
("BAAI/bge-large-en-v1.5") stored in the DB's embedding_model column for
provenance, NOT used to load anything. Keep it accurate even though the
actual file is local -- it documents what you embedded with months from now.

HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set before importing
sentence_transformers, and local_files_only=True is passed explicitly, as
two independent layers against any accidental network call -- a valid local
directory path should already prevent hub lookups on its own, but on a
genuinely air-gapped machine a hung connection attempt is worse than a
redundant safeguard.

1024-dim output is enforced at construction time and again per call -- not
because the model's default dimension is in question (it isn't: BGE-large
is 1024-dim by design), but because a manually-assembled local model folder
is exactly the kind of thing that can silently end up pointing at a
different checkpoint. The check turns that into an immediate, readable
error instead of a cryptic Postgres type-mismatch several steps later.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    def __init__(
        self,
        model_path: str,
        model_label: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        embedding_dim: int = 1024,
        normalize_embeddings: bool = True,
    ):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"model_path '{model_path}' is not a directory. This must point "
                f"at a local folder containing the full sentence-transformers "
                f"file set (see this module's docstring for the required file "
                f"list) -- not a Hugging Face model id, since this deployment "
                f"has no Hugging Face access."
            )

        self.model = SentenceTransformer(model_path, device=device, local_files_only=True)
        self.model_label = model_label
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.expected_dim = embedding_dim

        actual_dim = self.model.get_embedding_dimension()
        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model at '{model_path}' produces {actual_dim}-dim embeddings, "
                f"but this pipeline requires exactly {self.expected_dim} "
                f"(see db_schema.sql's VECTOR({self.expected_dim}) column). "
                f"Double-check this is genuinely BAAI/bge-large-en-v1.5 and not "
                f"a different checkpoint (bge-base is 768-dim, bge-small is "
                f"384-dim) -- a manually assembled model folder is exactly "
                f"where that kind of mix-up happens."
            )

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Used for BOTH intent utterances (ingestion) and live queries
        (retrieval) -- symmetric s2s matching, no instruction prefix on
        either side."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(
                f"Got {embeddings.shape[1]}-dim output, expected {self.expected_dim}"
            )
        return embeddings