"""
app/embedder.py

Pure text → vector conversion. No DB, no normalisation, no business logic.
This service's only job: receive text strings, return 1024-dim float vectors.

Offline-safe: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are set before any
import so the Hugging Face hub is never contacted at runtime.

Required files in MODEL_PATH (BAAI/bge-large-en-v1.5):
    config.json, config_sentence_transformers.json, modules.json,
    sentence_bert_config.json, special_tokens_map.json, tokenizer.json,
    tokenizer_config.json, vocab.txt, 1_Pooling/config.json,
    pytorch_model.bin  OR  model.safetensors  (~1.3 GB)

    1_Pooling/config.json is the most commonly missed file in manual
    downloads: without it sentence-transformers silently falls back to mean
    pooling instead of BGE's CLS pooling -- same dimensions, wrong vectors.
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
        device: str = "cpu",
        batch_size: int = 32,
        embedding_dim: int = 1024,
        normalize_embeddings: bool = True,
    ):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"MODEL_PATH '{model_path}' is not a directory. "
                "Point it at a local copy of the full sentence-transformers "
                "file set (see docstring for required files)."
            )

        self.model = SentenceTransformer(model_path, device=device, local_files_only=True)
        self.model_label = model_label
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.expected_dim = embedding_dim

        # get_sentence_embedding_dimension() was renamed in newer releases;
        # try the new name first, fall back to the old one.
        try:
            actual_dim = self.model.get_embedding_dimension()
        except AttributeError:
            actual_dim = self.model.get_sentence_embedding_dimension()

        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model at '{model_path}' produces {actual_dim}-dim embeddings "
                f"but EMBEDDING_DIM is set to {self.expected_dim}. "
                "bge-large=1024, bge-base=768, bge-small=384. "
                "Check you have the right checkpoint."
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts. Called for both ingestion (utterances) and
        retrieval (live query). No prefix, no instruction -- BGE s2s mode.
        Returns (N, 1024) float32 array, L2-normalised.
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
