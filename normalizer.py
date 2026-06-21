"""
normalizer.py

Shared text normalization for BOTH ingestion (intent utterances) and
retrieval (live user queries). Using the exact same function on both sides
is what makes "lowercase everything" actually hold over time -- any drift
between ingestion-time and query-time normalization silently breaks matching,
and that kind of bug is invisible until production traffic hits it.

Deliberately lightweight. Banking queries are short, and aggressive NLP
(lemmatization, stopword removal) tends to hurt sentence-embedding quality,
since the embedding model was trained on fluent natural language, not
mangled stems. We also can't safely strip stopwords here -- "transfer *to*
savings" and "transfer *from* savings" differ only in a stopword, and that
difference is the entire intent signal.
"""

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str, strip_punctuation: bool = False) -> str:
    """
    Canonical normalization, applied identically at ingestion and query time.

    - unicode NFKC normalization (e.g. full-width digits/punctuation -> ASCII)
    - lowercasing
    - whitespace collapsing
    - optional punctuation stripping (off by default; "?" rarely hurts BGE
      embeddings, and turning this on is one more place ingestion and query
      normalization could silently drift apart if you forget to flip it on
      both sides -- if you do enable it, enable it in exactly one place
      this function is called from, not per-caller)
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    if strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
