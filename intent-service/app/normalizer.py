"""
app/normalizer.py

Text normalisation applied identically at ingestion time (utterances) and
query time (user query). One function, used in both paths -- any divergence
silently breaks cosine similarity matching without raising an error.

Lightweight by design: BGE was trained on fluent natural language. Stopwords
like "to"/"from" can be the entire semantic signal in a banking context
("transfer to savings" vs "transfer from savings"), so we leave them in.
"""

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """
    Canonical normalisation:
      - unicode NFKC  (full-width / accented chars → ASCII equivalent)
      - lowercase
      - whitespace collapse

    Idempotent: normalize(normalize(x)) == normalize(x) for all x.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text
