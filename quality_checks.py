"""
quality_checks.py

spaCy's role here is much smaller than in a general document-RAG pipeline,
and a different role entirely. Utterances are short, atomic, hand-authored
phrases -- there's no sentence segmentation or chunking to do. What's
actually useful is validating the *authoring quality* of the taxonomy
before you embed it:

  - utterances with too little content to be discriminative
    (e.g. "what is my" with no object)
  - exact duplicate utterances -- wasted if duplicated within one intent,
    a real labeling contradiction if duplicated across two different intents

This runs on the YAML source BEFORE embedding. It does not touch the text
that actually gets embedded -- normalizer.py handles that, separately, with
no lemmatization or stopword removal (see that file's docstring for why).

Requires: python -m spacy download en_core_web_sm
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        # Tagger gives us real stopword/POS info cheaply; no need for NER,
        # parser, or lemmatizer for authoring-quality checks.
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    return _nlp


def content_token_count(text: str) -> int:
    """Non-stopword, non-punctuation token count. Low counts indicate an
    utterance too generic to be discriminative on its own."""
    doc = get_nlp()(text)
    return sum(1 for t in doc if not t.is_stop and not t.is_punct and not t.is_space)


def validate_utterance(text: str, min_content_tokens: int = 1, min_total_tokens: int = 2) -> List[str]:
    """Returns a list of warning strings; empty list means no issues found."""
    warnings = []
    doc = get_nlp()(text)
    total = sum(1 for t in doc if not t.is_space)
    content = content_token_count(text)

    if total < min_total_tokens:
        warnings.append(f"too short ({total} tokens)")
    if content < min_content_tokens:
        warnings.append(f"too few content words ({content}); mostly stopwords/punctuation")
    return warnings


def find_exact_duplicates(utterances: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    utterances: list of (normalized_text, intent_code) pairs.
    Returns {normalized_text: [intent_codes...]} for any text occurring more
    than once. Two intent_codes for the same text is the dangerous case --
    that utterance will pull retrieval toward whichever intent happens to
    rank marginally higher, which is a coin flip you don't want in a banking
    app.
    """
    seen: Dict[str, List[str]] = defaultdict(list)
    for text, intent_code in utterances:
        seen[text].append(intent_code)
    return {text: codes for text, codes in seen.items() if len(codes) > 1}
