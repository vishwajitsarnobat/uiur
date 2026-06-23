"""
query_intent.py

Runtime intent classification: takes a raw user query from the search bar,
normalizes it (same normalizer.py used at ingestion), embeds it, and matches
against stored intent utterances via cosine similarity in pgvector.

Returns the best-matching intent plus a confidence/ambiguity signal so the
caller can decide whether to act automatically or ask the user to clarify --
important here because a wrong guess (e.g. confusing "make payment" with
"check balance") has real consequences in a banking app.

Usage (as a library):
    classifier = IntentClassifier()
    result = classifier.classify("send 500 to john")
    if result.confident and not result.ambiguous:
        route_to(result.top.intent_code)
    else:
        ask_user_to_clarify(result.candidates)
"""

from dataclasses import dataclass
from typing import List, Optional

import psycopg2
from pgvector.psycopg2 import register_vector

from config import Config
from normalizer import normalize
from embedder import BGEEmbedder


@dataclass
class IntentMatch:
    intent_code: str
    parent_intent_code: Optional[str]
    display_name: str
    similarity: float
    matched_utterance: str


@dataclass
class ClassificationResult:
    query_raw: str
    query_normalized: str
    top: Optional[IntentMatch]
    candidates: List[IntentMatch]  # all candidates considered, best first
    confident: bool                # top similarity clears confidence_threshold
    ambiguous: bool                # top two candidates too close to call


class IntentClassifier:
    def __init__(
        self,
        cfg: Config = Config(),
        confidence_threshold: float = 0.75,
        ambiguity_margin: float = 0.03,
        top_k: int = 5,
    ):
        """
        confidence_threshold / ambiguity_margin are starting points, not
        calibrated values -- tune both against test_intent_matching.py's
        labeled eval set once you have real query logs.
        """
        self.cfg = cfg
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.top_k = top_k
        self.embedder = BGEEmbedder(
            model_path=cfg.model_path,
            model_label=cfg.model_name,
            device=cfg.device,
            batch_size=1,
            embedding_dim=cfg.embedding_dim,
        )
        self.conn = psycopg2.connect(cfg.dsn)
        register_vector(self.conn)

    def classify(self, raw_query: str) -> ClassificationResult:
        norm_query = normalize(raw_query)
        query_vec = self.embedder.embed_text([norm_query])[0]

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.intent_code, i.parent_intent_code, i.display_name,
                       u.utterance_raw, 1 - (u.embedding <=> %s) AS similarity
                FROM intent_utterances u
                JOIN intents i ON i.id = u.intent_id
                WHERE i.is_active = TRUE
                ORDER BY u.embedding <=> %s
                LIMIT %s
                """,
                (query_vec, query_vec, self.top_k * 3),  # over-fetch, collapse to one-per-intent below
            )
            rows = cur.fetchall()

        # Multiple utterances map to the same intent -- keep the best-scoring
        # utterance per intent, not raw top-k rows (which could all be the
        # same intent's paraphrases, hiding genuine alternative candidates).
        best_per_intent = {}
        for code, parent_code, display_name, utt, sim in rows:
            sim = float(sim)
            if code not in best_per_intent or sim > best_per_intent[code].similarity:
                best_per_intent[code] = IntentMatch(code, parent_code, display_name, sim, utt)

        candidates = sorted(best_per_intent.values(), key=lambda m: m.similarity, reverse=True)[: self.top_k]

        if not candidates:
            return ClassificationResult(raw_query, norm_query, None, [], False, False)

        top = candidates[0]
        confident = top.similarity >= self.confidence_threshold
        close_second = (
            len(candidates) > 1
            and (top.similarity - candidates[1].similarity) < self.ambiguity_margin
        )

        return ClassificationResult(
            query_raw=raw_query,
            query_normalized=norm_query,
            top=top,
            candidates=candidates,
            confident=confident,
            ambiguous=confident and close_second,  # only worth flagging if it would've been auto-routed
        )

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import sys

    classifier = IntentClassifier()
    test_queries = sys.argv[1:] or ["send 500 to john", "whats my balance", "block my card"]
    for q in test_queries:
        result = classifier.classify(q)
        print(f"\nQUERY: {q!r} (normalized: {result.query_normalized!r})")
        if not result.top:
            print("  no match found")
            continue
        flag = ""
        if not result.confident:
            flag = "  [LOW CONFIDENCE]"
        elif result.ambiguous:
            flag = "  [AMBIGUOUS -- consider asking user to clarify]"
        print(f"  -> {result.top.intent_code} (sim={result.top.similarity:.4f}){flag}")
        for c in result.candidates[1:4]:
            print(f"     also considered: {c.intent_code} (sim={c.similarity:.4f})")
    classifier.close()