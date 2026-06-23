"""
test_intent_matching.py

Functional correctness checks for the intent taxonomy -- these catch things
PCA/UMAP plots can't:

  1. Self-retrieval -- every stored utterance should retrieve its own intent
     as the top match. Failure = DB/index/normalization bug, not a taxonomy
     problem.
  2. Cross-intent ambiguity -- utterance pairs belonging to DIFFERENT intents
     that are nonetheless highly similar. This is the most important check
     for this domain specifically: it doesn't point at a code bug, it points
     at two intents whose example phrasing is too close for the embedding
     model to separate, which WILL cause real misclassifications in
     production.
  3. Labeled eval set -- hand-written (query, expected_intent_code) pairs run
     through the actual classify() path. Treat this as a regression suite:
     add a case every time you find a real misclassification in production.

Usage:
    python test_intent_matching.py
"""

from itertools import combinations
from typing import List, Tuple

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from config import Config
from embedder import BGEEmbedder
from query_intent import IntentClassifier


def check_self_retrieval(cfg: Config, embedder: BGEEmbedder):
    conn = psycopg2.connect(cfg.dsn)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.utterance_normalized, i.intent_code
            FROM intent_utterances u
            JOIN intents i ON i.id = u.intent_id
            """
        )
        rows = cur.fetchall()

        passed = 0
        for row_id, text, intent_code in rows:
            vec = embedder.embed_text([text])[0]
            cur.execute(
                """
                SELECT u.id, 1 - (u.embedding <=> %s) AS sim
                FROM intent_utterances u
                ORDER BY u.embedding <=> %s
                LIMIT 1
                """,
                (vec, vec),
            )
            top_id, sim = cur.fetchone()
            ok = top_id == row_id and sim > 0.999
            passed += int(ok)
            if not ok:
                print(f"[self-retrieval] FAIL id={row_id} intent={intent_code} top_match={top_id} sim={sim:.4f}")
    conn.close()
    print(f"[self-retrieval] {passed}/{len(rows)} passed")


def check_cross_intent_ambiguity(cfg: Config, threshold: float = 0.92):
    conn = psycopg2.connect(cfg.dsn)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT u.id, i.intent_code, u.utterance_raw, u.embedding
            FROM intent_utterances u
            JOIN intents i ON i.id = u.intent_id
            """
        )
        rows = cur.fetchall()
    conn.close()

    if len(rows) < 2:
        print("[cross-intent-ambiguity] not enough rows to check")
        return

    ids = [r[0] for r in rows]
    codes = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    embeddings = np.vstack([np.asarray(r[3]) for r in rows])
    sims = embeddings @ embeddings.T

    flagged = 0
    for i, j in combinations(range(len(ids)), 2):
        if codes[i] == codes[j]:
            continue  # same intent -- high similarity is expected and fine
        if sims[i, j] > threshold:
            flagged += 1
            print(
                f"[ambiguity] {sims[i, j]:.4f}  "
                f"[{codes[i]}] '{texts[i]}'  <->  [{codes[j]}] '{texts[j]}'"
            )
    print(f"[cross-intent-ambiguity] {flagged} cross-intent pairs above {threshold} similarity")
    if flagged:
        print(
            "  -> these intent pairs may need more distinctive example "
            "utterances, or genuinely overlap and should be merged."
        )


def run_labeled_eval(classifier: IntentClassifier, labeled_queries: List[Tuple[str, str]]):
    correct = 0
    for query, expected in labeled_queries:
        result = classifier.classify(query)
        got = result.top.intent_code if result.top else None
        ok = got == expected
        correct += int(ok)
        status = "OK  " if ok else "FAIL"
        conf_flag = "" if result.confident else " [low-confidence]"
        amb_flag = " [ambiguous]" if result.ambiguous else ""
        sim = result.top.similarity if result.top else 0.0
        print(f"[{status}] '{query}' -> got={got} expected={expected} sim={sim:.4f}{conf_flag}{amb_flag}")
    print(f"\n[labeled-eval] {correct}/{len(labeled_queries)} correct")


if __name__ == "__main__":
    cfg = Config()
    embedder = BGEEmbedder(
        model_path=cfg.model_path,
        model_label=cfg.model_name,
        device=cfg.device,
        batch_size=cfg.batch_size,
        embedding_dim=cfg.embedding_dim,
    )

    check_self_retrieval(cfg, embedder)
    print()
    check_cross_intent_ambiguity(cfg)

    # EDIT this list to match your real taxonomy -- this is your regression
    # suite, keep growing it as you find real production failures.
    labeled_queries = [
        ("what's my balance", "check_balance"),
        ("send 500 rupees to john", "make_payment"),
        ("block my debit card", "block_card"),
        ("how much is in my savings", "check_balance_savings"),
        ("transfer to my friend's mobile number", "make_payment_mobile"),
    ]
    print()
    classifier = IntentClassifier(cfg)
    run_labeled_eval(classifier, labeled_queries)
    classifier.close()