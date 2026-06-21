"""
verify_embeddings.py

Visual verification for the intent taxonomy: PCA/UMAP of intent_utterances
embeddings, colored by intent. With a small curated set you should see
TIGHT, well-separated clusters per intent -- unlike a document corpus,
there's no reason for one intent's utterances to spread out, since they're
all hand-written paraphrases of the same meaning. A spread-out or overlapping
cluster here usually means either (a) the utterances genuinely are
ambiguous with another intent, or (b) a bad example utterance got into the
taxonomy.

Usage:
    python verify_embeddings.py --method both --by leaf
    python verify_embeddings.py --method both --by parent
"""

import argparse
from typing import List

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import Config

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def fetch_embeddings(cfg: Config, color_by: str):
    conn = psycopg2.connect(cfg.dsn)
    register_vector(conn)
    label_col = "COALESCE(i.parent_intent_code, i.intent_code)" if color_by == "parent" else "i.intent_code"
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {label_col} AS label, u.utterance_raw, u.embedding
            FROM intent_utterances u
            JOIN intents i ON i.id = u.intent_id
            WHERE i.is_active = TRUE
            """
        )
        rows = cur.fetchall()
    conn.close()
    labels = [r[0] for r in rows]
    contents = [r[1] for r in rows]
    embeddings = np.vstack([np.asarray(r[2]) for r in rows]) if rows else np.empty((0, cfg.embedding_dim))
    return labels, contents, embeddings


def plot_2d(coords: np.ndarray, labels: List[str], title: str, out_path: str):
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    label_to_color = {l: cmap(i) for i, l in enumerate(unique_labels)}

    plt.figure(figsize=(10, 8))
    for l in unique_labels:
        idx = [i for i, x in enumerate(labels) if x == l]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=25, alpha=0.8, color=label_to_color[l], label=l)
    plt.title(title)
    plt.legend(markerscale=1.2, fontsize=7, loc="best", framealpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[verify] wrote {out_path}")


def run(method: str = "both", color_by: str = "leaf"):
    cfg = Config()
    labels, contents, embeddings = fetch_embeddings(cfg, color_by)
    print(f"[verify] fetched {embeddings.shape[0]} utterances across {len(set(labels))} intents")

    if embeddings.shape[0] < 5:
        print("[verify] not enough rows to visualize meaningfully, exiting")
        return

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[verify] embedding norm: mean={norms.mean():.4f} std={norms.std():.4f} (should be ~1.0)")
    if np.isnan(embeddings).any():
        print("[verify] WARNING: NaNs found in embeddings!")

    if method in ("pca", "both"):
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        plot_2d(coords, labels, f"PCA of intent utterances (by {color_by})", f"pca_intents_{color_by}.png")

    if method in ("umap", "both"):
        if not _HAS_UMAP:
            print("[verify] umap-learn not installed, skipping UMAP (pip install umap-learn)")
        else:
            n_neighbors = min(15, max(2, embeddings.shape[0] - 1))
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42)
            coords = reducer.fit_transform(embeddings)
            plot_2d(coords, labels, f"UMAP of intent utterances (by {color_by})", f"umap_intents_{color_by}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["pca", "umap", "both"], default="both")
    parser.add_argument(
        "--by", choices=["leaf", "parent"], default="leaf",
        help="color points by leaf intent code, or roll subintents up to their parent",
    )
    args = parser.parse_args()
    run(args.method, args.by)
