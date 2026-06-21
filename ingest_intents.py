"""
ingest_intents.py

Loads a YAML intent taxonomy (top-level intents + subintents, each with
example utterances), runs authoring-quality checks, normalizes text
(lowercase, same function used at query time), embeds with BGE, and writes
everything to pgvector.

Usage:
    python ingest_intents.py intents.yaml
"""

import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from config import Config
from normalizer import normalize
from quality_checks import validate_utterance, find_exact_duplicates
from embedder import BGEEmbedder
from db_writer import IntentWriter


def flatten_taxonomy(raw: dict):
    """
    Walks the YAML structure into:
      intents_meta: list of (code, parent_code, display_name, description)
      utterance_rows: list of (intent_code, utterance_text)
    covering both top-level intents and their subintents.
    """
    intents_meta: List[Tuple[str, str, str, str]] = []
    utterance_rows: List[Tuple[str, str]] = []

    for intent in raw["intents"]:
        code = intent["code"]
        intents_meta.append(
            (code, None, intent.get("display_name", code), intent.get("description", ""))
        )
        for utt in intent.get("utterances", []):
            utterance_rows.append((code, utt))

        for sub in intent.get("subintents", []):
            sub_code = sub["code"]
            intents_meta.append(
                (sub_code, code, sub.get("display_name", sub_code), sub.get("description", ""))
            )
            for utt in sub.get("utterances", []):
                utterance_rows.append((sub_code, utt))

    return intents_meta, utterance_rows


def run(yaml_path: str, cfg: Config = Config()):
    raw = yaml.safe_load(Path(yaml_path).read_text())
    intents_meta, utterance_rows = flatten_taxonomy(raw)
    print(f"[ingest_intents] {len(intents_meta)} intents/subintents, {len(utterance_rows)} utterances")

    # --- authoring-quality checks (spaCy), run BEFORE embedding ---
    normalized_pairs = [(normalize(text), code) for code, text in utterance_rows]
    dupes = find_exact_duplicates(normalized_pairs)
    if dupes:
        print("\n[ingest_intents] WARNING: duplicate utterances found:")
        for text, codes in dupes.items():
            tag = "same intent (wasted)" if len(set(codes)) == 1 else "DIFFERENT INTENTS -- conflict!"
            print(f"  '{text}' -> {codes}  [{tag}]")

    any_quality_issue = False
    for code, text in utterance_rows:
        warnings = validate_utterance(text)
        if warnings:
            any_quality_issue = True
            print(f"[ingest_intents] WARNING [{code}] '{text}': {', '.join(warnings)}")
    if not any_quality_issue:
        print("[ingest_intents] no utterance quality issues found")

    # --- normalize, embed, write ---
    embedder = BGEEmbedder(
        model_path=cfg.model_path,
        model_label=cfg.model_name,
        device=cfg.device,
        batch_size=cfg.batch_size,
        embedding_dim=cfg.embedding_dim,
    )
    writer = IntentWriter(cfg.dsn, embedding_model=cfg.model_name, embedding_dim=cfg.embedding_dim)

    writer.upsert_intents(intents_meta)

    codes = [code for code, _ in utterance_rows]
    raw_texts = [text for _, text in utterance_rows]
    norm_texts = [normalize(text) for text in raw_texts]

    embeddings = embedder.embed_text(norm_texts)

    inserted = writer.write_utterances(
        codes=codes,
        raw_texts=raw_texts,
        normalized_texts=norm_texts,
        embeddings=embeddings,
    )
    print(f"[ingest_intents] {inserted} new utterance rows inserted "
          f"({len(utterance_rows) - inserted} already existed)")
    writer.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python ingest_intents.py <intents.yaml>")
        sys.exit(1)
    run(sys.argv[1])