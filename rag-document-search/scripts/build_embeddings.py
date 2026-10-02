#!/usr/bin/env python3
"""Rebuild embed_matrix.npy from the current chunk_metadata.json.

Run after build_index.py — the chunk count changes and a stale embed_matrix.npy
will break Embedding/Hybrid search (the serving path raises on length mismatch).

Encodes every chunk's text with all-MiniLM-L6-v2 (model cached in ~/.cache/huggingface).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import index_fingerprint  # noqa: E402
REFS = REPO / "rag-document-search" / "references"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REFS
    chunks = json.load(open(out_dir / "chunk_metadata.json", encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(str(out_dir / "embed_matrix.npy"), vecs)
    index_fingerprint.write(out_dir, chunks)
    print(f"saved {len(vecs):,} embeddings ({vecs.shape}) -> {out_dir / 'embed_matrix.npy'}")


if __name__ == "__main__":
    main()
