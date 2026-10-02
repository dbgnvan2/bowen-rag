"""Fingerprint tying embed_matrix.npy to the exact chunk texts it was built from.

A rebuild that keeps the chunk COUNT but changes the text would otherwise pair old
embeddings with the wrong chunks and rank them silently. build_embeddings.py (and the
GUI's Build Embeddings) write embed_meta.json next to embed_matrix.npy; both apps verify
it when they load the index. A missing sidecar (an index built before this existed) is
accepted, so only a positive mismatch stops the load.
Spec:  CLAUDE.md "Building the embedding index"
Tests: test_index_consistency.py::test_idx_loaders_*embed*
"""
import hashlib
import json
from pathlib import Path

SIDECAR = "embed_meta.json"


def fingerprint(chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def write(refs_dir, chunks) -> None:
    Path(refs_dir, SIDECAR).write_text(json.dumps(
        {"chunks": len(chunks), "chunks_sha256": fingerprint(chunks)}), encoding="utf-8")


def verify(refs_dir, chunks) -> None:
    """Raise RuntimeError if the sidecar exists and does not match `chunks`."""
    p = Path(refs_dir, SIDECAR)
    if not p.exists():
        return
    meta = json.loads(p.read_text(encoding="utf-8"))
    if meta.get("chunks") != len(chunks) or meta.get("chunks_sha256") != fingerprint(chunks):
        raise RuntimeError(
            f"embed_matrix.npy in {refs_dir} was built from different chunk text than "
            "chunk_metadata.json. Rebuild the embeddings (build_embeddings.py).")


VECTORIZER = "vectorizer.json"


def saved_vocabulary(refs_dir):
    """The TF-IDF feature list the saved matrix's columns were built with, or None.

    The apps fix the vectorizer to this list instead of re-learning it: a different
    scikit-learn version can pick a different 8,000 features from the same text, which
    would pair the matrix's columns with the wrong words (or stop the app from loading).
    """
    p = Path(refs_dir, VECTORIZER)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("feature_names") or None


def verify_vectorizer(refs_dir, chunks) -> None:
    """Raise RuntimeError if vectorizer.json records a hash of different chunk text.

    Older files without the hash are accepted (only the matrix shape is checked then).
    """
    p = Path(refs_dir, VECTORIZER)
    if not p.exists():
        return
    saved = json.loads(p.read_text(encoding="utf-8")).get("chunks_sha256")
    if saved is not None and saved != fingerprint(chunks):
        raise RuntimeError(
            f"vectorizer.json in {refs_dir} does not match the chunk texts. "
            "Rebuild the index.")
